# kila_sm.py
# FastAPI app for handling Instagram DMs via ManyChat -> Social KILA
# Uses GPT-4 + Pinecone for persistent conversation memory
# Enhanced with booking link functionality

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone, PineconeApiException

# Import calendar functionality
from cal_api import router as calendar_router
from cal_help import detect_calendar_intent, process_calendar_intent

# ========================
# CONFIGURATION
# ========================

# Load environment variables (make sure these are set in your .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agently-memory")

# Message limits based on follower status
MAX_MESSAGES_FOLLOWER = 150  # Store more messages for followers
MAX_MESSAGES_NON_FOLLOWER = 50  # Store fewer messages for non-followers

# Maximum response length (characters)
MAX_RESPONSE_LENGTH = 900  # Setting to 900 to stay within Instagram's limits

# Your Google Calendar Appointment link
BOOKING_LINK = "https://calendar.app.google/GRWJ1537Uh7Kz1nB7"

# Validate required environment variables
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment variables")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment variables")
if not PINECONE_ENVIRONMENT:
    raise RuntimeError("Missing PINECONE_ENVIRONMENT in environment variables")

# Initialize clients
try:
    print(f"[INIT] Initializing OpenAI client...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print(f"[INIT] Initializing Pinecone client... (Index: {PINECONE_INDEX_NAME})")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    try:
        index_list = pc.list_indexes()
        if PINECONE_INDEX_NAME not in [index.name for index in index_list.indexes]:
            print(f"[WARNING] Pinecone index '{PINECONE_INDEX_NAME}' not found! Available indexes: {[index.name for index in index_list.indexes]}")
            raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist")
        else:
            print(f"[INIT] Pinecone index '{PINECONE_INDEX_NAME}' found and ready to use")
    except Exception as e:
        print(f"[ERROR] Failed to list Pinecone indexes: {str(e)}")
        raise
    
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    EMBEDDINGS = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    print(f"[INIT] Testing Pinecone connection...")
    # Test Pinecone connection with a simple operation
    try:
        stats = pinecone_index.describe_index_stats()
        print(f"[INIT] Pinecone connection successful. Index stats: {stats}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Pinecone: {str(e)}")
        raise
    
except Exception as e:
    print(f"[CRITICAL] Error during initialization: {str(e)}")
    raise

# Load system prompt
try:
    with open("kila_sm.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
    print(f"[INIT] System prompt loaded successfully ({len(SYSTEM_PROMPT)} characters)")
except Exception as e:
    print(f"[ERROR] Failed to load system prompt: {str(e)}")
    SYSTEM_PROMPT = "You are KILA, a helpful assistant."  # Fallback prompt

# ========================
# DATA MODELS
# ========================

class DMRequest(BaseModel):
    name: str = Field(..., description="User's full name")
    username: str = Field(..., description="Instagram username")
    message: str = Field(..., description="Current message from user")
    isFollower: str = Field(..., description="Whether user follows the account (1=yes, 0=no)")
    followCt: str = Field(..., description="User's follower count")
    lastInteract: str = Field(..., description="Timestamp of last interaction")
    mutualFollow: str = Field(..., description="Whether there's a mutual follow (1=yes, 0=no)")
    subscriberID: str = Field(..., description="Unique subscriber ID")
    # Can accept message_content too as an alternative to message
    message_content: Optional[str] = Field(None, description="Alternative field for message content")

class MessageResponse(BaseModel):
    version: str = "v2"
    content: Dict[str, Any]

# ========================
# FASTAPI APP SETUP
# ========================

app = FastAPI(title="Social KILA DM API")

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include calendar router
app.include_router(calendar_router)

# ========================
# LOCAL CONVERSATION CACHE
# ========================

# Simple in-memory cache as backup for Pinecone
# Maps username -> conversation history
conversation_cache = {}

# ========================
# UTILITY FUNCTIONS
# ========================

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    if not text:
        return 0
    # Split by whitespace and count non-empty items
    return len([word for word in text.split() if word.strip()])

def truncate_response(text: str, max_length: int = MAX_RESPONSE_LENGTH) -> str:
    """
    Truncate a response to stay within the character limit.
    
    Args:
        text: The text to truncate
        max_length: Maximum allowed length in characters
        
    Returns:
        Truncated text that ends at a sentence boundary if possible
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a sentence boundary
    truncated = text[:max_length]
    
    # Find the last sentence end
    last_period_pos = truncated.rfind('.')
    last_question_pos = truncated.rfind('?')
    last_exclamation_pos = truncated.rfind('!')
    
    # Take the latest sentence end that's at least 70% of the allowed length
    min_acceptable_pos = int(max_length * 0.7)
    sentence_ends = [pos for pos in [last_period_pos, last_question_pos, last_exclamation_pos] 
                    if pos > min_acceptable_pos]
    
    if sentence_ends:
        # Truncate at the last sentence end
        end_pos = max(sentence_ends) + 1
        return text[:end_pos]
    else:
        # If no good sentence boundary, truncate at a word boundary
        return truncated.rsplit(' ', 1)[0] + '...'

# ========================
# MEMORY FUNCTIONS
# ========================

def fetch_memory(username: str) -> str:
    """
    Fetch conversation memory for a user from Pinecone.
    
    Args:
        username: Instagram username to fetch memory for
    
    Returns:
        String containing conversation summary or empty string if no memory found
    """
    start_time = time.time()
    
    # First check our local cache
    if username in conversation_cache:
        print(f"[MEMORY] Found conversation in local cache for {username}")
        return conversation_cache[username]
    
    try:
        print(f"[MEMORY] Fetching memory for user: {username}")
        resp = pinecone_index.fetch(ids=[username], namespace="kila_sm")
        
        # Log the raw response for debugging
        print(f"[MEMORY] Raw Pinecone response: {resp}")
        
        # Check if the response has vectors attribute and contains our username
        vectors = getattr(resp, 'vectors', {})
        if username in vectors:
            metadata = getattr(vectors[username], 'metadata', {})
            summary = metadata.get('summary', '')
            print(f"[MEMORY] Successfully retrieved memory for {username} ({len(summary)} chars)")
            
            # Store in local cache
            conversation_cache[username] = summary
            
            duration = time.time() - start_time
            print(f"[MEMORY] Memory fetch took {duration:.2f} seconds")
            return summary
        else:
            print(f"[MEMORY] No memory found for {username}")
            return ""
    except PineconeApiException as e:
        print(f"[ERROR] Pinecone API error during fetch_memory: {str(e)}")
        return ""
    except Exception as e:
        print(f"[ERROR] Unexpected error during fetch_memory: {str(e)}")
        return ""

def store_memory(username: str, summary: str, message: str, last_interaction: str, is_follower: bool):
    """
    Store conversation memory for a user in Pinecone.
    
    Args:
        username: Instagram username to store memory for
        summary: User metadata and conversation context
        message: Latest message from user
        last_interaction: Timestamp of interaction
        is_follower: Whether user follows the account
    """
    start_time = time.time()
    
    # First update our local cache
    conversation_cache[username] = summary
    
    try:
        print(f"[MEMORY] Storing memory for user: {username}")
        
        # Combine summary and message for embedding
        to_embed = f"{summary}. Latest DM: {message}"
        
        # Create embedding vector
        try:
            vector = EMBEDDINGS.embed_query(to_embed)
            print(f"[MEMORY] Successfully created embedding vector ({len(vector)} dimensions)")
        except Exception as e:
            print(f"[ERROR] Failed to create embedding: {str(e)}")
            raise
        
        # Prepare metadata with message history limit based on follower status
        metadata = {
            "summary": summary,
            "lastInteraction": last_interaction,
            "is_follower": "1" if is_follower else "0", 
            "max_messages": MAX_MESSAGES_FOLLOWER if is_follower else MAX_MESSAGES_NON_FOLLOWER
        }
        
        # Upsert to Pinecone
        try:
            result = pinecone_index.upsert(
                vectors=[
                    {
                        "id": username,
                        "values": vector,
                        "metadata": metadata
                    }
                ],
                namespace="kila_sm"
            )
            duration = time.time() - start_time
            print(f"[MEMORY] Successfully stored memory for {username}. Upsert took {duration:.2f} seconds")
            print(f"[MEMORY] Upsert result: {result}")
        except Exception as e:
            print(f"[ERROR] Failed to upsert to Pinecone: {str(e)}")
            raise
    except Exception as e:
        print(f"[ERROR] Error in store_memory: {str(e)}")
        # Don't re-raise the exception here - log it but let the flow continue

def is_first_time_user(username: str, memory_str: str) -> bool:
    """
    Determine if this is the first conversation with this user.
    
    Args:
        username: Username to check
        memory_str: Memory string from Pinecone
        
    Returns:
        True if this appears to be the first conversation, False otherwise
    """
    # Check if we have any memory for this user
    if not memory_str:
        return True
    
    # Check if there's any conversation history in the memory string
    conversation_markers = ["User:", "KILA:", "conversation:", "said:"]
    return not any(marker in memory_str for marker in conversation_markers)

def should_suggest_follow(is_follower: str, text: str) -> bool:
    """Determine if we should suggest following the account."""
    keywords = ["book", "price", "help", "demo", "automate", "setup"]
    return (is_follower == "0") and any(kw in text.lower() for kw in keywords)

# ========================
# CALENDAR UTILS
# ========================

def format_calendar_events(events: List[dict], max_events: int = 5) -> str:
    """Format calendar events for display in messages."""
    if not events:
        return "You don't have any upcoming events."
    
    formatted = "Your upcoming events:\n"
    for i, event in enumerate(events[:max_events], 1):
        start_time = event['start'].split('T')[1].split('+')[0][:5]  # Extract HH:MM
        date = event['start'].split('T')[0]  # Extract YYYY-MM-DD
        formatted += f"{i}. {event['summary']} on {date} at {start_time}\n"
    
    if len(events) > max_events:
        formatted += f"...and {len(events) - max_events} more.\n"
    
    return formatted

def format_availability_slots(slots: List[dict], date: str) -> str:
    """Format availability slots for display in messages."""
    if not slots:
        return f"No available slots found for {date}."
    
    formatted = f"Available slots for {date}:\n"
    for i, slot in enumerate(slots, 1):
        start = slot['start'].split('T')[1].split('+')[0][:5]  # Extract HH:MM
        end = slot['end'].split('T')[1].split('+')[0][:5]      # Extract HH:MM
        formatted += f"{i}. {start} - {end}\n"
    
    return formatted

# ========================
# API ENDPOINTS
# ========================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "alive", "service": "KILA DM API"}

@app.get("/test-memory/{username}")
async def test_memory(username: str):
    """
    Test endpoint to verify Pinecone memory operations.
    
    This endpoint allows you to check if Pinecone is storing and retrieving memory correctly.
    """
    results = {}
    
    # Try to fetch existing memory
    try:
        memory = fetch_memory(username)
        results["fetch_status"] = "success"
        results["memory_exists"] = bool(memory)
        if memory:
            results["memory_content"] = memory
        else:
            results["memory_content"] = None
    except Exception as e:
        results["fetch_status"] = "error"
        results["fetch_error"] = str(e)
    
    # Try to store test memory
    try:
        test_text = f"Test memory for {username} at {datetime.now()}"
        test_memory = f"Name: Test User, Handle: {username}\nUser: Hello\nKILA: Hi there! How can I help you today?\nUser: {test_text}"
        
        store_memory(
            username=username,
            summary=test_memory,
            message=test_text,
            last_interaction=datetime.now().isoformat(),
            is_follower=True
        )
        results["store_status"] = "success"
    except Exception as e:
        results["store_status"] = "error"
        results["store_error"] = str(e)
    
    # Try to fetch again to confirm storage
    try:
        memory_after = fetch_memory(username)
        results["fetch_after_store"] = "success"
        results["memory_after_store_exists"] = bool(memory_after)
        if memory_after:
            results["memory_after_store"] = memory_after
    except Exception as e:
        results["fetch_after_store"] = "error"
        results["fetch_after_store_error"] = str(e)
    
    return results

@app.get("/clear-memory/{username}")
async def clear_memory(username: str):
    """Clear the memory for a specific user."""
    try:
        # Clear from local cache
        if username in conversation_cache:
            del conversation_cache[username]
        
        # Delete from Pinecone
        pinecone_index.delete(ids=[username], namespace="kila_sm")
        
        return {"status": "success", "message": f"Memory cleared for {username}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-calendar")
async def test_calendar():
    """Test the calendar integration."""
    results = {}
    
    # Test connection to Google Calendar
    try:
        from cal_api import get_calendar_service
        service = get_calendar_service()
        calendar = service.calendars().get(calendarId="ceo@agently-ai.com").execute()
        results["connection"] = "success"
        results["calendar_name"] = calendar.get('summary', 'Unknown')
    except Exception as e:
        results["connection"] = "error"
        results["error"] = str(e)
    
    # Test listing events
    try:
        from cal_api import list_events
        response = await list_events(max_results=5)
        results["list_events"] = "success" if response.success else "error"
        results["events_count"] = len(response.data.get("events", [])) if response.success else 0
    except Exception as e:
        results["list_events"] = "error"
        results["list_events_error"] = str(e)
    
    return results

@app.post("/dm")
async def handle_dm(payload: DMRequest):
    """
    Handle incoming DM requests from ManyChat.
    
    This is the main endpoint that processes Instagram DMs and returns responses.
    """
    request_time = datetime.now().isoformat()
    print(f"[REQUEST] Received at {request_time} from user: {payload.username}")
    
    # Use message_content if provided, otherwise use message
    message = payload.message_content if payload.message_content else payload.message
    print(f"[REQUEST] Message: '{message}'")
    print(f"[REQUEST] Word count: {count_words(message)}, Character count: {len(message)}")
    
    # Check for calendar intent
    calendar_intent = detect_calendar_intent(message)
    calendar_data = None
    if calendar_intent:
        print(f"[CALENDAR] Detected calendar intent: {calendar_intent}")
        instruction, extracted_data = process_calendar_intent(calendar_intent, message)
        print(f"[CALENDAR] Calendar instruction: {instruction}")
        print(f"[CALENDAR] Extracted data: {extracted_data}")
    
    # Extract user info
    name = payload.name.strip() if payload.name else "Unknown"
    is_follower = payload.isFollower == "1"  # Convert to boolean
    
    # Create metadata for storage and context
    meta = (
        f"Name: {name}, Handle: {payload.username}, "
        f"Followers: {payload.followCt}, Follower: {payload.isFollower}, "
        f"MutualFollow: {payload.mutualFollow}, Last interaction: {payload.lastInteract}"
    )
    
    # Fetch existing memory or create new one
    memory_str = fetch_memory(payload.username)
    print(f"[MEMORY] Retrieved memory: {memory_str[:100] if memory_str else 'None'}...")
    
    # Check if this is a first-time user
    first_time = is_first_time_user(payload.username, memory_str)
    print(f"[FLOW] Is first time user: {first_time}")
    
    # Initialize or update conversation history
    if not memory_str:
        # First time ever - just store user metadata
        current_memory = meta
        print(f"[FLOW] No existing memory found for {payload.username}. Creating new memory entry.")
    else:
        # We have existing memory
        current_memory = memory_str
    
    # Build conversation context for the AI
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add memory context if available
    if current_memory:
        # If this is not the first time user, explicitly tell KILA this is a returning user
        if not first_time:
            chat_messages.append({"role": "system", "content": f"IMPORTANT: This is a returning user that you've talked to before. DO NOT introduce yourself as if this is your first conversation. User context: {current_memory}"})
        else:
            chat_messages.append({"role": "system", "content": f"User context: {current_memory}"})
    
    # Suggest following if appropriate
    if should_suggest_follow(payload.isFollower, message):
        follow_prompt = "[NOTE FOR KILA] The user is not following us but is showing interest. If natural, let them know that following helps you remember them better next time."
        chat_messages.append({"role": "system", "content": follow_prompt})
    
    # Handle calendar actions and provide booking link instructions
    if calendar_intent:
        try:
            if calendar_intent == "check_availability":
                print(f"[CALENDAR] Detected availability check - providing scheduling link")
                calendar_data = "To see all my available appointment times, please use the booking link below."
                
            elif calendar_intent == "book_appointment":
                print(f"[CALENDAR] Detected booking intent - providing scheduling link")
                
                # Extract purpose if available
                purpose_text = ""
                if "purpose" in extracted_data and extracted_data["purpose"]:
                    purpose_text = f" to discuss {extracted_data['purpose']}"
                
                calendar_data = f"To book an appointment{purpose_text}, please use the booking link below."
                
            elif calendar_intent == "list_appointments":
                try:
                    # Call the calendar API directly
                    from cal_api import list_events
                    calendar_response = await list_events(max_results=5)
                    
                    if calendar_response.success:
                        events = calendar_response.data.get("events", [])
                        calendar_data = format_calendar_events(events) + f"\n\nTo schedule a new appointment, please use the booking link below."
                        
                        print(f"[CALENDAR] Retrieved events: {calendar_data}")
                except Exception as e:
                    print(f"[ERROR] Calendar list events error: {str(e)}")
                    calendar_data = "I'm having trouble retrieving your appointments. To schedule a new appointment, please use the booking link below."
                    
            # Handle rescheduling intent
            elif "reschedule" in message.lower() or "change" in message.lower():
                print(f"[CALENDAR] Detected rescheduling intent - providing scheduling link")
                calendar_data = "To reschedule your appointment, please use the booking link below."
            
            # Add calendar data to AI context
            if calendar_data:
                chat_messages.append({
                    "role": "system", 
                    "content": f"CALENDAR ACTION RESULT: {calendar_data}"
                })
            
            # Add calendar intent instructions
            chat_messages.append({
                "role": "system", 
                "content": f"The user has a calendar-related request. Instructions: {instruction}"
            })
            
            # Tell KILA to mention the booking link will be below the message
            chat_messages.append({
                "role": "system",
                "content": "IMPORTANT: Let the user know they can click the booking link below your message. Don't include the actual URL in your message text, as it will be provided as a clickable button below."
            })
            
        except Exception as e:
            print(f"[ERROR] Calendar processing error: {str(e)}")
            chat_messages.append({
                "role": "system", 
                "content": "IMPORTANT: There was an error processing the calendar request. Let the user know they can click the booking link below your message to schedule or reschedule an appointment."
            })
    
    # Add the current user message
    chat_messages.append({"role": "user", "content": message})
    
    # Generate response from OpenAI
    try:
        print(f"[OPENAI] Sending request to OpenAI with {len(chat_messages)} messages")
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # Or your preferred model
            messages=chat_messages,
            temperature=0.75,
            max_tokens=300
        )
        reply = response.choices[0].message.content.strip()
        print(f"[OPENAI] Received response: '{reply[:50]}...'")
        print(f"[OPENAI] Response word count: {count_words(reply)}, Character count: {len(reply)}")
        
        # Check if the response is too long and truncate if needed
        if len(reply) > MAX_RESPONSE_LENGTH:
            original_length = len(reply)
            reply = truncate_response(reply, MAX_RESPONSE_LENGTH)
            print(f"[OPENAI] Truncated response from {original_length} to {len(reply)} characters")
    except Exception as e:
        print(f"[ERROR] OpenAI error: {str(e)}")
        reply = "Sorry, I'm having trouble connecting right now. Could you try again in a moment?"
    
    # Update memory with this interaction
    try:
        # Add this interaction to the conversation history
        if first_time:
            # For first time users, start a new conversation
            updated_memory = f"{meta}\nconversation_started: {request_time}\nUser: {message}\nKILA: {reply}"
        else:
            # For returning users, add to existing conversation
            updated_memory = f"{current_memory}\nUser: {message}\nKILA: {reply}"
        
        # Store updated memory
        store_memory(
            payload.username,
            updated_memory,
            message,
            request_time,  # Use current timestamp
            is_follower
        )
    except Exception as e:
        print(f"[ERROR] Failed to update memory: {str(e)}")
    
    # Format response for ManyChat with or without booking button based on intent
    if calendar_intent:
        button_caption = "Book Appointment"
        
        if "reschedule" in message.lower() or "change" in message.lower():
            button_caption = "Reschedule Appointment"
        
        response_data = {
            "version": "v2",
            "content": {
                "type": "instagram",
                "messages": [
                    {
                        "type": "text",
                        "text": reply,
                        "buttons": [
                            {
                                "type": "url",
                                "caption": button_caption,
                                "url": BOOKING_LINK
                            }
                        ]
                    }
                ]
            }
        }
    else:
        # Standard response without buttons for non-calendar intents
        response_data = {
            "version": "v2",
            "content": {
                "type": "instagram",
                "messages": [
                    {
                        "type": "text",
                        "text": reply
                    }
                ]
            }
        }
    
    print(f"[RESPONSE] Sending response data: {json.dumps(response_data)[:100]}...")
    return response_data

# ========================
# ERROR HANDLERS
# ========================

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions."""
    print(f"[ERROR] Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An internal server error occurred", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def calendar_exception_handler(request: Request, exc: Exception):
    """Handle calendar-related exceptions."""
    if "calendar" in str(request.url).lower():
        print(f"[ERROR] Calendar API error: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "An error occurred with the calendar service", "detail": str(exc)}
        )
    raise exc

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    print(f"[STARTUP] Starting KILA DM API server...")
    uvicorn.run("kila_sm:app", host="0.0.0.0", port=8000, reload=True)