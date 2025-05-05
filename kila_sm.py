# kila_sm.py
# FastAPI app for handling Instagram DMs via ManyChat -> Social KILA
# Uses GPT-4 + Pinecone for persistent conversation memory

import os
import json
import tempfile
import time
from datetime import datetime, timedelta
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
MAX_RESPONSE_LENGTH = 950

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

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    if not text:
        return 0
    # Split by whitespace and count non-empty items
    return len([word for word in text.split() if word.strip()])

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

@app.get("/debug-calendar")
async def debug_calendar():
    """Diagnostic endpoint for calendar integration."""
    results = {}
    
    # 1. Check service account file
    try:
        service_account_path = "./kila-456404-d406b0474c4f.json"
        alt_path = "./.kila-456404-d406b0474c4f.json"
        
        results["file_checks"] = {
            "service_account_exists": os.path.exists(service_account_path),
            "alt_path_exists": os.path.exists(alt_path),
            "service_account_file_env": os.getenv("SERVICE_ACCOUNT_FILE"),
            "current_directory": os.getcwd(),
            "files_in_directory": os.listdir(".")[:10]  # List first 10 files
        }
    except Exception as e:
        results["file_checks"] = {"error": str(e)}
    
    # 2. Check calendar API connection
    try:
        from cal_api import get_calendar_service
        service = get_calendar_service()
        
        # Try to get calendar info
        try:
            calendar = service.calendars().get(calendarId="ceo@agently-ai.com").execute()
            results["calendar_access"] = {
                "success": True,
                "name": calendar.get("summary", "Unknown"),
                "timezone": calendar.get("timeZone", "Unknown")
            }
        except Exception as e:
            results["calendar_access"] = {
                "success": False,
                "error": str(e)
            }
    except Exception as e:
        results["service_initialization"] = {
            "success": False,
            "error": str(e)
        }
    
    # 3. Test event creation
    try:
        from cal_api import create_event, EventRequest
        
        # Create event for tomorrow
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        test_time = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0)
        
        event_data = EventRequest(
            summary="Debug Test Event",
            start_time=test_time.isoformat(),
            end_time=(test_time + timedelta(hours=1)).isoformat(),
            description="Created by debug endpoint"
        )
        
        # Try to create event
        result = await create_event(event_data)
        results["event_creation"] = {
            "success": result.success if result else False,
            "message": result.message if result else "No result returned",
            "data": result.data if result and result.success else None
        }
    except Exception as e:
        results["event_creation"] = {
            "success": False,
            "error": str(e)
        }
    
    return results

@app.get("/word-count")
async def word_count(text: str):
    """Count words in a text string."""
    count = count_words(text)
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "word_count": count,
        "character_count": len(text)
    }

@app.get("/test-date-parsing")
async def test_date_parsing(message: str):
    """Test date parsing from a message."""
    try:
        from cal_help import extract_dates
        import pytz
        
        # Extract dates
        dates = extract_dates(message)
        
        # Format results
        results = []
        for date in dates:
            results.append({
                "iso_format": date.isoformat(),
                "readable": date.strftime("%A, %B %d, %Y at %I:%M %p"),
                "day_of_week": date.strftime("%A")
            })
        
        return {
            "original_message": message,
            "extracted_dates": results
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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
    # Initialize chat_messages here to avoid reference errors
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
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
    
    # Handle calendar actions if needed
    if calendar_intent:
        try:
            if calendar_intent == "check_availability":
                # Check if we have a date
                if "dates" in extracted_data and extracted_data["dates"]:
                    date_str = extracted_data["dates"][- 1]
                    try:
                        # Call the calendar API directly
                        from cal_api import check_availability
                        calendar_response = await check_availability(date=date_str)
                        
                        if calendar_response.success:
                            available_slots = calendar_response.data.get('available_slots', [])
                            friendly_date = calendar_response.data.get('date', date_str)
                            calendar_data = format_availability_slots(available_slots, friendly_date)
                            print(f"[CALENDAR] Retrieved availability: {calendar_data}")
                    except Exception as e:
                        print(f"[ERROR] Calendar availability error: {str(e)}")
            
            # Handle booking appointments
            elif calendar_intent == "book_appointment":
                print(f"[CALENDAR] Processing booking intent")
                try:
                    # First, check if we have an email for this booking
                    has_email = "email" in extracted_data and extracted_data["email"]
                    
                    if not has_email:
                        # Add a system message to instruct KILA to ask for email or alternative contact
                        chat_messages.append({
                            "role": "system", 
                            "content": "IMPORTANT: The user wants to book a call but hasn't provided an email address. " +
                                      "Ask them for their email to send a calendar invitation. " +
                                      "Also mention they can provide a phone number or use their Instagram handle if they prefer."
                        })
                        # Set calendar data to indicate we need contact info
                        calendar_data = "Need to collect contact information first"
                    else:
                        # Check if we have date information
                        if "dates" in extracted_data and extracted_data["dates"]:
                            from cal_api import check_availability, create_event, EventRequest
                            from dateparser import parse
                            
                            # Parse the date
                            date_str = extracted_data["dates"][-1]
                            print(f"[CALENDAR] Date from extracted data: {date_str}")
                            
                            # Parse start and end times
                            start_time = parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
                            
                            if not start_time:
                                print(f"[CALENDAR] Failed to parse date: {date_str}")
                                calendar_data = f"Failed to understand the requested date/time: {date_str}. Please ask for a more specific date and time."
                                chat_messages.append({
                                    "role": "system",
                                    "content": f"CALENDAR ERROR: {calendar_data} Ask the user to specify the date and time more clearly, for example 'Monday May 12th at 2pm'."
                                })
                            else:
                                print(f"[CALENDAR] Parsed start time: {start_time.isoformat()}")
                                
                                # Set end time (1 hour later)
                                end_time = start_time + timedelta(hours=1)
                                
                                # First check if this time slot is available
                                try:
                                    calendar_response = await check_availability(date=start_time.isoformat())
                                    
                                    if calendar_response.success:
                                        # Extract available slots
                                        available_slots = calendar_response.data.get('available_slots', [])
                                        
                                        # Check if the requested time is in an available slot
                                        requested_time_available = False
                                        alternative_slots = []
                                        
                                        for slot in available_slots:
                                            slot_start = datetime.fromisoformat(slot['start'].replace('Z', '+00:00'))
                                            slot_end = datetime.fromisoformat(slot['end'].replace('Z', '+00:00'))
                                            
                                            # If requested time is within this slot
                                            if (start_time >= slot_start and end_time <= slot_end):
                                                requested_time_available = True
                                                break
                                            else:
                                                # Add to alternatives list
                                                alternative_slots.append((slot_start, slot_end))
                                        
                                        if not requested_time_available and alternative_slots:
                                            # Suggested alternatives
                                            alternatives_text = "The requested time is not available. Here are some alternatives:\n"
                                            for i, (alt_start, alt_end) in enumerate(alternative_slots[:3], 1):
                                                alternatives_text += f"{i}. {alt_start.strftime('%A, %B %d at %I:%M %p')} - {alt_end.strftime('%I:%M %p')}\n"
                                            
                                            calendar_data = alternatives_text
                                            chat_messages.append({
                                                "role": "system",
                                                "content": f"CALENDAR CONFLICT: {calendar_data} Please suggest one of these alternative times to the user."
                                            })
                                        else:
                                            # Create summary and description
                                            summary = f"Meeting with {payload.username}"
                                            if "purpose" in extracted_data:
                                                summary += f" - {extracted_data['purpose']}"
                                            
                                            description = f"Meeting requested by {payload.username} via Instagram DM"
                                            
                                            # Get attendee email from extracted data
                                            attendees = []
                                            if extracted_data["email"]:
                                                attendees.append(extracted_data["email"])
                                                print(f"[CALENDAR] Using attendee email from message: {extracted_data['email']}")
                                            
                                            # Create event request
                                            event_data = EventRequest(
                                                summary=summary,
                                                start_time=start_time.isoformat(),
                                                end_time=end_time.isoformat(),
                                                description=description,
                                                attendees=attendees
                                            )
                                            
                                            # Create the event
                                            print(f"[CALENDAR] Creating event with data: {event_data}")
                                            result = await create_event(event_data)
                                            
                                            # Check result - IMPROVED ERROR HANDLING HERE
                                            if result and result.success:
                                                print(f"[CALENDAR] Successfully created event: {result.data}")
                                                event_id = result.data.get("event", {}).get("id")
                                                calendar_data = f"BOOKING CONFIRMED: Appointment successfully booked for {start_time.strftime('%A, %B %d at %I:%M %p')}. Event ID: {event_id}"
                                            else:
                                                error_msg = result.message if result else "Unknown error occurred"
                                                print(f"[CALENDAR] Failed to create event: {error_msg}")
                                                calendar_data = f"BOOKING FAILED: I couldn't schedule your appointment due to a technical issue ({error_msg}). Please tell the user something casual like 'I'm having some trouble with my calendar right now, but I've noted your request for {start_time.strftime('%A, %B %d at %I:%M %p')}. I'll make sure Mr. Kotak gets the details and reaches out to confirm soon.' Make sure to confirm you have their contact information."
                                            
                                            # Add clear instruction for KILA in system message
                                            chat_messages.append({
                                                "role": "system",
                                                "content": f"CALENDAR ACTION RESULT: {calendar_data}"
                                            })
                                            
                                            # If the booking failed, add explicit instruction to collect contact info
                                            if not (result and result.success):
                                                chat_messages.append({
                                                    "role": "system",
                                                    "content": "IMPORTANT: Since the booking failed, make sure you have contact information from the user (email preferred, but also ask for phone or confirm their Instagram handle). Don't mention the technical error details - just be casual and reassure them that we'll follow up."
                                                })
                                    else:
                                        print(f"[CALENDAR] Error checking availability: {calendar_response.message}")
                                        calendar_data = f"Error checking availability: {calendar_response.message}"
                                        chat_messages.append({
                                            "role": "system",
                                            "content": f"CALENDAR ERROR: {calendar_data} Tell the user you're having trouble with your calendar system, but you'll note their request for {start_time.strftime('%A, %B %d')} and make sure someone follows up. Ask for their email or preferred contact method."
                                        })
                                except Exception as e:
                                    print(f"[CALENDAR] Error checking availability: {str(e)}")
                                    calendar_data = f"Error checking availability: {str(e)}"
                                    chat_messages.append({
                                        "role": "system",
                                        "content": f"CALENDAR ERROR: {calendar_data} Tell the user you're having trouble with your calendar system, but you'll note their request for {start_time.strftime('%A, %B %d')} and make sure someone follows up. Ask for their email or preferred contact method."
                                    })
                        else:
                            print("[CALENDAR] No date information found in extracted data")
                            calendar_data = "No date information found in the message"
                            chat_messages.append({
                                "role": "system",
                                "content": f"CALENDAR ERROR: {calendar_data} Ask the user to specify a date and time for the appointment."
                            })
                except Exception as e:
                    print(f"[CALENDAR] Error in calendar event creation: {str(e)}")
                    calendar_data = f"Error in calendar event creation: {str(e)}"
                    chat_messages.append({
                        "role": "system",
                        "content": f"CALENDAR ERROR: {calendar_data} Tell the user you're having trouble with your calendar, but you'll note their request and make sure someone follows up. Ask for their email or preferred contact method if you don't already have it."
                    })
            
            elif calendar_intent == "list_appointments":
                try:
                    # Call the calendar API directly
                    from cal_api import list_events
                    calendar_response = await list_events(max_results=5)
                    
                    if calendar_response.success:
                        events = calendar_response.data.get("events", [])
                        calendar_data = format_calendar_events(events)
                        print(f"[CALENDAR] Retrieved events: {calendar_data}")
                except Exception as e:
                    print(f"[ERROR] Calendar list events error: {str(e)}")
            
            # Handle rescheduling
            elif "reschedule" in message.lower() or "change" in message.lower():
                print(f"[CALENDAR] Processing reschedule request")
                try:
                    # First, check if we have an email for this booking
                    has_email = "email" in extracted_data and extracted_data["email"]
                    
                    if not has_email:
                        # Add a system message to instruct KILA to ask for email or alternative contact
                        chat_messages.append({
                            "role": "system", 
                            "content": "IMPORTANT: The user wants to reschedule but hasn't provided an email address. " +
                                      "Ask them for their email to send a calendar invitation. " +
                                      "Also mention they can provide a phone number or use their Instagram handle if they prefer."
                        })
                        # Set calendar data to indicate we need contact info
                        calendar_data = "Need to collect contact information first"
                    else:
                        # Extract the new date
                        from cal_help import extract_dates
                        dates = extract_dates(message)
                        
                        if dates:
                            from cal_api import check_availability, create_event, EventRequest
                            
                            # Use the last date mentioned
                            new_date = dates[-1] if dates else None
                            print(f"[CALENDAR] New date for rescheduling: {new_date.isoformat()}")
                            
                            # Set end time (1 hour later)
                            end_time = new_date + timedelta(hours=1)
                            
                            # Check if this time slot is available
                            try:
                                calendar_response = await check_availability(date=new_date.isoformat())
                                
                                if calendar_response.success:
                                    # Extract available slots
                                    available_slots = calendar_response.data.get('available_slots', [])
                                    
                                    # Check if the requested time is in an available slot
                                    requested_time_available = False
                                    alternative_slots = []
                                    
                                    for slot in available_slots:
                                        slot_start = datetime.fromisoformat(slot['start'].replace('Z', '+00:00'))
                                        slot_end = datetime.fromisoformat(slot['end'].replace('Z', '+00:00'))
                                        
                                        # If requested time is within this slot
                                        if (new_date >= slot_start and end_time <= slot_end):
                                            requested_time_available = True
                                            break
                                        else:
                                            # Add to alternatives list
                                            alternative_slots.append((slot_start, slot_end))
                                    
                                    if not requested_time_available and alternative_slots:
                                        # Suggested alternatives
                                        alternatives_text = "The requested time is not available for rescheduling. Here are some alternatives:\n"
                                        for i, (alt_start, alt_end) in enumerate(alternative_slots[:3], 1):
                                            alternatives_text += f"{i}. {alt_start.strftime('%A, %B %d at %I:%M %p')} - {alt_end.strftime('%I:%M %p')}\n"
                                        
                                        calendar_data = alternatives_text
                                        chat_messages.append({
                                            "role": "system",
                                            "content": f"CALENDAR CONFLICT: {calendar_data} Please suggest one of these alternative times to the user for rescheduling."
                                        })
                                    else:
                                        # Get attendee email from extracted data
                                        attendees = []
                                        if extracted_data["email"]:
                                            attendees.append(extracted_data["email"])
                                            print(f"[CALENDAR] Using attendee email from message: {extracted_data['email']}")
                                        
                                        # Create event
                                        event_data = EventRequest(
                                            summary=f"Meeting with {payload.username} (Rescheduled)",
                                            start_time=new_date.isoformat(),
                                            end_time=end_time.isoformat(),
                                            description=f"Rescheduled meeting via Instagram DM",
                                            attendees=attendees
                                        )
                                        
                                        # Create the event
                                        print(f"[CALENDAR] Creating rescheduled event: {event_data}")
                                        result = await create_event(event_data)
                                        
                                        # Check result - IMPROVED ERROR HANDLING HERE
                                        if result and result.success:
                                            print(f"[CALENDAR] Successfully created rescheduled event: {result.data}")
                                            event_id = result.data.get("event", {}).get("id")
                                            calendar_data = f"RESCHEDULE CONFIRMED: Appointment successfully rescheduled for {new_date.strftime('%A, %B %d at %I:%M %p')}. Event ID: {event_id}"
                                        else:
                                            error_msg = result.message if result else "Unknown error occurred"
                                            print(f"[CALENDAR] Failed to create rescheduled event: {error_msg}")
                                            calendar_data = f"RESCHEDULE FAILED: I couldn't reschedule your appointment due to a technical issue ({error_msg}). Please tell the user something casual like 'I'm having some trouble with my calendar right now, but I've noted your request to reschedule for {new_date.strftime('%A, %B %d at %I:%M %p')}. I'll make sure Mr. Kotak gets the details and reaches out to confirm soon.'"
                                        
                                        # Add clear instruction for KILA in system message
                                        chat_messages.append({
                                            "role": "system",
                                            "content": f"CALENDAR ACTION RESULT: {calendar_data}"
                                        })
                                        
                                        # If the rescheduling failed, add explicit instruction to collect contact info
                                        if not (result and result.success):
                                            chat_messages.append({
                                                "role": "system",
                                                "content": "IMPORTANT: Since the rescheduling failed, make sure you have contact information from the user (email preferred, but also ask for phone or confirm their Instagram handle). Don't mention the technical error details - just be casual and reassure them that we'll follow up."
                                            })
                                else:
                                    print(f"[CALENDAR] Error checking availability: {calendar_response.message}")
                                    calendar_data = f"Error checking availability: {calendar_response.message}"
                                    chat_messages.append({
                                        "role": "system",
                                        "content": f"CALENDAR ERROR: {calendar_data} Tell the user you're having trouble with your calendar system, but you'll note their request for {new_date.strftime('%A, %B %d')} and make sure someone follows up. Ask for their email or preferred contact method."
                                    })
                            except Exception as e:
                                print(f"[CALENDAR] Error checking availability: {str(e)}")
                                calendar_data = f"Error checking availability: {str(e)}"
                                chat_messages.append({
                                    "role": "system",
                                    "content": f"CALENDAR ERROR: {calendar_data} Tell the user you're having trouble with your calendar system, but you'll note their reschedule request and make sure someone follows up. Ask for their email or preferred contact method."
                                })
                        else:
                            print("[CALENDAR] Could not extract new date for rescheduling")
                            calendar_data = "Could not understand the new date for rescheduling."
                            chat_messages.append({
                                "role": "system",
                                "content": f"CALENDAR ERROR: {calendar_data} Ask the user to specify a clear date and time for rescheduling, for example 'Monday May 12th at 2pm'."
                            })
                except Exception as e:
                    print(f"[CALENDAR] Error in rescheduling: {str(e)}")
                    calendar_data = f"Error in rescheduling: {str(e)}"
                    chat_messages.append({
                        "role": "system",
                        "content": f"CALENDAR ERROR: {calendar_data} Tell the user you're having trouble with your calendar, but you'll note their rescheduling request and make sure someone follows up. Ask for their email or preferred contact method if you don't already have it."
                    })
        
        except Exception as e:
            print(f"[ERROR] Calendar processing error: {str(e)}")
    
    # Build conversation context for the AI
    # chat_messages was initialized at the start to avoid reference errors
    
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
    
    # Add calendar context if applicable
    if calendar_intent:
        chat_messages.append({
            "role": "system", 
            "content": f"The user has a calendar-related request. Instructions: {instruction}"
        })
        
        if calendar_data:
            chat_messages.append({
                "role": "system", 
                "content": f"CALENDAR API RESULT: {calendar_data}"
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
    
    # Format response for ManyChat
    response_data = {
        "version": "v2",
        "content": {
            "type": "instagram",  # This is crucial for ManyChat to process correctly
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