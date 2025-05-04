# kila_sm.py
# FastAPI app for handling Instagram DMs via ManyChat -> Social KILA
# Uses GPT-4 + Pinecone for persistent conversation memory

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
    previous_message: Optional[str] = Field(None, description="Previous message from user")
    isFollower: str = Field(..., description="Whether user follows the account (1=yes, 0=no)")
    followCt: str = Field(..., description="User's follower count")
    lastInteract: str = Field(..., description="Timestamp of last interaction")
    mutualFollow: str = Field(..., description="Whether there's a mutual follow (1=yes, 0=no)")
    subscriberID: str = Field(..., description="Unique subscriber ID")

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
    try:
        print(f"[MEMORY] Fetching memory for user: {username}")
        resp = pinecone_index.fetch(ids=[username], namespace="kila_sm")
        
        # Log the raw response for debugging
        print(f"[MEMORY] Raw Pinecone response: {resp}")
        
        # Check if the response has vectors attribute and contains our username
        vectors = resp.get('vectors', {})
        if username in vectors:
            metadata = vectors[username].get('metadata', {})
            summary = metadata.get('summary', '')
            print(f"[MEMORY] Successfully retrieved memory for {username} ({len(summary)} chars)")
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

def should_suggest_follow(is_follower: str, text: str) -> bool:
    """Determine if we should suggest following the account."""
    keywords = ["book", "price", "help", "demo", "automate", "setup"]
    return (is_follower == "0") and any(kw in text.lower() for kw in keywords)

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
    except Exception as e:
        results["fetch_status"] = "error"
        results["fetch_error"] = str(e)
    
    # Try to store test memory
    try:
        test_text = f"Test memory for {username} at {datetime.now()}"
        store_memory(
            username=username,
            summary=f"Test user metadata. Name: Test User",
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

@app.post("/dm")
async def handle_dm(payload: DMRequest):
    """
    Handle incoming DM requests from ManyChat.
    
    This is the main endpoint that processes Instagram DMs and returns responses.
    """
    request_time = datetime.now().isoformat()
    print(f"[REQUEST] Received at {request_time} from user: {payload.username}")
    print(f"[REQUEST] Message: '{payload.message}'")
    print(f"[REQUEST] Full payload: {payload.model_dump_json()}")
    
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
    if not memory_str:
        print(f"[FLOW] No existing memory found for {payload.username}. Creating new memory entry.")
        store_memory(
            payload.username, 
            meta, 
            payload.message, 
            payload.lastInteract,
            is_follower
        )
    else:
        print(f"[FLOW] Found existing memory for {payload.username} ({len(memory_str)} chars)")
    
    # Build conversation context for the AI
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add memory context if available
    if memory_str:
        chat_messages.append({"role": "system", "content": f"User context: {memory_str}"})
    
    # Suggest following if appropriate
    if should_suggest_follow(payload.isFollower, payload.message):
        follow_prompt = "[NOTE FOR KILA] The user is not following us but is showing interest. If natural, let them know that following helps you remember them better next time."
        chat_messages.append({"role": "system", "content": follow_prompt})
    
    # Add previous message context if provided
    if payload.previous_message:
        chat_messages.append({"role": "system", "content": f"User's previous message: {payload.previous_message}"})
    
    # Add the current user message
    chat_messages.append({"role": "user", "content": payload.message})
    
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
    except Exception as e:
        print(f"[ERROR] OpenAI error: {str(e)}")
        reply = "Sorry, I'm having trouble connecting right now. Could you try again in a moment?"
    
    # Update memory with this interaction
    try:
        # Add this interaction to the memory summary
        updated_summary = f"{memory_str or meta}\nUser: {payload.message}\nKILA: {reply}"
        
        # Store updated memory
        store_memory(
            payload.username,
            updated_summary,
            payload.message,
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

# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions."""
    print(f"[ERROR] Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An internal server error occurred", "detail": str(exc)}
    )

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    print(f"[STARTUP] Starting KILA DM API server...")
    uvicorn.run("kila_sm:app", host="0.0.0.0", port=8000, reload=True)