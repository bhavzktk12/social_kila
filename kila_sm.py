# kila_sm.py
# FastAPI app for handling Instagram DMs via ManyChat -> n8n -> Social KILA
# Uses GPT-4 Turbo + Pinecone (agently-memory) + multilingual smart memory

import os
from datetime import datetime
from typing import List, Dict, Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone

# -------------------------------
# 1. Configuration & Initialization
# -------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agently-memory")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise RuntimeError("Missing OpenAI or Pinecone keys in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
EMBEDDINGS = OpenAIEmbeddings()

with open("kila_sm.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# -------------------------------
# 2. FastAPI Setup
# -------------------------------

app = FastAPI(title="Social KILA DM API")

class DMRequest(BaseModel):
    name: str = Field(...)
    message: str = Field(...)
    username: str = Field(...)
    isFollower: str = Field(...)
    followCt: str = Field(...)
    lastInteract: str = Field(...)
    mutualFollow: str = Field(...)
    subscriberID: str = Field(...)

class ManyChatMessage(BaseModel):
    type: Literal["text"]
    text: str

class ManyChatContent(BaseModel):
    messages: List[ManyChatMessage]

class ManyChatResponse(BaseModel):
    version: Literal["v2"]
    content: ManyChatContent

# -------------------------------
# 3. Helper Functions
# -------------------------------

def fetch_memory(username: str) -> str:
    try:
        resp = pinecone_index.fetch(ids=[username], namespace="kila_sm")
        vectors = resp.vectors if hasattr(resp, "vectors") else {}
        if username in vectors and hasattr(vectors[username], "metadata"):
            return vectors[username].metadata.get("summary", "")
    except Exception as e:
        print("[Fetch Memory Error]", str(e))
    return ""

def store_memory(username: str, summary: str, message: str, last_interaction: str):
    try:
        to_embed = f"{summary}. Latest DM: {message}"
        vector = EMBEDDINGS.embed_query(to_embed)
        pinecone_index.upsert([
            (username, vector, {
                "summary": summary,
                "lastInteraction": last_interaction
            })
        ], namespace="kila_sm")
    except Exception as e:
        print("[Store Memory Error]", str(e))

def should_suggest_follow(is_follower: str, text: str) -> bool:
    keywords = ["book", "price", "help", "demo", "automate", "setup"]
    return (is_follower == "0") and any(kw in text.lower() for kw in keywords)

# -------------------------------
# 4. Endpoint: /dm
# -------------------------------

@app.post("/dm", response_model=ManyChatResponse)
async def handle_dm(payload: DMRequest) -> Dict[str, Any]:
    print("[DEBUG] Payload received:", payload.model_dump())

    name = payload.name.strip() if payload.name else "Unknown"

    meta = (
        f"Name: {name}, Handle: {payload.username}, "
        f"Followers: {payload.followCt}, Follower: {payload.isFollower}, "
        f"MutualFollow: {payload.mutualFollow}"
    )

    memory_str = fetch_memory(payload.username)
    if not memory_str:
        store_memory(payload.username, meta, payload.message, payload.lastInteract)

    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory_str:
        chat_messages.append({"role": "system", "content": memory_str})

    if should_suggest_follow(payload.isFollower, payload.message):
        chat_messages.append({"role": "system", "content": "[NOTE FOR KILA] The user is not following us but is showing interest. If natural, let them know that following helps you remember them next time."})

    chat_messages.append({"role": "user", "content": payload.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=chat_messages,
            temperature=0.75,
            max_tokens=300
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print("[OpenAI Error]", str(e))
        reply = "Sorry, I ran into an issue. Could you try again in a moment?"

        return {
        "version": "v2",
        "content": {
            "type": "instagram",
            "messages": [
                {
                    "type": "text",
                    "text": reply
                }
            ],
            "actions": [],
            "quick_replies": []
        }
    }



# End of kila_sm.py
