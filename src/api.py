"""
api.py
------

API layer for the Boma Yetu AI Assistant.

This module exposes HTTP endpoints that allow external
applications to communicate with the chatbot.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Boma Yetu AI Assistant",
    description="API for the Boma Yetu Affordable Housing Assistant",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    """
    Represents a question sent by the user.
    """

    question: str


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):
    """
    Represents the response returned by the chatbot.
    """

    answer: str


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    """
    Confirm that the API is running.
    """

    return {
        "message": "Boma Yetu AI Assistant API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Check whether the API is healthy.
    """

    return {
        "status": "healthy"
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Receive a user's question.

    The actual chatbot/RAG/LLM connection will be added
    after the API structure has been tested.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # Temporary response while the chatbot is being connected.
    answer = (
        f"Received your question: {question}"
    )

    return ChatResponse(
        answer=answer
    )