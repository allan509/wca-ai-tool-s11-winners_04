"""
api.py
------

FastAPI interface for the Boma Yetu AI Assistant.

Architecture:

HTTP Request
    ↓
FastAPI
    ↓
RAG Retrieval
    ↓
Prompt Construction
    ↓
OpenAI LLM
    ↓
Answer
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.chatbot import answer_with_llm
from src.pipeline import process_pdf_directory
from src.vector_store import VectorStore


# ============================================================
# CREATE KNOWLEDGE STORE
# ============================================================

knowledge_store = VectorStore()

process_pdf_directory(
    "data/pdfs",
    knowledge_store,
)


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
    Receive a user's question and generate
    a grounded answer using the RAG + LLM pipeline.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        answer = answer_with_llm(
            question=question,
            store=knowledge_store,
            top_k=3,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate answer: {exc}",
        ) from exc

    return ChatResponse(
        answer=answer,
    )