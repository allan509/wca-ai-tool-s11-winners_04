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
from src.memory import MemoryManager
from uuid import uuid4

# ============================================================
# CREATE KNOWLEDGE STORE
# ============================================================

knowledge_store = VectorStore()

process_pdf_directory(
    "data/pdfs",
    knowledge_store,
)
# Creat conversation memory
memory_manager = MemoryManager(
    max_messages=20
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

    question: str

    conversation_id: str | None = None

# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):
    """
    Represents the response returned by the chatbot.
    """

    answer: str
    conversation_id: str


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

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    conversation_id = (
        request.conversation_id
        or str(uuid4())
    )

    memory = memory_manager.get_or_create(
        conversation_id
    )

    try:

        
        answer = answer_with_llm(
            question=question,
            store=knowledge_store,
            top_k=3,
        conversation_history=memory.get_messages(),
        )
        

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate answer: {exc}",
        ) from exc

    memory.add_user_message(question)
    memory.add_assistant_message(answer)

    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
    )