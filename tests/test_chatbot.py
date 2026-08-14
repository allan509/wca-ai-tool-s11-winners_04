"""
Automated tests for the chatbot module.
"""

import pytest

from src.chatbot import (
    validate_question,
    format_context,
    build_prompt,
    no_context_response,
    SYSTEM_INSTRUCTIONS,
)


from src.chatbot import answer_from_knowledge
from src.vector_store import VectorStore
from src.pipeline import process_pdf

from src.vector_store import DocumentChunk
# Question validation

def test_validate_question():
    """Check that a valid question is returned correctly."""

    result = validate_question(
        "  Who qualifies for affordable housing?  "
    )

    assert result == "Who qualifies for affordable housing?"


def test_validate_empty_question():
    """Check that an empty question raises ValueError."""

    with pytest.raises(ValueError):
        validate_question("")


def test_validate_whitespace_question():
    """Check that a whitespace-only question raises ValueError."""

    with pytest.raises(ValueError):
        validate_question("   ")


def test_validate_question_type():
    """Check that the question must be a string."""

    with pytest.raises(TypeError):
        validate_question(123)
# Context formatting

def test_format_context():
    """Check that document information is formatted correctly."""

    document = DocumentChunk(
        chunk_id="001",
        text="Boma Yangu supports home ownership.",
        metadata={
            "source": "boma_yangu.pdf",
            "category": "About",
        },
    )

    result = format_context([document])

    assert "boma_yangu.pdf" in result
    assert "About" in result
    assert "Boma Yangu supports home ownership." in result


def test_format_empty_context():
    """Check that empty document lists are handled."""

    result = format_context([])

    assert result == "No relevant information was found."


def test_format_context_multiple_documents():
    """Check that multiple documents are included."""

    documents = [
        DocumentChunk(
            chunk_id="001",
            text="Boma Yangu portal information.",
            metadata={
                "source": "about.pdf",
                "category": "About",
            },
        ),
        DocumentChunk(
            chunk_id="002",
            text="Housing eligibility information.",
            metadata={
                "source": "eligibility.pdf",
                "category": "Eligibility",
            },
        ),
    ]

    result = format_context(documents)

    assert "about.pdf" in result
    assert "eligibility.pdf" in result
    assert "About" in result
    assert "Eligibility" in result


def test_format_context_invalid_type():
    """Check that documents must be provided as a list."""

    with pytest.raises(TypeError):
        format_context("not a list")


def test_format_context_invalid_document():
    """Check that list items must be DocumentChunk objects."""

    with pytest.raises(TypeError):
        format_context(["not a document"])

# Prompt building

def test_build_prompt():
    """Check that a complete prompt is created."""

    document = DocumentChunk(
        chunk_id="001",
        text="Boma Yangu supports home ownership.",
        metadata={
            "source": "boma_yangu.pdf",
            "category": "About",
        },
    )

    result = build_prompt(
        "What is Boma Yangu?",
        [document],
    )

    assert SYSTEM_INSTRUCTIONS.strip() in result
    assert "What is Boma Yangu?" in result
    assert "Boma Yangu supports home ownership." in result
    assert "boma_yangu.pdf" in result
    assert "ANSWER:" in result


def test_build_prompt_empty_context():
    """Check that a prompt can be created without context."""

    result = build_prompt(
        "What is Boma Yangu?",
        [],
    )

    assert "What is Boma Yangu?" in result
    assert "No relevant information was found." in result


def test_build_prompt_invalid_question():
    """Check that invalid questions are rejected."""

    with pytest.raises(ValueError):
        build_prompt("", [])

# Fallback response

def test_no_context_response():
    """Check the safe response when no context is available."""

    result = no_context_response()

    assert isinstance(result, str)
    assert len(result) > 0
    assert "information" in result.lower()

def test_answer_from_knowledge():
    """Test retrieval and chatbot integration."""

    store = VectorStore()

    process_pdf(
        "data/pdfs/About/test.pdf",
        store,
        document_id="TEST_DOC",
        category="About",
    )

    response = answer_from_knowledge(
        question="What is Boma Yangu?",
        store=store,
        top_k=3,
    )

    assert isinstance(response, str)
    assert response.strip() != ""


def test_answer_from_knowledge_invalid_question():
    """Test that invalid questions are rejected."""

    store = VectorStore()

    with pytest.raises((ValueError, TypeError)):
        answer_from_knowledge(
            question="",
            store=store,
        )


def test_answer_from_knowledge_no_results():
    """Test chatbot behavior when nothing relevant is retrieved."""

    store = VectorStore()

    response = answer_from_knowledge(
        question="xyz completely unrelated question",
        store=store,
        top_k=3,
    )

    assert isinstance(response, str)
    assert response.strip() != ""