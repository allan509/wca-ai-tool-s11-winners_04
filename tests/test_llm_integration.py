import pytest

from src.chatbot import answer_with_llm
from src.pipeline import process_pdf
from src.vector_store import VectorStore


TEST_PDF = "tests/data/test.pdf"


def test_answer_with_llm(monkeypatch):
    """
    Test the complete knowledge-to-LLM pipeline
    without making a real API request.
    """

    store = VectorStore()

    process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    def fake_generate_response(prompt):
        assert isinstance(prompt, str)
        assert "What is Boma Yangu?" in prompt
        assert "RELEVANT KNOWLEDGE:" in prompt
        assert "CURRENT USER QUESTION:" in prompt

        return "This is a mocked Boma Yangu response."

    monkeypatch.setattr(
        "src.llm.generate_response",
        fake_generate_response,
    )

    response = answer_with_llm(
        question="What is Boma Yangu?",
        store=store,
        top_k=3,
    )

    assert response == (
        "This is a mocked Boma Yangu response."
    )


def test_answer_with_llm_invalid_question():
    """
    Test that invalid questions are rejected.
    """

    store = VectorStore()

    with pytest.raises((ValueError, TypeError)):
        answer_with_llm(
            question="",
            store=store,
        )


def test_answer_with_llm_no_knowledge():
    """
    Test behavior when no relevant knowledge is found.
    """

    store = VectorStore()

    response = answer_with_llm(
        question="xyz completely unrelated information",
        store=store,
        top_k=3,
    )

    assert isinstance(response, str)
    assert response.strip() != ""
    