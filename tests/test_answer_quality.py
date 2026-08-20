"""
Answer quality checks for the Boma Yetu AI Assistant.

These tests verify that generated answers contain important
facts expected from the knowledge base.
"""

from src.chatbot import answer_with_llm
from src.pipeline import process_pdf_directory
from src.vector_store import VectorStore


def build_knowledge_store():

    store = VectorStore()

    process_pdf_directory(
        "data/pdfs",
        store,
    )

    return store


def test_contact_information():

    store = build_knowledge_store()

    answer = answer_with_llm(
        "What phone numbers can I use to contact Boma Yangu?",
        store,
        top_k=3,
    )

    answer_lower = answer.lower()

    assert "0700 832 832" in answer
    assert "0739 832 832" in answer
    assert "support@bomayangu.go.ke" in answer_lower


def test_ecitizen_access():

    store = build_knowledge_store()

    answer = answer_with_llm(
        "Can I access Boma Yangu through eCitizen?",
        store,
        top_k=3,
    )

    answer_lower = answer.lower()

    assert "ecitizen" in answer_lower
    assert (
        "yes" in answer_lower
        or "can access" in answer_lower
    )


def test_registration_answer():

    store = build_knowledge_store()

    answer = answer_with_llm(
        "How do I register for Boma Yangu?",
        store,
        top_k=3,
    )

    answer_lower = answer.lower()

    assert "bomayangu.go.ke" in answer_lower
    assert "*832#" in answer
    assert "huduma" in answer_lower


def test_kiambu_answer():

    store = build_knowledge_store()

    answer = answer_with_llm(
        "What housing projects are available in Kiambu County?",
        store,
        top_k=3,
    )

    answer_lower = answer.lower()

    assert "kiambu" in answer_lower
    assert (
        "kikuyu" in answer_lower
        or "ruiru" in answer_lower
    )


def test_nairobi_answer():

    store = build_knowledge_store()

    answer = answer_with_llm(
        "What housing projects are available in Nairobi?",
        store,
        top_k=3,
    )

    answer_lower = answer.lower()

    assert "nairobi" in answer_lower
    assert (
        "mukuru" in answer_lower
        or "kibera" in answer_lower
        or "ngong vet" in answer_lower
    )


def test_nyanza_answer():

    store = build_knowledge_store()

    answer = answer_with_llm(
        "What housing projects are available in Nyanza?",
        store,
        top_k=3,
    )

    answer_lower = answer.lower()

    assert "nyanza" in answer_lower
    assert (
        "kisumu" in answer_lower
        or "lumumba" in answer_lower
        or "ogembo" in answer_lower
    )