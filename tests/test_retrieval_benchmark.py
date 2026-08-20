"""
Real-corpus retrieval benchmark for the Boma Yetu AI Assistant.

These tests evaluate retrieval quality against the actual 38-PDF
knowledge base rather than small synthetic test documents.

The complete knowledge store is built once per test module to avoid
reprocessing the 38-PDF corpus for every individual test.
"""

import pytest

from src.pipeline import process_pdf_directory
from src.retrieval import retrieve_documents
from src.vector_store import VectorStore


@pytest.fixture(scope="module")
def knowledge_store():
    """
    Build the complete Boma Yetu knowledge store once.

    The module-scoped fixture allows all tests in this file to share
    the same 193-chunk knowledge store instead of rebuilding it for
    every test.
    """

    store = VectorStore()

    process_pdf_directory(
        "data/pdfs",
        store,
    )

    return store


def test_full_corpus_size(knowledge_store):
    """Confirm that the complete corpus is loaded."""

    assert knowledge_store.count() == 193


def test_kiambu_project_retrieval(knowledge_store):
    """Kiambu project query should retrieve the dedicated project document."""

    results = retrieve_documents(
        "Kiambu County land bank",
        knowledge_store,
        top_k=3,
    )

    document_ids = [
        document.chunk_id
        for document in results
    ]

    assert any(
        "Projects_Kiambu_County_Land_Bank" in document_id
        for document_id in document_ids
    )


def test_nyanza_project_retrieval(knowledge_store):
    """Nyanza/Rift/Western query should retrieve its project document."""

    results = retrieve_documents(
        "Nyanza Rift Western housing projects",
        knowledge_store,
        top_k=3,
    )

    document_ids = [
        document.chunk_id
        for document in results
    ]

    assert any(
        "Projects_Nyanza_Rift_Western" in document_id
        for document_id in document_ids
    )


def test_eligibility_retrieval(knowledge_store):
    """Eligibility questions should retrieve eligibility information."""

    results = retrieve_documents(
        "Who is eligible for the Affordable Housing Programme?",
        knowledge_store,
        top_k=3,
    )

    document_ids = [
        document.chunk_id
        for document in results
    ]

    assert any(
        "Eligibility" in document_id
        for document_id in document_ids
    )


def test_registration_retrieval(knowledge_store):
    """Registration questions should retrieve registration information."""

    results = retrieve_documents(
        "How do I register for Boma Yangu?",
        knowledge_store,
        top_k=3,
    )

    document_ids = [
        document.chunk_id
        for document in results
    ]

    assert any(
        "Application_Registration" in document_id
        for document_id in document_ids
    )


def test_housing_levy_retrieval(knowledge_store):
    """Housing levy questions should retrieve levy information."""

    results = retrieve_documents(
        "How does the housing levy work?",
        knowledge_store,
        top_k=3,
    )

    document_ids = [
        document.chunk_id
        for document in results
    ]

    assert any(
        "Housing_Levy" in document_id
        for document_id in document_ids
    )