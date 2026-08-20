"""
Real-corpus retrieval benchmark for the Boma Yetu AI Assistant.

These tests evaluate retrieval quality against the actual 38-PDF
knowledge base rather than small synthetic test documents.
"""

from src.pipeline import process_pdf_directory
from src.retrieval import retrieve_documents
from src.vector_store import VectorStore


def build_knowledge_store():
    """Build a VectorStore from the complete Boma Yetu PDF corpus."""

    store = VectorStore()

    process_pdf_directory(
        "data/pdfs",
        store,
    )

    return store


def test_full_corpus_size():
    """Confirm that the complete corpus is loaded."""

    store = build_knowledge_store()

    assert store.count() == 193


def test_kiambu_project_retrieval():
    """Kiambu project query should retrieve the dedicated project document."""

    store = build_knowledge_store()

    results = retrieve_documents(
        "Kiambu County land bank",
        store,
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


def test_nyanza_project_retrieval():
    """Nyanza/Rift/Western query should retrieve its project document."""

    store = build_knowledge_store()

    results = retrieve_documents(
        "Nyanza Rift Western housing projects",
        store,
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


def test_eligibility_retrieval():
    """Eligibility questions should retrieve eligibility information."""

    store = build_knowledge_store()

    results = retrieve_documents(
        "Who is eligible for the Affordable Housing Programme?",
        store,
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


def test_registration_retrieval():
    """Registration questions should retrieve registration information."""

    store = build_knowledge_store()

    results = retrieve_documents(
        "How do I register for Boma Yangu?",
        store,
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


def test_housing_levy_retrieval():
    """Housing levy questions should retrieve levy information."""

    store = build_knowledge_store()

    results = retrieve_documents(
        "How does the housing levy work?",
        store,
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
