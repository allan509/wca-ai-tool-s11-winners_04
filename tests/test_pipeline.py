"""
test_pipeline.py
----------------

Integration tests for the Boma Yetu knowledge pipeline.

Tests:

PDF
 ↓
parser.py
 ↓
rag.py
 ↓
vector_store.py
"""

from pathlib import Path

import pytest

from src.pipeline import (
    process_pdf,
    process_pdf_directory,
    search_knowledge,
)
from src.vector_store import VectorStore


# ============================================================
# TEST PDF LOCATION
# ============================================================

TEST_PDF = Path("data/pdfs/About/test.pdf")


# ============================================================
# TEST PROCESS PDF
# ============================================================

def test_process_pdf():
    """
    Test that a PDF can be processed and its chunks
    added to the vector store.
    """

    store = VectorStore()

    count = process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    assert count > 0

    documents = store.get_all_documents()

    assert len(documents) == count


# ============================================================
# TEST DOCUMENT CONTENT
# ============================================================

def test_processed_document_contains_text():
    """
    Test that processed chunks contain actual text.
    """

    store = VectorStore()

    process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    documents = store.get_all_documents()

    assert len(documents) > 0

    for document in documents:
        assert isinstance(document.text, str)
        assert document.text.strip() != ""


# ============================================================
# TEST DOCUMENT IDs
# ============================================================

def test_processed_document_ids():
    """
    Test that each generated chunk receives a unique ID.
    """

    store = VectorStore()

    process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    documents = store.get_all_documents()

    ids = [document.chunk_id for document in documents]

    assert len(ids) == len(set(ids))

    for chunk_id in ids:
        assert chunk_id.startswith("TEST_DOC_")


# ============================================================
# TEST METADATA
# ============================================================

def test_processed_document_metadata():
    """
    Test that document metadata is attached correctly.
    """

    store = VectorStore()

    process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    documents = store.get_all_documents()

    assert len(documents) > 0

    for document in documents:
        assert document.metadata["document_id"] == "TEST_DOC"
        assert document.metadata["category"] == "About"
        assert "source" in document.metadata


# ============================================================
# TEST MISSING PDF
# ============================================================

def test_process_missing_pdf():
    """
    Test that a missing PDF raises FileNotFoundError.
    """

    store = VectorStore()

    with pytest.raises(FileNotFoundError):
        process_pdf(
            "data/pdfs/About/not_found.pdf",
            store,
        )


# ============================================================
# TEST NON-PDF FILE
# ============================================================

def test_process_non_pdf():
    """
    Test that a non-PDF file is rejected.
    """

    store = VectorStore()

    non_pdf = Path("data/pdfs/About/not_a_pdf.txt")

    non_pdf.write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError):
            process_pdf(
                non_pdf,
                store,
            )

    finally:
        non_pdf.unlink(missing_ok=True)


# ============================================================
# TEST EMPTY STORE BEFORE PROCESSING
# ============================================================

def test_empty_store_before_processing():
    """
    Confirm that the vector store starts empty.
    """

    store = VectorStore()

    assert store.get_all_documents() == []


# ============================================================
# TEST PDF DIRECTORY
# ============================================================

def test_process_pdf_directory():
    """
    Test that all PDFs in the About directory can be processed.
    """

    store = VectorStore()

    count = process_pdf_directory(
        "data/pdfs/About",
        store,
    )

    assert count > 0

    documents = store.get_all_documents()

    assert len(documents) == count


# ============================================================
# TEST DIRECTORY METADATA
# ============================================================

def test_process_pdf_directory_category():
    """
    Test that the directory name is used as the category.
    """

    store = VectorStore()

    process_pdf_directory(
        "data/pdfs/About",
        store,
    )

    documents = store.get_all_documents()

    assert len(documents) > 0

    for document in documents:
        assert document.metadata["category"] == "About"

# ============================================================
# TEST KNOWLEDGE SEARCH
# ============================================================

def test_search_knowledge():
    """
    Test that the pipeline can search processed
    knowledge using retrieval.py.
    """

    store = VectorStore()

    process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    results = search_knowledge(
        query="Boma Yangu",
        store=store,
        top_k=3,
    )

    assert isinstance(results, list)

    assert len(results) > 0

    assert len(results) <= 3


# ============================================================
# TEST RELEVANT SEARCH CONTENT
# ============================================================

def test_search_knowledge_returns_relevant_content():
    """
    Test that searching the knowledge base returns
    document chunks containing relevant information.
    """

    store = VectorStore()

    process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    results = search_knowledge(
        query="Boma Yangu",
        store=store,
        top_k=3,
    )

    assert len(results) > 0

    combined_text = " ".join(
        document.text.lower()
        for document in results
    )

    assert "boma yangu" in combined_text


# ============================================================
# TEST TOP K
# ============================================================

def test_search_knowledge_top_k():
    """
    Test that top_k limits the number of returned
    documents.
    """

    store = VectorStore()

    process_pdf(
        TEST_PDF,
        store,
        document_id="TEST_DOC",
        category="About",
    )

    results = search_knowledge(
        query="housing",
        store=store,
        top_k=2,
    )

    assert len(results) <= 2