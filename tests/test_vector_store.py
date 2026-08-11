"""
test_vector_store.py
--------------------
Automated tests for the vector_store module.
"""

import pytest

from src.vector_store import (
    DocumentChunk,
    VectorStore,
)


# ============================================================
# DOCUMENT CHUNK TESTS
# ============================================================

def test_create_document_chunk():
    """Check that a DocumentChunk can be created."""

    document = DocumentChunk(
        chunk_id="001",
        text="Boma Yangu information",
    )

    assert document.chunk_id == "001"
    assert document.text == "Boma Yangu information"
    assert document.metadata == {}


def test_document_chunk_with_metadata():
    """Check that metadata is stored correctly."""

    metadata = {
        "source": "boma_yangu.pdf",
        "category": "About",
    }

    document = DocumentChunk(
        chunk_id="001",
        text="Boma Yangu information",
        metadata=metadata,
    )

    assert document.metadata == metadata
    assert document.metadata["source"] == "boma_yangu.pdf"


# ============================================================
# DOCUMENT CHUNK VALIDATION
# ============================================================

def test_empty_chunk_id():
    """Check that an empty chunk ID raises ValueError."""

    with pytest.raises(ValueError):
        DocumentChunk(
            chunk_id="",
            text="Boma Yangu information",
        )


def test_invalid_chunk_id_type():
    """Check that chunk_id must be a string."""

    with pytest.raises(TypeError):
        DocumentChunk(
            chunk_id=123,
            text="Boma Yangu information",
        )


def test_empty_text():
    """Check that empty text raises ValueError."""

    with pytest.raises(ValueError):
        DocumentChunk(
            chunk_id="001",
            text="",
        )


def test_invalid_text_type():
    """Check that text must be a string."""

    with pytest.raises(TypeError):
        DocumentChunk(
            chunk_id="001",
            text=123,
        )


def test_invalid_metadata_type():
    """Check that metadata must be a dictionary."""

    with pytest.raises(TypeError):
        DocumentChunk(
            chunk_id="001",
            text="Boma Yangu",
            metadata="invalid metadata",
        )


# ============================================================
# VECTOR STORE TESTS
# ============================================================

def test_create_vector_store():
    """Check that a new VectorStore starts empty."""

    store = VectorStore()

    assert store.count() == 0


def test_add_document():
    """Check that a document can be added."""

    store = VectorStore()

    document = DocumentChunk(
        chunk_id="001",
        text="Boma Yangu information",
    )

    store.add_document(document)

    assert store.count() == 1


def test_get_document():
    """Check that a document can be retrieved by ID."""

    store = VectorStore()

    document = DocumentChunk(
        chunk_id="001",
        text="Boma Yangu information",
    )

    store.add_document(document)

    result = store.get_document("001")

    assert result is document
    assert result.text == "Boma Yangu information"


def test_get_missing_document():
    """Check that a missing document returns None."""

    store = VectorStore()

    result = store.get_document("does-not-exist")

    assert result is None


# ============================================================
# MULTIPLE DOCUMENT TESTS
# ============================================================

def test_add_multiple_documents():
    """Check that multiple documents can be added."""

    store = VectorStore()

    documents = [
        DocumentChunk(
            chunk_id="001",
            text="Boma Yangu",
        ),
        DocumentChunk(
            chunk_id="002",
            text="Affordable Housing",
        ),
        DocumentChunk(
            chunk_id="003",
            text="Housing allocation",
        ),
    ]

    store.add_documents(documents)

    assert store.count() == 3


def test_get_all_documents():
    """Check that all stored documents can be retrieved."""

    store = VectorStore()

    documents = [
        DocumentChunk("001", "Boma Yangu"),
        DocumentChunk("002", "Affordable Housing"),
    ]

    store.add_documents(documents)

    result = store.get_all_documents()

    assert len(result) == 2
    assert result[0].chunk_id == "001"
    assert result[1].chunk_id == "002"


# ============================================================
# DELETE TESTS
# ============================================================

def test_delete_document():
    """Check that a document can be deleted."""

    store = VectorStore()

    document = DocumentChunk(
        chunk_id="001",
        text="Boma Yangu",
    )

    store.add_document(document)

    result = store.delete_document("001")

    assert result is True
    assert store.count() == 0
    assert store.get_document("001") is None


def test_delete_missing_document():
    """Check that deleting a missing document returns False."""

    store = VectorStore()

    result = store.delete_document("does-not-exist")

    assert result is False


# ============================================================
# CLEAR TEST
# ============================================================

def test_clear_store():
    """Check that clear removes all documents."""

    store = VectorStore()

    documents = [
        DocumentChunk("001", "Boma Yangu"),
        DocumentChunk("002", "Affordable Housing"),
    ]

    store.add_documents(documents)

    assert store.count() == 2

    store.clear()

    assert store.count() == 0
    assert store.get_all_documents() == []