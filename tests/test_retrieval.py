"""
Automated tests for the retrieval module.
"""

import pytest

from src.retrieval import (
    tokenize,
    calculate_keyword_score,
    retrieve_documents,
)

from src.vector_store import (
    DocumentChunk,
    VectorStore,
)

# Tokenize tests

def test_tokenize():
    """Check that text is converted to lowercase words."""

    result = tokenize("Boma Yangu Portal!")

    assert result == [
        "boma",
        "yangu",
        "portal",
    ]


def test_tokenize_removes_punctuation():
    """Check that punctuation is removed."""

    result = tokenize("Housing, Kenya!")

    assert "," not in result
    assert "!" not in result

    assert result == [
        "housing",
        "kenya",
    ]


def test_tokenize_invalid_type():
    """Check that tokenize requires a string."""

    with pytest.raises(TypeError):
        tokenize(123)

# Keyword Score Tests

def test_keyword_score():
    """Check that matching keywords produce a positive score."""

    score = calculate_keyword_score(
        "Boma Yangu",
        "The Boma Yangu portal supports housing.",
    )

    assert score == 2


def test_keyword_score_no_match():
    """Check that unrelated text produces a zero score."""

    score = calculate_keyword_score(
        "chicken farming",
        "Affordable housing in Kenya.",
    )

    assert score == 0


def test_keyword_score_case_insensitive():
    """Check that keyword matching ignores capitalization."""

    score = calculate_keyword_score(
        "BOMA YANGU",
        "boma yangu portal",
    )

    assert score == 2

# Retrieval tests

def create_test_store():
    """
    Create a VectorStore containing sample Boma Yetu
    documents for testing.
    """

    store = VectorStore()

    documents = [
        DocumentChunk(
            chunk_id="001",
            text=(
                "Boma Yangu is a housing portal "
                "that supports home ownership."
            ),
            metadata={
                "source": "boma_yangu.pdf",
                "category": "About",
            },
        ),
        DocumentChunk(
            chunk_id="002",
            text=(
                "The Affordable Housing Program "
                "provides affordable housing units."
            ),
            metadata={
                "source": "housing.pdf",
                "category": "Program",
            },
        ),
        DocumentChunk(
            chunk_id="003",
            text=(
                "Applicants can access housing "
                "allocation through the Boma Yangu portal."
            ),
            metadata={
                "source": "allocation.pdf",
                "category": "Allocation",
            },
        ),
    ]

    store.add_documents(documents)

    return store


def test_retrieve_documents():
    """Check that relevant documents are returned."""

    store = create_test_store()

    results = retrieve_documents(
        "Boma Yangu housing",
        store,
    )

    assert len(results) > 0


def test_retrieve_most_relevant_document():
    """Check that the highest-scoring document comes first."""

    store = create_test_store()

    results = retrieve_documents(
        "Boma Yangu home ownership",
        store,
    )

    assert results[0].chunk_id == "001"


def test_retrieve_top_k():
    """Check that top_k limits the number of results."""

    store = create_test_store()

    results = retrieve_documents(
        "housing Boma Yangu",
        store,
        top_k=2,
    )

    assert len(results) <= 2

def test_retrieve_no_match():
    """Check that unrelated queries return no documents."""

    store = create_test_store()

    results = retrieve_documents(
        "chicken farming",
        store,
    )

    assert results == []


def test_retrieve_empty_query():
    """Check that an empty query returns an empty list."""

    store = create_test_store()

    results = retrieve_documents(
        "",
        store,
    )

    assert results == []

# Input validation

def test_retrieve_invalid_query():
    """Check that query must be a string."""

    store = create_test_store()

    with pytest.raises(TypeError):
        retrieve_documents(
            123,
            store,
        )


def test_retrieve_invalid_store():
    """Check that store must be a VectorStore."""

    with pytest.raises(TypeError):
        retrieve_documents(
            "Boma Yangu",
            "not a store",
        )


def test_retrieve_invalid_top_k():
    """Check that top_k must be greater than zero."""

    store = create_test_store()

    with pytest.raises(ValueError):
        retrieve_documents(
            "Boma Yangu",
            store,
            top_k=0,
        )