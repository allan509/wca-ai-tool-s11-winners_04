"""
test_rag.py
----------
Automated tests for the RAG document-processing module.
"""

import pytest

from src.rag import (
    clean_text,
    split_text_into_chunks,
)


# ============================================================
# TEST TEXT CLEANING
# ============================================================

def test_clean_text():
    """Check that unnecessary whitespace is removed."""

    text = "  Hello   \n\n  Boma Yetu  "

    result = clean_text(text)

    assert result == "Hello\nBoma Yetu"


# ============================================================
# TEST EMPTY TEXT
# ============================================================

def test_clean_empty_text():
    """Check that empty text returns an empty string."""

    result = clean_text("")

    assert result == ""


# ============================================================
# TEST CHUNKING
# ============================================================

def test_split_text_into_chunks():
    """Check that long text is divided into chunks."""

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = split_text_into_chunks(
        text,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) > 1

    # Every chunk should contain text.
    assert all(chunks)


# ============================================================
# TEST CHUNK OVERLAP
# ============================================================

def test_chunk_overlap():
    """Check that neighboring chunks share characters."""

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = split_text_into_chunks(
        text,
        chunk_size=10,
        overlap=2,
    )

    # Last two characters of the first chunk
    # should appear at the beginning of the next chunk.
    assert chunks[0][-2:] == chunks[1][:2]


# ============================================================
# TEST EMPTY INPUT
# ============================================================

def test_empty_text_returns_empty_list():
    """Check that empty input produces no chunks."""

    result = split_text_into_chunks("")

    assert result == []


# ============================================================
# TEST INVALID CHUNK SIZE
# ============================================================

def test_invalid_chunk_size():
    """Check that an invalid chunk size raises ValueError."""

    with pytest.raises(ValueError):
        split_text_into_chunks(
            "Boma Yetu",
            chunk_size=0,
        )


# ============================================================
# TEST INVALID OVERLAP
# ============================================================

def test_invalid_overlap():
    """Check that overlap cannot equal or exceed chunk size."""

    with pytest.raises(ValueError):
        split_text_into_chunks(
            "Boma Yetu",
            chunk_size=10,
            overlap=10,
        )


# ============================================================
# TEST INVALID TEXT TYPE
# ============================================================

def test_invalid_text_type():
    """Check that text must be a string."""

    with pytest.raises(TypeError):
        split_text_into_chunks(12345)