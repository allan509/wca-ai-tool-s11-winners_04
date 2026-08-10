from pathlib import Path

import pytest

from src.parser import (
    extract_text_from_pdf,
    find_pdf_files,
)

from src.config import PDF_DIR


def test_find_pdf_files():
    """Check that the parser can find PDF files."""
    pdf_files = find_pdf_files(PDF_DIR)

    assert len(pdf_files) > 0
    assert all(file.suffix.lower() == ".pdf" for file in pdf_files)


def test_extract_text_from_pdf():
    """Check that text can be extracted from a PDF."""

    test_pdf = PDF_DIR / "About" / "test.pdf"

    text = extract_text_from_pdf(test_pdf)

    assert text
    assert isinstance(text, str)


def test_missing_pdf():
    """Check that a missing PDF raises FileNotFoundError."""

    missing_pdf = PDF_DIR / "About" / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(missing_pdf)


def test_non_pdf_file():
    """Check that a non-PDF file raises ValueError."""

    test_file = PDF_DIR / "About" / "test.txt"

    # Create a temporary text file for this test.
    test_file.write_text("This is not a PDF.")

    try:
        with pytest.raises(ValueError):
            extract_text_from_pdf(test_file)
    finally:
        # Remove the temporary test file after the test.
        test_file.unlink()