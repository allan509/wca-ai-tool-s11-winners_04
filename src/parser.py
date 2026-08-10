"""
parser.py
---------
PDF document parser for the Boma Yetu AI Assistant.

This module is responsible for:
1. Finding PDF files.
2. Opening PDF files.
3. Extracting text from PDF pages.
4. Returning the extracted text in a structured form.

It does NOT:
- Generate embeddings.
- Store documents in the vector database.
- Generate AI responses.
- Handle the user interface.
"""

from pathlib import Path

from pypdf import PdfReader


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        All extracted text from the PDF as one string.

    Raises:
        FileNotFoundError:
            If the PDF file does not exist.

        ValueError:
            If the supplied file is not a PDF.
    """

    # Convert the supplied path into a Path object.
    pdf_path = Path(pdf_path)

    # Check that the file actually exists.
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    # Check that the file has a PDF extension.
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, but received: {pdf_path}"
        )

    # Open the PDF.
    reader = PdfReader(pdf_path)

    # Store text from each page.
    pages_text = []

    # Extract text page by page.
    for page in reader.pages:

        # extract_text() may return None for pages
        # that contain no extractable text.
        text = page.extract_text()

        if text:
            pages_text.append(text)

    # Join all page text together.
    return "\n\n".join(pages_text)


# ============================================================
# FIND PDF FILES
# ============================================================

def find_pdf_files(pdf_directory: Path) -> list[Path]:
    """
    Find all PDF files inside a directory.

    The search is recursive, meaning it also searches
    inside subdirectories.

    Args:
        pdf_directory: Directory containing PDF files.

    Returns:
        A list of PDF file paths.
    """

    pdf_directory = Path(pdf_directory)

    # Check whether the directory exists.
    if not pdf_directory.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_directory}"
        )

    # Find PDF files recursively.
    return list(pdf_directory.rglob("*.pdf"))