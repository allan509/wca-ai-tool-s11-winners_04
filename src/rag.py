"""
Document preparation for the Boma Yetu AI Assistant.

This module is responsible for:
1. Taking extracted document text.
2. Cleaning the text.
3. Splitting long documents into smaller chunks.
4. Preparing chunks for later retrieval.

It does NOT:
- Read PDF files directly.
- Generate embeddings.
- Store vectors.
- Generate chatbot responses.
"""



# text cleaning


def clean_text(text: str) -> str:
    """
    Clean extracted document text.

    The function:
    - Removes unnecessary whitespace.
    - Removes excessive blank lines.
    - Returns clean text.

    Args:
        text: Extracted document text.

    Returns:
        Cleaned text.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Remove leading and trailing whitespace.
    text = text.strip()

    # Split into lines and remove unnecessary whitespace.
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Join the cleaned lines.
    return "\n".join(lines)



# text chunking

def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into smaller overlapping chunks.

    Args:
        text:
            The document text to split.

        chunk_size:
            Maximum approximate number of characters in
            each chunk.

        overlap:
            Number of characters shared between neighboring
            chunks.

    Returns:
        A list of text chunks.

    Raises:
        TypeError:
            If text is not a string.

        ValueError:
            If chunk_size or overlap is invalid.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    # Clean the text before splitting.
    text = clean_text(text)

    # If there is no text, return an empty list.
    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        # Get the current chunk.
        chunk = text[start:end]

        chunks.append(chunk)

        # Move forward while keeping the overlap.
        start = end - overlap

    return chunks