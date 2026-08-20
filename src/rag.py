"""

Text cleaning and chunking utilities for the Boma Yetu
AI Assistant knowledge pipeline.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Removes:
    - excessive whitespace
    - browser/file URI artifacts
    - common PDF page-number artifacts
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()

    if not text:
        return ""

    lines = []

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # Remove local HTML/PDF-generation file paths.
        if re.search(
            r"file:///.*deepseek_html_.*\.html",
            line,
            re.IGNORECASE,
        ):
            continue

        # Remove generated PDF page headers/footers.
        if re.match(
            r"^\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M\s+BY-[A-Z0-9-]+\s+[–-]",
            line,
            re.IGNORECASE,
        ):
            continue

        
        # Remove common page footer patterns such as:
        # "1/3", "2/3", "3/3"
        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue

        # Remove lines containing the generated HTML path.
        if "deepseek_html_" in line.lower():
            continue

        lines.append(line)

    return "\n".join(lines)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into smaller overlapping chunks.

    The function prefers paragraph/line/word boundaries when
    possible, but falls back to character boundaries when a
    single word is longer than the chunk size.

    Args:
        text:
            Document text to split.

        chunk_size:
            Maximum approximate number of characters per chunk.

        overlap:
            Number of characters shared between neighboring chunks.

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
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        # Initial target endpoint.
        target_end = min(
            start + chunk_size,
            text_length,
        )

        # If this is the final section, use the remaining text.
        if target_end == text_length:
            chunk = text[start:target_end].strip()

            if chunk:
                chunks.append(chunk)

            break

        # Prefer a newline boundary.
        newline_position = text.rfind(
            "\n",
            start,
            target_end,
        )

        # Otherwise prefer a whitespace boundary.
        whitespace_position = text.rfind(
            " ",
            start,
            target_end,
        )

        if newline_position > start:
            end = newline_position

        elif whitespace_position > start:
            end = whitespace_position

        else:
            # No safe word boundary exists.
            # Split at the character limit.
            end = target_end

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while preserving overlap.
        start = end - overlap

        # Prevent an infinite loop.
        if start <= 0:
            start = end

    return chunks
