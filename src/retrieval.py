"""
Retrieves information from the document store.

Current version:
- Uses simple keyword matching.
- Does not require an embedding model.
- Does not require an external vector database.

Later:
- This module can be upgraded to semantic/vector similarity
  search without changing the chatbot's overall architecture.
"""

import re

from src.vector_store import DocumentChunk, VectorStore

# Text processing

def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase words.

    Punctuation is removed so that words can be compared
    consistently.

    Args:
        text: Text to tokenize.

    Returns:
        A list of lowercase words.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(r"\b\w+\b", text.lower())

# Keyword scoring

def calculate_keyword_score(
    query: str,
    text: str,
) -> int:
    """
    Calculate how many query words appear in a document.

    Args:
        query: User's question or search query.
        text: Document chunk text.

    Returns:
        Number of matching keywords.
    """

    query_words = set(tokenize(query))
    text_words = set(tokenize(text))

    return len(query_words.intersection(text_words))

# Retrieval

def retrieve_documents(
    query: str,
    store: VectorStore,
    top_k: int = 3,
) -> list[DocumentChunk]:
    """
    Retrieve the most relevant document chunks.

    The current implementation ranks chunks according to
    the number of keywords they share with the query.

    Args:
        query:
            User's question or search query.

        store:
            VectorStore containing document chunks.

        top_k:
            Maximum number of results to return.

    Returns:
        List of the most relevant DocumentChunk objects.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if not query.strip():
        return []

    if not isinstance(store, VectorStore):
        raise TypeError("store must be a VectorStore")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    scored_documents = []

    for document in store.get_all_documents():

        score = calculate_keyword_score(
            query,
            document.text,
        )

        if score > 0:
            scored_documents.append(
                (score, document)
            )

    # Highest score first.
    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # Return only the requested number of documents.
    return [
        document
        for score, document in scored_documents[:top_k]
    ]