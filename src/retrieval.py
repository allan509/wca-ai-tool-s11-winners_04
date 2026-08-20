"""
Retrieves information from the document store.

Current version:
- Uses weighted keyword matching.
- Does not require an embedding model.
- Does not require an external vector database.

The retrieval strategy gives greater importance to distinctive
query terms such as "Nairobi", "Nyanza", and "eCitizen" while
reducing the influence of common words such as "housing".
"""

import math
import re

from src.vector_store import DocumentChunk, VectorStore


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "where",
    "who",
    "with",
    "you",
}


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase words.

    Punctuation is removed so that words can be compared
    consistently.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(r"\b\w+\b", text.lower())


def meaningful_tokens(text: str) -> list[str]:
    """
    Return meaningful tokens after removing common stop words.
    """

    return [
        token
        for token in tokenize(text)
        if token not in STOP_WORDS
    ]


# ============================================================
# ORIGINAL KEYWORD SCORING
# ============================================================

def calculate_keyword_score(
    query: str,
    text: str,
) -> int:
    """
    Calculate how many query words appear in a document.

    This function is intentionally retained for compatibility
    with the existing unit tests.
    """

    query_words = set(tokenize(query))
    text_words = set(tokenize(text))

    return len(query_words.intersection(text_words))


# ============================================================
# DOCUMENT FREQUENCY
# ============================================================

def calculate_document_frequency(
    query_tokens: list[str],
    documents: list[DocumentChunk],
) -> dict[str, int]:
    """
    Calculate how many documents contain each query token.

    Rare terms receive greater importance during retrieval.
    """

    frequency = {
        token: 0
        for token in query_tokens
    }

    for document in documents:

        document_tokens = set(
            meaningful_tokens(document.text)
        )

        for token in query_tokens:

            if token in document_tokens:
                frequency[token] += 1

    return frequency


# ============================================================
# WEIGHTED RELEVANCE SCORE
# ============================================================

def calculate_relevance_score(
    query: str,
    document: DocumentChunk,
    documents: list[DocumentChunk],
) -> float:
    """
    Calculate a weighted relevance score.

    Rare query terms receive higher weights than common terms.

    Exact phrase matches and source/document-name matches
    receive additional bonuses.
    """

    query_tokens = list(
        dict.fromkeys(
            meaningful_tokens(query)
        )
    )

    if not query_tokens:
        return 0.0

    document_tokens = set(
        meaningful_tokens(document.text)
    )

    frequencies = calculate_document_frequency(
        query_tokens,
        documents,
    )

    total_documents = max(
        len(documents),
        1,
    )

    score = 0.0

    # --------------------------------------------------------
    # Weighted keyword matching
    # --------------------------------------------------------

    for token in query_tokens:

        if token not in document_tokens:
            continue

        document_frequency = frequencies.get(
            token,
            0,
        )

        # IDF-style weighting.
        #
        # Rare words such as "nairobi", "nyanza",
        # and "ecitizen" receive higher scores.
        weight = math.log(
            (total_documents + 1)
            / (document_frequency + 1)
        ) + 1

        score += weight

    # --------------------------------------------------------
    # Exact phrase bonus
    # --------------------------------------------------------

    normalized_query = " ".join(
        meaningful_tokens(query)
    )

    normalized_text = " ".join(
        meaningful_tokens(document.text)
    )

    if (
        normalized_query
        and normalized_query in normalized_text
    ):
        score += 3.0

    # --------------------------------------------------------
    # Source/document filename bonus
    # --------------------------------------------------------

    source = str(
        document.metadata.get(
            "source",
            "",
        )
    ).lower()

    document_id = str(
        document.chunk_id
    ).lower()

    searchable_metadata = (
        source
        + " "
        + document_id
    )

    for token in query_tokens:

        if token in searchable_metadata:
            score += 1.5

    return score


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    store: VectorStore,
    top_k: int = 3,
) -> list[DocumentChunk]:
    """
    Retrieve the most relevant document chunks.

    Retrieval uses weighted keyword relevance rather than
    simple keyword frequency.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if not query.strip():
        return []

    if not isinstance(store, VectorStore):
        raise TypeError(
            "store must be a VectorStore"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    documents = store.get_all_documents()

    if not documents:
        return []

    scored_documents = []

    for document in documents:

        score = calculate_relevance_score(
            query,
            document,
            documents,
        )

        if score > 0:
            scored_documents.append(
                (score, document)
            )

    # Highest relevance first.
    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for score, document
        in scored_documents[:top_k]
    ]