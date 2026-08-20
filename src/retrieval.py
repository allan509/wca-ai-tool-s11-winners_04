"""
Retrieves information from the document store.

Retrieval strategy:
- Weighted keyword matching.
- IDF-style weighting for distinctive terms.
- Exact phrase matching bonus.
- Source/document-name matching bonus.
- No embedding model required.
- No external vector database required.

Performance:
- Query tokens are calculated once.
- Document tokens are calculated once per retrieval.
- Document frequencies are calculated once per retrieval.
- Preprocessed document text is reused during scoring.
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
    Convert text into lowercase word tokens.

    Punctuation is removed so words can be compared
    consistently.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def meaningful_tokens(text: str) -> list[str]:
    """
    Return meaningful tokens after removing stop words.
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

    Retained for compatibility with existing tests and
    application code.
    """

    query_words = set(tokenize(query))
    text_words = set(tokenize(text))

    return len(
        query_words.intersection(text_words)
    )


# ============================================================
# DOCUMENT FREQUENCY
# ============================================================

def calculate_document_frequency(
    query_tokens: list[str],
    documents: list[DocumentChunk],
) -> dict[str, int]:
    """
    Calculate how many documents contain each query token.

    Each document contributes at most one occurrence for
    each token.
    """

    frequency = {
        token: 0
        for token in query_tokens
    }

    if not query_tokens or not documents:
        return frequency

    query_token_set = set(query_tokens)

    for document in documents:

        document_tokens = set(
            meaningful_tokens(document.text)
        )

        matched_tokens = (
            query_token_set.intersection(
                document_tokens
            )
        )

        for token in matched_tokens:
            frequency[token] += 1

    return frequency


# ============================================================
# INTERNAL OPTIMIZED SCORING
# ============================================================

def _score_document(
    query_tokens: list[str],
    normalized_query: str,
    document: DocumentChunk,
    document_tokens: set[str],
    normalized_text: str,
    frequencies: dict[str, int],
    total_documents: int,
) -> float:
    """
    Calculate a document relevance score using
    precomputed values.
    """

    if not query_tokens:
        return 0.0

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

        weight = math.log(
            (total_documents + 1)
            / (document_frequency + 1)
        ) + 1

        score += weight

    # --------------------------------------------------------
    # Exact phrase bonus
    # --------------------------------------------------------

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
# PUBLIC RELEVANCE SCORE
# ============================================================

def calculate_relevance_score(
    query: str,
    document: DocumentChunk,
    documents: list[DocumentChunk],
) -> float:
    """
    Calculate a weighted relevance score.

    This public function retains the original interface.

    For repeated scoring across a corpus, retrieve_documents()
    uses the optimized cached scoring path.
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

    normalized_query = " ".join(
        query_tokens
    )

    normalized_text = " ".join(
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

    return _score_document(
        query_tokens=query_tokens,
        normalized_query=normalized_query,
        document=document,
        document_tokens=document_tokens,
        normalized_text=normalized_text,
        frequencies=frequencies,
        total_documents=total_documents,
    )


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

    The corpus is tokenized once per retrieval operation,
    document frequencies are calculated once, and the cached
    values are reused while scoring every document.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string"
        )

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

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = store.get_all_documents()

    if not documents:
        return []

    # --------------------------------------------------------
    # Process query once
    # --------------------------------------------------------

    query_tokens = list(
        dict.fromkeys(
            meaningful_tokens(query)
        )
    )

    if not query_tokens:
        return []

    normalized_query = " ".join(
        query_tokens
    )

    # --------------------------------------------------------
    # Preprocess documents once
    # --------------------------------------------------------

    document_cache = {}

    for document in documents:

        meaningful = meaningful_tokens(
            document.text
        )

        document_cache[
            document.chunk_id
        ] = {
            "tokens": set(meaningful),
            "text": " ".join(meaningful),
        }

    # --------------------------------------------------------
    # Calculate document frequencies once
    # --------------------------------------------------------

    frequencies = calculate_document_frequency(
        query_tokens,
        documents,
    )

    total_documents = max(
        len(documents),
        1,
    )

    # --------------------------------------------------------
    # Score documents
    # --------------------------------------------------------

    scored_documents = []

    for document in documents:

        cached = document_cache[
            document.chunk_id
        ]

        score = _score_document(
            query_tokens=query_tokens,
            normalized_query=normalized_query,
            document=document,
            document_tokens=cached["tokens"],
            normalized_text=cached["text"],
            frequencies=frequencies,
            total_documents=total_documents,
        )

        if score > 0:
            scored_documents.append(
                (score, document)
            )

    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # --------------------------------------------------------
    # Return top results
    # --------------------------------------------------------

    return [
        document
        for score, document
        in scored_documents[:top_k]
    ]