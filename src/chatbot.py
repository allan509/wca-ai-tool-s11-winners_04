"""
Core chatbot logic for the Boma Yetu AI Assistant.

Current responsibilities:
1. Receive the user's question.
2. Receive relevant document chunks from retrieval.py.
3. Build a prompt containing the question and trusted context.
4. Provide a simple response when no context is available.

This first version does NOT call an external LLM API.

Later, an LLM provider can be connected without changing
the retrieval architecture.
"""

from src.vector_store import DocumentChunk

# System instructions

SYSTEM_INSTRUCTIONS = """
You are Boma Yetu AI Assistant.

Your purpose is to provide accurate information about
Kenya's affordable housing programs and related services.

Use the provided context to answer the user's question.

Rules:
1. Use the provided context as the primary source.
2. Do not invent facts.
3. If the context does not contain enough information,
   clearly say that the available information is insufficient.
4. Keep answers clear and easy to understand.
5. When appropriate, mention the source of the information.
"""

# Input validation

def validate_question(question: str) -> str:
    """
    Validate and clean a user's question.

    Args:
        question: User's question.

    Returns:
        Cleaned question.

    Raises:
        TypeError: If question is not a string.
        ValueError: If question is empty.
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    question = question.strip()

    if not question:
        raise ValueError("question cannot be empty")

    return question


# Context Formatting

def format_context(
    documents: list[DocumentChunk],
) -> str:
    """
    Convert retrieved document chunks into context
    that can be supplied to the AI model.

    Args:
        documents:
            Retrieved DocumentChunk objects.

    Returns:
        Formatted context string.
    """

    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    if not documents:
        return "No relevant information was found."

    context_parts = []

    for index, document in enumerate(documents, start=1):

        if not isinstance(document, DocumentChunk):
            raise TypeError(
                "all documents must be DocumentChunk objects"
            )

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        category = document.metadata.get(
            "category",
            "Unknown category",
        )

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {source}\n"
            f"Category: {category}\n"
            f"Content:\n{document.text}"
        )

    return "\n\n".join(context_parts)

# Prompt building

def build_prompt(
    question: str,
    documents: list[DocumentChunk],
) -> str:
    """
    Build the complete prompt that will eventually be sent
    to the language model.

    Args:
        question:
            User's question.

        documents:
            Relevant document chunks.

    Returns:
        Complete prompt string.
    """

    question = validate_question(question)

    context = format_context(documents)

    prompt = (
        f"{SYSTEM_INSTRUCTIONS.strip()}\n\n"
        f"CONTEXT:\n"
        f"{context}\n\n"
        f"USER QUESTION:\n"
        f"{question}\n\n"
        f"ANSWER:"
    )

    return prompt


# Response Fallback

def no_context_response() -> str:
    """
    Return a safe response when no relevant information
    is available.
    """

    return (
        "I could not find enough relevant information in "
        "the available Boma Yetu knowledge base to answer "
        "that question accurately."
    )

def answer_from_knowledge(
    question: str,
    store,
    top_k: int = 3,
) -> str:
    """
    Retrieve relevant Boma Yangu knowledge and build
    a response using the existing chatbot logic.
    """

    from src.pipeline import search_knowledge

    validate_question(question)

    documents = search_knowledge(
        query=question,
        store=store,
        top_k=top_k,
    )

    if not documents:
        return no_context_response()

    return build_prompt(
        question=question,
        documents=documents,
    )
