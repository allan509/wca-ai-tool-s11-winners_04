"""
End-to-end evaluation of the Boma Yetu AI Assistant.

Tests the complete pipeline:

Question
    ↓
Knowledge retrieval
    ↓
Prompt construction
    ↓
OpenAI LLM
    ↓
Final answer
"""

from src.chatbot import answer_with_llm
from src.pipeline import process_pdf_directory
from src.vector_store import VectorStore


# ============================================================
# BUILD KNOWLEDGE STORE
# ============================================================

def build_knowledge_store():
    """Load the complete Boma Yetu knowledge base."""

    store = VectorStore()

    process_pdf_directory(
        "data/pdfs",
        store,
    )

    return store


# ============================================================
# TEST QUESTIONS
# ============================================================

QUESTIONS = [
    "How do I register for Boma Yangu?",
    "Who is eligible for the Affordable Housing Programme?",
    "How does the housing levy work?",
    "What payment options are available?",
    "What housing projects are available in Kiambu County?",
    "What housing projects are available in Nairobi?",
    "What housing projects are available in Nyanza?",
    "What phone numbers can I use to contact Boma Yangu?",
    "Can I access Boma Yangu through eCitizen?",
    "How does the allocation process work?",
]


# ============================================================
# END-TO-END TEST
# ============================================================

def test_end_to_end_answers():

    store = build_knowledge_store()

    assert store.count() == 193

    for question in QUESTIONS:

        print("\n" + "=" * 80)
        print("QUESTION:")
        print(question)
        print("-" * 80)

        answer = answer_with_llm(
            question=question,
            store=store,
            top_k=3,
        )

        print("ANSWER:")
        print(answer)

        print("=" * 80)

        # Basic safety checks.
        assert isinstance(answer, str)
        assert answer.strip() != ""

        # Make sure the fallback message was not returned
        # for questions that should have relevant knowledge.
        assert (
            "could not find enough relevant information"
            not in answer.lower()
        )