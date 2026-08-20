"""
Boma Yetu RAG Retrieval Evaluation
===================================

Evaluates retrieval against the complete 38-PDF knowledge base.

This script does NOT call the OpenAI API.
It evaluates only the retrieval layer.
"""

from src.pipeline import process_pdf_directory
from src.retrieval import retrieve_documents
from src.vector_store import VectorStore


QUERIES = [
    {
        "question": "Who is eligible for the Affordable Housing Programme?",
        "expected": [
            "Eligibility",
        ],
    },
    {
        "question": "How do I register for Boma Yangu?",
        "expected": [
            "Application_Registration",
        ],
    },
    {
        "question": "How does the housing levy work?",
        "expected": [
            "Housing_Levy",
        ],
    },
    {
        "question": "What payment options are available?",
        "expected": [
            "Savings_Pricing_Payments",
            "Homeownership_Journey",
        ],
    },
    {
        "question": "What housing projects are available in Kiambu County?",
        "expected": [
            "Projects_Kiambu_County_Land_Bank",
            "FAQs_Projects_Impact",
        ],
    },
    {
        "question": "What housing projects are available in Nairobi?",
        "expected": [
            "Projects_Nairobi_Region",
        ],
    },
    {
        "question": "What housing projects are available in Nyanza?",
        "expected": [
            "Projects_Nyanza_Rift_Western",
        ],
    },
    {
        "question": "What phone numbers can I use to contact Boma Yangu?",
        "expected": [
            "Contact_Support",
        ],
    },
    {
        "question": "Can I access Boma Yangu through eCitizen?",
        "expected": [
            "Contact_Support",
        ],
    },
    {
        "question": "How does the allocation process work?",
        "expected": [
            "Allocation_Process",
        ],
    },
]


def build_store():
    """Load the complete Boma Yetu PDF corpus."""

    store = VectorStore()

    count = process_pdf_directory(
        "data/pdfs",
        store,
    )

    print(f"PDF chunks loaded: {count}")
    print(f"Store count: {store.count()}")

    return store


def evaluate_query(store, item):
    """Evaluate one retrieval query."""

    question = item["question"]
    expected = item["expected"]

    results = retrieve_documents(
        question,
        store,
        top_k=3,
    )

    result_ids = [
        document.chunk_id
        for document in results
    ]

    matched = any(
        any(expected_part in document_id
            for expected_part in expected)
        for document_id in result_ids
    )

    print()
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("-" * 80)

    for rank, document in enumerate(results, start=1):
        print(
            f"{rank}. {document.chunk_id}"
            f" | category={document.metadata.get('category')}"
        )

    print("-" * 80)
    print(f"EXPECTED: {expected}")
    print(f"MATCH: {'PASS' if matched else 'FAIL'}")

    return matched


def main():
    """Run retrieval evaluation."""

    store = build_store()

    passed = 0
    failed = 0

    for item in QUERIES:
        if evaluate_query(store, item):
            passed += 1
        else:
            failed += 1

    total = passed + failed

    print()
    print("=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total queries: {total}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {failed}")

    if total:
        accuracy = (passed / total) * 100
        print(f"Retrieval accuracy: {accuracy:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()