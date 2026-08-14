"""
pipeline.py
-----------

Coordinates the Boma Yetu AI Assistant knowledge pipeline.

Pipeline:

PDF
 ↓
parser.py
 ↓
rag.py
 ↓
vector_store.py
 ↓
retrieval.py
"""

from pathlib import Path

from src.parser import extract_text_from_pdf, find_pdf_files
from src.rag import clean_text, split_text_into_chunks
from src.vector_store import DocumentChunk, VectorStore


def process_pdf(
    pdf_path: str | Path,
    store: VectorStore,
    document_id: str | None = None,
    category: str | None = None,
) -> int:
    """
    Process one PDF and add its chunks to the vector store.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    store:
        VectorStore where the processed chunks will be stored.

    document_id:
        Optional identifier for the source document.

    category:
        Optional knowledge-base category, for example "About".

    Returns
    -------
    int
        Number of chunks added to the vector store.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file: {pdf_path}"
        )

    # --------------------------------------------------------
    # STEP 1: Extract text
    # --------------------------------------------------------

    text = extract_text_from_pdf(pdf_path)

    # --------------------------------------------------------
    # STEP 2: Clean text
    # --------------------------------------------------------

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return 0

    # --------------------------------------------------------
    # STEP 3: Split text into chunks
    # --------------------------------------------------------

    chunks = split_text_into_chunks(cleaned_text)

    # --------------------------------------------------------
    # STEP 4: Create DocumentChunk objects
    # --------------------------------------------------------

    if document_id is None:
        document_id = pdf_path.stem

    added_count = 0

    for index, chunk in enumerate(chunks):

        chunk_id = f"{document_id}_{index + 1}"

        metadata = {
            "document_id": document_id,
            "source": str(pdf_path),
        }

        if category is not None:
            metadata["category"] = category

        document = DocumentChunk(
            chunk_id=chunk_id,
            text=chunk,
            metadata=metadata,
        )

        store.add_document(document)

        added_count += 1

    return added_count


def process_pdf_directory(
    pdf_directory: str | Path,
    store: VectorStore,
) -> int:
    """
    Process all PDF files in a directory.

    Returns
    -------
    int
        Total number of chunks added.
    """

    pdf_directory = Path(pdf_directory)

    if not pdf_directory.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_directory}"
        )

    pdf_files = find_pdf_files(pdf_directory)

    total_chunks = 0

    for pdf_path in pdf_files:

        category = pdf_path.parent.name

        total_chunks += process_pdf(
            pdf_path=pdf_path,
            store=store,
            category=category,
        )

    return total_chunks

def search_knowledge(
    query: str,
    store: VectorStore,
    top_k: int = 3,
):
    """
    Search the processed knowledge base for documents
    relevant to a user query.

    Parameters
    ----------
    query:
        User's question.

    store:
        VectorStore containing processed document chunks.

    top_k:
        Maximum number of relevant documents to return.

    Returns
    -------
    list
        Relevant document chunks.
    """

    from src.retrieval import retrieve_documents

    return retrieve_documents(
        query=query,
        store=store,
        top_k=top_k,
    )