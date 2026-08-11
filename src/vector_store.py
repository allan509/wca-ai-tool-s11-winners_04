"""
vector_store.py
---------------
Storage and retrieval of document chunks for the Boma Yetu
AI Assistant.

This first version uses an in-memory store.

Later, this module can be connected to a real vector database
and embedding model without changing the rest of the application.
"""


# ============================================================
# DOCUMENT CHUNK
# ============================================================

class DocumentChunk:
    """
    Represents one chunk of information from a document.

    Each chunk contains:
    - text
    - a unique ID
    - metadata about the original document
    """

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: dict | None = None,
    ):
        if not isinstance(chunk_id, str):
            raise TypeError("chunk_id must be a string")

        if not chunk_id.strip():
            raise ValueError("chunk_id cannot be empty")

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self):
        return (
            f"DocumentChunk("
            f"chunk_id='{self.chunk_id}', "
            f"text_length={len(self.text)}"
            f")"
        )


# ============================================================
# VECTOR STORE
# ============================================================

class VectorStore:
    """
    Simple in-memory document store.

    This version stores document chunks but does not yet
    perform mathematical vector similarity.

    It provides the basic interface that the real vector
    database will eventually implement.
    """

    def __init__(self):
        """Create an empty vector store."""

        self.documents = {}

    # --------------------------------------------------------
    # ADD DOCUMENT
    # --------------------------------------------------------

    def add_document(self, document: DocumentChunk):
        """
        Add a document chunk to the store.

        Args:
            document: DocumentChunk object.
        """

        if not isinstance(document, DocumentChunk):
            raise TypeError(
                "document must be a DocumentChunk"
            )

        self.documents[document.chunk_id] = document

    # --------------------------------------------------------
    # ADD MULTIPLE DOCUMENTS
    # --------------------------------------------------------

    def add_documents(self, documents: list[DocumentChunk]):
        """
        Add multiple document chunks.

        Args:
            documents: List of DocumentChunk objects.
        """

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        for document in documents:
            self.add_document(document)

    # --------------------------------------------------------
    # GET DOCUMENT
    # --------------------------------------------------------

    def get_document(self, chunk_id: str):
        """
        Retrieve a document chunk using its ID.

        Returns:
            DocumentChunk if found, otherwise None.
        """

        return self.documents.get(chunk_id)

    # --------------------------------------------------------
    # GET ALL DOCUMENTS
    # --------------------------------------------------------

    def get_all_documents(self):
        """
        Return all stored document chunks.

        Returns:
            List of DocumentChunk objects.
        """

        return list(self.documents.values())

    # --------------------------------------------------------
    # DELETE DOCUMENT
    # --------------------------------------------------------

    def delete_document(self, chunk_id: str):
        """
        Delete a document chunk.

        Returns:
            True if deleted, False if the ID did not exist.
        """

        if chunk_id in self.documents:
            del self.documents[chunk_id]
            return True

        return False

    # --------------------------------------------------------
    # COUNT DOCUMENTS
    # --------------------------------------------------------

    def count(self):
        """
        Return the number of stored document chunks.
        """

        return len(self.documents)

    # --------------------------------------------------------
    # CLEAR STORE
    # --------------------------------------------------------

    def clear(self):
        """Remove all documents from the store."""

        self.documents.clear()