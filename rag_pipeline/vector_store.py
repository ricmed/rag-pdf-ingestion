"""
FAISS vector store wrapper for the PDF RAG pipeline.

Wraps LangChain's FAISS integration with convenience methods for:
  - Adding chunks (with full metadata)
  - Persisting the index to disk
  - Loading an existing index from disk
  - Similarity search returning chunks with metadata

Metadata stored per vector:
  chunk_id, doc_id, page_start, page_end, bbox, section_heading,
  file_name, title, author, chunk_type, strategy
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class FAISSVectorStore:
    """Thin wrapper around LangChain's FAISS vector store."""

    def __init__(self, embedder: "Embeddings") -> None:
        self.embedder = embedder
        self._store = None

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        doc_meta: dict[str, Any],
    ) -> None:
        """
        Embed and add chunks to the FAISS index.

        Args:
            chunks:   List of chunk dicts (must have chunk_id, text, and standard fields).
            doc_meta: Document-level metadata (title, author, file_name, doc_id).
        """
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "Install langchain-community: pip install langchain-community"
            )

        documents: list[Document] = []
        for chunk in chunks:
            bbox = chunk.get("bbox")
            metadata: dict[str, Any] = {
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": chunk.get("doc_id", doc_meta.get("doc_id", "")),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "bbox": bbox if bbox else None,
                "section_heading": chunk.get("section_heading"),
                "chunk_type": chunk.get("chunk_type"),
                "strategy": chunk.get("strategy"),
                "file_name": doc_meta.get("file_name", ""),
                "title": doc_meta.get("title", ""),
                "author": doc_meta.get("author", ""),
            }
            documents.append(Document(page_content=chunk["text"], metadata=metadata))

        if not documents:
            print("[vector_store] No documents to add.")
            return

        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embedder)
        else:
            self._store.add_documents(documents)

        print(f"[vector_store] Added {len(documents)} vectors to FAISS index.")

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index to disk."""
        if self._store is None:
            raise RuntimeError("No vectors have been added yet.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(path))
        print(f"[vector_store] Index saved to '{path}'.")

    @classmethod
    def load(cls, path: str | Path, embedder: "Embeddings") -> "FAISSVectorStore":
        """Load a persisted FAISS index from disk."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError(
                "Install langchain-community: pip install langchain-community"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found at '{path}'.")
        instance = cls(embedder)
        instance._store = FAISS.load_local(
            str(path), embedder, allow_dangerous_deserialization=True
        )
        print(f"[vector_store] Index loaded from '{path}'.")
        return instance

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query.

        Returns a list of dicts with 'text' and 'metadata' keys.
        """
        if self._store is None:
            raise RuntimeError("No FAISS index loaded.")
        results = self._store.similarity_search(query, k=k)
        return [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]

    def as_retriever(self, k: int = 5):
        """Return a LangChain retriever for use in chains."""
        if self._store is None:
            raise RuntimeError("No FAISS index loaded.")
        return self._store.as_retriever(search_kwargs={"k": k})
