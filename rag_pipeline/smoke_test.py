"""
Smoke test for the PDF RAG pipeline.

Tests the full ingest → query happy path using:
  - A real PDF from samples/pdf/
  - Local sentence-transformers embedding (no API key required)
  - No LLM call (retrieval-only mode to avoid requiring API keys)
  - Temporary SQLite database and FAISS index (cleaned up after)

Usage:
    python rag_pipeline/smoke_test.py
    python -m rag_pipeline.smoke_test
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "rag_pipeline"

from rag_pipeline.chunker import get_chunks
from rag_pipeline.config import (
    ChunkingConfig,
    DatabaseConfig,
    EmbeddingConfig,
    HybridConfig,
    PipelineConfig,
)
from rag_pipeline.db import connect, init_schema, insert_document, insert_chunks, get_chunks_for_doc, list_documents

SAMPLE_PDF = Path(__file__).parent.parent / "samples" / "pdf" / "1901.03003.pdf"


def _make_temp_config(tmp_dir: str) -> PipelineConfig:
    return PipelineConfig(
        hybrid=HybridConfig(enabled=False),
        embedding=EmbeddingConfig(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2",
            device="cpu",
        ),
        chunking=ChunkingConfig(strategy="section", min_chars=100),
        database=DatabaseConfig(
            db_path=str(Path(tmp_dir) / "test.db"),
            faiss_index_path=str(Path(tmp_dir) / "faiss_index"),
        ),
    )


def test_chunker() -> None:
    """Unit test: chunker produces valid chunks from a mock document."""
    doc = {
        "tagged": True,
        "number of pages": 2,
        "kids": [
            {"type": "heading", "content": "Introduction", "page number": 1, "bounding box": None},
            {"type": "paragraph", "content": "This is the introduction paragraph text.", "page number": 1, "bounding box": None},
            {"type": "table", "content": "", "page number": 2, "bounding box": None, "rows": [
                {"cells": [{"content": "Model"}, {"content": "Accuracy"}]},
                {"cells": [{"content": "Ours"}, {"content": "98.5"}]},
            ]},
            {"type": "image", "content": "", "description": "Fig 1: Architecture diagram.", "page number": 2, "bounding box": None},
        ],
    }
    for strategy in ("element", "section", "merged", "table"):
        chunks = get_chunks(doc, strategy=strategy)
        assert len(chunks) > 0, f"[{strategy}] No chunks produced"
        for c in chunks:
            assert c["text"].strip(), f"[{strategy}] Empty chunk text at index {chunks.index(c)}"
    print("  [PASS] chunker: all strategies produce non-empty chunks")


def test_db() -> None:
    """Unit test: SQLite layer creates schema and round-trips a document+chunk."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = connect(db_path)
        init_schema(conn)

        doc_id = str(uuid.uuid4())
        insert_document(conn, {
            "doc_id": doc_id,
            "file_name": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "title": "Test Doc",
            "author": "Test Author",
            "subject": "",
            "keywords": "test",
            "num_pages": 3,
            "doc_type": "article",
            "is_tagged": 1,
            "is_scanned": 0,
            "chunking_strategy": "section",
            "ingested_at": "2026-01-01T00:00:00Z",
        })
        insert_chunks(conn, [{
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "chunk_index": 0,
            "text": "Sample chunk content",
            "chunk_type": "paragraph",
            "strategy": "section",
            "page_start": 1,
            "page_end": 1,
            "bbox": None,
            "section_heading": "Introduction",
            "file_name": "test.pdf",
            "file_path": "/tmp/test.pdf",
            "title": "Test Doc",
            "author": "Test Author",
            "doc_type": "article",
            "num_pages": 3,
            "ingested_at": "2026-01-01T00:00:00Z",
        }])
        docs = list_documents(conn)
        chunks = get_chunks_for_doc(conn, doc_id)
        assert len(docs) == 1
        assert len(chunks) == 1
        assert chunks[0]["title"] == "Test Doc"
        assert chunks[0]["author"] == "Test Author"
        assert chunks[0]["num_pages"] == 3
        conn.close()
    print("  [PASS] db: schema init, insert/query, denorm fields round-trip OK")


def test_ingest_and_retrieval() -> None:
    """Integration test: ingest a real PDF and retrieve chunks via FAISS."""
    try:
        import sentence_transformers  # noqa: F401
        import faiss  # noqa: F401
    except ImportError:
        print("  [SKIP] ingest+retrieval: sentence-transformers or faiss not installed")
        return

    if not SAMPLE_PDF.exists():
        print(f"  [SKIP] ingest+retrieval: sample PDF not found at {SAMPLE_PDF}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_temp_config(tmp)
        cfg.ensure_data_dir()

        from rag_pipeline.ingest import ingest_pdf
        doc_id = ingest_pdf(str(SAMPLE_PDF), cfg, edit_metadata=False)
        assert doc_id, "ingest_pdf returned empty doc_id"

        conn = connect(cfg.database.db_path)
        docs = list_documents(conn)
        chunks = get_chunks_for_doc(conn, doc_id)
        conn.close()
        assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
        assert len(chunks) > 0, "No chunks stored"
        assert chunks[0]["title"] is not None

        from rag_pipeline.embedder import get_embedder
        from rag_pipeline.vector_store import FAISSVectorStore
        embedder = get_embedder(cfg.embedding)
        vs = FAISSVectorStore.load(cfg.database.faiss_index_path, embedder)
        results = vs.similarity_search("What is MORAN?", k=3)
        assert len(results) > 0
        assert results[0]["text"].strip()

        print(
            f"  [PASS] ingest+retrieval: {len(chunks)} chunks stored, "
            f"{len(results)} results retrieved OK"
        )


def main() -> None:
    print("Running PDF RAG pipeline smoke tests ...\n")
    failed = []

    tests = [
        ("chunker", test_chunker),
        ("db", test_db),
        ("ingest+retrieval", test_ingest_and_retrieval),
    ]
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed.append(name)
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed.append(name)
        print()

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
