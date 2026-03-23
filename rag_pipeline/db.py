"""
SQLite storage layer for the PDF RAG pipeline.

Two tables are managed here:
  - documents  : one row per ingested PDF
  - chunks     : one row per text chunk, including denormalised doc-level
                 fields (title, author, file_name, file_path, doc_type,
                 num_pages) so that each chunk row is self-contained.

Schema is defined in schema.sql at the repository root and applied
automatically when the database connection is first opened.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def _get_schema_sql() -> str:
    if _SCHEMA_PATH.exists():
        return _SCHEMA_PATH.read_text(encoding="utf-8")
    return ""


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Explicitly apply the schema.sql DDL to an existing connection.

    This is called automatically by connect(), but can also be called
    directly when a connection is obtained externally (e.g. in tests).
    """
    schema_sql = _get_schema_sql()
    if schema_sql:
        conn.executescript(schema_sql)
    conn.commit()


def connect(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def insert_document(conn: sqlite3.Connection, doc: dict[str, Any]) -> None:
    """Insert a document record. Replaces any existing row with the same doc_id."""
    conn.execute(
        """
        INSERT OR REPLACE INTO documents
            (doc_id, file_name, file_path, title, author, subject, keywords,
             num_pages, doc_type, is_tagged, is_scanned, chunking_strategy, ingested_at)
        VALUES
            (:doc_id, :file_name, :file_path, :title, :author, :subject, :keywords,
             :num_pages, :doc_type, :is_tagged, :is_scanned, :chunking_strategy, :ingested_at)
        """,
        doc,
    )
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, chunks: list[dict[str, Any]]) -> None:
    """
    Bulk-insert chunk records.

    Each chunk dict should include both chunk-local fields and the denormalised
    document fields: file_name, file_path, title, author, doc_type, num_pages.
    """
    rows = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "chunk_type": c["chunk_type"],
            "strategy": c["strategy"],
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "bbox_json": json.dumps(c["bbox"]) if c.get("bbox") else None,
            "section_heading": c.get("section_heading"),
            "file_name": c.get("file_name", ""),
            "file_path": c.get("file_path", ""),
            "title": c.get("title", ""),
            "author": c.get("author", ""),
            "doc_type": c.get("doc_type", ""),
            "num_pages": c.get("num_pages"),
            "ingested_at": c["ingested_at"],
        }
        for c in chunks
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks
            (chunk_id, doc_id, chunk_index, text, chunk_type, strategy,
             page_start, page_end, bbox_json, section_heading,
             file_name, file_path, title, author, doc_type, num_pages,
             ingested_at)
        VALUES
            (:chunk_id, :doc_id, :chunk_index, :text, :chunk_type, :strategy,
             :page_start, :page_end, :bbox_json, :section_heading,
             :file_name, :file_path, :title, :author, :doc_type, :num_pages,
             :ingested_at)
        """,
        rows,
    )
    conn.commit()


def get_chunks_for_doc(
    conn: sqlite3.Connection, doc_id: str
) -> list[dict[str, Any]]:
    """Return all chunks for a document, ordered by chunk_index."""
    rows = conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_documents(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all document records."""
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY ingested_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_document(
    conn: sqlite3.Connection, doc_id: str
) -> Optional[dict[str, Any]]:
    """Return a single document record by doc_id."""
    row = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_document(conn: sqlite3.Connection, doc_id: str) -> None:
    """Delete a document and all its chunks (cascade)."""
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
