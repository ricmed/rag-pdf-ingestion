"""
Main PDF ingestion CLI for the RAG pipeline.

This script orchestrates the full ingestion flow:
  1. (Optional) Interactive metadata editor
  2. PDF conversion via OpenDataLoader PDF
  3. Semantic chunking
  4. Storage in SQLite
  5. Embedding + upsert into FAISS index

Usage:
    python rag_pipeline/ingest.py path/to/file.pdf [options]
    python -m rag_pipeline.ingest path/to/file.pdf [options]

Examples:
    # Ingest a single PDF with default settings (sentence-transformers, section chunking)
    python rag_pipeline/ingest.py path/to/document.pdf

    # Edit metadata interactively before ingesting
    python rag_pipeline/ingest.py path/to/document.pdf --edit-metadata

    # Hybrid mode with image descriptions (requires hybrid server running on port 5002)
    python rag_pipeline/ingest.py paper.pdf --hybrid --strategy element

    # Scanned PDF with OCR (start hybrid server with --force-ocr first)
    python rag_pipeline/ingest.py scan.pdf --hybrid --force-ocr

    # Use OpenAI embeddings and a custom strategy
    python rag_pipeline/ingest.py doc.pdf --embedding-provider openai --embedding-model text-embedding-3-small

    # GPU inference
    python rag_pipeline/ingest.py doc.pdf --device cuda

    # Ingest multiple files in one JVM invocation
    python rag_pipeline/ingest.py file1.pdf file2.pdf --strategy merged
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "rag_pipeline"

from rag_pipeline.config import (
    ChunkingConfig,
    DatabaseConfig,
    EmbeddingConfig,
    HybridConfig,
    PipelineConfig,
)
from rag_pipeline.converter import convert_pdf
from rag_pipeline.chunker import get_chunks
from rag_pipeline.db import connect, insert_document, insert_chunks
from rag_pipeline.edit_metadata import load_or_edit_metadata
from rag_pipeline.embedder import get_embedder
from rag_pipeline.vector_store import FAISSVectorStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ingest_pdf(
    pdf_path: str | Path,
    cfg: PipelineConfig,
    edit_metadata: bool = False,
) -> str:
    """
    Ingest a single PDF file into the RAG pipeline.

    Returns the doc_id of the ingested document.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cfg.ensure_data_dir()

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path, md_path, doc_json = convert_pdf(
            pdf_path=pdf_path,
            output_dir=tmp_dir,
            hybrid_cfg=cfg.hybrid if cfg.hybrid.enabled else None,
            quiet=True,
        )

        meta = load_or_edit_metadata(
            pdf_path,
            doc_json=doc_json,
            force_interactive=edit_metadata,
        )

        doc_id = str(uuid.uuid4())
        now = _now_iso()

        document_record = {
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "title": meta.get("title") or doc_json.get("title") or "",
            "author": meta.get("author") or doc_json.get("author") or "",
            "subject": meta.get("subject") or doc_json.get("subject") or "",
            "keywords": meta.get("keywords") or doc_json.get("keywords") or "",
            "num_pages": meta.get("num_pages") or doc_json.get("number of pages"),
            "doc_type": meta.get("doc_type") or "",
            "is_tagged": 1 if meta.get("is_tagged") else 0,
            "is_scanned": 1 if meta.get("is_scanned") else 0,
            "chunking_strategy": cfg.chunking.strategy,
            "ingested_at": now,
        }

        print(
            f"[ingest] Chunking with strategy='{cfg.chunking.strategy}' ...",
            flush=True,
        )
        raw_chunks = get_chunks(
            doc_json,
            strategy=cfg.chunking.strategy,
            min_chars=cfg.chunking.min_chars,
        )

        chunk_records = []
        for idx, chunk in enumerate(raw_chunks):
            chunk_record = {
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "chunk_index": idx,
                "text": chunk["text"],
                "chunk_type": chunk.get("chunk_type", "mixed"),
                "strategy": chunk.get("strategy", cfg.chunking.strategy),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "bbox": chunk.get("bbox"),
                "section_heading": chunk.get("section_heading"),
                # denormalised doc-level fields: each chunk is self-contained
                "file_name": document_record["file_name"],
                "file_path": document_record["file_path"],
                "title": document_record["title"],
                "author": document_record["author"],
                "doc_type": document_record["doc_type"],
                "num_pages": document_record["num_pages"],
                "ingested_at": now,
            }
            chunk_records.append(chunk_record)

        print(
            f"[ingest] Storing {len(chunk_records)} chunks in SQLite ...",
            flush=True,
        )
        conn = connect(cfg.database.db_path)
        insert_document(conn, document_record)
        insert_chunks(conn, chunk_records)
        conn.close()

        print("[ingest] Building embeddings ...", flush=True)
        embedder = get_embedder(cfg.embedding)

        faiss_path = Path(cfg.database.faiss_index_path)
        if faiss_path.exists():
            vs = FAISSVectorStore.load(faiss_path, embedder)
        else:
            vs = FAISSVectorStore(embedder)

        doc_meta = {
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "title": document_record["title"],
            "author": document_record["author"],
        }
        vs.add_chunks(chunk_records, doc_meta)
        vs.save(faiss_path)

    print(
        f"\n[ingest] Done.\n"
        f"  doc_id   : {doc_id}\n"
        f"  file     : {pdf_path.name}\n"
        f"  chunks   : {len(chunk_records)}\n"
        f"  strategy : {cfg.chunking.strategy}\n"
        f"  db       : {cfg.database.db_path}\n"
        f"  index    : {cfg.database.faiss_index_path}\n",
        flush=True,
    )
    return doc_id


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    hybrid_cfg = HybridConfig(
        enabled=args.hybrid,
        force_ocr=args.force_ocr,
    )
    embedding_cfg = EmbeddingConfig(
        provider=args.embedding_provider,
        model=args.embedding_model,
        device=args.device,
    )
    chunking_cfg = ChunkingConfig(
        strategy=args.strategy,
        min_chars=args.min_chars,
    )
    db_cfg = DatabaseConfig(
        db_path=args.db_path,
        faiss_index_path=args.faiss_path,
    )
    return PipelineConfig(
        hybrid=hybrid_cfg,
        embedding=embedding_cfg,
        chunking=chunking_cfg,
        database=db_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into the RAG pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        metavar="PDF",
        help="One or more PDF files to ingest.",
    )
    parser.add_argument(
        "--strategy",
        default="section",
        choices=["element", "section", "merged", "table"],
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        metavar="N",
        help="Minimum characters per chunk (merged strategy only).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid AI backend (requires hybrid server running on port 5002).",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages (start hybrid server with --force-ocr).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for local embedding models.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="sentence-transformers",
        choices=["sentence-transformers", "huggingface", "openai", "cohere"],
        help="Embedding model provider.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name or path.",
    )
    parser.add_argument(
        "--db-path",
        default="rag_pipeline/data/chunks.db",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--faiss-path",
        default="rag_pipeline/data/faiss_index",
        help="FAISS index directory path.",
    )
    parser.add_argument(
        "--edit-metadata",
        action="store_true",
        help="Open the interactive metadata editor before ingesting each PDF.",
    )

    args = parser.parse_args()
    cfg = build_config_from_args(args)

    for pdf_path in args.pdf_paths:
        try:
            ingest_pdf(pdf_path, cfg, edit_metadata=args.edit_metadata)
        except Exception as exc:
            print(f"[ingest] ERROR processing '{pdf_path}': {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
