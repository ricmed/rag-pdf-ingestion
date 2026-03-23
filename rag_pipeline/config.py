"""
Configuration dataclasses for the PDF RAG pipeline.

All settings can be overridden from CLI flags or by constructing a PipelineConfig
directly in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions based only on the provided context. "
    "If the answer is not in the context, say so clearly. "
    "Cite the source document and page number when possible."
)


@dataclass
class HybridConfig:
    """Settings for the OpenDataLoader hybrid (AI) backend."""

    enabled: bool = False
    backend: str = "docling-fast"
    url: str = "http://localhost:5002"
    timeout_ms: int = 30000
    fallback: bool = True
    mode: str = "auto"
    force_ocr: bool = False
    ocr_lang: str = "en"
    enrich_picture_description: bool = True
    enrich_formula: bool = False


@dataclass
class EmbeddingConfig:
    """Settings for the embedding model."""

    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass
class LLMConfig:
    """Settings for the LLM used in RAG query."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class DatabaseConfig:
    """Paths for persistent storage."""

    db_path: str = "rag_pipeline/data/chunks.db"
    faiss_index_path: str = "rag_pipeline/data/faiss_index"


@dataclass
class ChunkingConfig:
    """Settings for the chunking strategy."""

    strategy: str = "section"
    min_chars: int = 200
    table_method: str = "cluster"


@dataclass
class PipelineConfig:
    """Top-level configuration combining all subsystems."""

    hybrid: HybridConfig = field(default_factory=HybridConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    def ensure_data_dir(self) -> None:
        Path(self.database.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.database.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
