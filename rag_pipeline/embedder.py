"""
Pluggable embedding provider module.

Supports open/local and proprietary embedding models:
  - sentence-transformers  (local, CPU/GPU)
  - huggingface            (via langchain-huggingface)
  - openai                 (via langchain-openai)
  - cohere                 (via langchain-cohere)

All providers implement the LangChain Embeddings base class so they can be
dropped into any LangChain vector-store integration.

Usage:
    from rag_pipeline.embedder import get_embedder
    from rag_pipeline.config import EmbeddingConfig

    embedder = get_embedder(EmbeddingConfig(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2",
        device="cpu",
    ))
    vectors = embedder.embed_documents(["hello world", "another chunk"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_pipeline.config import EmbeddingConfig

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def _get_sentence_transformers(cfg: EmbeddingConfig) -> "Embeddings":
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "Install langchain-huggingface: pip install langchain-huggingface sentence-transformers"
        )
    return HuggingFaceEmbeddings(
        model_name=cfg.model,
        model_kwargs={"device": cfg.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_huggingface(cfg: EmbeddingConfig) -> "Embeddings":
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "Install langchain-huggingface: pip install langchain-huggingface"
        )
    return HuggingFaceEmbeddings(
        model_name=cfg.model,
        model_kwargs={"device": cfg.device},
    )


def _get_openai(cfg: EmbeddingConfig) -> "Embeddings":
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "Install langchain-openai: pip install langchain-openai"
        )
    import os

    kwargs: dict = {"model": cfg.model}
    return OpenAIEmbeddings(**kwargs)


def _get_cohere(cfg: EmbeddingConfig) -> "Embeddings":
    try:
        from langchain_cohere import CohereEmbeddings
    except ImportError:
        raise ImportError(
            "Install langchain-cohere: pip install langchain-cohere"
        )
    return CohereEmbeddings(model=cfg.model)


def get_embedder(cfg: EmbeddingConfig) -> "Embeddings":
    """
    Factory: return the appropriate LangChain Embeddings instance.

    Args:
        cfg: EmbeddingConfig with provider, model, and device.

    Returns:
        A LangChain Embeddings object ready to embed text.
    """
    provider = cfg.provider.lower().strip()
    dispatch = {
        "sentence-transformers": _get_sentence_transformers,
        "huggingface": _get_huggingface,
        "openai": _get_openai,
        "cohere": _get_cohere,
    }
    if provider not in dispatch:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            f"Choose from: {', '.join(dispatch.keys())}"
        )
    print(f"[embedder] Loading provider='{provider}' model='{cfg.model}' device='{cfg.device}'")
    return dispatch[provider](cfg)
