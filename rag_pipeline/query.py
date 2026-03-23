"""
RAG query CLI for the PDF RAG pipeline.

Retrieves relevant chunks from the FAISS index and uses a configurable LLM
to generate a grounded answer with source citations.

Usage:
    python rag_pipeline/query.py "What is X?" [options]
    python -m rag_pipeline.query "What is X?" [options]

Examples:
    # Query with default settings (OpenAI GPT-4o-mini)
    python rag_pipeline/query.py "What is MORAN?"

    # Use a local Ollama model
    python rag_pipeline/query.py "Summarize the paper." --llm-provider ollama --llm-model llama3

    # Use Claude with a custom system prompt
    python rag_pipeline/query.py "List key results." \\
        --llm-provider anthropic --llm-model claude-3-5-haiku-20241022 \\
        --system-prompt "You are a scientific paper analyst. Be concise and precise."

    # Use a local HuggingFace model
    python rag_pipeline/query.py "What architecture is proposed?" \\
        --llm-provider huggingface --llm-model meta-llama/Llama-3.2-1B-Instruct

    # Retrieve more context
    python rag_pipeline/query.py "What datasets were used?" --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "rag_pipeline"

from rag_pipeline.config import DatabaseConfig, EmbeddingConfig, LLMConfig, DEFAULT_SYSTEM_PROMPT
from rag_pipeline.embedder import get_embedder
from rag_pipeline.llm_provider import get_llm, build_rag_chain
from rag_pipeline.vector_store import FAISSVectorStore


def format_sources(results: list[dict]) -> str:
    """Format retrieved chunks as a readable citation list."""
    lines = ["\nSources:"]
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        parts = [f"  [{i}]"]
        if meta.get("title"):
            parts.append(f"'{meta['title']}'")
        if meta.get("file_name"):
            parts.append(f"({meta['file_name']})")
        if meta.get("page_start"):
            page = f"p.{meta['page_start']}"
            if meta.get("page_end") and meta["page_end"] != meta["page_start"]:
                page += f"-{meta['page_end']}"
            parts.append(page)
        if meta.get("section_heading"):
            parts.append(f"§ {meta['section_heading']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def run_query(
    question: str,
    faiss_path: str,
    embedding_cfg: EmbeddingConfig,
    llm_cfg: LLMConfig,
    top_k: int = 5,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    show_sources: bool = True,
) -> str:
    """
    Retrieve relevant chunks and generate an LLM answer.

    Args:
        question:      The user's question.
        faiss_path:    Path to the persisted FAISS index directory.
        embedding_cfg: Embedding provider settings.
        llm_cfg:       LLM provider settings.
        top_k:         Number of chunks to retrieve.
        system_prompt: System instruction injected into every LLM call.
        show_sources:  If True, print source citations to stdout.

    Returns:
        The generated answer string.
    """
    embedder = get_embedder(embedding_cfg)
    vs = FAISSVectorStore.load(faiss_path, embedder)

    if show_sources:
        raw_results = vs.similarity_search(question, k=top_k)

    retriever = vs.as_retriever(k=top_k)
    llm = get_llm(llm_cfg)
    chain = build_rag_chain(llm, retriever, system_prompt=system_prompt)

    print(f"\nQuestion: {question}\n", flush=True)
    answer = chain.invoke(question)
    print(f"Answer:\n{answer}\n", flush=True)

    if show_sources:
        print(format_sources(raw_results), flush=True)

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the PDF RAG pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", help="Question to ask the RAG system.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--faiss-path",
        default="rag_pipeline/data/faiss_index",
        help="FAISS index directory path.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="sentence-transformers",
        choices=["sentence-transformers", "huggingface", "openai", "cohere"],
        help="Embedding provider (must match what was used during ingest).",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model (must match what was used during ingest).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for local embedding models.",
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "anthropic", "cohere", "ollama", "huggingface"],
        help="LLM provider.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model name.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to inject into every LLM call.",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Suppress source citation output.",
    )

    args = parser.parse_args()

    embedding_cfg = EmbeddingConfig(
        provider=args.embedding_provider,
        model=args.embedding_model,
        device=args.device,
    )
    llm_cfg = LLMConfig(
        provider=args.llm_provider,
        model=args.llm_model,
        system_prompt=args.system_prompt,
    )

    try:
        run_query(
            question=args.question,
            faiss_path=args.faiss_path,
            embedding_cfg=embedding_cfg,
            llm_cfg=llm_cfg,
            top_k=args.top_k,
            system_prompt=args.system_prompt,
            show_sources=not args.no_sources,
        )
    except FileNotFoundError as exc:
        print(f"[query] ERROR: {exc}", file=sys.stderr)
        print(
            "[query] Hint: Run ingest.py first to build the index.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"[query] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
