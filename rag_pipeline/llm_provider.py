"""
LLM provider module for the PDF RAG pipeline.

Supports a variety of language models for the RAG query step:
  - openai      — OpenAI ChatGPT models (GPT-4o, GPT-4o-mini, etc.)
  - anthropic   — Anthropic Claude models
  - cohere      — Cohere Command models
  - ollama      — Local models via Ollama (Llama-3, Mistral, etc.)
  - huggingface — Local HuggingFace transformers pipeline

The system prompt is injected via the LangChain system message so it
is prepended to every invocation without modifying user messages.

Usage:
    from rag_pipeline.llm_provider import get_llm, build_rag_chain
    from rag_pipeline.config import LLMConfig

    llm = get_llm(LLMConfig(provider="openai", model="gpt-4o-mini"))
    chain = build_rag_chain(llm, retriever, system_prompt="...")
    answer = chain.invoke("What is X?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_pipeline.config import LLMConfig, DEFAULT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.vectorstores import VectorStoreRetriever


def _get_openai(cfg: LLMConfig) -> "BaseLanguageModel":
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Install langchain-openai: pip install langchain-openai")
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
    }
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return ChatOpenAI(**kwargs)


def _get_anthropic(cfg: LLMConfig) -> "BaseLanguageModel":
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
    }
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    return ChatAnthropic(**kwargs)


def _get_cohere(cfg: LLMConfig) -> "BaseLanguageModel":
    try:
        from langchain_cohere import ChatCohere
    except ImportError:
        raise ImportError("Install langchain-cohere: pip install langchain-cohere")
    return ChatCohere(model=cfg.model, temperature=cfg.temperature)


def _get_ollama(cfg: LLMConfig) -> "BaseLanguageModel":
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("Install langchain-ollama: pip install langchain-ollama")
    return ChatOllama(model=cfg.model, temperature=cfg.temperature)


def _get_huggingface(cfg: LLMConfig) -> "BaseLanguageModel":
    try:
        from langchain_huggingface import HuggingFacePipeline
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError(
            "Install langchain-huggingface and transformers: "
            "pip install langchain-huggingface transformers"
        )
    device = 0 if (torch.cuda.is_available()) else -1
    pipe = hf_pipeline(
        "text-generation",
        model=cfg.model,
        device=device,
        max_new_tokens=512,
        temperature=cfg.temperature if cfg.temperature > 0 else None,
        do_sample=cfg.temperature > 0,
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_llm(cfg: LLMConfig) -> "BaseLanguageModel":
    """
    Factory: return the configured LangChain language model.

    Args:
        cfg: LLMConfig specifying provider, model, and optional credentials.

    Returns:
        A LangChain BaseLanguageModel ready for use.
    """
    provider = cfg.provider.lower().strip()
    dispatch = {
        "openai": _get_openai,
        "anthropic": _get_anthropic,
        "cohere": _get_cohere,
        "ollama": _get_ollama,
        "huggingface": _get_huggingface,
    }
    if provider not in dispatch:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Choose from: {', '.join(dispatch.keys())}"
        )
    print(f"[llm_provider] Loading provider='{provider}' model='{cfg.model}'")
    return dispatch[provider](cfg)


def build_rag_chain(
    llm: "BaseLanguageModel",
    retriever: "VectorStoreRetriever",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """
    Build a simple RAG chain using LangChain LCEL.

    The chain:
      1. Retrieves top-k documents for the question.
      2. Formats them as context.
      3. Calls the LLM with the system prompt + context + question.
      4. Returns the answer string.

    Args:
        llm:           The language model to use.
        retriever:     A LangChain retriever (e.g. from FAISSVectorStore.as_retriever()).
        system_prompt: System-level instruction injected before every question.

    Returns:
        A callable chain that accepts a question string and returns an answer string.
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
    except ImportError:
        raise ImportError(
            "Install langchain-core: pip install langchain-core"
        )

    def format_docs(docs: list) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            header_parts = [f"[{i}]"]
            if meta.get("title"):
                header_parts.append(f"Title: {meta['title']}")
            if meta.get("file_name"):
                header_parts.append(f"File: {meta['file_name']}")
            if meta.get("page_start"):
                page_info = f"Page {meta['page_start']}"
                if meta.get("page_end") and meta["page_end"] != meta["page_start"]:
                    page_info += f"-{meta['page_end']}"
                header_parts.append(page_info)
            if meta.get("section_heading"):
                header_parts.append(f"Section: {meta['section_heading']}")
            header = " | ".join(header_parts)
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            parts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Use the following context to answer the question.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}",
            ),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
