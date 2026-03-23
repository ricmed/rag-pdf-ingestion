# PDF RAG Ingestion Pipeline

A complete, production-ready system for ingesting PDF files (books, articles, tutorials, handouts, presentations) into a Retrieval-Augmented Generation (RAG) pipeline, built on top of [OpenDataLoader PDF](https://opendataloader.org).

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [System Architecture](#system-architecture)
5. [Quick Start](#quick-start)
6. [Ingestion (`ingest.py`)](#ingestion-ingestpy)
7. [Querying (`query.py`)](#querying-querypy)
8. [Metadata Editor (`edit_metadata.py`)](#metadata-editor-edit_metadatapy)
9. [Chunking Strategies](#chunking-strategies)
10. [Embedding Providers](#embedding-providers)
11. [LLM Providers](#llm-providers)
12. [Hybrid Mode (OCR, Tables, Image Descriptions)](#hybrid-mode)
13. [Tagged PDFs](#tagged-pdfs)
14. [GPU Support](#gpu-support)
15. [Database Schema](#database-schema)
16. [Configuration Reference](#configuration-reference)
17. [Python API](#python-api)

---

## Overview

This pipeline converts PDF files into searchable vector chunks and stores them in both a SQLite metadata database and a FAISS vector index. A retrieval step then uses an LLM to answer questions grounded in the documents.

```
PDF Files
    │
    ▼
OpenDataLoader PDF (Java engine + optional AI hybrid backend)
    │   ├── JSON (structured: elements, tables, images, bounding boxes)
    │   └── Markdown (for reference)
    │
    ▼
Metadata Editor  ← edit title, author, doc type before ingesting
    │
    ▼
Chunker  ← element | section | merged | table strategy
    │
    ├── SQLite (chunks + document metadata)
    └── FAISS  (embeddings + metadata)
              │
              ▼
     Query: question → top-k retrieval → LLM → answer + citations
```

Key features:
- **Hybrid AI mode** — routes complex pages (tables, scans) to a local AI backend
- **OCR** — handles scanned/image PDFs via EasyOCR in the hybrid backend
- **Image descriptions** — AI-generated alt text for figures and charts (SmolVLM)
- **Tables** — preserved as structured Markdown chunks
- **Tagged PDF support** — exploits PDF structure trees for better reading order
- **Pluggable embeddings** — sentence-transformers, HuggingFace, OpenAI, Cohere
- **Pluggable LLMs** — OpenAI, Anthropic, Cohere, Ollama (local), HuggingFace (local)
- **Injectable system prompt** — instruct the LLM on response format
- **SQLite schema** (versioned in `schema.sql`) + FAISS persistence

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Java       | 11+     | Required by OpenDataLoader PDF |
| Python     | 3.10+   | Required |
| RAM        | 2–4 GB  | More for hybrid/OCR mode |
| Disk       | ~2 GB   | For AI model downloads (cached) |
| GPU        | Optional | Accelerates embedding and OCR |

Verify Java:
```bash
java -version
```

---

## Installation

### 1. Install the OpenDataLoader PDF package

```bash
pip install opendataloader-pdf
```

### 2. Install RAG pipeline dependencies

```bash
pip install -r rag_pipeline/requirements.txt
```

For hybrid mode (OCR, image descriptions):
```bash
pip install "opendataloader-pdf[hybrid]"
```

For GPU-accelerated FAISS (CUDA):
```bash
pip install faiss-gpu  # instead of faiss-cpu
```

### 3. Set API keys (if using cloud providers)

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
```

---

## System Architecture

```
rag_pipeline/
├── __init__.py          # Package marker
├── config.py            # Dataclass configuration
├── converter.py         # PDF → JSON/Markdown via OpenDataLoader PDF
├── chunker.py           # Chunking strategies
├── db.py                # SQLite layer
├── edit_metadata.py     # Interactive metadata editor
├── embedder.py          # Pluggable embedding providers
├── vector_store.py      # FAISS wrapper
├── llm_provider.py      # Pluggable LLM providers + RAG chain
├── ingest.py            # CLI: ingest PDFs
├── query.py             # CLI: ask questions
├── requirements.txt     # Dependencies
├── README.md            # This file
└── data/                # Created automatically
    ├── chunks.db        # SQLite database
    └── faiss_index/     # FAISS index files

schema.sql               # Database schema (versioned at repo root)
```

---

## Quick Start

```bash
# 1. Ingest a PDF
python rag_pipeline/ingest.py path/to/document.pdf

# 2. Ask a question (set OPENAI_API_KEY first, or use Ollama)
python rag_pipeline/query.py "What is the MORAN architecture?"

# 3. Use a local model instead of OpenAI
python rag_pipeline/query.py "Summarize the paper." \
  --llm-provider ollama --llm-model llama3
```

---

## Ingestion (`ingest.py`)

### Basic usage

```bash
python rag_pipeline/ingest.py path/to/file.pdf
```

### All options

```
usage: ingest.py [-h] [--strategy {element,section,merged,table}]
                 [--min-chars N] [--hybrid] [--force-ocr]
                 [--device {cpu,cuda,mps}]
                 [--embedding-provider {sentence-transformers,huggingface,openai,cohere}]
                 [--embedding-model MODEL]
                 [--db-path PATH] [--faiss-path PATH]
                 [--edit-metadata]
                 PDF [PDF ...]

Arguments:
  PDF                   One or more PDF files to ingest

Options:
  --strategy            Chunking strategy [default: section]
  --min-chars N         Min chars per chunk (merged strategy) [default: 200]
  --hybrid              Enable AI hybrid backend
  --force-ocr           Force OCR on all pages (hybrid server must be started with --force-ocr)
  --device              Device for local models: cpu | cuda | mps [default: cpu]
  --embedding-provider  Embedding provider [default: sentence-transformers]
  --embedding-model     Embedding model name [default: all-MiniLM-L6-v2]
  --db-path             SQLite database path [default: rag_pipeline/data/chunks.db]
  --faiss-path          FAISS index directory [default: rag_pipeline/data/faiss_index]
  --edit-metadata       Open interactive metadata editor before ingesting
```

### Examples

```bash
# Ingest with hybrid mode (start hybrid server first — see Hybrid Mode section)
python rag_pipeline/ingest.py paper.pdf --hybrid

# Scanned PDF with OCR
python rag_pipeline/ingest.py scan.pdf --hybrid --force-ocr

# Element chunking for precise citations
python rag_pipeline/ingest.py report.pdf --strategy element

# Merge small paragraphs (minimum 500 characters per chunk)
python rag_pipeline/ingest.py long_doc.pdf --strategy merged --min-chars 500

# OpenAI embeddings
python rag_pipeline/ingest.py doc.pdf \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small

# Batch multiple PDFs in one JVM invocation (faster)
python rag_pipeline/ingest.py file1.pdf file2.pdf folder/*.pdf

# Edit metadata interactively
python rag_pipeline/ingest.py handout.pdf --edit-metadata
```

---

## Querying (`query.py`)

### Basic usage

```bash
python rag_pipeline/query.py "Your question here"
```

### All options

```
usage: query.py [-h] [--top-k N] [--faiss-path PATH]
                [--embedding-provider ...] [--embedding-model MODEL]
                [--device {cpu,cuda,mps}]
                [--llm-provider {openai,anthropic,cohere,ollama,huggingface}]
                [--llm-model MODEL]
                [--system-prompt TEXT] [--no-sources]
                question

Options:
  --top-k N             Number of chunks to retrieve [default: 5]
  --faiss-path PATH     FAISS index directory [default: rag_pipeline/data/faiss_index]
  --embedding-provider  Must match what was used during ingest [default: sentence-transformers]
  --embedding-model     Must match what was used during ingest [default: all-MiniLM-L6-v2]
  --device              Device for local embedding models [default: cpu]
  --llm-provider        LLM provider [default: openai]
  --llm-model           LLM model name [default: gpt-4o-mini]
  --system-prompt       System prompt injected into every LLM call
  --no-sources          Suppress source citation output
```

### Examples

```bash
# OpenAI (default)
python rag_pipeline/query.py "What datasets were used?"

# Ollama (local, no API key needed)
python rag_pipeline/query.py "Summarize the paper." \
  --llm-provider ollama --llm-model llama3

# Anthropic Claude
python rag_pipeline/query.py "List the contributions." \
  --llm-provider anthropic --llm-model claude-3-5-haiku-20241022

# Cohere
python rag_pipeline/query.py "What are the limitations?" \
  --llm-provider cohere --llm-model command-r

# Local HuggingFace model
python rag_pipeline/query.py "What is the method?" \
  --llm-provider huggingface \
  --llm-model meta-llama/Llama-3.2-1B-Instruct

# Custom system prompt to control response format
python rag_pipeline/query.py "What are the key results?" \
  --system-prompt "You are a scientific paper analyst. Respond with bullet points only."

# Retrieve more context
python rag_pipeline/query.py "Explain the architecture." --top-k 10
```

---

## Metadata Editor (`edit_metadata.py`)

Before ingesting, you can inspect and override the auto-detected metadata.

```bash
python rag_pipeline/edit_metadata.py path/to/file.pdf
```

Fields you can edit:
- **Title** — document title
- **Author** — author(s)
- **Subject** — description or abstract
- **Keywords** — comma-separated search terms
- **Doc type** — book | article | tutorial | handout | presentation | other

Edits are saved to a sidecar file (`.pdf_meta.json` next to the PDF) and
reused automatically on subsequent ingest runs, so you won't be asked again.

To reset and re-edit:
```bash
python rag_pipeline/edit_metadata.py path/to/file.pdf --reset
```

---

## Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `element` | One chunk per paragraph, heading, or list item | Fine-grained retrieval, precise citations |
| `section` | Group all content under the nearest heading | Context-rich retrieval, topic-based search |
| `merged`  | Merge elements until `--min-chars` threshold | Balanced chunks, noisy/unstructured docs |
| `table`   | Same as `element` but tables always separate | Financial reports, table-heavy documents |

Tables and picture descriptions are always extracted as their own dedicated chunks,
regardless of strategy.

---

## Embedding Providers

| Provider | `--embedding-provider` | Default Model | Notes |
|----------|----------------------|---------------|-------|
| Sentence Transformers | `sentence-transformers` | `all-MiniLM-L6-v2` | Local, no API key |
| HuggingFace | `huggingface` | any HF model | Local, no API key |
| OpenAI | `openai` | `text-embedding-3-small` | Requires `OPENAI_API_KEY` |
| Cohere | `cohere` | `embed-english-v3.0` | Requires `COHERE_API_KEY` |

**Important**: The embedding provider and model must be the same at ingest and query time.

---

## LLM Providers

| Provider | `--llm-provider` | Example Model | Notes |
|----------|-----------------|---------------|-------|
| OpenAI | `openai` | `gpt-4o-mini` | Requires `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `claude-3-5-haiku-20241022` | Requires `ANTHROPIC_API_KEY` |
| Cohere | `cohere` | `command-r` | Requires `COHERE_API_KEY` |
| Ollama | `ollama` | `llama3`, `mistral` | Local, install [Ollama](https://ollama.ai) |
| HuggingFace | `huggingface` | `meta-llama/Llama-3.2-1B-Instruct` | Local, downloads model |

### Injecting a System Prompt

The `--system-prompt` flag lets you instruct the LLM on how to structure its response:

```bash
python rag_pipeline/query.py "What are the results?" \
  --system-prompt "You are a research assistant. Always cite page numbers. Use markdown formatting."
```

---

## Hybrid Mode

Hybrid mode routes complex pages to a local AI backend (docling) for:
- **Complex/borderless tables** — 90%+ table accuracy
- **OCR for scanned PDFs** — image-based PDF text extraction
- **Image descriptions** — AI-generated alt text for figures and charts
- **Formula extraction** — LaTeX from math-heavy documents

### Starting the hybrid server

```bash
# Basic
opendataloader-pdf-hybrid --port 5002

# With OCR (for scanned PDFs)
opendataloader-pdf-hybrid --port 5002 --force-ocr

# With image descriptions
opendataloader-pdf-hybrid --port 5002 --enrich-picture-description

# Full: OCR + image descriptions + formula extraction
opendataloader-pdf-hybrid --port 5002 --force-ocr --enrich-picture-description --enrich-formula

# Non-English OCR
opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "zh,en"
```

### Ingest with hybrid mode

```bash
# Start server first, then:
python rag_pipeline/ingest.py paper.pdf --hybrid

# Scanned PDF
python rag_pipeline/ingest.py scan.pdf --hybrid --force-ocr
```

---

## Tagged PDFs

When a PDF has a structure tree (tagged PDF), the converter automatically enables
`use_struct_tree=True` for better reading order and semantic structure. This is
detected automatically — no additional flags are needed.

To take advantage of tagged PDFs with LangChain directly, use:

```python
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
loader = OpenDataLoaderPDFLoader(
    file_path=["tagged.pdf"],
    use_struct_tree=True,
    format="text",
)
documents = loader.load()
```

---

## GPU Support

For faster embedding and local model inference:

```bash
# Ingest with GPU embeddings
python rag_pipeline/ingest.py doc.pdf --device cuda

# Query with GPU embeddings
python rag_pipeline/query.py "question" --device cuda

# GPU-accelerated FAISS (install instead of faiss-cpu)
pip install faiss-gpu
```

The hybrid backend automatically uses GPU if available (via PyTorch/CUDA).

---

## Database Schema

The schema is defined in `schema.sql` at the repository root and applied
automatically when the database is first created.

### `documents` table

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | TEXT PK | UUID generated at ingest time |
| `file_name` | TEXT | Basename of the PDF file |
| `file_path` | TEXT | Absolute path to the PDF file |
| `title` | TEXT | Document title |
| `author` | TEXT | Author(s) |
| `subject` | TEXT | Subject / description |
| `keywords` | TEXT | Comma-separated keywords |
| `num_pages` | INTEGER | Total page count |
| `doc_type` | TEXT | book, article, tutorial, handout, presentation |
| `is_tagged` | INTEGER | 1 if PDF has a structure tree |
| `is_scanned` | INTEGER | 1 if PDF appears to be image-only |
| `chunking_strategy` | TEXT | Strategy used: element, section, merged, table |
| `ingested_at` | TEXT | ISO-8601 UTC timestamp |

### `chunks` table

| Column | Type | Description |
|--------|------|-------------|
| `chunk_id` | TEXT PK | UUID |
| `doc_id` | TEXT FK | References `documents.doc_id` |
| `chunk_index` | INTEGER | 0-based position in document |
| `text` | TEXT | Chunk text content |
| `chunk_type` | TEXT | paragraph, heading, list, table, picture, mixed |
| `strategy` | TEXT | Chunking strategy that produced this chunk |
| `page_start` | INTEGER | First page (1-indexed) |
| `page_end` | INTEGER | Last page (1-indexed) |
| `bbox_json` | TEXT | JSON [left, bottom, right, top] in PDF points |
| `section_heading` | TEXT | Nearest ancestor heading |
| `ingested_at` | TEXT | ISO-8601 UTC timestamp |

---

## Configuration Reference

All settings live in `config.py` as Python dataclasses.

```python
from rag_pipeline.config import (
    PipelineConfig, HybridConfig, EmbeddingConfig,
    LLMConfig, DatabaseConfig, ChunkingConfig
)

cfg = PipelineConfig(
    hybrid=HybridConfig(
        enabled=True,
        backend="docling-fast",       # hybrid backend name
        url="http://localhost:5002",  # server URL
        timeout_ms=30000,             # ms
        fallback=True,                # fall back to Java on backend error
        mode="auto",                  # auto | full
        force_ocr=False,              # OCR all pages
        enrich_picture_description=True,
        enrich_formula=False,
    ),
    embedding=EmbeddingConfig(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2",
        device="cpu",                 # cpu | cuda | mps
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        system_prompt="You are a helpful assistant...",
    ),
    database=DatabaseConfig(
        db_path="rag_pipeline/data/chunks.db",
        faiss_index_path="rag_pipeline/data/faiss_index",
    ),
    chunking=ChunkingConfig(
        strategy="section",           # element | section | merged | table
        min_chars=200,                # for merged strategy
    ),
)
```

---

## Python API

Use the pipeline programmatically:

```python
from rag_pipeline.config import PipelineConfig, HybridConfig
from rag_pipeline.ingest import ingest_pdf
from rag_pipeline.query import run_query

# Ingest
cfg = PipelineConfig()
doc_id = ingest_pdf("paper.pdf", cfg)

# Query
answer = run_query(
    question="What is the main contribution?",
    faiss_path=cfg.database.faiss_index_path,
    embedding_cfg=cfg.embedding,
    llm_cfg=cfg.llm,
    top_k=5,
    system_prompt="Answer concisely in 3 bullet points.",
)
print(answer)
```

### Using the LangChain integration directly

```python
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

loader = OpenDataLoaderPDFLoader(file_path=["doc.pdf"], format="text", quiet=True)
documents = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```
