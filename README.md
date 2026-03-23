# rag-pdf-ingestion

A production-ready Python pipeline that ingests PDF files into a hybrid SQLite + FAISS vector store, ready for Retrieval-Augmented Generation (RAG). Built on top of [OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf) as the parsing engine.

## Features

- **Hybrid AI parsing** — local deterministic mode or AI-backed hybrid mode (OCR, table, formula, image descriptions)
- **Semantic chunking** — four strategies: `element`, `section`, `merged`, `table`
- **SQLite storage** — every chunk is stored with full provenance (page, bbox, heading, doc metadata)
- **FAISS vector store** — fast approximate nearest-neighbour similarity search
- **Pluggable embeddings** — open models via Sentence-Transformers / HuggingFace or proprietary via OpenAI / Cohere
- **Switchable LLMs** — OpenAI, Anthropic, Cohere, Ollama (local), HuggingFace
- **Tagged PDF support** — respects native PDF structure tags for accurate reading order
- **Interactive metadata editing** — override title, author, doc-type, keywords before ingesting
- **LangChain integration** — load via `OpenDataLoaderPDFLoader` for existing LangChain pipelines
- **Smoke test suite** — unit + end-to-end integration tests included

## Requirements

- Python 3.10+
- Java 11+ (required by OpenDataLoader PDF)
- `pip install opendataloader-pdf sentence-transformers faiss-cpu langchain`

Optional (for proprietary models):
```
pip install openai cohere anthropic
```

## Quick Start

### 1. Ingest a PDF

```bash
python -m rag_pipeline.ingest path/to/document.pdf
```

With options:

```bash
python -m rag_pipeline.ingest path/to/document.pdf \
  --strategy section \
  --embedding-provider sentence-transformers \
  --embedding-model all-MiniLM-L6-v2 \
  --hybrid \
  --edit-metadata
```

### 2. Query the vector store

```bash
python -m rag_pipeline.query "What are the main conclusions?" --top-k 5
```

With an LLM answer:

```bash
python -m rag_pipeline.query "Summarise the methodology." \
  --llm-provider openai \
  --llm-model gpt-4o
```

## CLI Reference

### `ingest.py`

| Option | Default | Description |
|--------|---------|-------------|
| `PDF` | (required) | One or more PDF files to ingest |
| `--strategy` | `section` | Chunking strategy: `element`, `section`, `merged`, `table` |
| `--min-chars` | `200` | Minimum characters per chunk (merged strategy only) |
| `--hybrid` | off | Enable hybrid AI backend (requires hybrid server running on port 5002) |
| `--force-ocr` | off | Force OCR on all pages (start hybrid server with `--force-ocr`) |
| `--embedding-provider` | `sentence-transformers` | Embedding backend: `sentence-transformers`, `huggingface`, `openai`, `cohere` |
| `--embedding-model` | `all-MiniLM-L6-v2` | Embedding model name or path |
| `--device` | `cpu` | Device for local models: `cpu`, `cuda`, `mps` |
| `--db-path` | `rag_pipeline/data/chunks.db` | SQLite database path |
| `--faiss-path` | `rag_pipeline/data/faiss_index` | FAISS index directory path |
| `--edit-metadata` | off | Open interactive metadata editor before ingesting each PDF |

### `query.py`

| Option | Default | Description |
|--------|---------|-------------|
| `question` | (required) | Question to ask the RAG system |
| `--top-k` | `5` | Number of chunks to retrieve |
| `--embedding-provider` | `sentence-transformers` | Must match provider used at ingest time |
| `--embedding-model` | `all-MiniLM-L6-v2` | Must match model used at ingest time |
| `--device` | `cpu` | Device for local models: `cpu`, `cuda`, `mps` |
| `--llm-provider` | `openai` | LLM for answer synthesis: `openai`, `anthropic`, `cohere`, `ollama`, `huggingface` |
| `--llm-model` | `gpt-4o-mini` | LLM model name |
| `--system-prompt` | *(built-in)* | System prompt injected into every LLM call |
| `--no-sources` | off | Suppress source citation output |
| `--faiss-path` | `rag_pipeline/data/faiss_index` | FAISS index directory path |

## Project Structure

```
rag_pipeline/
├── ingest.py          # Ingestion CLI
├── query.py           # Query / RAG CLI
├── config.py          # Shared configuration dataclasses
├── converter.py       # OpenDataLoader PDF wrapper (local + hybrid)
├── chunker.py         # Four chunking strategies
├── embedder.py        # Pluggable embedding backends
├── vector_store.py    # FAISS index (add / search)
├── db.py              # SQLite schema + CRUD helpers
├── llm_provider.py    # Switchable LLM backends
├── edit_metadata.py   # Interactive metadata editing
├── smoke_test.py      # Unit + integration tests
└── README.md          # Detailed module-level docs
schema.sql             # SQLite DDL (apply once to pre-create tables)
```

See [rag_pipeline/README.md](rag_pipeline/README.md) for detailed module documentation.

## Database Schema

Two tables are maintained in SQLite:

- **`documents`** — one row per ingested PDF, with full file metadata
- **`chunks`** — one row per text chunk, including denormalised doc fields so every row is self-contained for retrieval

Apply the schema manually if needed:

```bash
sqlite3 rag_pipeline/data/chunks.db < schema.sql
```

## Running the Smoke Tests

```bash
python -m rag_pipeline.smoke_test
```

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | embedder, LLM | OpenAI API key |
| `COHERE_API_KEY` | embedder, LLM | Cohere API key |
| `ANTHROPIC_API_KEY` | LLM | Anthropic API key |
| `OLLAMA_BASE_URL` | LLM | Ollama server URL (default `http://localhost:11434`) |

## License

Apache 2.0
