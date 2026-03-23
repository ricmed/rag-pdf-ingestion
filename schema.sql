-- PDF RAG Pipeline Database Schema
-- Version: 1.1.0
--
-- Two tables:
--   documents  — one row per ingested PDF file
--   chunks     — one row per text chunk produced from a document
--              — includes denormalised doc-level fields (title, author,
--                file_name, file_path, doc_type, num_pages) so that every
--                chunk row is self-contained for retrieval without a join
--
-- Apply with:
--   sqlite3 rag_pipeline/data/chunks.db < schema.sql

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id           TEXT PRIMARY KEY,          -- UUID, generated at ingest time
    file_name        TEXT NOT NULL,             -- basename of the PDF file
    file_path        TEXT NOT NULL,             -- absolute path to the PDF file
    title            TEXT,                      -- PDF title (from metadata or overridden)
    author           TEXT,                      -- PDF author (from metadata or overridden)
    subject          TEXT,                      -- PDF subject / description
    keywords         TEXT,                      -- comma-separated keywords
    num_pages        INTEGER,                   -- total page count
    doc_type         TEXT,                      -- e.g. "book", "article", "tutorial", "handout", "presentation"
    is_tagged        INTEGER NOT NULL DEFAULT 0, -- 1 if PDF has a structure tree (tagged PDF)
    is_scanned       INTEGER NOT NULL DEFAULT 0, -- 1 if PDF appears to be image-only / scanned
    chunking_strategy TEXT NOT NULL,            -- strategy used: element | section | merged | table
    ingested_at      TEXT NOT NULL              -- ISO-8601 UTC timestamp
);

CREATE INDEX IF NOT EXISTS idx_documents_file_name ON documents (file_name);
CREATE INDEX IF NOT EXISTS idx_documents_ingested_at ON documents (ingested_at);

-- ---------------------------------------------------------------------------
-- chunks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,           -- UUID
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,           -- 0-based position within the document
    text            TEXT NOT NULL,              -- the chunk text
    chunk_type      TEXT NOT NULL,              -- element type: paragraph | heading | list | table | picture | mixed
    strategy        TEXT NOT NULL,              -- chunking strategy that produced this chunk
    page_start      INTEGER,                    -- first page (1-indexed)
    page_end        INTEGER,                    -- last page (1-indexed); equals page_start for single-page chunks
    bbox_json       TEXT,                       -- JSON array [left, bottom, right, top] in PDF points, or null
    section_heading TEXT,                       -- nearest ancestor heading text, or null
    -- denormalised document fields for self-contained retrieval
    file_name       TEXT,                       -- basename of the source PDF
    file_path       TEXT,                       -- absolute path to the source PDF
    title           TEXT,                       -- document title
    author          TEXT,                       -- document author(s)
    doc_type        TEXT,                       -- document type
    num_pages       INTEGER,                    -- total page count of source document
    ingested_at     TEXT NOT NULL               -- ISO-8601 UTC timestamp
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks (doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks (doc_id, page_start);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks (chunk_type);
