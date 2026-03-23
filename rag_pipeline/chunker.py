"""
Chunking strategies for the PDF RAG pipeline.

Each strategy consumes an OpenDataLoader JSON document and returns a list of
chunk dicts with the following common fields:

    text           str   — the chunk text
    chunk_type     str   — element type: paragraph | heading | list | table | image | mixed
    page_start     int   — first page (1-indexed)
    page_end       int   — last page (1-indexed)
    bbox           list  — [left, bottom, right, top] in PDF points, or None
    section_heading str  — nearest ancestor heading, or None

Available strategies (selectable by name):
    element  — one chunk per paragraph, heading, or list element
    section  — group all content under the nearest heading
    merged   — merge adjacent elements until a minimum character count is reached
    table    — tables as separate chunks (text elements still get element strategy)

Tables are always extracted as dedicated chunks regardless of strategy.
Picture descriptions (if present from hybrid mode) are included as chunks.
"""

from __future__ import annotations

import json
from typing import Any


ELEMENT_TYPES = ("paragraph", "heading", "list")
CHUNK_TYPES_INCLUDE = ("paragraph", "heading", "list", "table", "image")


def _table_to_markdown(element: dict[str, Any]) -> str:
    """Serialize an OpenDataLoader table element to Markdown."""
    rows = element.get("rows", [])
    if not rows:
        return element.get("content", "")

    md_rows: list[str] = []
    for i, row in enumerate(rows):
        cells = row.get("cells", [])
        cell_texts = [c.get("content", "") or "" for c in cells]
        md_rows.append("| " + " | ".join(cell_texts) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")

    return "\n".join(md_rows)


def _extract_tables(
    doc: dict[str, Any],
    doc_id: str,
    strategy: str,
) -> list[dict[str, Any]]:
    """Extract all table elements as standalone chunks."""
    chunks = []
    for element in doc.get("kids", []):
        if element.get("type") != "table":
            continue
        text = _table_to_markdown(element)
        if not text.strip():
            continue
        page = element.get("page number")
        chunks.append(
            {
                "text": text,
                "chunk_type": "table",
                "page_start": page,
                "page_end": page,
                "bbox": element.get("bounding box"),
                "section_heading": None,
                "strategy": strategy,
            }
        )
    return chunks


def _extract_pictures(
    doc: dict[str, Any],
    strategy: str,
) -> list[dict[str, Any]]:
    """Extract image elements that have a description or alt text."""
    chunks = []
    for element in doc.get("kids", []):
        if element.get("type") != "image":
            continue
        description = element.get("description") or element.get("content") or ""
        if not description.strip():
            continue
        page = element.get("page number")
        chunks.append(
            {
                "text": description,
                "chunk_type": "image",
                "page_start": page,
                "page_end": page,
                "bbox": element.get("bounding box"),
                "section_heading": None,
                "strategy": strategy,
            }
        )
    return chunks


def chunk_by_element(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Strategy: element

    One chunk per paragraph, heading, or list element.
    Best for fine-grained retrieval and precise citations.
    Tables and pictures are always included as their own chunks.
    """
    strategy = "element"
    chunks: list[dict[str, Any]] = []
    current_heading: str | None = None

    for element in doc.get("kids", []):
        etype = element.get("type")

        if etype == "heading":
            current_heading = element.get("content") or ""

        if etype in ELEMENT_TYPES:
            text = element.get("content") or ""
            if not text.strip():
                continue
            page = element.get("page number")
            chunks.append(
                {
                    "text": text,
                    "chunk_type": etype,
                    "page_start": page,
                    "page_end": page,
                    "bbox": element.get("bounding box"),
                    "section_heading": current_heading if etype != "heading" else None,
                    "strategy": strategy,
                }
            )

    chunks.extend(_extract_tables(doc, "", strategy))
    chunks.extend(_extract_pictures(doc, strategy))
    return chunks


def chunk_by_section(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Strategy: section

    Group all content under the nearest heading into one chunk.
    Best for context-rich retrieval and topic-based search.
    Tables and pictures are included as their own chunks.
    """
    strategy = "section"
    chunks: list[dict[str, Any]] = []

    current_heading: str | None = None
    current_texts: list[str] = []
    current_start_page: int | None = None
    current_end_page: int | None = None

    def flush() -> None:
        nonlocal current_texts, current_start_page, current_end_page
        if not current_texts:
            return
        chunks.append(
            {
                "text": "\n\n".join(current_texts),
                "chunk_type": "mixed",
                "page_start": current_start_page,
                "page_end": current_end_page,
                "bbox": None,
                "section_heading": current_heading,
                "strategy": strategy,
            }
        )
        current_texts = []
        current_start_page = None
        current_end_page = None

    for element in doc.get("kids", []):
        etype = element.get("type")
        page = element.get("page number")

        if etype == "heading":
            flush()
            current_heading = element.get("content") or ""
            content = current_heading
            current_texts = [content]
            current_start_page = page
            current_end_page = page
        elif etype in ("paragraph", "list"):
            content = element.get("content") or ""
            if content.strip():
                current_texts.append(content)
                if current_start_page is None:
                    current_start_page = page
                current_end_page = page

    flush()

    chunks.extend(_extract_tables(doc, "", strategy))
    chunks.extend(_extract_pictures(doc, strategy))
    return chunks


def chunk_merged(doc: dict[str, Any], min_chars: int = 200) -> list[dict[str, Any]]:
    """
    Strategy: merged

    Merge adjacent text elements until min_chars threshold is reached.
    Best for balanced chunk sizes when documents lack clear headings.
    Tables and pictures are included as their own chunks.
    """
    strategy = "merged"
    chunks: list[dict[str, Any]] = []

    buffer_texts: list[str] = []
    buffer_pages: list[int] = []
    current_heading: str | None = None

    def flush() -> None:
        if not buffer_texts:
            return
        chunks.append(
            {
                "text": "\n\n".join(buffer_texts),
                "chunk_type": "mixed",
                "page_start": buffer_pages[0] if buffer_pages else None,
                "page_end": buffer_pages[-1] if buffer_pages else None,
                "bbox": None,
                "section_heading": current_heading,
                "strategy": strategy,
            }
        )
        buffer_texts.clear()
        buffer_pages.clear()

    for element in doc.get("kids", []):
        etype = element.get("type")
        if etype == "heading":
            current_heading = element.get("content") or ""

        if etype in ELEMENT_TYPES:
            content = element.get("content") or ""
            page = element.get("page number")
            if content.strip():
                buffer_texts.append(content)
                if page is not None and (not buffer_pages or buffer_pages[-1] != page):
                    buffer_pages.append(page)

            total = sum(len(t) for t in buffer_texts)
            if total >= min_chars:
                flush()

    flush()

    chunks.extend(_extract_tables(doc, "", strategy))
    chunks.extend(_extract_pictures(doc, strategy))
    return chunks


def chunk_tables_only(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Strategy: table

    Tables extracted as dedicated chunks; text elements use element strategy.
    Best for documents that are primarily table-heavy (financial reports, etc.).
    """
    strategy = "table"
    element_chunks = chunk_by_element(doc)
    for c in element_chunks:
        c["strategy"] = strategy
    return element_chunks


def get_chunks(
    doc: dict[str, Any],
    strategy: str = "section",
    min_chars: int = 200,
) -> list[dict[str, Any]]:
    """
    Dispatcher: return chunks produced by the named strategy.

    Args:
        doc:       Parsed OpenDataLoader JSON document.
        strategy:  One of: element | section | merged | table
        min_chars: Minimum character count for the 'merged' strategy.

    Returns:
        List of chunk dicts (without chunk_id, doc_id, chunk_index — added by ingest.py).
    """
    strategy = strategy.lower().strip()
    if strategy == "element":
        return chunk_by_element(doc)
    if strategy == "section":
        return chunk_by_section(doc)
    if strategy == "merged":
        return chunk_merged(doc, min_chars=min_chars)
    if strategy == "table":
        return chunk_tables_only(doc)
    raise ValueError(
        f"Unknown chunking strategy '{strategy}'. "
        "Choose from: element, section, merged, table"
    )
