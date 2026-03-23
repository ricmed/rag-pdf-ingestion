"""
PDF conversion module for the RAG pipeline.

Wraps opendataloader_pdf.convert() with smart defaults:
  - Detects tagged PDFs and enables use_struct_tree
  - Detects scanned PDFs (and warns when force_ocr is requested but
    the hybrid server must be started with --force-ocr separately)
  - Enables image description by switching to hybrid_mode=full when images
    are present and enrich_picture_description=True
  - Outputs both JSON (for structured chunking) and Markdown (for fallback)
  - Provides a LangChain-based loader path for tagged PDFs via
    OpenDataLoaderPDFLoader when the langchain-opendataloader-pdf package
    is available

OCR and image-enrichment flags are hybrid-SERVER options, not client-side
convert() kwargs.  The correct workflow is:

    # Start the hybrid server with the desired server-side flags:
    opendataloader-pdf-hybrid --port 5002 --force-ocr --enrich-picture-description

    # Then call ingest with --hybrid:
    python rag_pipeline/ingest.py scan.pdf --hybrid

The converter detects scanned docs and warns the user if they have not
enabled --force-ocr on the server.

Supported opendataloader_pdf.convert() parameters (verified against SDK 2.0.2):
  input_path, output_dir, password, format, quiet, content_safety_off,
  sanitize, keep_line_breaks, replace_invalid_chars, use_struct_tree,
  table_method, reading_order, markdown_page_separator,
  text_page_separator, html_page_separator, image_output, image_format,
  image_dir, pages, include_header_footer,
  hybrid, hybrid_mode, hybrid_url, hybrid_timeout, hybrid_fallback
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import opendataloader_pdf

from rag_pipeline.config import HybridConfig

_CONVERT_PARAMS: frozenset[str] = frozenset(
    inspect.signature(opendataloader_pdf.convert).parameters
)


def _safe_convert(**kwargs: Any) -> None:
    """
    Call opendataloader_pdf.convert() filtering to only supported parameters.

    This prevents TypeError if a future caller passes an unsupported kwarg.
    Any unsupported kwarg is silently dropped with a warning message.
    """
    unsupported = {k for k in kwargs if k not in _CONVERT_PARAMS}
    if unsupported:
        print(
            f"[converter] WARNING: unsupported convert() kwargs ignored: {unsupported}",
            flush=True,
        )
    safe_kwargs = {k: v for k, v in kwargs.items() if k in _CONVERT_PARAMS}
    opendataloader_pdf.convert(**safe_kwargs)


def _has_images(doc_json: dict[str, Any]) -> bool:
    """Return True if the document contains any image elements."""
    return any(e.get("type") == "image" for e in doc_json.get("kids", []))


def _is_tagged(doc_json: dict[str, Any]) -> bool:
    """Return True if the document has a PDF structure tree."""
    return bool(doc_json.get("tagged", False))


def _is_scanned(doc_json: dict[str, Any]) -> bool:
    """Heuristic: scanned PDF has very little selectable text."""
    kids = doc_json.get("kids", [])
    total_text = sum(len(e.get("content", "") or "") for e in kids)
    return total_text < 50 and len(kids) < 5


def load_via_langchain(
    pdf_path: Path,
    use_struct_tree: bool = False,
) -> list[Any] | None:
    """
    Load a PDF using the LangChain OpenDataLoader integration.

    This is the preferred path for tagged PDFs when you want LangChain
    Document objects for downstream chain use.  Returns None if the
    langchain-opendataloader-pdf package is not installed.

    Args:
        pdf_path:        Path to the PDF.
        use_struct_tree: Set True for tagged PDFs to exploit the structure tree.

    Returns:
        List of LangChain Document objects, or None if unavailable.
    """
    try:
        from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
    except ImportError:
        return None

    loader = OpenDataLoaderPDFLoader(
        file_path=[str(pdf_path)],
        use_struct_tree=use_struct_tree,
        format="text",
        quiet=True,
    )
    docs = loader.load()
    return docs


def convert_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    hybrid_cfg: HybridConfig | None = None,
    password: str | None = None,
    quiet: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Convert a single PDF file using OpenDataLoader PDF.

    Two-pass approach:
      1. Fast first-pass (JSON only, no hybrid) to inspect the document structure.
      2. Full second-pass with all appropriate flags applied (hybrid, tagged, etc.).

    OCR / image enrichment:
      These are controlled by how the hybrid server is started, not by
      client-side flags.  If hybrid_cfg.force_ocr is True the user is warned
      that the hybrid server must be started with --force-ocr.

    Tagged PDFs:
      When a tagged PDF is detected, use_struct_tree=True is passed to the
      converter.  The load_via_langchain() helper is also available for
      obtaining LangChain Document objects directly from a tagged PDF.

    Returns:
        (json_path, markdown_path, doc_json)  where doc_json is the parsed JSON.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    if hybrid_cfg is None:
        hybrid_cfg = HybridConfig()

    print(f"[converter] Converting '{pdf_path.name}' ...", flush=True)

    first_pass_kwargs: dict[str, Any] = {
        "input_path": str(pdf_path),
        "output_dir": str(output_dir),
        "format": "json",
        "reading_order": "xycut",
        "table_method": "cluster",
        "quiet": quiet,
    }
    if password:
        first_pass_kwargs["password"] = password

    _safe_convert(**first_pass_kwargs)

    if not json_path.exists():
        raise FileNotFoundError(
            f"[converter] Expected JSON output not found: {json_path}"
        )

    with open(json_path, encoding="utf-8") as f:
        doc_json = json.load(f)

    tagged = _is_tagged(doc_json)
    scanned = _is_scanned(doc_json)
    has_images_flag = _has_images(doc_json)

    if tagged:
        print("[converter] Tagged PDF detected — enabling structure tree.", flush=True)
    if scanned:
        if hybrid_cfg.force_ocr:
            print(
                "[converter] Scanned PDF detected + force_ocr requested. "
                "Ensure the hybrid server was started with --force-ocr.",
                flush=True,
            )
        else:
            print(
                "[converter] Scanned/image PDF detected — "
                "start the hybrid server with --force-ocr for best OCR results.",
                flush=True,
            )

    second_pass_kwargs: dict[str, Any] = {
        "input_path": str(pdf_path),
        "output_dir": str(output_dir),
        "format": "json,markdown",
        "reading_order": "xycut",
        "table_method": "cluster",
        "use_struct_tree": tagged,
        "quiet": quiet,
    }
    if password:
        second_pass_kwargs["password"] = password

    if hybrid_cfg.enabled:
        second_pass_kwargs["hybrid"] = hybrid_cfg.backend
        second_pass_kwargs["hybrid_url"] = hybrid_cfg.url
        second_pass_kwargs["hybrid_timeout"] = str(hybrid_cfg.timeout_ms)
        second_pass_kwargs["hybrid_fallback"] = hybrid_cfg.fallback

        if has_images_flag and hybrid_cfg.enrich_picture_description:
            second_pass_kwargs["hybrid_mode"] = "full"
            print(
                "[converter] Images found — using hybrid_mode=full for picture descriptions.",
                flush=True,
            )
        else:
            second_pass_kwargs["hybrid_mode"] = hybrid_cfg.mode

    print("[converter] Running full conversion ...", flush=True)
    _safe_convert(**second_pass_kwargs)

    with open(json_path, encoding="utf-8") as f:
        doc_json = json.load(f)

    print(
        f"[converter] Done. Pages: {doc_json.get('number of pages')} | "
        f"Elements: {len(doc_json.get('kids', []))}",
        flush=True,
    )

    return json_path, md_path, doc_json
