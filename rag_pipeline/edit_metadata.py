"""
Interactive metadata editor for PDF files.

Before ingesting a PDF into the RAG pipeline you can inspect and override
the auto-detected metadata (title, author, subject, keywords, doc_type).

When run as a standalone CLI the module does a quick first-pass conversion
via OpenDataLoader PDF to auto-detect title, author, page count, tagged/scanned
status before prompting.  Edits are persisted in a JSON sidecar file placed
next to the PDF so they survive across multiple ingestion runs.

Usage (standalone):
    python -m rag_pipeline.edit_metadata path/to/file.pdf
    python rag_pipeline/edit_metadata.py path/to/file.pdf
    python rag_pipeline/edit_metadata.py path/to/file.pdf --reset

Usage (from code):
    from rag_pipeline.edit_metadata import load_or_edit_metadata
    meta = load_or_edit_metadata(pdf_path, doc_json=..., force_interactive=True)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".pdf_meta.json"

_EDITABLE_FIELDS: list[tuple[str, str]] = [
    ("title", "Document title"),
    ("author", "Author(s)"),
    ("subject", "Subject / description"),
    ("keywords", "Keywords (comma-separated)"),
    ("doc_type", "Document type  [book | article | tutorial | handout | presentation | other]"),
]


def _sidecar_path(pdf_path: Path) -> Path:
    return pdf_path.parent / (pdf_path.stem + SIDECAR_SUFFIX)


def _detect_metadata_from_json(doc_json: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata fields from an OpenDataLoader JSON document."""
    return {
        "title": doc_json.get("title") or "",
        "author": doc_json.get("author") or "",
        "subject": doc_json.get("subject") or "",
        "keywords": doc_json.get("keywords") or "",
        "doc_type": "",
        "num_pages": doc_json.get("number of pages"),
        "is_tagged": bool(doc_json.get("tagged", False)),
        "is_scanned": _detect_scanned(doc_json),
    }


def _detect_scanned(doc_json: dict[str, Any]) -> bool:
    """Heuristic: a PDF is likely scanned if it has no extractable text."""
    kids = doc_json.get("kids", [])
    total_text = sum(len(e.get("content", "") or "") for e in kids)
    return total_text < 50 and len(kids) < 5


def _auto_detect_from_pdf(pdf_path: Path) -> dict[str, Any]:
    """
    Do a fast first-pass conversion to extract metadata directly from a PDF.

    This is used by the standalone CLI so that the user sees auto-detected
    values (title, author, pages, tagged/scanned) before the edit prompt.
    Falls back to empty defaults if conversion fails.
    """
    try:
        import opendataloader_pdf
        import json as _json

        with tempfile.TemporaryDirectory() as tmp:
            opendataloader_pdf.convert(
                input_path=str(pdf_path),
                output_dir=tmp,
                format="json",
                quiet=True,
            )
            json_path = Path(tmp) / f"{pdf_path.stem}.json"
            if json_path.exists():
                with open(json_path, encoding="utf-8") as fh:
                    doc_json = _json.load(fh)
                return _detect_metadata_from_json(doc_json)
    except Exception:
        pass

    return {
        "title": "",
        "author": "",
        "subject": "",
        "keywords": "",
        "doc_type": "",
        "num_pages": None,
        "is_tagged": False,
        "is_scanned": False,
    }


def load_sidecar(pdf_path: Path) -> dict[str, Any] | None:
    """Load existing sidecar metadata if it exists."""
    sp = _sidecar_path(pdf_path)
    if sp.exists():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_sidecar(pdf_path: Path, meta: dict[str, Any]) -> None:
    """Persist metadata sidecar next to the PDF."""
    sp = _sidecar_path(pdf_path)
    sp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadata saved to: {sp}")


def _prompt_field(label: str, current: str) -> str:
    """Prompt the user to edit a single field. Return current value on empty input."""
    display = current if current else "(empty)"
    try:
        value = input(f"  {label}\n    Current: {display}\n    New value (Enter to keep): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return current
    return value if value else current


def interactive_edit(meta: dict[str, Any]) -> dict[str, Any]:
    """Run an interactive terminal session to edit metadata fields."""
    print("\n--- Metadata Editor ---")
    print("Press Enter to keep the current value for each field.\n")

    updated = dict(meta)
    for field_key, field_label in _EDITABLE_FIELDS:
        updated[field_key] = _prompt_field(field_label, str(meta.get(field_key) or ""))

    print("\nUpdated metadata:")
    for k, v in updated.items():
        print(f"  {k}: {v}")
    print()
    return updated


def load_or_edit_metadata(
    pdf_path: str | Path,
    doc_json: dict[str, Any] | None = None,
    force_interactive: bool = False,
) -> dict[str, Any]:
    """
    Return metadata for a PDF, optionally opening an interactive editor.

    Priority:
    1. Existing sidecar file (unless force_interactive)
    2. Auto-detected from doc_json (if provided) or by converting the PDF
    3. Empty defaults

    Args:
        pdf_path:          Path to the PDF file.
        doc_json:          Pre-parsed OpenDataLoader JSON output (optional).
        force_interactive: Always open the interactive editor even if a sidecar exists.

    Returns:
        A dict with keys: title, author, subject, keywords, doc_type,
        num_pages, is_tagged, is_scanned.
    """
    pdf_path = Path(pdf_path)

    existing = load_sidecar(pdf_path) if not force_interactive else None
    if existing:
        return existing

    if doc_json:
        meta = _detect_metadata_from_json(doc_json)
    else:
        meta = {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "doc_type": "",
            "num_pages": None,
            "is_tagged": False,
            "is_scanned": False,
        }

    if force_interactive or sys.stdin.isatty():
        print(f"\nFile: {pdf_path.name}")
        if meta.get("num_pages"):
            print(f"Pages: {meta['num_pages']}")
        if meta.get("is_tagged"):
            print("Detected: Tagged PDF (structure tree present)")
        if meta.get("is_scanned"):
            print("Detected: Likely scanned / image-based PDF (OCR may be needed)")
        meta = interactive_edit(meta)
        save_sidecar(pdf_path, meta)
    else:
        save_sidecar(pdf_path, meta)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively edit PDF metadata before RAG ingestion. "
            "The PDF is auto-parsed to pre-populate title, author, page count, "
            "tagged/scanned detection before the edit prompt."
        )
    )
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore any existing sidecar and start fresh (re-detect from PDF).",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    force_interactive = True
    if not args.reset:
        existing = load_sidecar(pdf_path)
        if existing:
            print(f"Found existing sidecar for '{pdf_path.name}'.")
            print(json.dumps(existing, indent=2, ensure_ascii=False))
            try:
                ans = input("Re-edit? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                ans = "n"
            if ans not in ("y", "yes"):
                print("Keeping existing metadata.")
                return

    print(f"[edit_metadata] Auto-detecting metadata from '{pdf_path.name}' ...")
    meta = _auto_detect_from_pdf(pdf_path)
    print(f"\nFile: {pdf_path.name}")
    if meta.get("num_pages"):
        print(f"Pages: {meta['num_pages']}")
    if meta.get("is_tagged"):
        print("Detected: Tagged PDF (structure tree present)")
    if meta.get("is_scanned"):
        print("Detected: Likely scanned / image-based PDF (OCR may be needed)")

    meta = interactive_edit(meta)
    save_sidecar(pdf_path, meta)
    print("\nFinal metadata:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
