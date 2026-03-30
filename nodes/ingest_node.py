"""
RD.011 Agent — Ingest node.

Reads each file in state["input_files"], routes to the appropriate parser,
and stores parsed text in state["raw_texts"] with filename as key.
"""

from __future__ import annotations

import logging
import os

from parsers.docx_parser import parse_docx
from parsers.excel_parser import parse_excel

logger = logging.getLogger(__name__)


def ingest_node(state: dict) -> dict:
    """
    Parse all input files and store their text content.

    Routes ``.docx`` files to ``parse_docx`` and ``.xlsx`` files to
    ``parse_excel``.  Preserves the source filename in the key to track
    provenance.
    """
    input_files = state.get("input_files", [])
    raw_texts: dict[str, str] = {}
    errors: list[str] = list(state.get("errors", []))

    if not input_files:
        errors.append("No input files provided.")
        return {"raw_texts": raw_texts, "errors": errors, "last_completed_node": "ingest"}

    for file_path in input_files:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".docx":
                text = parse_docx(file_path)
            elif ext == ".xlsx":
                text = parse_excel(file_path)
            else:
                logger.warning("Unsupported file type '%s' for %s — skipping", ext, filename)
                errors.append(f"Unsupported file type: {filename}")
                continue

            raw_texts[filename] = text
            logger.info("Ingested %s (%d chars)", filename, len(text))

        except Exception as exc:
            error_msg = f"Failed to parse {filename}: {exc}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info("Ingest complete: %d files parsed", len(raw_texts))
    return {
        "raw_texts": raw_texts,
        "errors": errors,
        "last_completed_node": "ingest",
    }
