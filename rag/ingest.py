"""
RD.011 Agent - RAG ingest pipeline.

Parses example RD.011 .docx files, chunks them using structure-aware rules,
and stores chunks in Chroma with rich metadata for retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Allow running as `python rag/ingest.py ...` as well as `python -m rag.ingest ...`.
# When run as a script, Python sets sys.path[0] to `.../rag`, so `import rag.*` fails
# unless we add the project root to sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config_rag import CHROMA_DB_PATH, COLLECTION_NAME, get_embedding_function

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 200

SECTION_TYPES = {
    "document_control",
    "intro",
    "how_organized",
    "enterprise_structure",
    "ledger",
    "coa",
    "module_overview",
    "process_outline",
    "process_narrative",
    "process_steps",
    "journal_entries",
    "key_requirements",
    "process_diagram",
    "issues_open",
    "issues_closed",
    "other",
}

INGEST_BATCH_SIZE = 80
INGEST_MAX_RETRIES = 2


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _heading_level(paragraph: Any) -> int | None:
    style_name = paragraph.style.name if paragraph.style else ""
    if style_name.startswith("Heading"):
        try:
            return int(style_name.split()[-1])
        except (ValueError, IndexError):
            return None
    return None


def _iter_block_items(doc: Any) -> Iterable[Any]:
    """
    Yield paragraphs and tables in the order they appear in the document body.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _table_to_markdown(table: Any) -> str:
    """
    Convert a python-docx table to a markdown-like string.
    """
    rows: list[list[str]] = []
    for row in table.rows:
        row_cells: list[str] = []
        for cell in row.cells:
            row_cells.append(
                " ".join((p.text or "").strip() for p in cell.paragraphs if (p.text or "").strip())
            )
        rows.append(row_cells)
    if not rows:
        return ""

    # Deduplicate repeated merged-cell text per row
    cleaned: list[list[str]] = []
    for r in rows:
        out: list[str] = []
        prev = None
        for cell in r:
            if cell != prev:
                out.append(cell)
            prev = cell
        cleaned.append(out)

    max_cols = max(len(r) for r in cleaned)
    for r in cleaned:
        while len(r) < max_cols:
            r.append("")

    lines: list[str] = []
    header = cleaned[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in cleaned[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _infer_module(text: str) -> str | None:
    t = _norm(text)
    if "accounts payable" in t or re.search(r"\bpayables\b", t):
        return "AP"
    if "accounts receivable" in t or re.search(r"\breceivables\b", t):
        return "AR"
    if "general ledger" in t:
        return "GL"
    if "fixed asset" in t or re.search(r"\bfixed assets\b", t):
        return "FA"
    if "cash management" in t or re.search(r"\btreasury\b", t):
        return "CM"
    return None


def _parse_process_heading(text: str) -> tuple[str | None, str | None]:
    """
    Extract (process_id, process_name) from headings like:
    - "AP.02 Create PO Invoice"
    - "FA-07 Asset Retirement (Disposal)"
    """
    raw = (text or "").strip()
    m = re.match(r"^(AP|AR|GL|FA|CM|CE)\s*[.\-]\s*(\d{2})\b[:.\-\s]*(.*)$", raw, flags=re.I)
    if not m:
        return None, None
    mod = m.group(1).upper()
    if mod == "CE":
        mod = "CM"
    num = m.group(2)
    name = (m.group(3) or "").strip(" -:\t")
    return f"{mod}.{num}", (name or None)


def _infer_process_type(module: str | None, process_name: str) -> str | None:
    t = _norm(process_name)
    if any(k in t for k in ("supplier", "vendor", "customer", "chart of accounts", "coa")):
        return "master_data"
    if "prepayment" in t:
        return "prepayment"
    if "credit memo" in t or "debit memo" in t or "debit/credit" in t:
        return "memo"
    if "po invoice" in t or ("po" in t and "invoice" in t):
        return "invoice_po"
    if "manual invoice" in t or "non-po" in t or "non po" in t or "direct invoice" in t:
        return "invoice_non_po"
    if "payment" in t:
        return "payment"
    if "receipt" in t or "collection" in t:
        return "receipt"
    if "reconciliation" in t or "statement" in t:
        return "reconciliation"
    if "close" in t or "period end" in t or "month end" in t:
        return "month_end_close"
    if "journal" in t:
        return "journal"
    if "revaluation" in t:
        return "revaluation"
    if "budget" in t:
        return "budgeting"
    if module == "FA":
        if "retire" in t or "disposal" in t:
            return "asset_retirement"
        if "transfer" in t:
            return "asset_transfer"
        if "impair" in t:
            return "asset_impairment"
        if "physical count" in t or "count" in t:
            return "physical_count"
        if "capitalize" in t or "capitalise" in t:
            return "asset_capitalization"
        if "add" in t or "addition" in t:
            return "asset_addition"
    return None


def _classify_heading(text: str) -> str | None:
    """
    Map a heading/subheading to an internal section_type label.

    This does not rely on exact template wording.
    """
    t = _norm(text)
    if not t:
        return None

    if t == "document control":
        return "document_control"
    if t == "introduction":
        return "intro"
    if "how this document is organized" in t or "how this document is organised" in t:
        return "how_organized"
    if "enterprise structure" in t or "structure segments" in t or "business architecture" in t:
        return "enterprise_structure"
    if t == "narrative" or "process details" in t:
        return "process_narrative"
    if "process steps" in t or "process step" in t:
        return "process_steps"
    if "journal entry" in t or "journal entries" in t:
        return "journal_entries"
    if "key requirements" in t or "highlights" in t:
        return "key_requirements"
    if "process diagram" in t or "process flow diagram" in t:
        return "process_diagram"
    if "open issues" in t and "closed" not in t:
        return "issues_open"
    if "closed issues" in t:
        return "issues_closed"
    if "ledger" in t:
        return "ledger"
    if "chart of accounts" in t or re.search(r"\bcoa\b", t):
        return "coa"

    return None


def _compute_retry_delay_seconds(error: Exception, attempt: int) -> float | None:
    """
    Return a retry delay in seconds for quota/rate-limit errors, else None.
    """
    text = str(error)
    lower = text.lower()
    if "429" not in text and "resource_exhausted" not in lower and "quota" not in lower:
        return None

    # Prefer provider-supplied delays when available.
    m = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", lower)
    if m:
        return float(m.group(1)) + 1.0

    m = re.search(r"retrydelay['\"]?\s*[:=]\s*['\"]?([0-9]+)s", lower)
    if m:
        return float(m.group(1)) + 1.0

    # Fallback exponential backoff capped at 60s.
    return min(60.0, 5.0 * (2**attempt))


def _stable_chunk_id(doc: dict[str, Any]) -> str:
    """
    Build a deterministic ID so repeated ingests upsert instead of duplicating.
    """
    meta = doc.get("metadata") or {}
    source_path = str(meta.get("source_path") or meta.get("source") or "")
    chunk_index = str(meta.get("chunk_index") or "")
    section_type = str(meta.get("section_type") or "")
    process_id = str(meta.get("process_id") or "")
    raw = f"{source_path}|{chunk_index}|{section_type}|{process_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_and_chunk_examples(docs_dir: Path) -> list[dict[str, Any]]:
    """
    Parse all .docx files in docs_dir into structure-aware chunks.

    The output is a list of dicts:
      {"page_content": str, "metadata": dict}
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise ValueError(f"Documents directory does not exist: {docs_dir}")

    logger.info("Scanning %s for .docx files...", docs_dir)
    docx_files = sorted(docs_dir.glob("**/*.docx"))
    logger.info("Found %d .docx files", len(docx_files))

    if not docx_files:
        logger.warning("No .docx files found in %s", docs_dir)
        return []

    all_documents: list[dict[str, Any]] = []

    for docx_file in docx_files:
        logger.info("Loading: %s", docx_file.name)
        try:
            from docx import Document
        except Exception as exc:
            raise RuntimeError("python-docx is required for RAG ingest.") from exc

        try:
            doc = Document(str(docx_file))

            current_module: str | None = None
            current_process_id: str | None = None
            current_process_name: str | None = None
            current_process_type: str | None = None
            current_section_type: str = "other"
            current_section_heading: str | None = None
            in_process_list = False

            buf: list[str] = []
            chunk_index = 0

            def flush() -> None:
                nonlocal buf, chunk_index
                text = "\n".join(line for line in buf if line).strip()
                buf = []
                if len(text) < MIN_CHUNK_CHARS:
                    return

                st = current_section_type if current_section_type in SECTION_TYPES else "other"
                meta: dict[str, Any] = {
                    "source": docx_file.name,
                    "source_path": str(docx_file),
                    "style_family": docx_file.name,
                    "section_type": st,
                    "section": (current_section_heading or "")[:140],
                    "chunk_index": chunk_index,
                }
                if current_module:
                    meta["module"] = current_module
                if current_process_id:
                    meta["process_id"] = current_process_id
                if current_process_name:
                    meta["process_name"] = current_process_name
                if current_process_type:
                    meta["process_type"] = current_process_type

                all_documents.append({"page_content": text, "metadata": meta})
                chunk_index += 1

            for block in _iter_block_items(doc):
                # Paragraph
                if hasattr(block, "text"):
                    text = (block.text or "").strip()
                    if not text:
                        continue

                    lvl = _heading_level(block)
                    if lvl is not None:
                        flush()

                        current_section_heading = text

                        # Module detection resets process context.
                        mod = _infer_module(text)
                        if mod:
                            current_module = mod
                            current_process_id = None
                            current_process_name = None
                            current_process_type = None
                            in_process_list = False

                        ht = _norm(text)
                        if "business processes" in ht or "list of processes" in ht or "business process" in ht:
                            in_process_list = True

                        pid, pname = _parse_process_heading(text)
                        if pid:
                            current_module = pid.split(".")[0]
                            current_process_id = pid
                            current_process_name = pname or current_process_name
                            current_process_type = _infer_process_type(current_module, current_process_name or "")
                            in_process_list = True

                        cls = _classify_heading(text)

                        # If we're inside a process list, treat short headings as process titles.
                        if (
                            in_process_list
                            and current_module
                            and not pid
                            and cls is None
                            and len(text) <= 120
                            and lvl <= 3
                        ):
                            current_process_id = None
                            current_process_name = text
                            current_process_type = _infer_process_type(current_module, current_process_name)

                        if cls:
                            current_section_type = cls
                        else:
                            # Default to module overview when not a recognized subsection marker.
                            current_section_type = "module_overview" if current_module else "other"

                        buf.append(f"# {text}")
                        continue

                    buf.append(text)
                    continue

                # Table
                table_md = ""
                try:
                    table_md = _table_to_markdown(block)
                except Exception:
                    table_md = ""
                if table_md.strip():
                    buf.append(table_md)

            flush()

        except Exception as exc:
            logger.error("Failed to parse %s: %s", docx_file.name, exc)
            continue

    logger.info("Total chunks created: %d", len(all_documents))
    return all_documents


def ingest_to_chroma(documents: list[dict[str, Any]], collection_name: str = COLLECTION_NAME) -> int:
    """
    Ingest documents into Chroma vector database.
    """
    if not documents:
        logger.info("No documents to ingest.")
        return 0

    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Using Chroma DB at: %s", CHROMA_DB_PATH)

    embedding_func = get_embedding_function()

    try:
        from langchain_chroma import Chroma
    except Exception as exc:
        raise RuntimeError(
            "langchain_chroma is required for RAG ingest. Install with: pip install langchain-chroma"
        ) from exc

    vector_db = Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embedding_func,
    )

    total_batches = (len(documents) + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE
    ingested_count = 0
    stopped_due_to_quota = False

    for start in range(0, len(documents), INGEST_BATCH_SIZE):
        end = min(start + INGEST_BATCH_SIZE, len(documents))
        batch = documents[start:end]
        texts = [doc["page_content"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        ids = [_stable_chunk_id(doc) for doc in batch]
        batch_no = (start // INGEST_BATCH_SIZE) + 1

        for attempt in range(INGEST_MAX_RETRIES + 1):
            try:
                vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                ingested_count += len(batch)
                logger.info(
                    "Ingested batch %d/%d (%d chunks).",
                    batch_no,
                    total_batches,
                    len(batch),
                )
                break
            except Exception as exc:
                delay_seconds = _compute_retry_delay_seconds(exc, attempt)
                if delay_seconds is None:
                    raise
                if attempt >= INGEST_MAX_RETRIES:
                    logger.error(
                        "Quota/rate limit persisted for batch %d/%d after %d attempts. "
                        "Stopping ingest early with %d/%d chunks stored. Re-run later to continue.",
                        batch_no,
                        total_batches,
                        INGEST_MAX_RETRIES + 1,
                        ingested_count,
                        len(documents),
                    )
                    stopped_due_to_quota = True
                    break
                logger.warning(
                    "Embedding rate limit on batch %d/%d (attempt %d/%d). Retrying in %.1fs.",
                    batch_no,
                    total_batches,
                    attempt + 1,
                    INGEST_MAX_RETRIES,
                    delay_seconds,
                )
                time.sleep(delay_seconds)

        if stopped_due_to_quota:
            break

    persist_fn = getattr(vector_db, "persist", None)
    if callable(persist_fn):
        persist_fn()
    else:
        logger.info("Chroma persist() not available; relying on automatic persistence.")

    logger.info("Ingested %d chunks into Chroma collection '%s'.", ingested_count, collection_name)
    if stopped_due_to_quota:
        logger.warning(
            "Ingestion finished early due to quota limits. Stored %d/%d chunks.",
            ingested_count,
            len(documents),
        )
    if documents:
        logger.info("Sample metadata: %s", json.dumps(documents[0]["metadata"], indent=2))
    return ingested_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest example RD.011 documents into a Chroma vector store.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="Directory containing .docx example documents.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="Chroma collection name (default: rd011_examples).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting RAG ingest pipeline...")
    logger.info("Examples directory: %s", args.docs_dir)

    documents = parse_and_chunk_examples(args.docs_dir)
    ingested = ingest_to_chroma(documents, collection_name=args.collection)
    logger.info("Ingestion complete: %d chunks stored.", ingested)


if __name__ == "__main__":
    main()
