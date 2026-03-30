"""
RD.011 Agent — Word (.docx) parser.

Extracts structured text from Minutes of Meeting and Scope documents.
Preserves heading levels, paragraph text, and table content as
markdown-style output.  Handles Arabic/English mixed content and merged
table cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

from docx import Document
from docx.table import Table, _Cell
from docx.oxml.ns import qn

logger = logging.getLogger(__name__)


def _heading_level(paragraph) -> int | None:
    """Return the heading level (1-9) or None if not a heading."""
    style_name = paragraph.style.name if paragraph.style else ""
    if style_name.startswith("Heading"):
        try:
            return int(style_name.split()[-1])
        except (ValueError, IndexError):
            pass
    # Check outline level in XML
    pPr = paragraph._element.find(qn("w:pPr"))
    if pPr is not None:
        outlineLvl = pPr.find(qn("w:outlineLvl"))
        if outlineLvl is not None:
            val = outlineLvl.get(qn("w:val"))
            if val is not None:
                return int(val) + 1
    return None


def _get_cell_text(cell: _Cell) -> str:
    """Extract text from a table cell, joining paragraphs with spaces."""
    return " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())


def _iter_merged_cells(table: Table):
    """
    Yield (row_idx, col_idx, cell_text) for each cell in the table,
    handling merged cells by reading the actual cell span.
    """
    for row_idx, row in enumerate(table.rows):
        for col_idx, cell in enumerate(row.cells):
            # python-docx reports the same _Cell object for all cells
            # spanned by a merge.  Check if this cell's top-left is the
            # current position to avoid duplicates.
            tc = cell._tc
            grid_span = tc.find(qn("w:tcPr"))
            if grid_span is not None:
                v_merge = grid_span.find(qn("w:vMerge"))
                if v_merge is not None:
                    val = v_merge.get(qn("w:val"), "")
                    # val == "restart" means start of vertical merge
                    # val == "" (or absent attribute) means continuation
                    if val != "restart":
                        continue  # skip continuation cells
            yield row_idx, col_idx, _get_cell_text(cell)


def _table_to_markdown(table: Table) -> str:
    """Convert a python-docx Table to a markdown-style table string."""
    rows_data: list[list[str]] = []
    max_cols = 0

    for row in table.rows:
        row_cells = []
        for cell in row.cells:
            row_cells.append(_get_cell_text(cell))
        # Deduplicate cells from horizontal merges:
        # python-docx returns the same cell object for merged spans
        deduped: list[str] = []
        prev_text = None
        for text in row_cells:
            if text != prev_text:
                deduped.append(text)
            prev_text = text
        rows_data.append(deduped)
        max_cols = max(max_cols, len(deduped))

    if not rows_data:
        return ""

    # Pad rows to max_cols
    for row in rows_data:
        while len(row) < max_cols:
            row.append("")

    lines = []
    # Header row
    header = rows_data[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    # Data rows
    for row in rows_data[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def parse_docx(file_path: str) -> str:
    """
    Parse a ``.docx`` file into a structured text string.

    Uses ``python-docx``.  Preserves:

    - Paragraph text with heading level prefix (e.g. ``## Section Title``)
    - Table content as markdown-style tables
    - Merges adjacent runs.  Strips track changes markup.
    - Handles Arabic/English mixed content (preserves both).

    Parameters
    ----------
    file_path
        Path to the ``.docx`` file.

    Returns
    -------
    str
        The full document as a single string.
    """
    path = Path(file_path)
    if not path.suffix.lower() == ".docx":
        raise ValueError(f"Expected .docx file, got: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Parsing DOCX: %s", file_path)
    doc = Document(str(path))

    output_parts: list[str] = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            # It's a paragraph
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, doc)
            text = para.text.strip()
            if not text:
                continue

            level = _heading_level(para)
            if level is not None:
                prefix = "#" * level
                output_parts.append(f"{prefix} {text}")
            else:
                output_parts.append(text)

        elif tag == "tbl":
            # It's a table
            table = Table(element, doc)
            md_table = _table_to_markdown(table)
            if md_table:
                output_parts.append(md_table)

    result = "\n\n".join(output_parts)
    logger.info("Parsed %d characters from %s", len(result), path.name)
    return result
