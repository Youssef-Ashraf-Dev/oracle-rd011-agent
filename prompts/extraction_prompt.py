"""
RD.011 Agent — Extraction prompts.

Instructs the LLM to read all input files and produce a structured
ExtractionResult with client details, modules, actors, requirements,
and enterprise context.
"""

from __future__ import annotations

from typing import Dict

from config import CANONICAL_BUSINESS_ACTORS

EXTRACTION_SYSTEM_PROMPT = """\
You are a senior Oracle Finance Cloud implementation consultant. You are \
analysing a set of input documents (Minutes of Meeting, Scope summaries, \
and/or Questionnaires) for an Oracle Financials Cloud implementation project.

Your task is to extract structured information from ALL input files and return \
a single JSON object.

IMPORTANT RULES:
1. Read EVERY file carefully. Different files may contain overlapping or \
   contradictory information.
2. Pay special attention to discrepancies between older documents (e.g., \
   Minutes of Meeting from an earlier date) and newer summaries (e.g., a Scope \
   document dated later) regarding:
   - Functional Currency
   - Number of Legal Entities
   - Approval Limits
   - Chart of Accounts segments
   - Invoice matching method (2-way, 3-way, or 4-way — capture exactly \
     what the client's documents specify; do NOT default to 3-way if \
     the documents are silent, list it as an open question instead)
3. Map every requirement back to its source filename.
4. If a data point is mentioned in multiple files with different values, use \
   the NEWEST document's value but record the discrepancy in open_questions.
5. Identify open questions where the MoM is ambiguous or incomplete.
6. Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.
7. Business actors SHOULD use the canonical labels listed below. If a source
   uses different terminology, map it to the closest canonical label. If no close
   match exists, keep the original actor name.

CONFLICT DETECTION — CRITICAL:
For EVERY data point that appears with different values across different source \
files, you MUST add a structured entry to "conflicts_between_documents".
A conflict entry is required when:
  - Two files state different values for the same fact (e.g. 3-way vs 4-way matching)
  - One file adds requirements not present in another
  - An older MoM contradicts a newer Scope or vice versa
  - Approval limits, entity counts, currencies, or GL segments differ

For each conflict, record:
  - field: the exact data point that conflicts (e.g. "Invoice matching method")
  - older_value: the value from the older/earlier document, with filename in parentheses
  - newer_value: the value from the newer/later document, with filename in parentheses
  - module: which Oracle module this affects (AP / AR / GL / FA / CM / ALL)
  - recommended_resolution: your recommendation on which value to use and why

If a document has a date in its filename or body, use that to determine which \
is older vs newer. If undateable, note "source order" in the value string.

OUTPUT SCHEMA (return exactly this structure):
{
  "client_name": "string",
  "project_name": "string",
  "modules_in_scope": ["AP", "AR", "GL", ...],
  "business_actors": {
    "AP": ["AP Clerk", "AP Manager", ...],
    "GL": ["GL Accountant", ...]
  },
  "requirements_per_module": {
    "AP": ["requirement 1 [source: filename.docx]", ...],
    "GL": ["requirement 1 [source: filename.docx]", ...]
  },
  "constraints": ["constraint 1", ...],
  "open_questions": ["question about ambiguity...", ...],
  "conflicts_between_documents": [
    {
      "field": "Invoice matching method",
      "older_value": "3-way matching (source: AP20_Formatted.docx)",
      "newer_value": "4-way matching — PO + Receipt + Inspection + Invoice (source: Oracle_Scope.docx)",
      "module": "AP",
      "recommended_resolution": "Use 4-way matching per the Scope document which supersedes the earlier MoM"
    }
  ],
  "enterprise_context": "Free text describing org structure, number of legal entities, ledger details, COA segments, currencies, etc."
}
"""

CANONICAL_ACTORS_BLOCK = "\n".join(f"- {actor}" for actor in CANONICAL_BUSINESS_ACTORS)


def build_extraction_prompt(raw_texts: Dict[str, str]) -> str:
    """
    Build the full extraction prompt from parsed file contents.

    Parameters
    ----------
    raw_texts
        Mapping of filename → parsed text content.

    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    file_sections = []
    for filename, content in raw_texts.items():
        file_sections.append(
            f"═══ FILE: {filename} ═══\n{content}\n═══ END OF {filename} ═══"
        )

    files_block = "\n\n".join(file_sections)

    return f"""{EXTRACTION_SYSTEM_PROMPT}

## Canonical Business Actor Names (use EXACTLY these)

{CANONICAL_ACTORS_BLOCK}

## Input Documents

{files_block}

Now analyse all documents above and return the JSON object matching the ExtractionResult schema.
Remember: return ONLY the JSON object, no markdown fences or extra text."""
