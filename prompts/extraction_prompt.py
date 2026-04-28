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
  - Transaction control settings and process-specific policies (capture exactly \
    what the client's documents specify; if silent, record as open question)
3. Map every requirement back to its source filename.
4. If a data point is mentioned in multiple files with different values, use \
   the NEWEST document's value but record the discrepancy in open_questions.
5. Identify open questions where the MoM is ambiguous or incomplete.
6. Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.
  JSON MUST be strict syntax: double quotes, no trailing commas, no comments.
7. business_actors must be derived from roles that are actually present in
  this client's source documents. The canonical list below is only a
  normalization vocabulary, not a source of truth. Do NOT invent actors
  from the canonical list when they are not mentioned in source files.
8. Extract the actual organizational roles and departments that exist at 
   this client from the source documents. Record them in "org_roles". 
   This list will be used to constrain business actor names in the generated 
   document — only roles that actually exist at the client should appear.
   If the MoMs mention specific department names or role titles, capture them.
   If a role from the canonical list is NOT mentioned anywhere in the source 
   documents, do NOT include it in org_roles.
9. requirements_per_module must contain ONLY current-state requirements to be \
   used for section generation. Do NOT include unresolved conflict prose, older \
   superseded values, comparison text (e.g., X vs Y), or open-question wording.
   If two values conflict, keep the selected current value in requirements and \
   record the contradiction in conflicts_between_documents.

CONFLICT DETECTION — CRITICAL:
For EVERY data point that appears with different values across different source \
files, you MUST add a structured entry to "conflicts_between_documents".
A conflict entry is required when:
  - Two files BOTH state explicit, incompatible values for the same fact
  - An older MoM and newer Scope explicitly disagree on that same fact
  - Approval limits, entity counts, currencies, control rules, or GL segments differ

A conflict entry is NOT required when:
  - One file is silent and another file provides a value
  - One file adds detail while not contradicting the other
  - The values are equivalent but phrased differently
  - The difference is only wording, not business meaning

For each conflict, record:
  - field: the exact data point that conflicts (generic business fact name)
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
    "AP": ["AP Accountant", "Finance Manager", ...],
    "GL": ["GL Accountant", ...]
  },
  "org_roles": {
    "AP": ["role 1", "role 2"],
    "AR": ["role 1", "role 2"],
    "GL": ["role 1"],
    "FA": ["role 1"],
    "CM": ["role 1", "role 2"]
  },
  "requirements_per_module": {
    "AP": ["requirement 1 [source: filename.docx]", ...],
    "GL": ["requirement 1 [source: filename.docx]", ...]
  },
  "constraints": ["constraint 1", ...],
  "open_questions": ["question about ambiguity...", ...],
  "conflicts_between_documents": [
    {
      "field": "Data point name",
      "older_value": "Older explicit value (source: older_file.ext)",
      "newer_value": "Newer explicit value (source: newer_file.ext)",
      "module": "AP|AR|GL|FA|CM|ALL",
      "recommended_resolution": "Use newer value because document date indicates supersession"
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
        f"[BEGIN FILE: {filename}]\n{content}\n[END FILE: {filename}]"
        )

    files_block = "\n\n".join(file_sections)

    return f"""{EXTRACTION_SYSTEM_PROMPT}

## Canonical Business Actor Names (normalization hints only)

Use this list only to normalize naming when a role is explicitly present in
the source documents. Do not add roles from this list unless evidence exists.

{CANONICAL_ACTORS_BLOCK}

## Input Documents

{files_block}

Now analyse all documents above and return the JSON object matching the ExtractionResult schema.
Remember: return ONLY the JSON object, no markdown fences or extra text."""
