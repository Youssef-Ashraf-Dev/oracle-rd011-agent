"""
RD.011 Agent — Generate Intro node.

Generates the introduction and enterprise structure content for the
RD.011 document using the GENERATION model with optional RAG context.
"""

from __future__ import annotations

import json
import logging

from config import RAG_ENABLED, TOP_K_RETRIEVAL, TaskType
from llm.retry import call_with_retry
from models.schemas import ExtractionResult, IntroContent

logger = logging.getLogger(__name__)

INTRO_SYSTEM_PROMPT = """\
You are a senior Oracle Finance Cloud consultant writing the Introduction \
and Enterprise Structure sections of an RD.011 Future Process Model document.

Based on the extraction results, write:

1. **introduction_paragraphs** (3-6 paragraphs):
   - Paragraph 1: Purpose of this document
   - Paragraph 2: Project overview and scope
   - Paragraph 3: Oracle Cloud modules being implemented
   - Paragraph 4-6 (optional): Additional context about the implementation approach

2. **how_organized_text** (1-2 paragraphs):
   Explain how the document is structured — one chapter per module, each \
   containing process descriptions, step catalogs, flow diagrams, journal \
   entries, and key requirements.

3. **enterprise_context_paragraphs** (2-4 paragraphs):
   Describe the client's organisational structure, legal entities, business \
   units, and how they map to Oracle Cloud.

4. **ledger_facts** (bullet points):
   Key facts about the ledger configuration: primary ledger, accounting \
   calendar, currency, accounting method, chart of accounts.

5. **coa_description** (1-2 paragraphs):
   Describe the Chart of Accounts structure — segments, segment purposes, \
   and how they support the client's reporting needs.

Write in professional consultant style. Reference the client by name.
Return ONLY valid JSON matching the IntroContent schema.
JSON MUST be strict syntax: double quotes, no trailing commas, no comments.
Do not include markdown, code fences, or extra top-level keys.

OUTPUT SCHEMA:
{
  "introduction_paragraphs": ["paragraph 1", "paragraph 2", ...],
  "how_organized_text": "string",
  "enterprise_context_paragraphs": ["paragraph 1", ...],
  "ledger_facts": ["fact 1", "fact 2", ...],
  "coa_description": "string"
}
"""


def generate_intro_node(state: dict) -> dict:
    """
    Generate introduction and enterprise structure content.

    Attempts RAG retrieval for quality benchmarks; gracefully degrades
    if RAG is not yet available.
    """
    extraction_data = state.get("extraction_result", {})
    errors = list(state.get("errors", []))

    # Try RAG (optional): intro + enterprise structure exemplars (style only).
    rag_style_context = ""
    if RAG_ENABLED:
        try:
            from rag.retriever import build_exemplar_blocks

            rag_style_context, _ = build_exemplar_blocks(
                query="introduction enterprise structure ledger chart of accounts oracle finance cloud",
                needed_types=["intro", "enterprise_structure", "ledger", "coa"],
                module=None,
            )
            if rag_style_context:
                logger.info("RAG intro exemplars selected: style_chars=%d", len(rag_style_context))
        except Exception as exc:
            logger.warning("RAG query failed: %s (continuing without RAG)", exc)
    else:
        logger.info("RAG disabled - generating intro without reference examples")

    extraction_json = json.dumps(extraction_data, indent=2, ensure_ascii=False)

    rag_section = ""
    if rag_style_context:
        rag_section = f"\n## Reference Examples (patterns only - do not copy verbatim)\n{rag_style_context}\n"

    prompt = f"""{INTRO_SYSTEM_PROMPT}
{rag_section}
## Extraction Results

{extraction_json}

Write the introduction and enterprise structure content now.
Return ONLY the JSON object."""

    try:
        intro = call_with_retry(TaskType.GENERATION, prompt, IntroContent)
        logger.info("Intro generation complete: %d paragraphs", len(intro.introduction_paragraphs))
        return {
            "intro_content": intro.model_dump(),
            "last_completed_node": "generate_intro",
        }
    except RuntimeError as exc:
        error_msg = f"Intro generation failed: {exc}"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "intro_content": None,
            "errors": errors,
            "last_completed_node": "generate_intro",
        }
