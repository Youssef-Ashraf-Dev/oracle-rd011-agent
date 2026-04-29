"""
RD.011 Agent — Plan update prompts.

Instructs the LLM to revise a DocumentPlan based on consultant feedback.
"""

from __future__ import annotations

import json
from typing import List

from models.schemas import DocumentConflict, DocumentPlan, ExtractionResult

UPDATE_SYSTEM_PROMPT = """\
You are a senior Oracle Finance Cloud solution architect. The consultant \
has reviewed your proposed RD.011 document plan and provided feedback.

Your task is to revise the DocumentPlan based on their feedback while \
maintaining consistency and correctness.

RULES:
1. Apply ALL feedback points from the consultant.
2. If the consultant overrides a document conflict resolution, use their \
   chosen value — not the originally recommended one.
3. If the consultant requests adding a new process or modifying an existing one, \
   refer to the provided Extraction Context to accurately fill in the process details, \
   descriptions, outputs, and responsible roles. Do not hallucinate details if they \
   are available in the Extraction Context.
4. Maintain valid process_id format: ClientName.Module.NN (e.g., Contoso.AP.01).
   Always include the client name prefix — never use bare Module.NN format.
4. Keep processes in logical order; the system will renumber IDs after your update.
5. Update module_intro paragraphs if the feedback changes the module scope.
6. Preserve all unchanged sections exactly as they are.
7. Update confidence levels and missing_info as appropriate.
8. Keep org_roles and business_actors consistent:
    - org_roles = client-confirmed roles
    - business_actors = canonicalized labels derived from org_roles
9. JSON output must be strict syntax: double quotes, no trailing commas, no comments.
10. Return the COMPLETE revised DocumentPlan — not just the changed parts.

Return ONLY valid JSON matching the DocumentPlan schema. No markdown fences.
"""


def build_update_prompt(
    current_plan: DocumentPlan,
    feedback: str,
    conflicts: List[DocumentConflict] | None = None,
    extraction: ExtractionResult | None = None,
) -> str:
    """
    Build the plan update prompt from the current plan, feedback, and
    any document conflicts identified during extraction.

    Parameters
    ----------
    current_plan
        The current DocumentPlan to be revised.
    feedback
        Free-text feedback from the consultant.
    conflicts
        Structured cross-document conflicts from the extraction result.
        Included so the LLM can apply the consultant's resolution choices.
    extraction
        The complete extraction result from the documents, containing candidate
        processes and enterprise context to ground the LLM's updates.

    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    plan_json = json.dumps(current_plan.model_dump(), indent=2, ensure_ascii=False)

    conflicts_block = ""
    if conflicts:
        conflict_lines = []
        for i, c in enumerate(conflicts, start=1):
            conflict_lines.append(
                f"[{i}] {c.field} [{c.module}]\n"
                f"     Older: {c.older_value}\n"
                f"     Newer: {c.newer_value}\n"
                f"     Default resolution: {c.recommended_resolution}"
            )
        conflicts_block = (
            "\n\n## Document Conflicts (apply consultant's overrides if specified)\n\n"
            + "\n\n".join(conflict_lines)
        )

    extraction_block = ""
    if extraction:
        # Exclude conflicts since we already formatted them
        ext_dict = extraction.model_dump(exclude={"conflicts_between_documents"})
        ext_json = json.dumps(ext_dict, indent=2, ensure_ascii=False)
        extraction_block = f"\n\n## Extraction Context\n\n{ext_json}"

    return f"""{UPDATE_SYSTEM_PROMPT}

## Current Document Plan

{plan_json}{conflicts_block}{extraction_block}

## Consultant Feedback

{feedback}

Apply the consultant's feedback to the document plan above and return \
the complete revised DocumentPlan as JSON.
Return ONLY the JSON object."""
