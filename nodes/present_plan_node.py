"""
RD.011 Agent — Present Plan node.

Formats and prints the document plan to stdout for consultant review.
No LLM call — purely deterministic formatting.
"""

from __future__ import annotations

import logging

from models.schemas import DocumentConflict, DocumentPlan, ExtractionResult, IssueReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# False-conflict filter helpers
# ---------------------------------------------------------------------------

_NOT_SPECIFIED = {
    "not specified",
    "not mentioned",
    "not stated",
    "not provided",
    "not defined",
    "not described",
    "not detailed",
    "not documented",
    "not found",
    "not available",
    "not included",
    "not addressed",
    "not covered",
    "n/a",
    "na",
    "tbd",
    "tbc",
    "unknown",
    "unspecified",
    "unclear",
    "no information",
    "no data",
    "no value",
    "no mention",
    "absent",
    "missing",
    "silent",
    "only mentioned in one document",
    "only in one document",
    "mentioned in only one",
    "only one document",
    "single source",
    "one document",
}


def _is_empty_value(val: str) -> bool:
    """Return True if val is essentially 'not specified' (a false conflict side)."""
    v = val.strip().lower()
    if v in _NOT_SPECIFIED:
        return True
    for phrase in _NOT_SPECIFIED:
        if phrase in v:
            return True
    return False


def _filter_false_conflicts(conflicts: list) -> list:
    """
    Remove spurious conflicts where one side is 'not mentioned' / absent.

    Works on:
    - List[DocumentConflict]  — checks older_value / newer_value attributes
    - List[str]               — checks the full string for NOT_SPECIFIED phrases
    """
    result = []
    for conflict in conflicts:
        if hasattr(conflict, "older_value"):
            # DocumentConflict object
            if _is_empty_value(conflict.older_value) or _is_empty_value(conflict.newer_value):
                logger.debug(
                    "Filtered false conflict '%s': one side is empty/not-specified",
                    conflict.field,
                )
                continue
        else:
            # Plain string from issue_report.contradictions
            text = str(conflict).lower()
            if any(phrase in text for phrase in _NOT_SPECIFIED):
                logger.debug("Filtered false contradiction string: %s", str(conflict)[:80])
                continue
        result.append(conflict)
    return result


def present_plan_node(state: dict) -> dict:
    """
    Print a formatted summary of the document plan and open items,
    then prompt the consultant to approve or provide feedback.
    """
    plan_data = state.get("document_plan")
    issue_data = state.get("issue_report")
    extraction_data = state.get("extraction_result")

    if not plan_data:
        print("\n[ERROR] No document plan available to present.\n")
        return {"last_completed_node": "present_plan"}

    plan = DocumentPlan.model_validate(plan_data)
    issue_report = (
        IssueReport.model_validate(issue_data)
        if issue_data
        else IssueReport(contradictions=[], ambiguities_by_section={}, missing_required_fields=[])
    )

    # Pull structured document conflicts from the extraction result
    doc_conflicts: list[DocumentConflict] = []
    if extraction_data:
        try:
            extraction = ExtractionResult.model_validate(extraction_data)
            doc_conflicts = _filter_false_conflicts(extraction.conflicts_between_documents)
        except Exception:
            pass

    # ── Header ────────────────────────────────────────────────────────────
    print()
    print("\u2550" * 60)
    print(f"  RD.011 Document Plan \u2014 {plan.client_name}")
    print("\u2550" * 60)
    print()

    # ── Module summary ────────────────────────────────────────────────────
    total_processes = sum(len(s.processes) for s in plan.sections)
    print(f"  Modules ({len(plan.sections)} total):")
    for section in plan.sections:
        print(
            f"    {section.section_id} \u2014 {section.module_name} "
            f"\u2014 {len(section.processes)} processes"
        )
    print(f"\n  Total processes: {total_processes}")

    # ── Document conflicts (cross-file contradictions with resolution) ────
    if doc_conflicts:
        print()
        print(f"  \u26a0  DOCUMENT CONFLICTS REQUIRING YOUR DECISION ({len(doc_conflicts)}):")
        print("  " + "\u2500" * 56)
        print("  These are factual contradictions found between your source")
        print("  documents. Provide resolution in your feedback if the")
        print("  recommended resolution below is wrong.")
        print()
        for i, conflict in enumerate(doc_conflicts, start=1):
            print(f"  [{i}] {conflict.field}  [{conflict.module}]")
            print(f"       Older : {conflict.older_value}")
            print(f"       Newer : {conflict.newer_value}")
            print(f"       \u2714 Recommended: {conflict.recommended_resolution}")
            print()

    # ── Open items (from issue detection, consolidated and deduplicated) ──
    conflicts = _filter_false_conflicts(list(issue_report.contradictions))

    seen: set[str] = set()
    gaps: list[str] = []

    def _add_gap(text: str) -> None:
        key = text.strip().lower()
        if key not in seen:
            seen.add(key)
            gaps.append(text.strip())

    for field in issue_report.missing_required_fields:
        _add_gap(field)
    for section in plan.sections:
        for item in issue_report.ambiguities_by_section.get(section.section_id, []):
            _add_gap(item)
        for item in section.ambiguities:
            _add_gap(item)

    total_open = len(conflicts) + len(gaps)
    if total_open:
        print()
        print(f"  Open items for your review ({total_open} total):")

        if conflicts:
            print()
            print("    Additional conflicts (from cross-fact audit):")
            for item in conflicts:
                print(f"      \u2022 {item}")

        if gaps:
            print()
            print("    Missing information / ambiguities:")
            for item in gaps:
                print(f"      \u2022 {item}")
    else:
        print()
        print("  No additional open items detected.")

    # ── Process breakdown ─────────────────────────────────────────────────
    print()
    print("  Process breakdown:")
    for section in plan.sections:
        print(f"\n    [{section.section_id}] {section.module_name}")
        for proc in section.processes:
            conf_marker = {"high": "\u2713", "medium": "~", "low": "?"}.get(
                proc.confidence, "?"
            )
            print(f"      {conf_marker} {proc.process_id} {proc.process_name}")
            for info in proc.missing_info:
                print(f"        \u2514\u2500 Missing: {info}")

    print()
    print("\u2500" * 60)
    if doc_conflicts:
        print("  To override a recommended resolution, include it in your feedback.")
        print("  Example: 'Conflict 1: Use 3-way matching, not 4-way.'")
        print()
    print("  Type APPROVE to proceed, or type feedback to revise the plan:")
    print("\u2500" * 60)
    print()

    return {"last_completed_node": "present_plan"}
