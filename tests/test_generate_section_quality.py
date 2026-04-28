import re

from models.schemas import ProcessEntry, SectionContent, SectionPlan
from nodes.generate_section_node import _collect_quality_violations, _sanitize_content


def _build_section_plan() -> SectionPlan:
    process = ProcessEntry(
        process_id="AP.04",
        process_name="Payment Processing - UAT-AP04-MARKER",
        process_description="Process supplier payments using approved invoices.",
        output="Processed supplier payments",
    )
    return SectionPlan(
        section_id="AP",
        module_name="Accounts Payable",
        module_intro="AP module introduction",
        business_actors=["AP Accountant"],
        org_roles=["AP Accountant"],
        processes=[process],
    )


def _build_content(narrative: str, key_requirements: list[str]) -> SectionContent:
    return SectionContent(
        process_id="AP.04",
        process_name="Payment Processing - UAT-AP04-MARKER",
        narrative=narrative,
        process_steps=[
            {
                "step_id": "AP-04-01",
                "action": "Prepare payment batch",
                "description": "System prepares approved invoices for payment.",
                "business_actor": "AP Accountant",
            }
        ],
        journal_entries=[],
        key_requirements=key_requirements,
        missing_info=[],
        diagram_code="digraph G { A -> B; }",
    )


def test_identifier_marker_numbers_are_not_flagged_as_invented() -> None:
    section_plan = _build_section_plan()
    process = section_plan.processes[0]
    content = _build_content(
        narrative=(
            "Payment Processing - UAT-AP04-MARKER ensures approved supplier invoices "
            "are scheduled and released through the AP payment workflow."
        ),
        key_requirements=[
            "Keep UAT-AP04-MARKER traceability for review notes."
        ],
    )

    violations = _collect_quality_violations(
        content=content,
        section_plan=section_plan,
        process=process,
        requirements_text="No specific requirements captured.",
    )

    assert not any(v.startswith("Potential invented numeric values:") for v in violations)


def test_real_numeric_values_are_not_flagged_without_requirements_grounding() -> None:
    section_plan = _build_section_plan()
    process = section_plan.processes[0]
    content = _build_content(
        narrative=(
            "Payment Processing - UAT-AP04-MARKER applies a service fee of 13 per "
            "payment run when urgent processing is requested by operations."
        ),
        key_requirements=[
            "Urgent payments can be prioritized based on business need."
        ],
    )

    violations = _collect_quality_violations(
        content=content,
        section_plan=section_plan,
        process=process,
        requirements_text="No specific requirements captured.",
    )

    assert not any(v == "Potential invented numeric values: 13" for v in violations)


def test_real_numeric_values_still_flagged_when_grounding_exists() -> None:
    section_plan = _build_section_plan()
    process = section_plan.processes[0]
    content = _build_content(
        narrative=(
            "Payment Processing - UAT-AP04-MARKER applies a service fee of 13 per "
            "payment run when urgent processing is requested by operations."
        ),
        key_requirements=[
            "Urgent payment requests are handled per approved policy."
        ],
    )

    violations = _collect_quality_violations(
        content=content,
        section_plan=section_plan,
        process=process,
        requirements_text="Service fee of 7 may apply for urgent approved cases.",
    )

    assert any(v == "Potential invented numeric values: 13" for v in violations)


def test_leading_article_in_role_phrase_does_not_trigger_out_of_scope_warning() -> None:
    section_plan = _build_section_plan()
    process = section_plan.processes[0]
    content = _build_content(
        narrative=(
            "The AP Accountant prepares and approves the payment package before "
            "submission to Oracle Payables for execution."
        ),
        key_requirements=[
            "AP Accountant validates payment batch completeness before release."
        ],
    )

    violations = _collect_quality_violations(
        content=content,
        section_plan=section_plan,
        process=process,
        requirements_text="No specific requirements captured.",
    )

    assert not any(v.startswith("Out-of-scope role mentions") for v in violations)


def test_sanitize_content_removes_ungrounded_numeric_values() -> None:
    section_plan = _build_section_plan()
    process = section_plan.processes[0]
    content = _build_content(
        narrative=(
            "Payment Processing - UAT-AP04-MARKER applies a service fee of 13 per "
            "payment run when urgent processing is requested by operations."
        ),
        key_requirements=[
            "Urgent processing fee of 13 is charged for each run."
        ],
    )

    requirements_text = "Service fee of 7 may apply for urgent approved cases."
    sanitized = _sanitize_content(
        content=content,
        section_plan=section_plan,
        requirements_text=requirements_text,
    )

    violations = _collect_quality_violations(
        content=sanitized,
        section_plan=section_plan,
        process=process,
        requirements_text=requirements_text,
    )

    assert not any(v.startswith("Potential invented numeric values:") for v in violations)
    assert not re.search(r"(?<![A-Za-z0-9])13(?![A-Za-z0-9])", sanitized.narrative)
    assert any(
        msg.startswith("Removed ungrounded numeric values:")
        for msg in sanitized.missing_info
    )


def test_sanitize_content_rewrites_ungrounded_threshold_language() -> None:
    section_plan = _build_section_plan()
    process = section_plan.processes[0]
    content = _build_content(
        narrative=(
            "Payment Processing - UAT-AP04-MARKER routes approvals based on threshold "
            "rules configured by finance."
        ),
        key_requirements=[
            "Threshold validation is required before release."
        ],
    )

    requirements_text = "No specific requirements captured."
    sanitized = _sanitize_content(
        content=content,
        section_plan=section_plan,
        requirements_text=requirements_text,
    )

    violations = _collect_quality_violations(
        content=sanitized,
        section_plan=section_plan,
        process=process,
        requirements_text=requirements_text,
    )

    assert not any(v == "Threshold mentioned but not present in MoM requirements" for v in violations)
    assert "threshold" not in sanitized.narrative.lower()
    assert any(
        msg == "Approval threshold amount not confirmed in MoM"
        for msg in sanitized.missing_info
    )
