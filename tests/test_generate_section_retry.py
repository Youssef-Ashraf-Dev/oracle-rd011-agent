import importlib

from models.schemas import DocumentPlan, ProcessEntry, SectionContent, SectionPlan
from nodes.generate_section_node import (
    _HARDENED_RETRY_MARKER,
    generate_section_node,
)

generate_section_module = importlib.import_module("nodes.generate_section_node")


def _build_plan() -> DocumentPlan:
    process = ProcessEntry(
        process_id="AP.01",
        process_name="Supplier Invoice Entry",
        process_description="Capture supplier invoices for validation and posting.",
        output="Validated supplier invoice",
    )
    section = SectionPlan(
        section_id="AP",
        module_name="Accounts Payable",
        module_intro="AP module introduction",
        business_actors=["AP Accountant"],
        org_roles=["AP Accountant"],
        processes=[process],
    )
    return DocumentPlan(
        client_name="Contoso",
        project_name="Finance Transformation",
        sections=[section],
    )


def _valid_section_content() -> SectionContent:
    return SectionContent(
        process_id="AP.01",
        process_name="Supplier Invoice Entry",
        narrative=(
            "Supplier Invoice Entry in Oracle Payables captures supplier billing "
            "records, validates mandatory fields, and routes exceptions before "
            "posting to the payable ledger. This process keeps invoice handling "
            "traceable for the AP team and reduces posting delays during period close."
        ),
        process_steps=[
            {
                "step_id": "AP-01-01",
                "action": "Enter supplier invoice",
                "description": "AP Accountant records supplier invoice data in Oracle Payables.",
                "business_actor": "AP Accountant",
            }
        ],
        journal_entries=[
            {
                "debit_account": "AP Accrual",
                "credit_account": "Liabilities",
                "amount_label": "Invoice Amount",
                "label": "Invoice Posting",
            }
        ],
        key_requirements=[
            "Supplier invoice must include mandatory header and line attributes before posting."
        ],
        missing_info=[],
        diagram_code="digraph G { AP01_start -> AP01_end; }",
    )


def _base_state() -> dict:
    return {
        "current_section_index": 0,
        "section_queue": ["AP.AP.01"],
        "document_plan": _build_plan().model_dump(),
        "extraction_result": {},
        "generated_sections": {},
        "failed_sections": [],
        "errors": [],
    }


def test_generate_section_same_node_hardened_retry_recovers(monkeypatch) -> None:
    prompts_seen: list[str] = []
    section_content = _valid_section_content()

    monkeypatch.setattr(generate_section_module, "build_generation_prompt", lambda **_: "BASE PROMPT")

    def fake_call_with_retry(task_type, prompt, schema, max_retries=3):
        prompts_seen.append(prompt)
        if _HARDENED_RETRY_MARKER in prompt:
            return section_content
        raise RuntimeError("Primary generation failed with malformed JSON")

    monkeypatch.setattr(generate_section_module, "call_with_retry", fake_call_with_retry)

    result = generate_section_node(_base_state())

    assert result["current_section_index"] == 1
    assert result["failed_sections"] == []
    assert result["errors"] == []
    assert "AP.AP.01" in result["generated_sections"]
    assert len(prompts_seen) == 2
    assert _HARDENED_RETRY_MARKER not in prompts_seen[0]
    assert _HARDENED_RETRY_MARKER in prompts_seen[1]


def test_generate_section_same_node_hardened_retry_still_fails(monkeypatch) -> None:
    prompts_seen: list[str] = []

    monkeypatch.setattr(generate_section_module, "build_generation_prompt", lambda **_: "BASE PROMPT")

    def fake_call_with_retry(task_type, prompt, schema, max_retries=3):
        prompts_seen.append(prompt)
        raise RuntimeError("Generation failed on all retries")

    monkeypatch.setattr(generate_section_module, "call_with_retry", fake_call_with_retry)

    result = generate_section_node(_base_state())

    assert result["current_section_index"] == 1
    assert result["generated_sections"] == {}
    assert result["failed_sections"] == ["AP.AP.01"]
    assert len(result["errors"]) == 1
    assert "Generation failed for AP.AP.01" in result["errors"][0]
    assert len(prompts_seen) == 2
    assert _HARDENED_RETRY_MARKER in prompts_seen[1]
