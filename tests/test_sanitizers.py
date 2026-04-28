import pytest

from llm.retry import _sanitize_process_id, _sanitize_step_ids
from models.schemas import ProcessEntry, SectionContent


def test_sanitize_process_id_extracts_module_nn() -> None:
    assert _sanitize_process_id("AP.01") == "AP.01"
    assert _sanitize_process_id("Contoso.AP.01") == "AP.01"
    assert _sanitize_process_id("Client Name.AP.01") == "AP.01"
    assert _sanitize_process_id("Client Name..AP.03") == "AP.03"
    assert _sanitize_process_id("GL.03") == "GL.03"
    assert _sanitize_process_id("AP-05-01") == "AP.05"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("AP09-01", "AP-09-01"),
        ("AP-04-01-01", "AP-04-01"),
        ("ap-04-01", "AP-04-01"),
        ("AP-4-1", "AP-04-01"),
        ("AP.04.01", "AP-04-01"),
        ("AP.04-01-01", "AP-04-01"),
    ],
)
def test_sanitize_step_ids_common_patterns(raw: str, expected: str) -> None:
    data = {"process_steps": [{"step_id": raw}]}
    out = _sanitize_step_ids(data)
    assert out["process_steps"][0]["step_id"] == expected


@pytest.mark.parametrize(
    "pid, expected",
    [
        ("AP.01", "AP.01"),
        ("Client Name.AP.01", "Client Name.AP.01"),
        ("Client Name..AP.01", "Client Name.AP.01"),
        ("contoso.ap.01", "contoso.AP.01"),
    ],
)
def test_process_entry_accepts_prefixed_process_id(pid: str, expected: str) -> None:
    pe = ProcessEntry(
        process_id=pid,
        process_name="T",
        process_description="D",
        output="O",
        confidence="high",
    )
    assert pe.process_id == expected


def test_section_content_accepts_prefixed_process_id() -> None:
    content = SectionContent(
        process_id="Client Name.AP.01",
        process_name="Create Invoice",
        narrative=(
            "This process validates supplier invoices, applies approvals, and posts accounting "
            "entries in Oracle Fusion to ensure accurate AP reporting and period close readiness."
        ),
        process_steps=[
            {
                "step_id": "AP-01-01",
                "action": "Create invoice",
                "description": "The AP Accountant records an invoice with supplier, amount, and tax details.",
                "business_actor": "AP Accountant",
            }
        ],
        key_requirements=[
            "Invoice must reference a valid supplier and business unit.",
            "Tax must be calculated or provided per policy.",
            "Approval rules must be satisfied before posting.",
        ],
        diagram_code="digraph G { A -> B; }",
    )
    assert content.process_id == "Client Name.AP.01"
