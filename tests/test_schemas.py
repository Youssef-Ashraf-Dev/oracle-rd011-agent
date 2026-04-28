"""
RD.011 Agent — Pydantic schema unit tests.

Tests all field validators, model validators, and schema constraints
without making any LLM calls.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.schemas import (
    DocumentPlan,
    ExtractionResult,
    IntroContent,
    IssueReport,
    JournalEntry,
    ProcessEntry,
    ProcessStep,
    SectionContent,
    SectionPlan,
)


# ── ProcessEntry ──────────────────────────────────────────────────────────

class TestProcessEntry:
    def test_valid_process_id_two_letter(self):
        entry = ProcessEntry(
            process_id="AP.01",
            process_name="Invoice Processing",
            process_description="Processes supplier invoices in Oracle Cloud.",
            output="Posted Invoice",
        )
        assert entry.process_id == "AP.01"

    def test_valid_process_id_three_letter(self):
        entry = ProcessEntry(
            process_id="OTH.12",
            process_name="Other Process",
            process_description="Some other process.",
            output="Output",
        )
        assert entry.process_id == "OTH.12"

    def test_process_id_normalizes_double_dot(self):
        entry = ProcessEntry(
            process_id="Client..AP.08",
            process_name="Invoice Processing",
            process_description="Processes supplier invoices.",
            output="Posted Invoice",
        )
        assert entry.process_id == "Client.AP.08"

    def test_invalid_process_id_lowercase(self):
        with pytest.raises(ValidationError) as exc_info:
            ProcessEntry(
                process_id="ap.1",
                process_name="Invoice Processing",
                process_description="Processes invoices.",
                output="Invoice",
            )
        assert "process_id" in str(exc_info.value).lower() or "must match" in str(exc_info.value)

    def test_invalid_process_id_no_dot(self):
        with pytest.raises(ValidationError):
            ProcessEntry(
                process_id="AP01",
                process_name="Invoice Processing",
                process_description="Processes invoices.",
                output="Invoice",
            )

    def test_invalid_process_id_single_digit(self):
        with pytest.raises(ValidationError):
            ProcessEntry(
                process_id="AP.1",
                process_name="Invoice Processing",
                process_description="Processes invoices.",
                output="Invoice",
            )

    def test_confidence_default(self):
        entry = ProcessEntry(
            process_id="GL.03",
            process_name="Journal Entry",
            process_description="Posts manual journal entries.",
            output="Posted Journal",
        )
        assert entry.confidence == "medium"

    def test_missing_info_default_empty(self):
        entry = ProcessEntry(
            process_id="FA.05",
            process_name="Asset Depreciation",
            process_description="Runs depreciation for fixed assets.",
            output="Depreciation Schedule",
        )
        assert entry.missing_info == []


# ── ProcessStep ───────────────────────────────────────────────────────────

class TestProcessStep:
    def test_valid_step_id(self):
        step = ProcessStep(
            step_id="AP-02-01",
            action="Create Invoice",
            description="Navigate to Payables > Invoices > Create",
            business_actor="AP Specialist",
        )
        assert step.step_id == "AP-02-01"

    def test_valid_step_id_three_digits(self):
        step = ProcessStep(
            step_id="GL-01-001",
            action="Open Journal",
            description="Create a manual journal entry.",
            business_actor="GL Accountant",
        )
        assert step.step_id == "GL-01-001"

    def test_invalid_step_id_lowercase(self):
        with pytest.raises(ValidationError):
            ProcessStep(
                step_id="ap-01-01",
                action="Create Invoice",
                description="Navigate to create invoice screen.",
                business_actor="AP Specialist",
            )

    def test_invalid_step_id_missing_second_segment(self):
        with pytest.raises(ValidationError):
            ProcessStep(
                step_id="AP-01",
                action="Create Invoice",
                description="Navigate to create invoice screen.",
                business_actor="AP Specialist",
            )


# ── SectionContent ────────────────────────────────────────────────────────

class TestSectionContent:
    def _minimal_valid_step(self):
        return ProcessStep(
            step_id="AP-01-01",
            action="Navigate to Payables",
            description="Open Oracle Cloud Payables module.",
            business_actor="AP Specialist",
        )

    def _valid_section(self, **overrides):
        defaults = dict(
            process_id="AP.01",
            process_name="Standard Invoice Processing",
            narrative=(
                "This process covers the end-to-end flow of processing standard "
                "supplier invoices in Oracle Cloud Payables. The AP Specialist "
                "receives invoices and enters them into the system for approval "
                "and payment processing according to client-defined approval rules."
            ),
            process_steps=[self._minimal_valid_step()],
            journal_entries=[],
            key_requirements=["Invoices require 3-way match for PO invoices"],
            diagram_code=(
                'digraph G {\n'
                '  rankdir=LR;\n'
                '  subgraph cluster_AP { label="AP Specialist"; A [label="Enter Invoice"]; }\n'
                '  A -> B;\n'
                '}'
            ),
        )
        defaults.update(overrides)
        return defaults

    def test_valid_section_content(self):
        data = self._valid_section()
        section = SectionContent(**data)
        assert section.process_id == "AP.01"
        assert section.missing_info == []

    def test_section_process_id_normalizes_double_dot(self):
        section = SectionContent(
            **self._valid_section(process_id="Client Name..AP.08")
        )
        assert section.process_id == "Client Name.AP.08"

    def test_narrative_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            SectionContent(**self._valid_section(narrative="Too short"))
        assert "50" in str(exc_info.value) or "narrative" in str(exc_info.value).lower()

    def test_narrative_exact_minimum(self):
        # exactly 50 chars — should pass
        section = SectionContent(
            **self._valid_section(narrative="A" * 50)
        )
        assert len(section.narrative) == 50

    def test_narrative_below_minimum(self):
        with pytest.raises(ValidationError):
            SectionContent(**self._valid_section(narrative="A" * 49))

    def test_diagram_code_must_start_with_digraph(self):
        section = SectionContent(
            **self._valid_section(
                diagram_code='digraph G { rankdir=LR; A -> B; }'
            )
        )
        assert section.diagram_code.startswith("digraph")

    def test_diagram_code_accepts_digraph_with_name(self):
        section = SectionContent(
            **self._valid_section(
                diagram_code='digraph InvoiceFlow { A -> B -> C; }'
            )
        )
        assert section.diagram_code.startswith("digraph")

    def test_diagram_code_invalid_mermaid_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            SectionContent(
                **self._valid_section(diagram_code="flowchart TD\n  A --> B")
            )
        assert "digraph" in str(exc_info.value)

    def test_diagram_code_invalid_prefix_raises(self):
        with pytest.raises(ValidationError):
            SectionContent(
                **self._valid_section(diagram_code="graph TD\n  A --> B")
            )

    def test_empty_process_steps_raises(self):
        with pytest.raises(ValidationError):
            SectionContent(**self._valid_section(process_steps=[]))

    def test_empty_key_requirements_raises(self):
        with pytest.raises(ValidationError):
            SectionContent(**self._valid_section(key_requirements=[]))

    def test_journal_entry_missing_debit_raises(self):
        """Journal entry with empty debit_account should fail model_validator."""
        bad_je = JournalEntry(
            debit_account="",
            credit_account="21010",
            amount_label="Invoice Amount",
            label="Accounts Payable",
        )
        with pytest.raises(ValidationError) as exc_info:
            SectionContent(**self._valid_section(journal_entries=[bad_je]))
        assert "Dr" in str(exc_info.value) or "debit" in str(exc_info.value).lower()

    def test_journal_entry_missing_credit_raises(self):
        """Journal entry with empty credit_account should fail model_validator."""
        bad_je = JournalEntry(
            debit_account="60010",
            credit_account="",
            amount_label="Invoice Amount",
            label="Expense Account",
        )
        with pytest.raises(ValidationError) as exc_info:
            SectionContent(**self._valid_section(journal_entries=[bad_je]))
        assert "Cr" in str(exc_info.value) or "credit" in str(exc_info.value).lower()

    def test_valid_journal_entries_pass(self):
        je = JournalEntry(
            debit_account="60010",
            credit_account="21010",
            amount_label="Invoice Amount",
            label="Expense / Accounts Payable",
        )
        section = SectionContent(**self._valid_section(journal_entries=[je]))
        assert len(section.journal_entries) == 1


# ── IntroContent ──────────────────────────────────────────────────────────

class TestIntroContent:
    def test_valid_intro(self):
        intro = IntroContent(
            introduction_paragraphs=["Para 1", "Para 2", "Para 3"],
            how_organized_text="This document is organized by module.",
            enterprise_context_paragraphs=["The client operates in MENA."],
            ledger_facts=["Primary Ledger: Main Ledger"],
            coa_description="The COA has 6 segments.",
        )
        assert len(intro.introduction_paragraphs) == 3

    def test_too_few_intro_paragraphs(self):
        with pytest.raises(ValidationError):
            IntroContent(
                introduction_paragraphs=["Only one para"],
                how_organized_text="Organized.",
                enterprise_context_paragraphs=["Context."],
                ledger_facts=["Ledger fact."],
                coa_description="COA.",
            )

    def test_too_many_intro_paragraphs(self):
        with pytest.raises(ValidationError):
            IntroContent(
                introduction_paragraphs=[f"Para {i}" for i in range(7)],
                how_organized_text="Organized.",
                enterprise_context_paragraphs=["Context."],
                ledger_facts=["Ledger fact."],
                coa_description="COA.",
            )


# ── DocumentPlan ──────────────────────────────────────────────────────────

class TestDocumentPlan:
    def test_valid_plan(self):
        section = SectionPlan(
            section_id="AP",
            module_name="Accounts Payable",
            module_intro="AP module handles all supplier payments.",
            processes=[
                ProcessEntry(
                    process_id="AP.01",
                    process_name="Invoice Processing",
                    process_description="Standard invoice workflow.",
                    output="Posted Invoice",
                )
            ],
        )
        plan = DocumentPlan(
            client_name="Acme Corp",
            project_name="Oracle Cloud Implementation",
            creation_date="2026-03-10",
            author="Test Author",
            sections=[section],
        )
        assert plan.client_name == "Acme Corp"
        assert plan.version == "DRAFT 1A"

    def test_default_document_ref(self):
        plan = DocumentPlan(
            client_name="Test Client",
            project_name="Test Project",
            creation_date="2026-01-01",
            sections=[],
        )
        assert plan.document_ref == "RD.011"
