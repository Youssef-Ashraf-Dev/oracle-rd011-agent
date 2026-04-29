"""
RD.011 Agent — Pydantic v2 schemas for all structured data.

Every LLM call in the pipeline returns JSON validated against one of these models.
Deterministic Python validators (regex, length, type) replace any need for
a second LLM validation pass.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from config import CANONICAL_BUSINESS_ACTORS, normalize_business_actor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_process_id(value: str) -> str:
    """
    Normalize process_id values to avoid avoidable validation failures.

    Fixes:
    - Double dots: "Client..AP.08" -> "Client.AP.08"
    - Dot spacing: "Client . AP . 08" -> "Client.AP.08"
    - Lowercase module: "ap.08" -> "AP.08"
    """
    if not value:
        return value

    v = str(value).strip()
    if not v:
        return v

    # Normalize dot spacing and collapse double dots
    v = re.sub(r"\s*\.\s*", ".", v)
    v = re.sub(r"\.{2,}", ".", v)

    # Upper-case the last module code if present
    matches = list(re.finditer(r"([A-Za-z]{2,3})[.\- ](\d{2})", v))
    if matches:
        last = matches[-1]
        mod = last.group(1).upper()
        num = last.group(2)
        v = v[: last.start()] + f"{mod}.{num}" + v[last.end() :]

    return v

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class DocumentConflict(BaseModel):
    """A factual contradiction detected between two dated source documents."""

    field: str = Field(
        ...,
        description="Name of the conflicting data point (for example, 'approval matrix version')",
    )
    older_value: str = Field(
        ...,
        description="Value from the older document, including source filename",
        min_length=1,
    )
    newer_value: str = Field(
        ...,
        description="Value from the newer document, including source filename",
        min_length=1,
    )
    module: str = Field(
        ...,
        description="Module affected — e.g. 'AP', 'GL', or 'ALL'",
    )
    recommended_resolution: str = Field(
        ...,
        description=(
            "Which value to apply and why, based on explicit source evidence "
            "and document recency"
        ),
        min_length=1,
    )


class ExtractionResult(BaseModel):
    """Output of the EXTRACT node — structured facts from all input files."""

    client_name: str = Field(..., min_length=1)
    project_name: str = Field(..., min_length=1)
    modules_in_scope: List[str] = Field(
        ..., description='e.g. ["AP", "FA", "AR", "GL", "CM"]'
    )
    business_actors: Dict[str, List[str]] = Field(
        ..., description="module code → list of actor names"
    )
    org_roles: dict[str, list[str]] = Field(
    default_factory=dict,
    description="Actual org roles per module confirmed from MoM documents"
    )
    requirements_per_module: Dict[str, List[str]] = Field(
        ..., description="module code → list of requirement strings"
    )
    candidate_processes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "module code → candidate processes captured from MoM Key Points Discussed bullets"
        ),
    )
    constraints: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    conflicts_between_documents: List[DocumentConflict] = Field(
        default_factory=list,
        description=(
            "Explicitly flagged contradictions between dated source documents. "
            "Each entry records both values, their sources, and your recommended resolution."
        ),
    )
    enterprise_context: str = Field(
        ...,
        description="Free text: org structure, ledger, COA facts",
    )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class ProcessEntry(BaseModel):
    """A single Oracle Finance Cloud process within a module."""

    process_id: str = Field(
        ..., description="Must match pattern XX.NN  e.g. AP.01"
    )
    process_name: str = Field(..., min_length=1)
    process_description: str = Field(..., min_length=1)
    output: str = Field(
        ..., description='e.g. "Validated PO Invoices"'
    )
    confidence: Literal["high", "medium", "low"] = "medium"
    missing_info: List[str] = Field(default_factory=list)

    @field_validator("process_id")
    @classmethod
    def validate_process_id(cls, v: str) -> str:
        v = _normalize_process_id(v)
        # Accept both bare (AP.01) and client-prefixed (Client Name.AP.01) formats.
        # Pattern: optional [prefix.] followed by 2-3 uppercase letters and 2 digits.
        if not re.match(r"^([A-Za-z0-9][A-Za-z0-9 &,.'()-]*\.)?[A-Z]{2,3}\.\d{2}$", v):
            # Fallback: extract the last Module.NN pattern if present
            m = re.findall(r"([A-Z]{2,3})\.(\d{2})", v.upper())
            if m:
                mod, num = m[-1]
                return f"{mod}.{num}"
            raise ValueError(
                f"process_id '{v}' must be Module.NN (e.g. AP.01) or "
                f"ClientName.Module.NN (e.g. Contoso.AP.01)"
            )
        return v


class SectionPlan(BaseModel):
    """Plan for one module chapter in the RD.011 document."""

    section_id: str = Field(..., description='e.g. "AP"')
    module_name: str = Field(..., description='e.g. "Accounts Payable"')
    module_intro: str = Field(
        ..., description="Key highlights paragraph for this module"
    )
    business_actors: List[str] = Field(
        default_factory=list,
        description="Canonical actor labels suggested for this module section",
    )
    org_roles: List[str] = Field(
        default_factory=list,
        description="Client-specific organizational roles confirmed from MoM",
    )
    processes: List[ProcessEntry]
    ambiguities: List[str] = Field(default_factory=list)


class DocumentPlan(BaseModel):
    """Complete plan for the RD.011 document — output of the PLAN node."""

    client_name: str
    project_name: str
    document_ref: str = Field(default="RD.011")
    author: str = Field(default="RD.011 Agent")
    version: str = Field(default="DRAFT 1A")
    sections: List[SectionPlan]


# ---------------------------------------------------------------------------
# Generation — per-process output
# ---------------------------------------------------------------------------

class ProcessStep(BaseModel):
    """One step in a process step catalog."""

    step_id: str = Field(
        ..., description="Must match pattern XX-NN-NNN  e.g. AP-02-01"
    )
    action: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    business_actor: str = Field(..., min_length=1)

    @field_validator("step_id")
    @classmethod
    def validate_step_id(cls, v: str) -> str:
        # Strict format: MODULE-PROCESS-STEP e.g. AP-02-01, GL-01-001
        if re.match(r"^[A-Z]{2,3}-\d{2}-\d{2,3}$", v):
            return v

        # Auto-correct common LLM hallucinations:
        # "CM-CE-01" → extract module "CM", discard fake sub-code "CE", keep step
        # "AP.04-01-01" → strip dots, normalize
        # "AP03-01" → insert missing dash
        cleaned = v.replace(".", "-").strip()

        # Try to extract a valid MODULE-NN-NN(N) from anywhere in the string
        m = re.search(r"([A-Z]{2,3})-(\d{2})-(\d{2,3})", cleaned)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        # Handle "MODULE-XX-NN" where XX is a hallucinated alpha sub-code
        # e.g. CM-CE-01 → CM-01-01 (treat as process 01, step 01)
        m = re.match(r"^([A-Z]{2,3})-[A-Z]+-(\d{2,3})$", cleaned)
        if m:
            return f"{m.group(1)}-01-{m.group(2)}"

        # Handle "MODULENN-NN" missing first dash e.g. AP03-01 → AP-03-01
        m = re.match(r"^([A-Z]{2,3})(\d{2})-(\d{2,3})$", cleaned)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        raise ValueError(
            f"step_id '{v}' does not match expected format "
            f"(e.g. AP-02-01, GL-01-001)"
        )

    @field_validator("business_actor")
    @classmethod
    def validate_business_actor(cls, v: str) -> str:
        v = normalize_business_actor(v)
        # Allow non-canonical actors to pass through (for newly observed roles).
        return v


class JournalEntry(BaseModel):
    """One accounting journal entry line."""

    debit_account: str = Field(default="", description="Account code or empty string")
    credit_account: str = Field(default="", description="Account code or empty string")
    amount_label: str = Field(
        default="", description='e.g. "Invoice Amount", "Tax Amount"'
    )
    label: str = Field(default="", description="Account name or description")

    @field_validator("debit_account", "credit_account", "amount_label", "label", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, v):
        """LLMs occasionally return null instead of an empty string."""
        return v if v is not None else ""


class SectionContent(BaseModel):
    """Generated content for a single process — output of GENERATE_SECTION node."""

    process_id: str = Field(...)
    process_name: str = Field(..., min_length=1)
    narrative: str = Field(
        ..., description="2-4 paragraphs, must be >= 50 chars"
    )
    process_steps: List[ProcessStep] = Field(
        ..., min_length=1, description="5-15 process steps"
    )
    journal_entries: List[JournalEntry] = Field(
        default_factory=list,
        description="Real Oracle accounting entries; empty list = N/A",
    )
    key_requirements: List[str] = Field(
        default_factory=list, description="0-8 bullet points. Leave empty if none are specified."
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="Process-level unresolved facts not confirmed in the MoM",
    )
    diagram_code: str = Field(
        ...,
        description='Must be a valid Graphviz DOT string starting with "digraph"',
    )

    @field_validator("process_id")
    @classmethod
    def validate_process_id(cls, v: str) -> str:
        v = _normalize_process_id(v)
        # Accept both bare (AP.01) and client-prefixed (Client Name.AP.01) formats.
        if not re.match(r"^([A-Za-z0-9][A-Za-z0-9 &,.'()-]*\.)?[A-Z]{2,3}\.\d{2}$", v):
            # Fallback: extract the last Module.NN pattern if present
            m = re.findall(r"([A-Z]{2,3})\.(\d{2})", v.upper())
            if m:
                mod, num = m[-1]
                return f"{mod}.{num}"
            raise ValueError(
                f"process_id '{v}' must be Module.NN (e.g. AP.01) or "
                f"ClientName.Module.NN (e.g. Contoso.AP.01)"
            )
        return v

    @field_validator("narrative")
    @classmethod
    def validate_narrative_length(cls, v: str) -> str:
        if len(v) < 50:
            raise ValueError(
                f"narrative must be >= 50 characters (got {len(v)}). "
                "Write 2-4 substantive paragraphs."
            )
        return v

    @field_validator("diagram_code")
    @classmethod
    def validate_diagram_code(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped.startswith("digraph"):
            raise ValueError(
                "diagram_code must be a Graphviz DOT string starting with 'digraph'. "
                f"Got: '{stripped[:40]}...'"
            )
        return v

    @model_validator(mode="after")
    def validate_accounting_logic(self) -> "SectionContent":
        """Ensure that journal entries, if present, have both Dr and Cr sides."""
        for entry in self.journal_entries:
            if not entry.debit_account or not entry.credit_account:
                raise ValueError(
                    f"Journal entry '{entry.label}' must have both "
                    "debit_account (Dr) and credit_account (Cr) sides."
                )
        return self


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------

class IssueReport(BaseModel):
    """Output of DETECT_ISSUES node — cross-file audit results."""

    contradictions: List[str] = Field(
        default_factory=list,
        description="Specifically for cross-file discrepancies",
    )
    ambiguities_by_section: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="section_id → list of ambiguity strings",
    )
    missing_required_fields: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------------

class IntroContent(BaseModel):
    """Generated introduction and enterprise structure — output of GENERATE_INTRO node."""

    introduction_paragraphs: List[str] = Field(
        ..., min_length=3, max_length=6, description="3-6 paragraphs"
    )
    how_organized_text: str = Field(..., min_length=1)
    enterprise_context_paragraphs: List[str] = Field(...)
    ledger_facts: List[str] = Field(
        ..., description="Bullet points about ledger configuration"
    )
    coa_description: str = Field(
        ..., description="Chart of Accounts structure description"
    )
