"""
RD.011 Agent — Section generation prompts.

Instructs the LLM to write consultant-quality content for a single Oracle
Finance Cloud process: narrative, process steps, journal entries, key
requirements, and a Graphviz DOT swimlane diagram.
"""

from __future__ import annotations

import re
from typing import List

from config import CANONICAL_BUSINESS_ACTORS

GENERATION_SYSTEM_PROMPT = """\
You are a senior Oracle Finance Cloud consultant writing one section of an \
RD.011 Future Process Model document for a client implementation project.

CRITICAL RULE: Every business-specific detail — matching method, approval levels, \
currencies, netting, custom fields, document types, integration points — MUST \
come from the "Requirements from MoM" section provided below. \
Do NOT invent requirements. Do NOT assume any configuration not explicitly stated.

═══════════════════════════════════════════════════
SECTION 1 — NARRATIVE
═══════════════════════════════════════════════════

Write 2–4 short paragraphs, minimum 150 words total.

STRUCTURE (follow this order):
  Paragraph 1: State the BUSINESS PROBLEM this process solves. What manual \
effort, risk, or inefficiency does it address? Be specific to this client.
  Paragraph 2: Describe how Oracle Cloud handles this process — name the exact \
Oracle module or feature (e.g. "Oracle Payables", "Oracle Revenue Management", \
"the AP Invoice Approval Workflow"). Describe the flow at a high level.
  Paragraph 3+ (if needed): Describe key exception paths, integrations, or \
configuration decisions specific to this client.

STYLE RULES:
  - Start the first sentence with the process subject, NOT with \
"The client will utilize Oracle Cloud to..."
  - Do NOT embed step-type labels like [Manual] or [System] inside the narrative.
  - Write as a consultant explaining to a client what will happen, not as a \
product marketer. Use declarative sentences.
  - Good example: "Maintaining supplier master data is critical to the \
procure-to-pay cycle; duplicate or incomplete records lead to payment errors \
and compliance failures. Oracle Supplier Portal will be configured to capture \
new supplier registrations through a self-service workflow, routing each \
submission through two sequential approval levels defined in the Authority Matrix."
  - Bad example: "The client will utilize Oracle Cloud to streamline their \
supplier onboarding process through a dual-approval workflow."

═══════════════════════════════════════════════════
SECTION 2 — PROCESS STEPS (5–15 steps)
═══════════════════════════════════════════════════

Each step is a real Oracle Cloud UI action or a manual business activity.

STEP TYPE — use exactly these four values:
  "Manual Step"       — performed entirely outside Oracle (physical receipt, \
phone call, paper form, physical delivery)
  "System Assisted"   — user performs an action WITHIN Oracle Cloud \
(entering data, running a program, clicking approve) — THIS IS THE MOST COMMON TYPE
  "System Automated"  — Oracle performs this step with NO user action required \
(background process, automatic notification, system-generated output)
  "Decision"          — a yes/no branch point (approval gate, validation check, \
threshold test)

STEP ID FORMAT: {MODULE}-{PROCESS_NUM}-{NN}
  Examples: AP-02-01, GL-03-05, FA-01-08
  Use zero-padded two-digit sequence numbers.

BUSINESS ACTOR NAMES — use ONLY the canonical labels listed below. \
Pick the correct role for each step.
  Use "System" ONLY for System Automated steps with no human actor.

DECISION STEPS — every process with conditional logic MUST include at least \
one Decision step. Real Oracle processes always have approval gates or \
validation checks. Add Yes/No labels in the diagram for every diamond node.

═══════════════════════════════════════════════════
SECTION 3 — JOURNAL ENTRIES
═══════════════════════════════════════════════════

If this process creates accounting entries, provide them.
If this process does NOT create journal entries (e.g. master data setup, \
physical count, COA maintenance), set journal_entries to an empty list [] \
and the document builder will render "N/A" automatically.

Journal entry rules:
  - Use descriptive account names ONLY. Examples: "AP Accrual", "Liabilities", \
"Prepayment / Advance to Suppliers", "Bank Clearing Account", "CIP Account", \
"Depreciation Expense", "Accumulated Depreciation"
  - NEVER use numeric placeholders like 12345, 654321, 123456.
  - Use multiple entries when the process has distinct accounting events \
(e.g. invoice creation AND payment are separate entries).
  - Keep entries minimal — match what Oracle actually posts, not every \
possible downstream effect.

Standard Oracle journal patterns by process type:
  AP Invoice:          Dr. AP Accrual / Cr. Liabilities
  AP Payment:          Dr. Supplier / Cr. Bank Clearing Account
  Bank Clearing:       Dr. Bank Clearing Account / Cr. Bank
  AR Invoice:          Dr. Accounts Receivable / Cr. Revenue
  AR Receipt (EFT):    Dr. Remitted / Cr. Accounts Receivable; then Dr. Bank / Cr. Remitted
  AR Receipt (Check):  Dr. Confirmed / Cr. AR; Dr. Remitted / Cr. Confirmed; Dr. Bank / Cr. Remitted
  Asset Addition:      Dr. Assets / Cr. Asset Clearing
  CIP:                 Dr. CIP / Cr. CIP Clearing
  Depreciation:        Dr. Depreciation Expense / Cr. Accumulated Depreciation
  Asset Retirement:    Dr. Accumulated Depreciation, Dr. Net Book Value (Loss) / Cr. Asset, Cr. Net Book Value (Gain)
  Bank Transfer:       Sender: Dr. Bank Transfer Reconciliation / Cr. Cash Clearing; Receiver: Dr. Cash Clearing / Cr. Bank Transfer Reconciliation
  Bank Charges:        Dr. Expenses – Bank Charge / Cr. Bank

═══════════════════════════════════════════════════
SECTION 4 — KEY REQUIREMENTS
═══════════════════════════════════════════════════

Write 4–10 bullet points. Each bullet is a SPECIFIC BUSINESS RULE for THIS \
process, extracted from the MoM requirements provided.

RULES:
  - Use an unnumbered bullet list (not numbered).
  - Each bullet must state a client decision or business rule, not a generic \
Oracle product capability.
  - GOOD: "Management selects 3 levels of approval for manual AP invoices \
and 1 level for SCM-generated invoices."
  - GOOD: "The standard count date starts from the invoice date, not from \
the Promised Delivery Date (PPD) on the PO."
  - BAD: "Automate VAT and Withholding Tax calculation using Oracle Tax Cloud."
  - BAD: "Integration with Customer Master for credit limit validation."
  - Scope requirements ONLY to this specific process. Do not include \
requirements that belong to a different process in the same module.
  - If a requirement involves a number (approval levels, VAT %, tolerance %), \
include the number.

═══════════════════════════════════════════════════
SECTION 5 — PROCESS FLOW DIAGRAM (Graphviz DOT)
═══════════════════════════════════════════════════

## Graphviz DOT Diagram Rules — FOLLOW EXACTLY

RULE 1 — UNIQUE NODE IDs:
  Every node ID must be globally unique within the digraph.
  Format: {MODULE}{PROCESS_NUM}_{short_name}
  Examples: CM02_start, AP04_approve, GL01_end
  NEVER use bare IDs like a1, a2, s1, n1, start, end, approve.
  NEVER place the same node ID in two different subgraph clusters.

RULE 2 — ONE CLUSTER PER NODE:
  Every node must be declared inside exactly one subgraph cluster_ block.
  Never reference a node in a cluster if declared outside.

RULE 3 — NO BACKWARD EDGES:
  All flow goes left to right. For rejection or retry loops use:
    NODE_A -> NODE_B [label="Retry", constraint=false, style=dashed]

RULE 4 — REQUIRED STRUCTURE:
  - First node:  shape=ellipse, label="Start"
  - Last node:   shape=ellipse, label="End"
  - Decisions:   shape=diamond — ALWAYS add [label="Yes"] and [label="No"] \
on outgoing edges
  - All others:  shape=box, style="rounded,filled", fillcolor="#3C6E99", \
fontcolor=white

RULE 5 — SWIMLANE LABELS:
  Use the business_actor values from the process steps as cluster labels.
  Each unique actor gets exactly one cluster.

RULE 6 — VALID TEMPLATE:

digraph G {
  rankdir=LR; splines=polyline;
  nodesep=0.5; ranksep=1.0; pad=0.4;
  node [shape=box, style="rounded,filled", fillcolor="#3C6E99",
        fontcolor=white, fontname="Arial", fontsize=10];
  edge [fontname="Arial", fontsize=9];

  subgraph cluster_APAccountant {
    label="AP Accountant"; style=filled; fillcolor="#F0F4F8";
    AP02_start [shape=ellipse, label="Start", fillcolor="#3C6E99"];
    AP02_enter [label="Enter Invoice"];
    AP02_match [label="Match PO/Receipt/\nInspection"];
  }
  subgraph cluster_System {
    label="Oracle System"; style=filled; fillcolor="#EBF3E8";
    AP02_validate [label="Validate Invoice"];
    AP02_decision [shape=diamond, label="Valid?"];
    AP02_post [label="Post to GL"];
    AP02_exception [label="Exception Queue"];
  }
  subgraph cluster_APManager {
    label="AP Manager"; style=filled; fillcolor="#F8F8F0";
    AP02_review [label="Review Exception"];
    AP02_end [shape=ellipse, label="End", fillcolor="#3C6E99"];
  }

  AP02_start -> AP02_enter;
  AP02_enter -> AP02_match;
  AP02_match -> AP02_validate;
  AP02_validate -> AP02_decision;
  AP02_decision -> AP02_post [label="Yes"];
  AP02_decision -> AP02_exception [label="No"];
  AP02_post -> AP02_end;
  AP02_exception -> AP02_review;
  AP02_review -> AP02_end;
}

═══════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════

Return ONLY a valid JSON object matching this schema. No markdown fences. \
No text before or after the JSON.
The "process_id" value MUST be exactly the Process ID shown above (Module.NN only, no client prefix).

{
  "process_id": "AP.01",
  "process_name": "string",
  "narrative": "string (minimum 150 words, 2-4 paragraphs)",
  "process_steps": [
    {
      "step_id": "AP-01-01",
      "action": "string (concise label, max 6 words)",
      "description": "string (1-2 sentences, what happens)",
      "step_type": "Manual Step|System Assisted|System Automated|Decision",
      "business_actor": "string (Oracle-standard role title)"
    }
  ],
  "journal_entries": [
    {
      "debit_account": "string (descriptive account name, no numbers)",
      "credit_account": "string (descriptive account name, no numbers)",
      "amount_label": "string (e.g. Invoice Amount, Tax Amount)",
      "label": "string (event name, e.g. Invoice Posting, Payment)"
    }
  ],
  "key_requirements": ["string (specific business rule)", ...],
  "diagram_code": "digraph G { ... }"
}
"""


def build_generation_prompt(
    section_plan,
    process,
    requirements_text: str,
    rag_context: str,
) -> str:
    """
    Build the full generation prompt for a single process.

    Parameters
    ----------
    section_plan
        The SectionPlan for the module containing this process.
    process
        The ProcessEntry being generated.
    requirements_text
        Formatted requirements text for this module.
    rag_context
        Retrieved examples from the knowledge base (may be empty).

    Returns
    -------
    str
        The complete prompt to send to the LLM.
    """
    # Build business actors list — used as swimlane hints for the LLM
    actors_list = getattr(section_plan, "business_actors", []) \
        if hasattr(section_plan, "business_actors") else []
    actors_text = ", ".join(actors_list) if actors_list else "(see requirements)"
    canonical_actors = ", ".join(CANONICAL_BUSINESS_ACTORS)

    rag_section = ""
    if rag_context:
        rag_section = f"""
## Retrieved Examples (quality benchmark — do not copy verbatim)
{rag_context}
"""

    # Sanitize process_id: extract just the MODULE.NN portion (e.g. "AP.01")
    # LLMs sometimes output full client name like "AL Gosaibi Co.AP.01"
    pid_raw = process.process_id
    pid_match = re.search(r'([A-Z]{2,3}\.\d{2})(?:\s|$|\.)', pid_raw)
    clean_pid = pid_match.group(1) if pid_match else pid_raw

    return f"""{GENERATION_SYSTEM_PROMPT}
{rag_section}
## Client and Process Context
Module: {section_plan.module_name}
Process ID: {clean_pid}
Process Name: {process.process_name}
Process Description: {process.process_description}
Expected Output: {process.output}
Business Actors (use these as swimlane labels): {actors_text}
Canonical Business Actor Names (use EXACTLY these): {canonical_actors}
Confidence Level: {process.confidence}

Requirements from MoM:
{requirements_text}

Missing Information:
{chr(10).join('- ' + mi for mi in process.missing_info) if process.missing_info else '- None identified'}

Module Introduction Context:
{section_plan.module_intro}

Now write the complete SectionContent for process {clean_pid} "{process.process_name}".
Return ONLY the JSON object."""
