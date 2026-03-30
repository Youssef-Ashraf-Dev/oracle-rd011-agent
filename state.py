"""
RD.011 Agent — LangGraph state definition.

The RD011State TypedDict is the single source of truth for all data
flowing through the agent graph.  Every node reads from and writes to
this state.  SQLite checkpointing persists a snapshot after every node.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class RD011State(TypedDict):
    """Complete state for the RD.011 document generation pipeline."""

    # Identity
    thread_id: str

    # Input
    input_files: List[str]
    raw_texts: Dict[str, str]

    # Extraction
    extraction_result: Optional[dict]

    # Planning
    document_plan: Optional[dict]
    issue_report: Optional[dict]

    # Approval loop
    consultant_approved: bool
    consultant_feedback: str
    approval_iteration: int
    approval_maxed: bool

    # Generation
    intro_content: Optional[dict]
    section_queue: List[str]
    current_section_index: int
    generated_sections: Dict[str, dict]
    failed_sections: List[str]

    # Diagrams
    diagram_registry: Dict[str, str]

    # Output
    output_path: Optional[str]

    # Error tracking
    errors: List[str]
    last_completed_node: str
