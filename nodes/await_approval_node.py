"""
RD.011 Agent — Await Approval node.

Uses LangGraph's interrupt mechanism for human-in-the-loop approval.
The consultant can type APPROVE to proceed or provide feedback text
to revise the plan.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from config import MAX_APPROVAL_ITERATIONS

logger = logging.getLogger(__name__)


def await_approval_node(state: dict) -> dict:
    """
    Interrupt execution to wait for consultant input.

    If the response is "APPROVE" (case-insensitive), sets
    ``consultant_approved=True``.  Otherwise stores the feedback
    for the update_plan node.
    """
    response = interrupt({"prompt": "APPROVE or feedback: "})

    # Handle the response
    if isinstance(response, dict):
        response_text = response.get("response", response.get("prompt", ""))
    else:
        response_text = str(response)

    response_text = response_text.strip()

    approval_iteration = state.get("approval_iteration", 0) + 1

    if response_text.upper() == "APPROVE":
        logger.info("Consultant approved the plan (iteration %d)", approval_iteration)
        return {
            "consultant_approved": True,
            "consultant_feedback": "",
            "approval_iteration": approval_iteration,
            "approval_maxed": False,
        }
    else:
        approval_maxed = approval_iteration >= MAX_APPROVAL_ITERATIONS
        errors = list(state.get("errors", []))
        if approval_maxed:
            errors.append(
                "Approval loop exceeded MAX_APPROVAL_ITERATIONS without approval."
            )
        logger.info(
            "Consultant provided feedback (iteration %d): %s",
            approval_iteration,
            response_text[:100],
        )
        return {
            "consultant_approved": False,
            "consultant_feedback": response_text,
            "approval_iteration": approval_iteration,
            "approval_maxed": approval_maxed,
            "errors": errors,
        }
