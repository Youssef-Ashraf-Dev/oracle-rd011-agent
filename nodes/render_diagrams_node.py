"""
RD.011 Agent — Render Diagrams node.

Iterates through all generated sections and converts each Graphviz DOT
diagram string to a PNG file using the two-strategy fallback renderer.
"""

from __future__ import annotations

import logging

from doc_builder.diagram_renderer import render_dot_to_png

logger = logging.getLogger(__name__)


def render_diagrams_node(state: dict) -> dict:
    """
    Render all Graphviz DOT diagrams to PNG.

    Builds a ``diagram_registry`` mapping process_id to the absolute
    path of the rendered PNG file.
    """
    generated_sections = state.get("generated_sections", {})
    diagram_registry: dict[str, str] = {}
    errors = list(state.get("errors", []))

    for section_key, section_data in generated_sections.items():
        if isinstance(section_data, dict):
            dot_str = section_data.get("diagram_code", "")
        else:
            dot_str = section_data.diagram_code

        # Derive process_id from section_key to ALWAYS match plan's proc.process_id.
        # section_key format: "SECTION_ID.PROCESS_ID"  e.g. "AP.AP.01"
        # The LLM sometimes generates a wrong process_id in its JSON; using the
        # key guarantees the diagram_registry key matches document_builder's lookup.
        parts = section_key.split(".", 1)
        process_id = parts[1] if len(parts) == 2 else section_key

        if not dot_str:
            logger.warning("No diagram code for %s — skipping", process_id)
            continue

        try:
            png_path = render_dot_to_png(dot_str, process_id)
            diagram_registry[process_id] = png_path
            logger.info("Rendered diagram for %s: %s", process_id, png_path)
        except Exception as exc:
            error_msg = f"Diagram rendering failed for {process_id}: {exc}"
            logger.warning(error_msg)
            errors.append(error_msg)

    logger.info("Diagram rendering complete: %d diagrams", len(diagram_registry))

    return {
        "diagram_registry": diagram_registry,
        "errors": errors,
        "last_completed_node": "render_diagrams",
    }
