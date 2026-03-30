"""
RD.011 Agent — Assemble Document node.

Reconstructs all data objects from state, invokes the DocumentBuilder,
and produces the final RD.011 Word document.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from config import OUTPUT_DIR
from models.schemas import (
    DocumentPlan,
    IntroContent,
    IssueReport,
    SectionContent,
    SectionPlan,
)
from doc_builder.document_builder import DocumentBuilder

logger = logging.getLogger(__name__)


def assemble_document_node(state: dict) -> dict:
    """
    Build the final RD.011 Word document from all generated content.

    Steps:
    1. Reconstruct Pydantic objects from state dicts
    2. Build cover page, document control, introduction, enterprise section
    3. Build each module chapter with diagrams
    4. Build issues section
    5. Build Mermaid source appendix
    6. Save the document
    """
    plan_data = state.get("document_plan")
    intro_data = state.get("intro_content")
    generated_sections = state.get("generated_sections", {})
    diagram_registry = state.get("diagram_registry", {})
    issue_data = state.get("issue_report")
    thread_id = state.get("thread_id", "unknown")
    errors = list(state.get("errors", []))

    if not plan_data:
        errors.append("No document plan — cannot assemble document.")
        return {"errors": errors, "last_completed_node": "assemble"}

    # Reconstruct objects
    plan = DocumentPlan.model_validate(plan_data)
    intro = IntroContent.model_validate(intro_data) if intro_data else None
    issue_report = IssueReport.model_validate(issue_data) if issue_data else None

    # Build section content objects dict
    section_contents: dict[str, SectionContent] = {}
    for key, data in generated_sections.items():
        try:
            if isinstance(data, dict):
                section_contents[key] = SectionContent.model_validate(data)
            else:
                section_contents[key] = data
        except Exception as exc:
            logger.warning("Could not validate section %s: %s", key, exc)

    # Initialize builder
    builder = DocumentBuilder()

    # 1. Cover page
    builder.build_cover_page(plan)

    # 2. Table of contents
    builder.build_table_of_contents()

    # 3. Document control
    builder.build_document_control(plan)

    # 4. Introduction
    if intro:
        builder.build_introduction(intro, plan)

        # 5. Enterprise structure
        builder.build_enterprise_section(intro, plan)

    # 6. Module chapters
    for chapter_num, section_plan in enumerate(plan.sections, start=1):
        # Collect section contents for this module
        module_sections: dict[str, SectionContent] = {}
        for key, content in section_contents.items():
            if key.startswith(f"{section_plan.section_id}."):
                module_sections[key] = content

        builder.build_module_chapter(
            section_plan=section_plan,
            sections=module_sections,
            diagram_registry=diagram_registry,
            chapter_number=chapter_num,
        )

    # 7. Open and Closed Issues
    open_issues = []
    closed_issues = []
    if issue_report and issue_report.contradictions:
        for i, contradiction in enumerate(issue_report.contradictions, start=1):
            open_issues.append({
                "id": i,
                "issue": contradiction,
                "resolution": "",
                "responsibility": "",
                "target_date": "",
                "impact_date": "",
            })
    builder.build_issues_section(open_issues, closed_issues)

    # 8. Graphviz DOT source appendix
    builder.build_diagram_appendix(generated_sections)

    # 9. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    client_slug = plan.client_name.replace(" ", "_")[:30]
    output_filename = f"RD011_{client_slug}_{timestamp}.docx"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    builder.save(output_path)
    logger.info("Document assembled: %s", output_path)

    return {
        "output_path": output_path,
        "last_completed_node": "assemble",
    }
