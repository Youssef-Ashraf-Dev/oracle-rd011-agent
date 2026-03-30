"""
Tests for implicit process handling.

Tests cover:
- Loading implicit processes from config file
- Augmenting plans with implicit processes
- Duplicate detection
- Module matching
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from models.schemas import ProcessEntry, SectionPlan, DocumentPlan


class TestImplicitProcessLoad:
    """Test loading implicit processes from config."""

    def test_load_implicit_processes_returns_list(self, tmp_path):
        """Test that load_implicit_processes returns list preserving order."""
        from nodes.plan_node import _load_implicit_processes

        # Create test config
        test_config = {
            "implicit_processes": [
                {
                    "process_id": "GL.10",
                    "process_name": "Month End Closing",
                    "module": "GL",
                    "description": "Period close",
                    "required_always": True,
                    "business_actors": ["Manager"],
                    "default_confidence": "high",
                }
            ]
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(test_config, f)

        # Patch the config path
        with patch("nodes.plan_node.IMPLICIT_PROCESSES_CONFIG", config_file):
            procs = _load_implicit_processes()
            assert isinstance(procs, list)
            assert procs[0]["process_id"] == "GL.10"
            assert procs[0]["process_name"] == "Month End Closing"

    def test_load_implicit_processes_missing_file(self):
        """Test graceful handling of missing config file."""
        from nodes.plan_node import _load_implicit_processes

        with patch("nodes.plan_node.IMPLICIT_PROCESSES_CONFIG") as mock_path:
            mock_path.exists.return_value = False
            procs = _load_implicit_processes()
            assert procs == []

    def test_load_implicit_processes_corrupt_json(self, tmp_path):
        """Test graceful handling of corrupt JSON config."""
        from nodes.plan_node import _load_implicit_processes

        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json")

        with patch("nodes.plan_node.IMPLICIT_PROCESSES_CONFIG", config_file):
            procs = _load_implicit_processes()
            assert procs == []


class TestAugmentPlanWithImplicit:
    """Test augmenting DocumentPlan with implicit processes."""

    def _make_section_plan(self, section_id: str, module_name: str) -> SectionPlan:
        """Helper to create a SectionPlan."""
        return SectionPlan(
            section_id=section_id,
            module_name=module_name,
            module_intro="Test intro",
            processes=[
                ProcessEntry(
                    process_id=f"{section_id}.01",
                    process_name="Existing Process",
                    process_description="Already in plan",
                    output="Output",
                    confidence="high",
                )
            ],
        )

    def _make_document_plan(self) -> DocumentPlan:
        """Helper to create a DocumentPlan with sample sections."""
        return DocumentPlan(
            client_name="Test Client",
            project_name="Test Project",
            document_ref="RD.011",
            author="Agent",
            version="DRAFT 1A",
            sections=[
                self._make_section_plan("GL", "General Ledger"),
                self._make_section_plan("AP", "Accounts Payable"),
            ],
        )

    def test_augment_plan_with_implicit_process_added(self):
        """Test that implicit processes are added to plan."""
        from nodes.plan_node import _augment_plan_with_implicit_processes

        plan = self._make_document_plan()
        implicit_procs = [
            {
                "process_id": "GL.10",
                "process_name": "Month End Closing",
                "module": "GL",
                "description": "Period-end closing",
                "default_confidence": "high",
            }
        ]

        augmented = _augment_plan_with_implicit_processes(plan, implicit_procs)

        # Check GL section now has 2 processes (original + implicit)
        gl_section = next(s for s in augmented.sections if s.section_id == "GL")
        assert len(gl_section.processes) == 2
        assert gl_section.processes[1].process_name == "Month End Closing"

    def test_augment_plan_skips_duplicate_process(self):
        """Test that duplicate processes are not added."""
        from nodes.plan_node import _augment_plan_with_implicit_processes

        plan = self._make_document_plan()

        # Modify GL section to already have a Month End Closing process
        gl_section = next(s for s in plan.sections if s.section_id == "GL")
        gl_section.processes.append(
            ProcessEntry(
                process_id="GL.02",
                process_name="Month End Closing",
                process_description="Already exists",
                output="Output",
                confidence="high",
            )
        )

        implicit_procs = [
            {
                "process_id": "GL.10",
                "process_name": "Month End Closing",
                "module": "GL",
                "description": "Period-end closing",
                "default_confidence": "high",
            }
        ]

        augmented = _augment_plan_with_implicit_processes(plan, implicit_procs)

        # Check GL section still has only 2 processes (duplicate not added)
        gl_section = next(s for s in augmented.sections if s.section_id == "GL")
        assert len(gl_section.processes) == 2

    def test_augment_plan_skips_missing_module(self):
        """Test that implicit processes for missing modules are skipped."""
        from nodes.plan_node import _augment_plan_with_implicit_processes

        plan = self._make_document_plan()

        implicit_procs = [
            {
                "process_id": "FA_DEPRECIATION",
                "process_name": "Depreciation Posting",
                "module": "FA",  # Not in plan
                "description": "Depreciation",
                "default_confidence": "high",
            }
        ]

        augmented = _augment_plan_with_implicit_processes(plan, implicit_procs)

        # Plan should be unchanged (no FA section)
        assert len(augmented.sections) == 2
        assert all(s.section_id in ("GL", "AP") for s in augmented.sections)

    def test_augment_plan_with_empty_implicit_dict(self):
        """Test that empty implicit dict returns unchanged plan."""
        from nodes.plan_node import _augment_plan_with_implicit_processes

        plan = self._make_document_plan()
        original_process_count = sum(len(s.processes) for s in plan.sections)

        augmented = _augment_plan_with_implicit_processes(plan, [])

        new_process_count = sum(len(s.processes) for s in augmented.sections)
        assert original_process_count == new_process_count

    def test_augment_plan_multiple_implicit_processes(self):
        """Test adding multiple implicit processes to same module."""
        from nodes.plan_node import _augment_plan_with_implicit_processes

        plan = self._make_document_plan()

        implicit_procs = [
            {
                "process_id": "GL.10",
                "process_name": "Month End Closing",
                "module": "GL",
                "description": "Period-end closing",
                "default_confidence": "high",
            },
            {
                "process_id": "GL.11",
                "process_name": "Tax Reconciliation",
                "module": "GL",
                "description": "Tax reconciliation",
                "default_confidence": "high",
            },
        ]

        augmented = _augment_plan_with_implicit_processes(plan, implicit_procs)

        # Check GL section now has 3 processes (original + 2 implicit)
        gl_section = next(s for s in augmented.sections if s.section_id == "GL")
        assert len(gl_section.processes) == 3
        process_names = {p.process_name for p in gl_section.processes}
        assert "Month End Closing" in process_names
        assert "Tax Reconciliation" in process_names

    def test_augment_plan_inserts_implicit_between_mom_processes(self):
        """Implicit processes should be inserted between MoM anchors in logical order."""
        from nodes.plan_node import _augment_plan_with_implicit_processes

        plan = self._make_document_plan()
        gl_section = next(s for s in plan.sections if s.section_id == "GL")
        gl_section.processes = [
            ProcessEntry(
                process_id="GL.01",
                process_name="Process A",
                process_description="MoM A",
                output="Output",
                confidence="high",
            ),
            ProcessEntry(
                process_id="GL.03",
                process_name="Process C",
                process_description="MoM C",
                output="Output",
                confidence="high",
            ),
        ]

        implicit_procs = [
            {
                "process_id": "GL.01",
                "process_name": "Process A",
                "module": "GL",
                "description": "Implicit A",
                "default_confidence": "high",
            },
            {
                "process_id": "GL.02",
                "process_name": "Process B",
                "module": "GL",
                "description": "Implicit B",
                "default_confidence": "high",
            },
            {
                "process_id": "GL.03",
                "process_name": "Process C",
                "module": "GL",
                "description": "Implicit C",
                "default_confidence": "high",
            },
        ]

        augmented = _augment_plan_with_implicit_processes(plan, implicit_procs)
        gl_section = next(s for s in augmented.sections if s.section_id == "GL")
        names = [p.process_name for p in gl_section.processes]
        assert names == ["Process A", "Process B", "Process C"]

    def test_augment_plan_skips_duplicate_by_id(self):
        """If MoM has same process_id, implicit should be skipped even if names differ."""
        from nodes.plan_node import _augment_plan_with_implicit_processes

        plan = self._make_document_plan()
        gl_section = next(s for s in plan.sections if s.section_id == "GL")
        gl_section.processes.append(
            ProcessEntry(
                process_id="GL.10",
                process_name="MoM Month End Close",
                process_description="Already exists with different name",
                output="Output",
                confidence="high",
            )
        )

        implicit_procs = [
            {
                "process_id": "GL.10",
                "process_name": "Month End Closing",
                "module": "GL",
                "description": "Implicit version",
                "default_confidence": "high",
            }
        ]

        augmented = _augment_plan_with_implicit_processes(plan, implicit_procs)
        gl_section = next(s for s in augmented.sections if s.section_id == "GL")
        assert len(gl_section.processes) == 2


class TestOrderAndRenumber:
    """Test renumbering keeps order and assigns sequential IDs."""

    def test_order_and_renumber_preserves_order(self):
        from nodes.plan_node import _order_and_renumber_processes

        plan = DocumentPlan(
            client_name="Test Client",
            project_name="Test Project",
            document_ref="RD.011",
            author="Agent",
            version="DRAFT 1A",
            sections=[
                SectionPlan(
                    section_id="GL",
                    module_name="General Ledger",
                    module_intro="Intro",
                    processes=[
                        ProcessEntry(
                            process_id="GL.10",
                            process_name="Proc A",
                            process_description="A",
                            output="Output",
                            confidence="high",
                        ),
                        ProcessEntry(
                            process_id="GL.20",
                            process_name="Proc B",
                            process_description="B",
                            output="Output",
                            confidence="high",
                        ),
                    ],
                )
            ],
        )

        updated = _order_and_renumber_processes(plan, [])
        gl_section = updated.sections[0]
        ids = [p.process_id for p in gl_section.processes]
        assert ids == ["Test Client.GL.01", "Test Client.GL.02"]


class TestImplicitProcessConfig:
    """Test loading and structure of implicit processes config file."""

    def test_config_file_exists_and_valid_json(self):
        """Test that config_implicit_processes.json exists and is valid JSON."""
        config_path = Path(__file__).parent.parent / "config_implicit_processes.json"
        assert config_path.exists(), f"Config file not found at {config_path}"

        with open(config_path) as f:
            data = json.load(f)
            assert "implicit_processes" in data
            assert isinstance(data["implicit_processes"], list)
            assert len(data["implicit_processes"]) > 0

    def test_config_process_structure(self):
        """Test that each implicit process has required fields."""
        config_path = Path(__file__).parent.parent / "config_implicit_processes.json"

        with open(config_path) as f:
            data = json.load(f)

        required_fields = {
            "process_id",
            "process_name",
            "module",
            "description",
            "required_always",
            "business_actors",
            "default_confidence",
        }

        for proc in data["implicit_processes"]:
            missing_fields = required_fields - set(proc.keys())
            assert (
                not missing_fields
            ), f"Process {proc.get('process_id')} missing fields: {missing_fields}"

    def test_config_module_codes_valid(self):
        """Test that all module codes are valid Oracle module codes."""
        config_path = Path(__file__).parent.parent / "config_implicit_processes.json"

        with open(config_path) as f:
            data = json.load(f)

        valid_modules = {"GL", "AP", "AR", "FA", "CM"}

        for proc in data["implicit_processes"]:
            module = proc.get("module")
            assert (
                module in valid_modules
            ), f"Invalid module '{module}' in process {proc.get('process_id')}"

    def test_config_has_gl_processes(self):
        """Test that GL module has required implicit processes."""
        config_path = Path(__file__).parent.parent / "config_implicit_processes.json"

        with open(config_path) as f:
            data = json.load(f)

        gl_processes = {p["process_name"] for p in data["implicit_processes"] if p["module"] == "GL"}

        # Check for expected GL processes per Oracle OUM standard list
        expected = {"Manual Journal Entry", "Journal Revision and Posting", "Period Closing"}
        assert expected.issubset(
            gl_processes
        ), f"Missing GL processes. Have: {gl_processes}, Expected: {expected}"

    def test_config_has_cm_processes(self):
        """Test that CM (Cash Management) module has required implicit processes."""
        config_path = Path(__file__).parent.parent / "config_implicit_processes.json"

        with open(config_path) as f:
            data = json.load(f)

        cm_processes = {p["process_name"] for p in data["implicit_processes"] if p["module"] == "CM"}

        expected = {"Bank Statement Reconciliation", "Bank Transfer", "External Transaction"}
        assert expected.issubset(
            cm_processes
        ), f"Missing CM processes. Have: {cm_processes}, Expected: {expected}"
