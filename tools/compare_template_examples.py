"""
Compare RD011 template vs RAG example documents.

Purpose:
- Estimate how closely example docs align with the current template styles.
- Highlight missing style names (DocumentBuilder will fall back when absent).

Run:
  conda run -n rd011-env python tools/compare_template_examples.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StyleReport:
    path: Path
    style_names: set[str]


def _load_style_names(doc_path: Path) -> set[str]:
    from docx import Document

    doc = Document(str(doc_path))
    names: set[str] = set()
    for s in doc.styles:
        try:
            if s and s.name:
                names.add(str(s.name))
        except Exception:
            continue
    return names


def _print_presence(title: str, required: list[str], present: set[str]) -> None:
    print(title)
    for name in required:
        print(f"  {name}: {'YES' if name in present else 'no'}")


def main() -> None:
    try:
        import docx  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "python-docx is required. Run via the project env, e.g.\n"
            "  conda run -n rd011-env python tools/compare_template_examples.py\n"
            f"Error: {exc}"
        )

    template_path = Path("templates/RD011_TEMPLATE.docx")
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    examples_dir = Path("rag/examples")
    example_files = sorted(examples_dir.glob("*.docx"))

    # These are the style names our output builder uses (from `config.py:WORD_STYLES`).
    expected_template_styles = [
        "Table Heading",
        "Table Text",
        "Body Text",
        "Heading 1",
        "Heading 2",
        "Heading 3",
        "Number List",
        "Bullet",
        "Title",
    ]

    template_styles = _load_style_names(template_path)
    print(f"template={template_path.resolve()}")
    print(f"template_style_count={len(template_styles)}")
    _print_presence("template_expected_styles:", expected_template_styles, template_styles)

    print(f"\nexamples_dir={examples_dir.resolve()}")
    print(f"examples_count={len(example_files)}")

    for ex in example_files:
        ex_styles = _load_style_names(ex)
        overlap = len(ex_styles & template_styles)
        union = len(ex_styles | template_styles)
        jaccard = (overlap / union) if union else 0.0

        print(f"\n--- {ex.name}")
        print(f"example_style_count={len(ex_styles)}")
        print(f"template_overlap_count={overlap}")
        print(f"style_jaccard={jaccard:.3f}")
        _print_presence("example_expected_template_styles:", expected_template_styles, ex_styles)

        missing = [s for s in expected_template_styles if s not in ex_styles]
        if missing:
            print("missing_expected_styles:", ", ".join(missing))


if __name__ == "__main__":
    main()
