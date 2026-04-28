"""
Compare the template outline to example outlines.

This answers: which example document's section naming is closest to the template?

Run:
  conda run -n rd011-env python tools/compare_template_outline.py
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from docx_outline import extract_outline


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def main() -> None:
    template = Path("templates/RD011_TEMPLATE.docx")
    examples_dir = Path("rag/examples")
    examples = sorted(examples_dir.glob("*.docx"))

    if not template.exists():
        raise SystemExit(f"Template not found: {template}")

    t_headings, t_markers = extract_outline(template)
    t_names = [h.text for h in t_headings]
    t_norm = [_norm(x) for x in t_names if x]

    print(f"template={template.resolve()}")
    print(f"template_headings={len(t_headings)}")
    t_present = {k: v for k, v in t_markers.items() if v}
    if t_present:
        print("template_markers:", ", ".join(f"{k}={v}" for k, v in t_present.items()))

    print(f"\nexamples_dir={examples_dir.resolve()}")
    print(f"examples_count={len(examples)}")

    for ex in examples:
        e_headings, e_markers = extract_outline(ex)
        e_names = [h.text for h in e_headings]
        e_norm = [_norm(x) for x in e_names if x]

        # Cheap similarity proxies:
        # 1) marker overlap score (same named markers used)
        marker_overlap = 0
        marker_union = 0
        for k in set(t_markers) | set(e_markers):
            tv = 1 if (t_markers.get(k, 0) > 0) else 0
            ev = 1 if (e_markers.get(k, 0) > 0) else 0
            marker_overlap += 1 if (tv == 1 and ev == 1) else 0
            marker_union += 1 if (tv == 1 or ev == 1) else 0
        marker_jaccard = (marker_overlap / marker_union) if marker_union else 0.0

        # 2) heading-name hit rate: for each template heading, find best match in example.
        # (If the template has no headings, this is 0 by definition.)
        hit_scores: list[float] = []
        for tn in t_names:
            if not tn.strip():
                continue
            best = 0.0
            for en in e_names:
                best = max(best, _similar(tn, en))
            hit_scores.append(best)
        avg_best = (sum(hit_scores) / len(hit_scores)) if hit_scores else 0.0

        print(f"\n--- {ex.name}")
        print(f"example_headings={len(e_headings)}")
        print(f"marker_jaccard={marker_jaccard:.3f}")
        print(f"avg_best_heading_match={avg_best:.3f}")
        e_present = {k: v for k, v in e_markers.items() if v}
        if e_present:
            print("example_markers:", ", ".join(f"{k}={v}" for k, v in e_present.items()))


if __name__ == "__main__":
    main()
