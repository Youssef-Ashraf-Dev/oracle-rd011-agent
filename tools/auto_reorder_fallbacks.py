"""Generate capability override file by auto-reordering fallbacks from safety report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("large_context", "reasoning", "generation")


def parse_model_id(model_id: str) -> tuple[str, str] | None:
    if "/" not in model_id:
        return None
    provider, model = model_id.split("/", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        return None
    return provider, model


def build_overrides(report: dict, top_n: int, min_score: float) -> dict:
    out: dict[str, dict] = {}
    recs_all = report.get("task_recommendations", {})

    for task in TASKS:
        recs = recs_all.get(task, [])
        selected = []
        for rec in recs:
            score = float(rec.get("score", 0.0) or 0.0)
            model_id = str(rec.get("model", ""))
            parsed = parse_model_id(model_id)
            if parsed is None:
                continue
            if score < min_score:
                continue
            provider, model = parsed
            selected.append({"provider": provider, "model": model, "score": score})
            if len(selected) >= top_n:
                break

        if not selected:
            continue

        primary = selected[0]
        fallback_chain = [{"provider": x["provider"], "model": x["model"]} for x in selected[1:]]
        out[task] = {
            "provider": primary["provider"],
            "model": primary["model"],
            "fallback_chain": fallback_chain,
            "source": "auto_reorder_fallbacks",
        }

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-reorder fallbacks from safety report")
    parser.add_argument("--report", default="outputs/llm_safety_report.json", help="Safety report JSON")
    parser.add_argument("--out", default="outputs/capability_overrides.json", help="Override output JSON")
    parser.add_argument("--top-n", type=int, default=3, help="Max models per task (primary + fallbacks)")
    parser.add_argument("--min-score", type=float, default=1.0, help="Minimum recommendation score")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Safety report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    overrides = build_overrides(report, top_n=max(1, args.top_n), min_score=args.min_score)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    print(f"Wrote capability overrides: {out_path}")
    print("Set CAPABILITY_OVERRIDES_PATH to this file to activate at runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
