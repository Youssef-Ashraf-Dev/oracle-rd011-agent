"""Build per-task/per-schema model quality scoreboard from retry telemetry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Row:
    task_type: str
    schema: str
    provider: str
    model: str
    cascade_attempts: int = 0
    attempt_success: int = 0
    model_success: int = 0
    validation_failures: int = 0
    retries_built: int = 0
    latencies_ms: list[int] = field(default_factory=list)

    def success_rate(self) -> float:
        if self.cascade_attempts <= 0:
            return 0.0
        return self.model_success / self.cascade_attempts

    def schema_pass_rate(self) -> float:
        denom = self.attempt_success + self.validation_failures
        if denom <= 0:
            return 0.0
        return self.attempt_success / denom

    def p95_latency_ms(self) -> int:
        if not self.latencies_ms:
            return 0
        values = sorted(self.latencies_ms)
        idx = int(0.95 * (len(values) - 1))
        return int(values[idx])


def parse_events(path: Path) -> dict[tuple[str, str, str, str], Row]:
    rows: dict[tuple[str, str, str, str], Row] = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue

            task = str(event.get("task_type", ""))
            schema = str(event.get("schema", ""))
            provider = str(event.get("provider", ""))
            model = str(event.get("model", ""))
            if not task or not schema or not provider or not model:
                continue

            key = (task, schema, provider, model)
            row = rows.get(key)
            if row is None:
                row = Row(task, schema, provider, model)
                rows[key] = row

            etype = str(event.get("event", ""))
            if etype == "cascade_attempt":
                row.cascade_attempts += 1
            elif etype == "attempt_success":
                row.attempt_success += 1
                lat = event.get("latency_ms")
                if isinstance(lat, int):
                    row.latencies_ms.append(lat)
            elif etype == "model_success":
                row.model_success += 1
            elif etype == "attempt_validation_failed":
                row.validation_failures += 1
            elif etype == "attempt_retry_prompt_built":
                row.retries_built += 1

    return rows


def build_report(rows: dict[tuple[str, str, str, str], Row]) -> dict:
    items = []
    for row in rows.values():
        quality_score = (
            row.success_rate() * 55.0
            + row.schema_pass_rate() * 35.0
            - min(20.0, row.retries_built * 0.8)
            - min(15.0, row.p95_latency_ms() / 1200.0)
        )
        items.append(
            {
                "task_type": row.task_type,
                "schema": row.schema,
                "provider": row.provider,
                "model": row.model,
                "cascade_attempts": row.cascade_attempts,
                "model_success": row.model_success,
                "attempt_success": row.attempt_success,
                "validation_failures": row.validation_failures,
                "retries_built": row.retries_built,
                "success_rate": round(row.success_rate(), 4),
                "schema_pass_rate": round(row.schema_pass_rate(), 4),
                "p95_latency_ms": row.p95_latency_ms(),
                "quality_score": round(max(0.0, min(100.0, quality_score)), 3),
            }
        )

    items.sort(
        key=lambda x: (x["task_type"], x["schema"], -x["quality_score"], -x["cascade_attempts"])
    )
    return {
        "summary": {
            "rows": len(items),
        },
        "scoreboard": items,
    }


def to_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Model Quality Scoreboard")
    lines.append("")
    lines.append(f"Rows: {report['summary']['rows']}")
    lines.append("")
    lines.append("| Task | Schema | Provider | Model | Attempts | Success | Schema Pass | P95 ms | Score |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in report["scoreboard"]:
        lines.append(
            "| {task_type} | {schema} | {provider} | {model} | {cascade_attempts} | {success_rate:.2f} | {schema_pass_rate:.2f} | {p95_latency_ms} | {quality_score:.2f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build model quality scoreboard from telemetry")
    parser.add_argument("--telemetry", default="outputs/llm_telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--json-out", default="outputs/model_quality_scoreboard.json", help="JSON output path")
    parser.add_argument("--md-out", default="outputs/model_quality_scoreboard.md", help="Markdown output path")
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry)
    if telemetry_path.exists():
        rows = parse_events(telemetry_path)
    else:
        rows = {}
        print(f"Telemetry file not found: {telemetry_path}. Writing empty scoreboard.")
    report = build_report(rows)

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_out.write_text(to_markdown(report), encoding="utf-8")

    print(f"Wrote JSON: {json_out}")
    print(f"Wrote markdown: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
