"""Estimate safe minimum RPM/TPM/burst by provider/model from telemetry rolling windows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        value = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


@dataclass
class Point:
    ts: datetime
    provider: str
    model: str
    tokens: int


def p95(values: list[int]) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = int(0.95 * (len(values) - 1))
    return int(values[idx])


def build_points(path: Path) -> list[Point]:
    out: list[Point] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue

            if event.get("event") != "attempt_success":
                continue

            ts = parse_ts(str(event.get("ts", "")))
            provider = str(event.get("provider", ""))
            model = str(event.get("model", ""))
            if ts is None or not provider or not model:
                continue

            prompt_chars = int(event.get("prompt_chars", 0) or 0)
            response_chars = int(event.get("response_chars", 0) or 0)
            # Approximate tokens from chars; provider-level token counters can replace this later.
            est_tokens = max(1, int((prompt_chars + response_chars) / 4))
            out.append(Point(ts=ts, provider=provider, model=model, tokens=est_tokens))

    out.sort(key=lambda x: x.ts)
    return out


def rolling_counts(points: list[Point], window_seconds: int) -> tuple[list[int], list[int]]:
    req_values: list[int] = []
    tok_values: list[int] = []

    left = 0
    for right in range(len(points)):
        current = points[right]
        while left <= right and (current.ts - points[left].ts).total_seconds() > window_seconds:
            left += 1
        window = points[left : right + 1]
        req_values.append(len(window))
        tok_values.append(sum(p.tokens for p in window))

    return req_values, tok_values


def summarize(points: list[Point]) -> dict:
    by_model: dict[tuple[str, str], list[Point]] = defaultdict(list)
    for p in points:
        by_model[(p.provider, p.model)].append(p)

    rows = []
    for (provider, model), bucket in by_model.items():
        req60, tok60 = rolling_counts(bucket, 60)
        req10, tok10 = rolling_counts(bucket, 10)

        safe_rpm = max(1, int(p95(req60) * 0.8))
        safe_tpm = max(1, int(p95(tok60) * 0.8))
        safe_burst_rps10 = max(1, int(p95(req10) * 0.8))
        safe_burst_tps10 = max(1, int(p95(tok10) * 0.8))

        rows.append(
            {
                "provider": provider,
                "model": model,
                "samples": len(bucket),
                "safe_rpm": safe_rpm,
                "safe_tpm": safe_tpm,
                "safe_burst_req_10s": safe_burst_rps10,
                "safe_burst_tok_10s": safe_burst_tps10,
                "p95_req_60s": p95(req60),
                "p95_tok_60s": p95(tok60),
            }
        )

    rows.sort(key=lambda x: (x["provider"], x["model"]))
    return {
        "summary": {"models": len(rows)},
        "limits": rows,
    }


def to_markdown(report: dict) -> str:
    lines = ["# Estimated Safe Limits", ""]
    lines.append("| Provider | Model | Samples | Safe RPM | Safe TPM | Safe Burst Req (10s) | Safe Burst Tok (10s) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in report["limits"]:
        lines.append(
            "| {provider} | {model} | {samples} | {safe_rpm} | {safe_tpm} | {safe_burst_req_10s} | {safe_burst_tok_10s} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("- Safe limits are 80% of rolling-window p95 utilization to preserve headroom.")
    lines.append("- Replace token approximation with provider-native usage metrics when available.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate safe RPM/TPM/burst from telemetry")
    parser.add_argument("--telemetry", default="outputs/llm_telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--json-out", default="outputs/rate_limit_sizing.json", help="JSON output path")
    parser.add_argument("--md-out", default="outputs/rate_limit_sizing.md", help="Markdown output path")
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry)
    if telemetry_path.exists():
        points = build_points(telemetry_path)
    else:
        points = []
        print(f"Telemetry file not found: {telemetry_path}. Writing empty sizing report.")
    report = summarize(points)

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
