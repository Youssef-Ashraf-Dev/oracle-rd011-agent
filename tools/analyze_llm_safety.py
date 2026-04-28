"""Analyze RD011 LLM logs and recommend safer model routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


PROVIDERS = ("google", "groq", "mistral", "openrouter")

SCHEMA_TO_TASK = {
    "ExtractionResult": "large_context",
    "DocumentPlan": "reasoning",
    "IssueReport": "reasoning",
    "IntroContent": "generation",
    "SectionContent": "generation",
}


@dataclass
class ModelStats:
    model: str
    attempts: int = 0
    successes: int = 0
    skipped: int = 0
    skip_rate_limit: int = 0
    skip_bad_request: int = 0
    skip_other: int = 0
    failed: int = 0
    fail_503: int = 0
    fail_validation_or_json: int = 0
    policy_skips: int = 0
    rpd_hits: int = 0
    tpm_hits: int = 0
    tpd_hits: int = 0
    http_429: int = 0
    http_503: int = 0
    http_404: int = 0
    schemas_success: Counter = field(default_factory=Counter)
    tasks_attempts: Counter = field(default_factory=Counter)
    tasks_success: Counter = field(default_factory=Counter)

    def success_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return self.successes / self.attempts

    def penalty_points(self) -> int:
        return (
            8 * self.rpd_hits
            + 6 * self.tpm_hits
            + 6 * self.tpd_hits
            + 4 * self.skip_rate_limit
            + 3 * self.fail_503
            + 2 * self.fail_validation_or_json
            + 2 * self.skip_bad_request
            + self.failed
        )

    def stability_score(self) -> float:
        base = 40.0
        base += min(30.0, self.successes * 1.2)
        base += self.success_rate() * 30.0
        score = base - float(self.penalty_points())
        return max(0.0, min(100.0, score))



def normalize_model_id(raw: str) -> str:
    value = (raw or "").strip().rstrip(".")
    if not value:
        return value

    parts = value.split("/")
    if len(parts) >= 3 and parts[0] in {"large_context", "reasoning", "generation"} and parts[1] in PROVIDERS:
        return parts[1] + "/" + "/".join(parts[2:])
    if len(parts) >= 3 and parts[0] not in PROVIDERS and parts[1] in PROVIDERS:
        return parts[1] + "/" + "/".join(parts[2:])

    if "/" in value and value.split("/", 1)[0] in PROVIDERS:
        return value

    for provider in PROVIDERS:
        prefix = provider + "-"
        if value.startswith(prefix):
            return provider + "/" + value[len(prefix) :]

    if value.startswith("gemini"):
        return "google/" + value
    if value.startswith("llama") or value.startswith("meta-") or value.startswith("qwen"):
        return "groq/" + value

    return value



def get_stats(store: Dict[str, ModelStats], model_id: str) -> ModelStats:
    if model_id not in store:
        store[model_id] = ModelStats(model=model_id)
    return store[model_id]



def between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        return ""
    return text[i + len(start) : j]



def parse_log(log_path: Path) -> Dict[str, ModelStats]:
    stats: Dict[str, ModelStats] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if "Cascade attempt" in line and " for " in line and ": trying " in line:
                tail = line.split(" for ", 1)[1]
                task, model_id_raw = tail.split(": trying ", 1)
                model_id = normalize_model_id(model_id_raw)
                st = get_stats(stats, model_id)
                st.attempts += 1
                st.tasks_attempts[task] += 1
                continue

            if "Successfully generated " in line and " via " in line:
                tail = line.split("Successfully generated ", 1)[1]
                schema, model_id_raw = tail.split(" via ", 1)
                model_id = normalize_model_id(model_id_raw)
                st = get_stats(stats, model_id)
                st.successes += 1
                st.schemas_success[schema] += 1
                task = SCHEMA_TO_TASK.get(schema)
                if task:
                    st.tasks_success[task] += 1
                continue

            if "Model " in line and " skipped (" in line:
                tail = line.split("Model ", 1)[1]
                model_id_raw, reason_tail = tail.split(" skipped (", 1)
                reason = reason_tail.split(")", 1)[0].lower()
                model_id = normalize_model_id(model_id_raw)
                st = get_stats(stats, model_id)
                st.skipped += 1
                if "rate limit" in reason:
                    st.skip_rate_limit += 1
                elif "bad request" in reason:
                    st.skip_bad_request += 1
                else:
                    st.skip_other += 1
                continue

            if "Model " in line and " failed (" in line and "trying next fallback" in line:
                tail = line.split("Model ", 1)[1]
                model_id_raw, reason_tail = tail.split(" failed (", 1)
                reason = reason_tail.split(")", 1)[0].lower()
                model_id = normalize_model_id(model_id_raw)
                st = get_stats(stats, model_id)
                st.failed += 1
                if "503 unavailable" in reason or "service unavailable" in reason:
                    st.fail_503 += 1
                if "validation" in reason or "json" in reason or "expecting value" in reason or "unterminated string" in reason:
                    st.fail_validation_or_json += 1
                continue

            if "Policy skip for " in line and " on " in line and ": " in line:
                tail = line.split("Policy skip for ", 1)[1]
                _task, rest = tail.split(" on ", 1)
                model_id_raw, _reason = rest.split(": ", 1)
                model_id = normalize_model_id(model_id_raw)
                st = get_stats(stats, model_id)
                st.policy_skips += 1
                continue

            if "googleapis.com" in line and "/models/" in line and ":generateContent" in line and "HTTP/1.1 " in line:
                model = between(line, "/models/", ":generateContent")
                status_text = line.split("HTTP/1.1 ", 1)[1].strip('"')
                status_code = int(status_text.split(" ", 1)[0])
                model_id = normalize_model_id(model)
                st = get_stats(stats, model_id)
                if status_code == 429:
                    st.http_429 += 1
                elif status_code == 503:
                    st.http_503 += 1
                elif status_code == 404:
                    st.http_404 += 1
                continue

            if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in line and "model': '" in line:
                model = between(line, "model': '", "'")
                if model:
                    model_id = normalize_model_id(model)
                    st = get_stats(stats, model_id)
                    st.rpd_hits += 1
                continue

            if "Rate limit reached for model `" in line and "TP" in line:
                model = between(line, "Rate limit reached for model `", "`")
                model_id = normalize_model_id(model)
                st = get_stats(stats, model_id)
                if "(TPM)" in line:
                    st.tpm_hits += 1
                if "(TPD)" in line:
                    st.tpd_hits += 1
                continue

            if "Error calling model '" in line and "(NOT_FOUND): 404" in line:
                model = between(line, "Error calling model '", "'")
                model_id = normalize_model_id(model)
                st = get_stats(stats, model_id)
                st.http_404 += 1
                continue

    return stats



def task_success_rate(st: ModelStats, task: str) -> float:
    attempts = st.tasks_attempts.get(task, 0)
    if attempts <= 0:
        return 0.0
    return st.tasks_success.get(task, 0) / attempts



def recommend_for_task(stats: Dict[str, ModelStats], task: str, min_attempts: int) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for model_id, st in stats.items():
        if st.tasks_attempts.get(task, 0) < min_attempts:
            continue
        task_sr = task_success_rate(st, task)
        availability_penalty = float(st.rpd_hits + st.tpm_hits + st.tpd_hits + st.fail_503)
        score = (task_sr * 70.0) + (st.stability_score() * 0.30) - (availability_penalty * 2.5)
        scored.append((model_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored



def build_json(stats: Dict[str, ModelStats], min_attempts: int) -> dict:
    models = sorted(stats.values(), key=lambda s: s.stability_score(), reverse=True)

    task_recommendations = {}
    for task in ("large_context", "reasoning", "generation"):
        task_recommendations[task] = [
            {
                "model": model_id,
                "score": round(score, 3),
            }
            for model_id, score in recommend_for_task(stats, task, min_attempts)
        ]

    return {
        "summary": {
            "models_seen": len(models),
            "min_attempts": min_attempts,
        },
        "models": [
            {
                **asdict(st),
                "schemas_success": dict(st.schemas_success),
                "tasks_attempts": dict(st.tasks_attempts),
                "tasks_success": dict(st.tasks_success),
                "success_rate": round(st.success_rate(), 4),
                "stability_score": round(st.stability_score(), 3),
            }
            for st in models
        ],
        "task_recommendations": task_recommendations,
    }



def build_markdown(report: dict) -> str:
    lines: List[str] = []
    lines.append("# LLM Safety Analysis")
    lines.append("")
    lines.append(f"Models seen: {report['summary']['models_seen']}")
    lines.append(f"Minimum attempts filter: {report['summary']['min_attempts']}")
    lines.append("")

    lines.append("## Top Model Stability")
    lines.append("")
    lines.append("| Model | Stability | Success Rate | Attempts | RPD | TPM | TPD | 503 | Validation/JSON |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in report["models"][:12]:
        lines.append(
            "| {model} | {stability_score:.2f} | {success_rate:.2f} | {attempts} | {rpd_hits} | {tpm_hits} | {tpd_hits} | {fail_503} | {fail_validation_or_json} |".format(
                **model
            )
        )
    lines.append("")

    lines.append("## Task Recommendations")
    lines.append("")
    for task in ("large_context", "reasoning", "generation"):
        lines.append(f"### {task}")
        recs = report["task_recommendations"].get(task, [])
        if not recs:
            lines.append("No models passed the minimum attempts threshold.")
            lines.append("")
            continue
        for idx, rec in enumerate(recs[:5], start=1):
            lines.append(f"{idx}. {rec['model']} (score={rec['score']})")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Stability favors successful structured outputs and penalizes quota/availability pressure.")
    lines.append("- Use this output to reorder task-specific fallback arrays in config routing.")
    lines.append("- Re-run weekly or after major model/provider changes.")
    lines.append("")

    return "\n".join(lines)



def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RD011 LLM log reliability")
    parser.add_argument("--log", default="rd011_agent.log", help="Path to log file")
    parser.add_argument(
        "--json-out",
        default="outputs/llm_safety_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--md-out",
        default="outputs/llm_safety_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--min-attempts",
        type=int,
        default=2,
        help="Minimum task attempts required for task recommendations",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    stats = parse_log(log_path)
    report = build_json(stats, args.min_attempts)

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_out.write_text(build_markdown(report), encoding="utf-8")

    print(f"Wrote JSON report: {json_out}")
    print(f"Wrote markdown report: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
