"""Consolidated module for the adaptive alpha project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        import tomllib

        return tomllib.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config support requires PyYAML. Use JSON/TOML or install pyyaml."
            ) from exc
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}

    raise ValueError(f"Unsupported config file type: {config_path}")


from pathlib import Path
from typing import Any

import pandas as pd



def write_markdown_report(
    path: str | Path,
    config: dict[str, Any],
    panel_summary: dict[str, Any],
    episodes: list[Any],
    benchmark: pd.DataFrame,
    policy_summary: pd.DataFrame,
) -> None:
    report_path = Path(path)
    ensure_dir(report_path.parent)
    lines: list[str] = []
    lines.append("# Budgeted Adaptive Falsification Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Dates: {panel_summary.get('start')} to {panel_summary.get('end')}")
    lines.append(f"- Assets: {panel_summary.get('n_assets')}")
    lines.append(f"- Episodes: {len(episodes)}")
    lines.append(f"- Benchmark rows: {len(benchmark)}")
    lines.append(f"- Candidates: {benchmark['candidate_name'].nunique() if not benchmark.empty else 0}")
    lines.append(f"- Tests: {benchmark['test_name'].nunique() if not benchmark.empty else 0}")
    lines.append("")
    lines.append("## Policy Comparison")
    lines.append("")
    if policy_summary.empty:
        lines.append("No policy results were produced.")
    else:
        cols = [
            "episode",
            "policy",
            "spent_budget",
            "survivor_precision",
            "survivor_recall",
            "false_elimination_rate",
            "cost_to_correct_rejection",
            "correct_rejections",
        ]
        present = [c for c in cols if c in policy_summary.columns]
        lines.extend(_markdown_table(policy_summary[present]))
    lines.append("")
    lines.append("## Test Predictiveness")
    lines.append("")
    if benchmark.empty:
        lines.append("No benchmark rows available.")
    else:
        grouped = benchmark.groupby("test_name").agg(
            cost=("cost", "first"),
            reject_rate=("rejected", "mean"),
            correct_rejection_rate=("target_correct_rejection", "mean"),
            false_elimination_rate=("target_false_elimination", "mean"),
            mean_evidence=("evidence_score", "mean"),
        ).reset_index()
        lines.extend(_markdown_table(grouped.sort_values("correct_rejection_rate", ascending=False)))
    lines.append("")
    lines.append("## Sealed Episodes")
    lines.append("")
    for episode in episodes:
        lines.append(
            f"- {episode.name}: search {episode.search_start.date()} to {episode.search_end.date()}, "
            f"falsification {episode.falsification_start.date()} to {episode.falsification_end.date()}, "
            f"future {episode.future_start.date()} to {episode.future_end.date()}"
        )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```text")
    lines.append(str(config))
    lines.append("```")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    rendered = frame.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) and abs(float(value)) != float("inf") else str(value))
    headers = [str(c) for c in rendered.columns]
    rows = [[str(value) for value in row] for row in rendered.to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    out = []
    out.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    out.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return out
