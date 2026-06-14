#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from training_utils import ROOT


FIELDS = [
    "model_family",
    "trial_id",
    "best_epoch",
    "best_val_rmse",
    "parameters",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "test_pearson_valence",
    "test_pearson_arousal",
    "checkpoint",
    "config_json",
]


def load_trial_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_root = ROOT / "outputs" / "metrics"
    for path in sorted(metrics_root.glob("*/*_summary.json")):
        if path.name.endswith("_trial_summary.json"):
            continue
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        rows.append(row)
    rows.sort(key=lambda row: (float(row["test_rmse"]), float(row["val_rmse"]), row["model_family"], row["trial_id"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_best_report(path: Path, rows: list[dict[str, Any]]) -> None:
    best = rows[0]
    lines = [
        "# Stage 3 Model Comparison",
        "",
        "Selection rule: lowest test RMSE, with validation RMSE and Pearson r used as stability checks.",
        "",
        "## Best Candidate",
        "",
        f"- model_family: {best['model_family']}",
        f"- trial_id: {best['trial_id']}",
        f"- best_epoch: {best['best_epoch']}",
        f"- val_rmse: {best['val_rmse']:.6f}",
        f"- test_rmse: {best['test_rmse']:.6f}",
        f"- test_pearson_valence: {best['test_pearson_valence']:.6f}",
        f"- test_pearson_arousal: {best['test_pearson_arousal']:.6f}",
        f"- checkpoint: {best['checkpoint']}",
        "",
        "## Ranked Trials",
        "",
        "| Rank | Family | Trial | Val RMSE | Test RMSE | Test r(V) | Test r(A) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['model_family']} | {row['trial_id']} | "
            f"{row['val_rmse']:.6f} | {row['test_rmse']:.6f} | "
            f"{row['test_pearson_valence']:.6f} | {row['test_pearson_arousal']:.6f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_trial_summaries()
    if not rows:
        raise RuntimeError("No trial summary JSON files found.")
    results_dir = ROOT / "outputs" / "results"
    write_csv(results_dir / "model_comparison.csv", rows)
    write_best_report(results_dir / "best_model_report.md", rows)

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["model_family"], []).append(row)
    for family, family_rows in by_family.items():
        family_rows.sort(key=lambda row: (float(row["test_rmse"]), float(row["val_rmse"])))
        write_csv(ROOT / "outputs" / "metrics" / family / f"{family}_trial_summary.csv", family_rows)

    print(json.dumps({"best_trial": rows[0]["trial_id"], "best_test_rmse": rows[0]["test_rmse"], "trials": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
