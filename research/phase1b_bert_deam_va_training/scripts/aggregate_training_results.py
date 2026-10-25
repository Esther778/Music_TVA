#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training_utils import ROOT, metrics_np, prefixed_metrics


FIELDS = [
    "model_family",
    "trial_id",
    "best_epoch",
    "best_val_rmse",
    "parameters",
    "train_rmse",
    "train_mae",
    "train_rmse_valence",
    "train_rmse_arousal",
    "train_ccc_valence",
    "train_ccc_arousal",
    "train_mean_ccc",
    "val_rmse",
    "val_mae",
    "val_rmse_valence",
    "val_rmse_arousal",
    "val_ccc_valence",
    "val_ccc_arousal",
    "val_mean_ccc",
    "test_rmse",
    "test_mae",
    "test_rmse_valence",
    "test_rmse_arousal",
    "test_pearson_valence",
    "test_pearson_arousal",
    "test_ccc_valence",
    "test_ccc_arousal",
    "test_mean_ccc",
    "train_val_rmse_gap",
    "checkpoint",
    "config_json",
]


def load_targets(model_family: str, split: str) -> np.ndarray:
    dataset_root = ROOT / "data" / "datasets" / "mert_va"
    if model_family == "mlp":
        return np.load(dataset_root / "frame" / f"Y_{split}.npy").astype(np.float32)

    index = pd.read_csv(dataset_root / "sequence_index.csv")
    rows = index[index["split"] == split].sort_values("song_id").to_dict("records")
    targets = []
    for row in rows:
        data = np.load(row["sequence_path"])
        targets.append(data["Y"].astype(np.float32))
    return np.concatenate(targets, axis=0)


def enrich_metrics(row: dict[str, Any]) -> dict[str, Any]:
    model_family = row["model_family"]
    trial_id = row["trial_id"]
    pred_dir = ROOT / "outputs" / "predictions" / model_family
    enriched = dict(row)
    for split in ["train", "val", "test"]:
        if split == "train" and model_family in {"bilstm", "transformer"}:
            continue
        pred_path = pred_dir / f"{trial_id}_{split}_pred.npy"
        if not pred_path.is_file():
            continue
        pred = np.load(pred_path).astype(np.float32)
        target = load_targets(model_family, split)
        split_metrics = metrics_np(pred, target)
        enriched.update(prefixed_metrics(split, split_metrics))
    if "train_rmse" in enriched and "val_rmse" in enriched:
        enriched["train_val_rmse_gap"] = float(enriched["val_rmse"]) - float(enriched["train_rmse"])
    return enriched


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, int, str, str]:
    return (
        float(row.get("val_rmse", float("inf"))),
        -float(row.get("val_mean_ccc", row.get("test_mean_ccc", 0.0))),
        float(row.get("train_val_rmse_gap", float("inf"))),
        int(row.get("parameters", 10**12)),
        row["model_family"],
        row["trial_id"],
    )


def load_trial_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_root = ROOT / "outputs" / "metrics"
    for path in sorted(metrics_root.glob("*/*_summary.json")):
        if path.name.endswith("_trial_summary.json"):
            continue
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        rows.append(enrich_metrics(row))
    rows.sort(key=sort_key)
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
        "Selection rule: lowest validation RMSE. Test metrics are reported only after model selection.",
        "",
        "## Best Candidate",
        "",
        f"- model_family: {best['model_family']}",
        f"- trial_id: {best['trial_id']}",
        f"- best_epoch: {best['best_epoch']}",
        f"- val_rmse: {best['val_rmse']:.6f}",
        f"- val_mean_ccc: {best.get('val_mean_ccc', 0.0):.6f}",
        f"- test_rmse: {best['test_rmse']:.6f}",
        f"- test_mean_ccc: {best.get('test_mean_ccc', 0.0):.6f}",
        f"- test_pearson_valence: {best['test_pearson_valence']:.6f}",
        f"- test_pearson_arousal: {best['test_pearson_arousal']:.6f}",
        f"- checkpoint: {best['checkpoint']}",
        "",
        "## Ranked Trials",
        "",
        "| Rank | Family | Trial | Val RMSE | Val CCC | Test RMSE | Test CCC | Test r(V) | Test r(A) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['model_family']} | {row['trial_id']} | "
            f"{row['val_rmse']:.6f} | {row.get('val_mean_ccc', 0.0):.6f} | "
            f"{row['test_rmse']:.6f} | {row.get('test_mean_ccc', 0.0):.6f} | "
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
        family_rows.sort(key=sort_key)
        write_csv(ROOT / "outputs" / "metrics" / family / f"{family}_trial_summary.csv", family_rows)

    print(json.dumps({"best_trial": rows[0]["trial_id"], "best_val_rmse": rows[0]["val_rmse"], "best_test_rmse": rows[0]["test_rmse"], "trials": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
