#!/usr/bin/env python3
"""RMSE-aware song-level bias correction for PMEmo Valence predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_song_bias_correction import (
    INPUT_DIR,
    PREVIOUS_BEST,
    apply_bias,
    eval_df,
    fit_and_predict,
    song_features,
)
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_song_bias_correction_rmse_aware_20260628"
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.00 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.00 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.00 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.00 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def evaluate_split(split_dir: str) -> tuple[dict, pd.DataFrame]:
    frames = {split: pd.read_csv(INPUT_DIR / split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    song_stats = {split: song_features(frames[split]) for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache = {}
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        pred_bias = {split: fit_and_predict(song_stats["train"], song_stats[split], alpha) for split in ["train", "val", "test"]}
        for strength in [0.25, 0.50, 0.75, 1.00]:
            for cap_name, cap in [("cap004", 0.04), ("cap008", 0.08), ("cap012", 0.12), ("cap018", 0.18), ("none", None)]:
                split_metrics = {}
                for split in ["train", "val", "test"]:
                    corrected = apply_bias(frames[split], pred_bias[split], strength, cap)
                    split_metrics[split] = eval_df(corrected)
                    cache[(split, alpha, strength, cap_name)] = corrected
                rows.append(
                    {
                        "split_name": split_dir,
                        "alpha": alpha,
                        "strength": strength,
                        "cap": cap_name,
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = df.iloc[0].to_dict()
    split_out = OUT_DIR / selected["split_name"]
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        cache[(split, float(selected["alpha"]), float(selected["strength"]), str(selected["cap"]))].to_csv(split_out / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    return summary, df


def aggregate(rows: list[dict]) -> dict:
    out = {"created_at": "2026-06-28", "previous_best": PREVIOUS_BEST, "objective": "rmse_mae_aware", "num_splits": len(rows), "per_split": rows}
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["baseline"]["test"][metric] for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_previous_best"] = {metric: float(selected_mean[metric] - PREVIOUS_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p.name for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_bias_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
