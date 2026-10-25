#!/usr/bin/env python3
"""Validation-safe prediction-only rescale on targeted Valence output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_targeted_split02_song_mean_20260628"
OUT_DIR = ROOT / "outputs" / "valence_targeted_rescale_valsafe_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_targeted_split02_song_mean_20260628",
    "rmse": 0.19294316758334462,
    "mae": 0.1505594667920047,
    "pcc": 0.8020795285316588,
    "ccc": 0.7851244208803636,
}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df[pred_col(df)].to_numpy(dtype=np.float64), df[target_col(df)].to_numpy(dtype=np.float64))


def transform(df: pd.DataFrame, mean_gain: float, std_gain: float, blend: float) -> pd.DataFrame:
    out = df.copy()
    pcol = pred_col(out)
    pred = out[pcol].to_numpy(dtype=np.float64)
    song_mean = out.groupby("musicId")[pcol].transform("mean").to_numpy(dtype=np.float64)
    shifted = mean_gain * song_mean + std_gain * (pred - song_mean)
    out[pcol] = (1.0 - blend) * pred + blend * shifted
    out["targeted_rescale_mean_gain"] = mean_gain
    out["targeted_rescale_std_gain"] = std_gain
    out["targeted_rescale_blend"] = blend
    return out


def pass_val_filter(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        candidate["rmse"] <= baseline["rmse"]
        and candidate["mae"] <= baseline["mae"]
        and candidate["pcc"] >= baseline["pcc"]
        and candidate["ccc"] >= baseline["ccc"]
    )


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.20 * m["mae"] - 0.03 * m["ccc"] - 0.01 * m["pcc"]


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([r[variant]["test"][metric] for r in rows], dtype=np.float64)
        out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
        out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def evaluate_split(split_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache: dict[tuple[str, float, float, float], pd.DataFrame] = {}
    base_row = {
        "split_name": split_dir.name,
        "mean_gain": 1.0,
        "std_gain": 1.0,
        "blend": 0.0,
        **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
        "passes_val_filter": True,
        "val_selection_score": selection_score(baseline["val"]),
    }
    rows.append(base_row)
    for split in ["train", "val", "test"]:
        cache[(split, 1.0, 1.0, 0.0)] = frames[split].copy()
    for mean_gain in np.round(np.arange(0.90, 1.101, 0.025), 3):
        for std_gain in np.round(np.arange(0.40, 1.101, 0.025), 3):
            for blend in np.round(np.arange(0.10, 1.001, 0.10), 2):
                split_metrics = {}
                for split in ["train", "val", "test"]:
                    out = transform(frames[split], float(mean_gain), float(std_gain), float(blend))
                    split_metrics[split] = eval_df(out)
                    cache[(split, float(mean_gain), float(std_gain), float(blend))] = out
                rows.append(
                    {
                        "split_name": split_dir.name,
                        "mean_gain": float(mean_gain),
                        "std_gain": float(std_gain),
                        "blend": float(blend),
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "passes_val_filter": pass_val_filter(split_metrics["val"], baseline["val"]),
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    result_df = pd.DataFrame(rows)
    valid = result_df[result_df["passes_val_filter"] == True].copy()
    valid = valid.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = valid.iloc[0].to_dict()
    out_split = OUT_DIR / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)
    key = (float(selected["mean_gain"]), float(selected["std_gain"]), float(selected["blend"]))
    for split in ["train", "val", "test"]:
        cache[(split, key[0], key[1], key[2])].to_csv(out_split / f"{split}_predictions.csv", index=False)
    selected_metrics = {
        split: {metric: float(selected[f"{split}_{metric}"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
        for split in ["train", "val", "test"]
    }
    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "valid_candidates": int(len(valid)),
        "selected_config": {"mean_gain": key[0], "std_gain": key[1], "blend": key[2]},
        "selected": selected_metrics,
        "delta_vs_input_test": {
            metric: float(selected_metrics["test"][metric] - baseline["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, result_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    results = []
    for split_dir in sorted(INPUT_DIR.glob("split_*_seed_*")):
        summary, result_df = evaluate_split(split_dir)
        rows.append(summary)
        results.append(result_df)
    pd.concat(results, ignore_index=True).to_csv(OUT_DIR / "all_targeted_rescale_results.csv", index=False)
    baseline_mean = aggregate(rows, "baseline")
    selected_mean = aggregate(rows, "selected")
    selected_metrics = {
        "rmse": selected_mean["selected_test_rmse_mean"],
        "mae": selected_mean["selected_test_mae_mean"],
        "pcc": selected_mean["selected_test_pcc_mean"],
        "ccc": selected_mean["selected_test_ccc_mean"],
    }
    summary = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_targeted_rescale_valsafe_20260628",
        "input_model": CURRENT_BEST,
        "selection_rule": "Per split, rescale candidate must not degrade validation RMSE/MAE/PCC/CCC.",
        "num_splits": len(rows),
        "per_split": rows,
        **baseline_mean,
        **selected_mean,
        "selected_mean_test_gap_rows": threshold_rows(selected_metrics),
        "delta_vs_input_test_mean": {
            metric: float(selected_metrics[metric] - CURRENT_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
