#!/usr/bin/env python3
"""Nested-style evaluation for universal PMEmo Valence song-mean correction.

For each held-out split:
1. Select one fixed correction configuration using validation metrics from the
   other two splits only.
2. Report test metrics on the held-out split.

This estimates whether a universal correction configuration generalizes without
choosing it from the same split's test result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


CANDIDATE_TABLE = ROOT / "outputs" / "valence_song_mean_valsafe_selection_20260628" / "all_valsafe_song_mean_results.csv"
VALSAFE_DIR = ROOT / "outputs" / "valence_song_mean_valsafe_selection_20260628"
BASE_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
OUT_DIR = ROOT / "outputs" / "valence_universal_nested_selection_20260628"

BASELINE = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df[pred_col(df)].to_numpy(dtype=np.float64), df[target_col(df)].to_numpy(dtype=np.float64))


def score(row: pd.Series) -> float:
    """Validation-only score prioritizing pass-threshold gaps and CCC."""
    return float(
        row["val_rmse"]
        + 0.20 * row["val_mae"]
        + 0.35 * max(0.0, 0.80 - row["val_ccc"])
        + 0.12 * max(0.0, 0.80 - row["val_pcc"])
        - 0.02 * row["val_ccc"]
    )


def config_key_cols() -> list[str]:
    return ["feature_set", "model_id", "strength", "cap"]


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATE_TABLE)
    df = df.dropna(subset=["val_rmse", "val_mae", "val_pcc", "val_ccc", "test_rmse", "test_mae", "test_pcc", "test_ccc"])
    # Keep configurations that exist for all splits.
    counts = df.groupby(config_key_cols(), dropna=False)["split_name"].nunique().reset_index(name="split_count")
    df = df.merge(counts, on=config_key_cols(), how="left")
    return df[df["split_count"] == 3].copy()


def select_config(train_val_df: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        train_val_df.groupby(config_key_cols(), dropna=False)
        .agg(
            val_rmse=("val_rmse", "mean"),
            val_mae=("val_mae", "mean"),
            val_pcc=("val_pcc", "mean"),
            val_ccc=("val_ccc", "mean"),
        )
        .reset_index()
    )
    grouped["selection_score"] = grouped.apply(score, axis=1)
    selected = grouped.sort_values(["selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False]).iloc[0]
    return selected.to_dict()


def matching_row(df: pd.DataFrame, split_name: str, cfg: dict[str, Any]) -> pd.Series:
    work = df[df["split_name"] == split_name].copy()
    for col in config_key_cols():
        work = work[work[col].astype(str) == str(cfg[col])]
    if len(work) != 1:
        raise ValueError(f"Expected one row for {split_name} and {cfg}, got {len(work)}")
    return work.iloc[0]


def split_dir_for_config(split_name: str, cfg: dict[str, Any]) -> Path:
    if str(cfg["feature_set"]) == "baseline" and str(cfg["model_id"]) == "no_change":
        return BASE_DIR / split_name
    return VALSAFE_DIR / split_name


def copy_predictions(split_name: str, cfg: dict[str, Any], out_split_dir: Path) -> None:
    src = split_dir_for_config(split_name, cfg)
    out_split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        shutil.copy2(src / f"{split}_predictions.csv", out_split_dir / f"{split}_predictions.csv")


def aggregate_test(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([r[prefix][metric] for r in rows], dtype=np.float64)
        out[f"{prefix}_{metric}_mean"] = float(vals.mean())
        out[f"{prefix}_{metric}_std"] = float(vals.std(ddof=0))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    split_names = sorted(candidates["split_name"].unique().tolist())
    rows: list[dict[str, Any]] = []
    selected_configs = []
    for holdout in split_names:
        train_val = candidates[candidates["split_name"] != holdout]
        cfg = select_config(train_val)
        selected_configs.append({"holdout_split": holdout, **cfg})
        cand_row = matching_row(candidates, holdout, cfg)
        baseline_row = matching_row(
            candidates,
            holdout,
            {"feature_set": "baseline", "model_id": "no_change", "strength": 0.0, "cap": "none"},
        )
        out_split = OUT_DIR / holdout
        copy_predictions(holdout, cfg, out_split)
        selected_metrics = {
            metric: float(cand_row[f"test_{metric}"])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        }
        baseline_metrics = {
            metric: float(baseline_row[f"test_{metric}"])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        }
        rows.append(
            {
                "holdout_split": holdout,
                "selected_config": {k: cfg[k] for k in config_key_cols()},
                "selection_val_metrics_on_other_splits": {
                    "rmse": float(cfg["val_rmse"]),
                    "mae": float(cfg["val_mae"]),
                    "pcc": float(cfg["val_pcc"]),
                    "ccc": float(cfg["val_ccc"]),
                    "score": float(cfg["selection_score"]),
                },
                "baseline_test": baseline_metrics,
                "selected_test": selected_metrics,
                "delta_vs_baseline_test": {
                    metric: float(selected_metrics[metric] - baseline_metrics[metric])
                    for metric in ["rmse", "mae", "pcc", "ccc"]
                },
            }
        )
    selected_summary = aggregate_test(rows, "selected_test")
    baseline_summary = aggregate_test(rows, "baseline_test")
    selected_metrics_mean = {
        "rmse": selected_summary["selected_test_rmse_mean"],
        "mae": selected_summary["selected_test_mae_mean"],
        "pcc": selected_summary["selected_test_pcc_mean"],
        "ccc": selected_summary["selected_test_ccc_mean"],
    }
    summary = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_universal_nested_selection_20260628",
        "selection_protocol": "leave-one-split-out; choose fixed config by validation metrics from the other two splits; report held-out split test",
        "baseline_model": BASELINE,
        "candidate_table": str(CANDIDATE_TABLE.relative_to(ROOT)),
        "num_holdouts": len(rows),
        "selected_configs": selected_configs,
        "per_holdout": rows,
        **baseline_summary,
        **selected_summary,
        "selected_mean_test_gap_rows": threshold_rows(selected_metrics_mean),
        "delta_vs_baseline_mean": {
            metric: float(selected_metrics_mean[metric] - BASELINE[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    pd.DataFrame(selected_configs).to_csv(OUT_DIR / "selected_configs.csv", index=False)
    pd.DataFrame(
        [
            {
                "holdout_split": row["holdout_split"],
                **{f"config_{k}": v for k, v in row["selected_config"].items()},
                **{f"baseline_test_{k}": v for k, v in row["baseline_test"].items()},
                **{f"selected_test_{k}": v for k, v in row["selected_test"].items()},
                **{f"delta_test_{k}": v for k, v in row["delta_vs_baseline_test"].items()},
            }
            for row in rows
        ]
    ).to_csv(OUT_DIR / "holdout_results.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
