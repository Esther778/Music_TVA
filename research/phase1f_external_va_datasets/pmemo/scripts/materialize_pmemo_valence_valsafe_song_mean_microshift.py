#!/usr/bin/env python3
"""Materialize validation-safe song-mean microshift for PMEmo Valence.

The fixed candidate was selected from the existing song-mean fusion search by
validation-only non-degradation:

- feature_set: pred_only
- model_id: rf_pca32
- strength: 1.0
- cap: 0.010

It predicts a conservative song-level mean from the current prediction curve and
shifts each frame by at most 0.01.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_song_mean_fusion import (
    CURRENT_BEST,
    INPUT_DIR,
    ROOT,
    apply_song_mean_shift,
    eval_df,
    model_candidates,
    prediction_song_features,
)
from train_pmemo_mert_valence_baseline import threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_valsafe_song_mean_microshift_20260628"
FEATURE_SET = "pred_only"
MODEL_ID = "rf_pca32"
STRENGTH = 1.0
CAP_VALUE = 0.010
CAP_NAME = "cap010"


def select_model(train_rows: int, feature_dim: int):
    for model_id, model in model_candidates(train_rows, feature_dim):
        if model_id == MODEL_ID:
            return model
    raise ValueError(f"Model candidate not found: {MODEL_ID}")


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([row[variant]["test"][metric] for row in rows], dtype=np.float64)
        out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
        out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def materialize_split(split_dir: Path) -> dict[str, Any]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    song_tables = {split: prediction_song_features(frames[split]) for split in ["train", "val", "test"]}
    feature_cols = [
        "frames",
        "duration_proxy",
        "pred_mean",
        "pred_std",
        "pred_min",
        "pred_max",
        "pred_range",
        "pred_q10",
        "pred_q90",
        "pred_head_mean",
        "pred_tail_mean",
    ]

    x_train = song_tables["train"][feature_cols].to_numpy(dtype=np.float64)
    y_train = song_tables["train"]["target_mean"].to_numpy(dtype=np.float64)
    model = select_model(len(x_train), x_train.shape[1])
    model.fit(x_train, y_train)
    pred_means = {
        split: dict(
            zip(
                song_tables[split]["musicId"].astype(int).tolist(),
                model.predict(song_tables[split][feature_cols].to_numpy(dtype=np.float64)),
            )
        )
        for split in ["train", "val", "test"]
    }

    corrected = {
        split: apply_song_mean_shift(frames[split], pred_means[split], STRENGTH, CAP_VALUE)
        for split in ["train", "val", "test"]
    }
    selected = {split: eval_df(corrected[split]) for split in ["train", "val", "test"]}

    split_out = OUT_DIR / split_dir.name
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        corrected[split]["song_mean_microshift_feature_set"] = FEATURE_SET
        corrected[split]["song_mean_microshift_model"] = MODEL_ID
        corrected[split]["song_mean_microshift_strength"] = STRENGTH
        corrected[split]["song_mean_microshift_cap"] = CAP_VALUE
        corrected[split].to_csv(split_out / f"{split}_predictions.csv", index=False)

    return {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected,
        "selected_config": {
            "feature_set": FEATURE_SET,
            "model_id": MODEL_ID,
            "strength": STRENGTH,
            "cap": CAP_NAME,
        },
        "delta_vs_input_test": {
            metric: float(selected["test"][metric] - baseline["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "delta_vs_input_val": {
            metric: float(selected["val"][metric] - baseline["val"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [materialize_split(split_dir) for split_dir in sorted(INPUT_DIR.glob("split_*_seed_*"))]
    baseline_mean = aggregate(rows, "baseline")
    selected_mean = aggregate(rows, "selected")
    selected_metrics = {
        "rmse": selected_mean["selected_test_rmse_mean"],
        "mae": selected_mean["selected_test_mae_mean"],
        "pcc": selected_mean["selected_test_pcc_mean"],
        "ccc": selected_mean["selected_test_ccc_mean"],
    }
    current_best = {
        "rmse": CURRENT_BEST["rmse"],
        "mae": CURRENT_BEST["mae"],
        "pcc": CURRENT_BEST["pcc"],
        "ccc": CURRENT_BEST["ccc"],
    }
    summary = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_valsafe_song_mean_microshift_20260628",
        "input_model": CURRENT_BEST,
        "selection_basis": "Existing search; fixed candidate selected by validation-only non-degradation across RMSE/MAE/PCC/CCC.",
        "fixed_config": {
            "feature_set": FEATURE_SET,
            "model_id": MODEL_ID,
            "strength": STRENGTH,
            "cap": CAP_NAME,
        },
        "num_splits": len(rows),
        "per_split": rows,
        **baseline_mean,
        **selected_mean,
        "selected_mean_test_gap_rows": threshold_rows(selected_metrics),
        "delta_vs_current_best_test_mean": {
            metric: float(selected_metrics[metric] - current_best[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
