#!/usr/bin/env python3
"""Materialize the latest deployable Valence refinement parameters.

The current formal CV best is:

    pmemo_valence_current_best_tonal_correction_20260707

Despite the name, the validation-selected config is still pred-only:

    Ridge(alpha=3.0), strength=0.35, cap=0.020

It is applied on top of the previous global-policy predictions. This script
fits the same policy form on the train predictions from all current-best CV
splits and saves a lightweight JSON parameter file for raw-MP3 inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_song_mean_global_policy_20260707"
OUT_DIR = ROOT / "outputs" / "valence_latest_refinement_params_20260708"
PARAMS_PATH = OUT_DIR / "pred_only_ridge_a3_strength035_cap020.json"

FEATURE_COLS = [
    "frames",
    "duration_proxy",
    "pred_mean",
    "pred_std",
    "pred_min",
    "pred_max",
    "pred_range",
    "pred_q10",
    "pred_q25",
    "pred_q75",
    "pred_q90",
    "pred_head_mean",
    "pred_tail_mean",
    "pred_slope_proxy",
    "pred_abs_delta_mean",
]
RIDGE_ALPHA = 3.0
STRENGTH = 0.35
CAP = 0.020

FORMAL_CV_REFERENCE = {
    "model_id": "pmemo_valence_current_best_tonal_correction_20260707",
    "rmse": 0.19067754835039066,
    "mae": 0.14868131614617217,
    "pcc": 0.8080536791187845,
    "ccc": 0.7961903151568043,
}


def collect_song_rows(data_split: str) -> pd.DataFrame:
    tables = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        df = pd.read_csv(split_dir / f"{data_split}_predictions.csv")
        songs = prediction_song_features(df)
        songs["split_name"] = split_dir.name
        songs["data_split"] = data_split
        tables.append(songs)
    return pd.concat(tables, ignore_index=True)


def apply_params_to_frame_df(df: pd.DataFrame, scaler: StandardScaler, model: Ridge) -> pd.DataFrame:
    songs = prediction_song_features(df)
    x = songs[FEATURE_COLS].to_numpy(dtype=np.float64)
    target_mean_pred = model.predict(scaler.transform(x))
    mean_map = dict(zip(songs["musicId"].astype(int).tolist(), target_mean_pred.tolist()))
    return apply_song_mean_shift(df, mean_map, STRENGTH, CAP)


def eval_split_predictions(scaler: StandardScaler, model: Ridge) -> dict[str, Any]:
    rows = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        item: dict[str, Any] = {"split_name": split_dir.name}
        for data_split in ["train", "val", "test"]:
            df = pd.read_csv(split_dir / f"{data_split}_predictions.csv")
            corrected = apply_params_to_frame_df(df, scaler, model)
            base_m = eval_df(df)
            corr_m = eval_df(corrected)
            for metric in ["rmse", "mae", "pcc", "ccc"]:
                item[f"baseline_{data_split}_{metric}"] = float(base_m[metric])
                item[f"corrected_{data_split}_{metric}"] = float(corr_m[metric])
        rows.append(item)
    out: dict[str, Any] = {"per_split": rows}
    for variant in ["baseline", "corrected"]:
        for data_split in ["train", "val", "test"]:
            for metric in ["rmse", "mae", "pcc", "ccc"]:
                vals = np.asarray([row[f"{variant}_{data_split}_{metric}"] for row in rows], dtype=np.float64)
                out[f"{variant}_{data_split}_{metric}_mean"] = float(vals.mean())
    corrected_test = {metric: out[f"corrected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["corrected_test_gap_rows"] = threshold_rows(corrected_test)
    out["delta_vs_formal_cv_reference"] = {
        metric: float(corrected_test[metric] - FORMAL_CV_REFERENCE[metric])
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_songs = collect_song_rows("train")
    x_train = train_songs[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_songs["target_mean"].to_numpy(dtype=np.float64)

    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    pipeline.fit(x_train, y_train)
    scaler: StandardScaler = pipeline.named_steps["standardscaler"]
    model: Ridge = pipeline.named_steps["ridge"]

    eval_summary = eval_split_predictions(scaler, model)
    params = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_latest_refinement_pred_only_ridge_a3_trainfolds_20260708",
        "formal_cv_reference": FORMAL_CV_REFERENCE,
        "input_predictions": str(INPUT_DIR.relative_to(ROOT)),
        "fit_data": "train_predictions.csv from each previous-global-policy split",
        "feature_cols": FEATURE_COLS,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_coef": model.coef_.astype(float).tolist(),
        "ridge_intercept": float(model.intercept_),
        "policy_strength": STRENGTH,
        "shift_cap": CAP,
        "label_scale": "[-1, 1]",
        "training_song_rows": int(len(train_songs)),
        "training_unique_music_ids": int(train_songs["musicId"].nunique()),
        "evaluation_note": (
            "These deployable parameters are fitted after CV policy selection for raw-MP3 inference. "
            "The formal model-selection metrics remain those in formal_cv_reference."
        ),
        "evaluation": eval_summary,
    }
    PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(params, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
