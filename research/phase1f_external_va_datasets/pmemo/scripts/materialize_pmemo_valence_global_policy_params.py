#!/usr/bin/env python3
"""Materialize deployable pred-only Valence global policy parameters.

The metric-best global policy selected the configuration
`pred_only + Ridge(alpha=1.0) + strength=0.5 + cap=0.070`.

For raw-MP3 deployment we need a concrete fitted Ridge model. This script fits
that policy form on the current best train predictions only and saves the
StandardScaler/Ridge parameters as JSON for lightweight inference.
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

import search_pmemo_valence_song_mean_fusion as song_mean
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_parsimony_output_guarded_20260707"
OUT_DIR = ROOT / "outputs" / "valence_global_policy_params_20260707"
PARAMS_PATH = OUT_DIR / "pred_only_ridge_a1_strength050_cap070.json"

FEATURE_COLS = [
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
STRENGTH = 0.5
CAP = 0.070


def collect_song_rows(data_split: str) -> pd.DataFrame:
    tables = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        df = pd.read_csv(split_dir / f"{data_split}_predictions.csv")
        songs = song_mean.prediction_song_features(df)
        songs["split_name"] = split_dir.name
        songs["data_split"] = data_split
        tables.append(songs)
    return pd.concat(tables, ignore_index=True)


def apply_params_to_frame_df(df: pd.DataFrame, scaler: StandardScaler, model: Ridge) -> pd.DataFrame:
    songs = song_mean.prediction_song_features(df)
    x = songs[FEATURE_COLS].to_numpy(dtype=np.float64)
    target_mean_pred = model.predict(scaler.transform(x))
    mean_map = dict(zip(songs["musicId"].astype(int).tolist(), target_mean_pred.tolist()))
    return song_mean.apply_song_mean_shift(df, mean_map, STRENGTH, CAP)


def eval_split_predictions(scaler: StandardScaler, model: Ridge) -> dict[str, Any]:
    rows = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        item: dict[str, Any] = {"split_name": split_dir.name}
        for data_split in ["train", "val", "test"]:
            df = pd.read_csv(split_dir / f"{data_split}_predictions.csv")
            corrected = apply_params_to_frame_df(df, scaler, model)
            base_m = metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))
            corr_m = metrics(
                corrected["pred_valence"].to_numpy(dtype=np.float64),
                corrected["target_valence"].to_numpy(dtype=np.float64),
            )
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
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_songs = collect_song_rows("train")
    x_train = train_songs[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_songs["target_mean"].to_numpy(dtype=np.float64)

    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipeline.fit(x_train, y_train)
    scaler: StandardScaler = pipeline.named_steps["standardscaler"]
    model: Ridge = pipeline.named_steps["ridge"]

    eval_summary = eval_split_predictions(scaler, model)
    params = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_global_policy_pred_only_ridge_a1_trainfolds_20260707",
        "input_predictions": str(INPUT_DIR.relative_to(ROOT)),
        "fit_data": "train_predictions.csv from each current-best split",
        "feature_cols": FEATURE_COLS,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "ridge_alpha": 1.0,
        "ridge_coef": model.coef_.astype(float).tolist(),
        "ridge_intercept": float(model.intercept_),
        "policy_strength": STRENGTH,
        "shift_cap": CAP,
        "label_scale": "[-1, 1]",
        "training_song_rows": int(len(train_songs)),
        "training_unique_music_ids": int(train_songs["musicId"].nunique()),
        "evaluation_note": (
            "These deployable parameters are fitted after policy selection for raw-MP3 inference. "
            "Use CV summary metrics from pmemo_valence_current_best_song_mean_global_policy_20260707 for formal model selection."
        ),
        "evaluation": eval_summary,
    }
    PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(params, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
