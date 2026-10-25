#!/usr/bin/env python3
"""Train+val refit for the validation-selected HGB Valence song-mean policy."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from search_pmemo_valence_correlated_song_mean import PRED_COLS, load_static_song_table, macro, select_audio_cols
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from search_pmemo_valence_temporal_shift import apply_shift
from search_pmemo_valence_current_best_smoothing import apply_smoothing
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
OUT_DIR = ROOT / "outputs" / "valence_hgb_trainval_refit_shift_smoothing_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}
CONFIG = {
    "model_id": "hgb_l2_001_leaf7_lr003",
    "target_mode": "residual_mean",
    "top_k": 64,
    "strength": 0.80,
    "cap": None,
    "temporal_shift_frames": -9,
    "smoothing_method": "ma",
    "smoothing_param": 21.0,
    "smoothing_blend": 1.0,
}


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=12,
        l2_regularization=0.01,
        random_state=17,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(s, d): pd.read_csv(INPUT_DIR / s / f"{d}_predictions.csv") for s in SPLITS for d in DATA_SPLITS}
    song_tables = {(s, d): prediction_song_features(frames[(s, d)]) for s in SPLITS for d in DATA_SPLITS}
    for table in song_tables.values():
        table["base_residual_mean"] = table["target_mean"].to_numpy(dtype=np.float64) - table["pred_mean"].to_numpy(dtype=np.float64)
    all_ids = sorted({int(sid) for table in song_tables.values() for sid in table["musicId"].tolist()})
    audio = pd.DataFrame({"musicId": all_ids}).merge(load_static_song_table(), on="musicId", how="left").fillna(0.0)
    audio_cols = [c for c in audio.columns if c != "musicId"]
    merged = {(s, d): song_tables[(s, d)].merge(audio, on="musicId", how="left").fillna(0.0) for s in SPLITS for d in DATA_SPLITS}

    per_split: list[dict[str, Any]] = []
    for split_name in SPLITS:
        train_val = pd.concat([merged[(split_name, "train")], merged[(split_name, "val")]], ignore_index=True)
        selected_cols = select_audio_cols(train_val, audio_cols, "base_residual_mean", int(CONFIG["top_k"]))
        cols = PRED_COLS + selected_cols
        reg = model()
        reg.fit(train_val[cols].to_numpy(dtype=np.float64), train_val["base_residual_mean"].to_numpy(dtype=np.float64))
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name, "selected_audio_cols": selected_cols}
        for data_split in DATA_SPLITS:
            table = merged[(split_name, data_split)]
            residual = reg.predict(table[cols].to_numpy(dtype=np.float64))
            pred_target_mean = table["pred_mean"].to_numpy(dtype=np.float64) + residual
            target_map = dict(zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist()))
            corrected = apply_song_mean_shift(frames[(split_name, data_split)], target_map, float(CONFIG["strength"]), None)
            shifted = apply_shift(corrected, int(CONFIG["temporal_shift_frames"]))
            smoothed = apply_smoothing(
                shifted,
                str(CONFIG["smoothing_method"]),
                float(CONFIG["smoothing_param"]),
                float(CONFIG["smoothing_blend"]),
            )
            smoothed.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(smoothed).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    selected_val = macro(per_split, "val")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_hgb_trainval_refit_shift_smoothing_20260708",
        "selection_rule": "final refit only: HGB hyperparameters fixed from prior validation selection; correction refit on train+val; test labels unused",
        "fixed_config_from_prior_validation": CONFIG,
        "current_best_reference": CURRENT_BEST,
        "selected_val_after_refit_diagnostic": selected_val,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
