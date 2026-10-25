#!/usr/bin/env python3
"""Materialize a validation CCC-guarded Ridge song-mean Valence candidate."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_correlated_song_mean import PRED_COLS, cap_value, load_static_song_table, macro, select_audio_cols
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_second_residual_smoothing_20260709"
OUT_DIR = ROOT / "outputs" / "valence_latest_ridge_ccc_guard_20260709"
SOURCE_CANDIDATES = ROOT / "outputs" / "valence_latest_ridge_song_mean_fast_20260709" / "all_ridge_song_mean_candidates.csv"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_smoothing_20260708",
    "rmse": 0.1878117801132452,
    "mae": 0.14589902068622534,
    "pcc": 0.8140948778047831,
    "ccc": 0.8013555689217714,
}


def select_config() -> dict[str, Any]:
    df = pd.read_csv(SOURCE_CANDIDATES)
    baseline = df.sort_values(["val_rmse", "val_ccc"], ascending=[True, False]).iloc[0]
    guarded = df[
        (df["val_ccc"] >= float(baseline["val_ccc"]) + 0.00025)
        & (df["val_rmse"] <= float(baseline["val_rmse"]) + 0.00040)
        & (df["val_mae"] <= float(baseline["val_mae"]) + 0.00040)
    ].copy()
    if guarded.empty:
        selected = baseline
        reason = "fallback to baseline; no candidate passed validation CCC guard"
    else:
        selected = guarded.sort_values(["val_rmse", "val_ccc"], ascending=[True, False]).iloc[0]
        reason = "selected by validation CCC guard then lowest validation RMSE"
    return {
        "target_col": str(selected["target_col"]),
        "top_k": int(selected["top_k"]),
        "alpha": float(selected["alpha"]),
        "strength": float(selected["strength"]),
        "cap": str(selected["cap"]),
        "selection_reason": reason,
        "source_validation_row": {k: float(selected[k]) if isinstance(selected[k], (int, float, np.number)) else str(selected[k]) for k in selected.index},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = select_config()
    frames = {(s, d): pd.read_csv(INPUT_DIR / s / f"{d}_predictions.csv") for s in SPLITS for d in DATA_SPLITS}
    song_tables = {(s, d): prediction_song_features(frames[(s, d)]) for s in SPLITS for d in DATA_SPLITS}
    for table in song_tables.values():
        table["base_residual_mean"] = table["target_mean"].to_numpy(dtype=np.float64) - table["pred_mean"].to_numpy(dtype=np.float64)
    all_ids = sorted({int(sid) for table in song_tables.values() for sid in table["musicId"].tolist()})
    audio = pd.DataFrame({"musicId": all_ids}).merge(load_static_song_table(), on="musicId", how="left").fillna(0.0)
    audio_cols = [c for c in audio.columns if c != "musicId"]
    merged = {(s, d): song_tables[(s, d)].merge(audio, on="musicId", how="left").fillna(0.0) for s in SPLITS for d in DATA_SPLITS}

    rows = []
    for split_name in SPLITS:
        cols = PRED_COLS + select_audio_cols(merged[(split_name, "train")], audio_cols, config["target_col"], int(config["top_k"]))
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(config["alpha"])))
        train = merged[(split_name, "train")]
        model.fit(train[cols].to_numpy(dtype=np.float64), train[config["target_col"]].to_numpy(dtype=np.float64))
        row: dict[str, Any] = {"split_name": split_name}
        split_dir = OUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            table = merged[(split_name, data_split)]
            pred = model.predict(table[cols].to_numpy(dtype=np.float64))
            if config["target_col"] == "base_residual_mean":
                pred_target_mean = table["pred_mean"].to_numpy(dtype=np.float64) + pred
            else:
                pred_target_mean = pred
            target_map = dict(zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist()))
            corrected = apply_song_mean_shift(
                frames[(split_name, data_split)],
                target_map,
                float(config["strength"]),
                cap_value(config["cap"]),
            )
            corrected.to_csv(split_dir / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
        rows.append(row)

    selected_val = macro(rows, "val")
    selected_test = macro(rows, "test")
    summary = {
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_latest_ridge_ccc_guard_20260709",
        "input_model": CURRENT_BEST,
        "selection_rule": (
            "from the Ridge candidate table, require validation CCC improvement over validation-RMSE baseline "
            "and near-equal validation RMSE/MAE, then choose lowest validation RMSE; test labels unused"
        ),
        "selected_config": config,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
        "per_split": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
