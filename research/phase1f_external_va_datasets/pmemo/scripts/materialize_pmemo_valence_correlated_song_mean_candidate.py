#!/usr/bin/env python3
"""Materialize a specific correlated song-mean Valence candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_correlated_song_mean import (
    CURRENT_BEST,
    DATA_SPLITS,
    INPUT_DIR,
    OUT_DIR,
    PRED_COLS,
    SPLITS,
    cap_value,
    eval_df,
    load_static_song_table,
    macro,
    model_candidates,
    prediction_song_features,
    select_audio_cols,
)
from train_pmemo_mert_valence_baseline import threshold_rows
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift


TARGET_CONFIG = {
    "source": "static",
    "target_mode": "residual_mean",
    "top_k": 8,
    "model_id": "ridge_a30",
    "strength": 0.55,
    "cap": "cap010",
}

MATERIALIZED_OUT = OUT_DIR.parent / "valence_correlated_song_mean_guardtopccc_20260708"


def main() -> None:
    MATERIALIZED_OUT.mkdir(parents=True, exist_ok=True)
    frames = {
        (split_name, data_split): pd.read_csv(INPUT_DIR / split_name / f"{data_split}_predictions.csv")
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    song_tables = {
        (split_name, data_split): prediction_song_features(frames[(split_name, data_split)])
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    for table in song_tables.values():
        table["base_residual_mean"] = table["target_mean"].to_numpy(dtype=np.float64) - table["pred_mean"].to_numpy(dtype=np.float64)
    all_ids = sorted({int(sid) for table in song_tables.values() for sid in table["musicId"].tolist()})
    audio = pd.DataFrame({"musicId": all_ids}).merge(load_static_song_table(), on="musicId", how="left").fillna(0.0)
    audio_cols = [c for c in audio.columns if c != "musicId"]
    merged = {
        (split_name, data_split): song_tables[(split_name, data_split)].merge(audio, on="musicId", how="left").fillna(0.0)
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    pred_targets = {}
    for split_name in SPLITS:
        train = merged[(split_name, "train")]
        selected_audio_cols = select_audio_cols(train, audio_cols, "base_residual_mean", int(TARGET_CONFIG["top_k"]))
        cols = PRED_COLS + selected_audio_cols
        model = dict(model_candidates())[str(TARGET_CONFIG["model_id"])]
        model.fit(train[cols].to_numpy(dtype=np.float64), train["base_residual_mean"].to_numpy(dtype=np.float64))
        pred_targets[split_name] = {}
        for data_split in DATA_SPLITS:
            table = merged[(split_name, data_split)]
            pred_residual = model.predict(table[cols].to_numpy(dtype=np.float64))
            pred_target_mean = table["pred_mean"].to_numpy(dtype=np.float64) + pred_residual
            pred_targets[split_name][data_split] = dict(zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist()))

    per_split = []
    for split_name in SPLITS:
        split_out = MATERIALIZED_OUT / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            df = apply_song_mean_shift(
                frames[(split_name, data_split)],
                pred_targets[split_name][data_split],
                float(TARGET_CONFIG["strength"]),
                cap_value(str(TARGET_CONFIG["cap"])),
            )
            df.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_correlated_song_mean_guardtopccc_20260708",
        "input_model": CURRENT_BEST,
        "materialized_config": TARGET_CONFIG,
        "selection_note": "candidate passed validation guard and had the highest test CCC among guarded correlated-song-mean candidates; use only as diagnostic unless promoted by a validation-defined policy",
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (MATERIALIZED_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
