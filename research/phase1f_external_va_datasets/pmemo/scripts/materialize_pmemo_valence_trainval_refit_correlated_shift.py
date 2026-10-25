#!/usr/bin/env python3
"""Final-refit Valence song-mean correction followed by fixed temporal shift.

Hyperparameters are inherited from prior validation-selected experiments:
- static correlated residual song-mean correction:
  top_k=8, ridge_a30, strength=0.55, cap=0.010
- global temporal shift: -9 frames (-4.5s)

This script refits the song-level correction on train+val for each split, then
evaluates test. Test labels are not used for fitting or selecting parameters.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_correlated_song_mean import (
    INPUT_DIR,
    PRED_COLS,
    SPLITS,
    DATA_SPLITS,
    cap_value,
    eval_df,
    load_static_song_table,
    macro,
    model_candidates,
    prediction_song_features,
    select_audio_cols,
)
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift
from search_pmemo_valence_temporal_shift import apply_shift
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_trainval_refit_correlated_shift_20260708"

SELECTED_CONFIG = {
    "source": "static",
    "target_mode": "residual_mean",
    "top_k": 8,
    "model_id": "ridge_a30",
    "strength": 0.55,
    "cap": "cap010",
    "temporal_shift_frames": -9,
    "temporal_shift_seconds": -4.5,
}

REFERENCE_BEST = {
    "model_id": "pmemo_valence_guardtopccc_temporal_shift_wide_20260708",
    "rmse": 0.1896185697242768,
    "mae": 0.14745982046674147,
    "pcc": 0.8105040088472681,
    "ccc": 0.797385312909266,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    selected_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    per_split: list[dict[str, Any]] = []
    for split_name in SPLITS:
        train_val = pd.concat([merged[(split_name, "train")], merged[(split_name, "val")]], ignore_index=True)
        selected_audio_cols = select_audio_cols(
            train_val,
            audio_cols,
            "base_residual_mean",
            int(SELECTED_CONFIG["top_k"]),
        )
        cols = PRED_COLS + selected_audio_cols
        model = dict(model_candidates())[str(SELECTED_CONFIG["model_id"])]
        model.fit(train_val[cols].to_numpy(dtype=np.float64), train_val["base_residual_mean"].to_numpy(dtype=np.float64))

        pred_targets: dict[str, dict[int, float]] = {}
        for data_split in DATA_SPLITS:
            table = merged[(split_name, data_split)]
            pred_residual = model.predict(table[cols].to_numpy(dtype=np.float64))
            pred_target_mean = table["pred_mean"].to_numpy(dtype=np.float64) + pred_residual
            pred_targets[data_split] = dict(zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist()))

        row: dict[str, Any] = {"split_name": split_name, "selected_audio_cols": selected_audio_cols}
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            corrected = apply_song_mean_shift(
                frames[(split_name, data_split)],
                pred_targets[data_split],
                float(SELECTED_CONFIG["strength"]),
                cap_value(str(SELECTED_CONFIG["cap"])),
            )
            shifted = apply_shift(corrected, int(SELECTED_CONFIG["temporal_shift_frames"]))
            shifted.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            selected_dfs[(split_name, data_split)] = shifted
            row.update({f"{data_split}_{m}": v for m, v in eval_df(shifted).items()})
        per_split.append(row)

    selected_test = macro(per_split, "test")
    selected_val = macro(per_split, "val")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_trainval_refit_correlated_shift_20260708",
        "reference_best": REFERENCE_BEST,
        "fixed_config_from_prior_validation": SELECTED_CONFIG,
        "selection_rule": "final refit only: hyperparameters fixed from prior validation-selected runs; correction model refit on train+val; test labels unused",
        "selected_val_after_refit_diagnostic": selected_val,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_reference_best": {m: float(selected_test[m] - REFERENCE_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
