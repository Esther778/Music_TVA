#!/usr/bin/env python3
"""Fine search around the validation-effective Ridge song-mean correction."""

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
OUT_DIR = ROOT / "outputs" / "valence_latest_ridge_strength_fine_20260709"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
REFERENCE = {
    "model_id": "pmemo_valence_latest_ridge_ccc_guard_smoothing_20260709",
    "rmse": 0.187623279893854,
    "mae": 0.14601650974911215,
    "pcc": 0.8141844446811181,
    "ccc": 0.8002325331205622,
}


def score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.08 * max(0.0, 0.80 - m["ccc"]) + 0.005 * m["mae"]


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

    cols_by_split = {
        split: PRED_COLS + select_audio_cols(merged[(split, "train")], audio_cols, "target_mean", 64)
        for split in SPLITS
    }
    target_maps_by_alpha: dict[float, dict[tuple[str, str], dict[int, float]]] = {}
    for alpha in [300.0, 600.0, 1000.0, 1600.0, 2400.0]:
        maps: dict[tuple[str, str], dict[int, float]] = {}
        for split in SPLITS:
            cols = cols_by_split[split]
            train = merged[(split, "train")]
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[cols].to_numpy(dtype=np.float64), train["target_mean"].to_numpy(dtype=np.float64))
            for data_split in DATA_SPLITS:
                table = merged[(split, data_split)]
                pred_target_mean = model.predict(table[cols].to_numpy(dtype=np.float64))
                maps[(split, data_split)] = dict(zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist()))
        target_maps_by_alpha[alpha] = maps

    rows_out = []
    best: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] | None = None
    best_frames: dict[tuple[str, str], pd.DataFrame] | None = None
    for alpha, target_maps in target_maps_by_alpha.items():
        for strength in [0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.125, 0.135, 0.145]:
            for cap in ["none", "cap020", "cap015", "cap012", "cap010", "cap008"]:
                split_rows = []
                dfs: dict[tuple[str, str], pd.DataFrame] = {}
                for split in SPLITS:
                    row: dict[str, Any] = {"split_name": split}
                    for data_split in DATA_SPLITS:
                        corrected = apply_song_mean_shift(
                            frames[(split, data_split)],
                            target_maps[(split, data_split)],
                            strength,
                            cap_value(cap),
                        )
                        dfs[(split, data_split)] = corrected
                        row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
                    split_rows.append(row)
                val_m = macro(split_rows, "val")
                test_m = macro(split_rows, "test")
                candidate = {
                    "alpha": alpha,
                    "strength": strength,
                    "cap": cap,
                    **{f"val_{m}": v for m, v in val_m.items()},
                    **{f"test_{m}": v for m, v in test_m.items()},
                    "val_score": score(val_m),
                }
                candidate["passes_guard"] = (
                    val_m["rmse"] <= 0.20035
                    and val_m["mae"] <= 0.15780
                    and val_m["ccc"] >= 0.77770
                )
                rows_out.append(candidate)
                if candidate["passes_guard"] and (best is None or candidate["val_score"] < best["val_score"]):
                    best = candidate
                    best_rows = split_rows
                    best_frames = dfs

    pd.DataFrame(rows_out).to_csv(OUT_DIR / "all_fine_strength_candidates.csv", index=False)
    if best is None or best_rows is None or best_frames is None:
        raise RuntimeError("No fine Ridge candidate passed validation guard.")
    for split in SPLITS:
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            best_frames[(split, data_split)].to_csv(split_dir / f"{data_split}_predictions.csv", index=False)

    selected_val = macro(best_rows, "val")
    selected_test = macro(best_rows, "test")
    summary = {
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_latest_ridge_strength_fine_20260709",
        "input_reference": REFERENCE,
        "selection_rule": "fine Ridge target-mean correction selected by validation RMSE+CCC guard; test labels unused",
        "selected_config": best,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "delta_vs_reference": {m: float(selected_test[m] - REFERENCE[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
        "per_split": best_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
