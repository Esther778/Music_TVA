#!/usr/bin/env python3
"""Fast Ridge-only song-mean correction on the latest Valence artifact."""

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
OUT_DIR = ROOT / "outputs" / "valence_latest_ridge_song_mean_fast_20260709"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_smoothing_20260708",
    "rmse": 0.1878117801132452,
    "mae": 0.14589902068622534,
    "pcc": 0.8140948778047831,
    "ccc": 0.8013555689217714,
}


def score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.04 * max(0.0, 0.80 - m["ccc"]) + 0.005 * m["mae"]


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

    baseline_rows = []
    for split_name in SPLITS:
        row: dict[str, Any] = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    col_cache: dict[tuple[str, str, int], list[str]] = {}
    for split_name in SPLITS:
        for target_col in ["base_residual_mean", "target_mean"]:
            for top_k in [0, 8, 16, 32, 64]:
                col_cache[(split_name, target_col, top_k)] = select_audio_cols(
                    merged[(split_name, "train")],
                    audio_cols,
                    target_col,
                    top_k,
                )

    candidates = []
    best: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] | None = None
    best_frames: dict[tuple[str, str], pd.DataFrame] | None = None
    best_score = score(base_val)
    for target_col in ["base_residual_mean", "target_mean"]:
        for top_k in [0, 8, 16, 32, 64]:
            for alpha in [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
                target_maps: dict[tuple[str, str], dict[int, float]] = {}
                for split_name in SPLITS:
                    cols = PRED_COLS + col_cache[(split_name, target_col, top_k)]
                    train = merged[(split_name, "train")]
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(train[cols].to_numpy(dtype=np.float64), train[target_col].to_numpy(dtype=np.float64))
                    for data_split in DATA_SPLITS:
                        table = merged[(split_name, data_split)]
                        pred = model.predict(table[cols].to_numpy(dtype=np.float64))
                        if target_col == "base_residual_mean":
                            pred_target_mean = table["pred_mean"].to_numpy(dtype=np.float64) + pred
                        else:
                            pred_target_mean = pred
                        target_maps[(split_name, data_split)] = dict(
                            zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist())
                        )
                for strength in [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15]:
                    for cap in ["cap003", "cap005", "cap008", "cap010", "cap015", "none"]:
                        rows = []
                        dfs: dict[tuple[str, str], pd.DataFrame] = {}
                        for split_name in SPLITS:
                            row: dict[str, Any] = {"split_name": split_name}
                            for data_split in DATA_SPLITS:
                                corrected = apply_song_mean_shift(
                                    frames[(split_name, data_split)],
                                    target_maps[(split_name, data_split)],
                                    strength,
                                    cap_value(cap),
                                )
                                dfs[(split_name, data_split)] = corrected
                                row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
                            rows.append(row)
                        val_m = macro(rows, "val")
                        test_m = macro(rows, "test")
                        candidate = {
                            "target_col": target_col,
                            "top_k": top_k,
                            "alpha": alpha,
                            "strength": strength,
                            "cap": cap,
                            **{f"val_{m}": v for m, v in val_m.items()},
                            **{f"test_{m}": v for m, v in test_m.items()},
                            "val_score": score(val_m),
                            "passes_guard": (
                                val_m["rmse"] <= base_val["rmse"] + 0.0004
                                and val_m["mae"] <= base_val["mae"] + 0.0006
                                and val_m["ccc"] >= base_val["ccc"] - 0.0008
                            ),
                        }
                        candidates.append(candidate)
                        if candidate["passes_guard"] and candidate["val_score"] < best_score:
                            best_score = candidate["val_score"]
                            best = candidate
                            best_rows = rows
                            best_frames = dfs

    pd.DataFrame(candidates).to_csv(OUT_DIR / "all_ridge_song_mean_candidates.csv", index=False)
    if best is None or best_rows is None or best_frames is None:
        selected_config = {"selection_reason": "identity retained; no Ridge candidate beat validation-guarded baseline"}
        selected_rows = baseline_rows
        selected_frames = frames
    else:
        selected_config = {k: best[k] for k in ["target_col", "top_k", "alpha", "strength", "cap", "selection_reason"] if k in best}
        selected_config["selection_reason"] = "selected by validation guard"
        selected_rows = best_rows
        selected_frames = best_frames

    for split_name in SPLITS:
        split_dir = OUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            selected_frames[(split_name, data_split)].to_csv(split_dir / f"{data_split}_predictions.csv", index=False)

    selected_val = macro(selected_rows, "val")
    selected_test = macro(selected_rows, "test")
    summary = {
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_latest_ridge_song_mean_fast_20260709",
        "input_model": CURRENT_BEST,
        "selection_rule": "fast Ridge song-mean correction selected by validation guard; test labels unused",
        "baseline_val": base_val,
        "baseline_test": base_test,
        "selected_config": selected_config,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
        "per_split": selected_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
