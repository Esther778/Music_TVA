#!/usr/bin/env python3
"""Lightweight latest-base HGB/Ridge song-mean correction search."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_correlated_song_mean import (
    PRED_COLS,
    cap_value,
    load_static_song_table,
    macro,
    select_audio_cols,
)
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_second_residual_smoothing_20260709"
OUT_DIR = ROOT / "outputs" / "valence_latest_hgb_song_mean_light_20260709"
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
    return m["rmse"] + 0.03 * max(0.0, 0.80 - m["ccc"]) + 0.005 * m["mae"]


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        (
            "hgb_l2_010_leaf7_lr002",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=120,
                max_leaf_nodes=7,
                min_samples_leaf=16,
                l2_regularization=0.10,
                random_state=17,
            ),
        ),
        (
            "hgb_l2_100_leaf5_lr002",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=120,
                max_leaf_nodes=5,
                min_samples_leaf=20,
                l2_regularization=1.00,
                random_state=17,
            ),
        ),
    ]


def load_frames() -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (split_name, data_split): pd.read_csv(INPUT_DIR / split_name / f"{data_split}_predictions.csv")
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }


def build_tables(frames: dict[tuple[str, str], pd.DataFrame]) -> tuple[dict[tuple[str, str], pd.DataFrame], list[str]]:
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
        key: table.merge(audio, on="musicId", how="left").fillna(0.0)
        for key, table in song_tables.items()
    }
    return merged, audio_cols


def evaluate_config(
    frames: dict[tuple[str, str], pd.DataFrame],
    merged: dict[tuple[str, str], pd.DataFrame],
    audio_cols: list[str],
    config: dict[str, Any],
    write_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], pd.DataFrame]]:
    model_map = dict(model_candidates())
    corrected_frames: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        train = merged[(split_name, "train")]
        selected_cols = select_audio_cols(train, audio_cols, str(config["target_col"]), int(config["top_k"]))
        cols = PRED_COLS + selected_cols
        model = model_map[str(config["model_id"])]
        model.fit(train[cols].to_numpy(dtype=np.float64), train[str(config["target_col"])].to_numpy(dtype=np.float64))
        row: dict[str, Any] = {"split_name": split_name, "selected_audio_cols": selected_cols}
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
                cap_value(str(config["cap"])),
            )
            corrected_frames[(split_name, data_split)] = corrected
            row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
        rows.append(row)
    return rows, corrected_frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    merged, audio_cols = build_tables(frames)
    baseline_rows = []
    for split_name in SPLITS:
        row: dict[str, Any] = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates = []
    best_config: dict[str, Any] | None = None
    best_score = score(base_val)
    for model_id, _ in model_candidates():
        for target_col in ["base_residual_mean", "target_mean"]:
            for top_k in [0, 16, 64]:
                for strength in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
                    for cap in ["cap003", "cap005", "cap010", "cap020", "none"]:
                        config = {
                            "model_id": model_id,
                            "target_col": target_col,
                            "top_k": top_k,
                            "strength": strength,
                            "cap": cap,
                        }
                        rows, _ = evaluate_config(frames, merged, audio_cols, config)
                        val_m = macro(rows, "val")
                        test_m = macro(rows, "test")
                        candidate = {
                            **config,
                            **{f"val_{m}": v for m, v in val_m.items()},
                            **{f"test_{m}": v for m, v in test_m.items()},
                            "val_score": score(val_m),
                            "passes_guard": (
                                val_m["rmse"] <= base_val["rmse"] + 0.0006
                                and val_m["mae"] <= base_val["mae"] + 0.0008
                                and val_m["ccc"] >= base_val["ccc"] - 0.001
                            ),
                        }
                        candidates.append(candidate)
                        if candidate["passes_guard"] and candidate["val_score"] < best_score:
                            best_score = float(candidate["val_score"])
                            best_config = config

    pd.DataFrame(candidates).to_csv(OUT_DIR / "all_light_hgb_song_mean_candidates.csv", index=False)
    if best_config is None:
        selected_rows = baseline_rows
        selected_frames = frames
        selected_config: dict[str, Any] = {
            "model_id": "identity",
            "selection_reason": "identity retained; no candidate beat validation-guarded baseline",
        }
    else:
        selected_rows, selected_frames = evaluate_config(frames, merged, audio_cols, best_config)
        selected_config = {**best_config, "selection_reason": "selected by validation guard"}

    for split_name in SPLITS:
        split_dir = OUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            selected_frames[(split_name, data_split)].to_csv(split_dir / f"{data_split}_predictions.csv", index=False)

    selected_val = macro(selected_rows, "val")
    selected_test = macro(selected_rows, "test")
    summary = {
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_latest_hgb_song_mean_light_20260709",
        "input_model": CURRENT_BEST,
        "selection_rule": "lightweight nonlinear song-mean correction selected by validation guard; test labels unused",
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
