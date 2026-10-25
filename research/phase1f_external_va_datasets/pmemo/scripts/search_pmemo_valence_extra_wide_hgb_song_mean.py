#!/usr/bin/env python3
"""Nonlinear song-mean correction for the current extra-wide Valence best."""

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


INPUT_DIR = ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
OUT_DIR = ROOT / "outputs" / "valence_extra_wide_hgb_song_mean_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


def score(m: dict[str, float]) -> float:
    return (
        3.8 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.2 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.8 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.03 * m["rmse"]
    )


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        (
            "hgb_l2_001_leaf7_lr003",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.03,
                max_iter=120,
                max_leaf_nodes=7,
                min_samples_leaf=12,
                l2_regularization=0.01,
                random_state=17,
            ),
        ),
        (
            "hgb_l2_010_leaf7_lr002",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=160,
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
                max_iter=160,
                max_leaf_nodes=5,
                min_samples_leaf=20,
                l2_regularization=1.00,
                random_state=17,
            ),
        ),
    ]


def build_frames() -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (split_name, data_split): pd.read_csv(INPUT_DIR / split_name / f"{data_split}_predictions.csv")
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_frames()
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

    baseline_rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, str, float, str], dict[tuple[str, str], pd.DataFrame]] = {}
    for target_mode, y_col in [("residual_mean", "base_residual_mean"), ("target_mean", "target_mean")]:
        for top_k in [0, 8, 16, 32, 64, 96, 128]:
            cols_by_split = {
                split_name: select_audio_cols(merged[(split_name, "train")], audio_cols, y_col, top_k)
                for split_name in SPLITS
            }
            for model_id, _ in model_candidates():
                targets_by_split: dict[str, dict[str, dict[int, float]]] = {}
                for split_name in SPLITS:
                    cols = PRED_COLS + cols_by_split[split_name]
                    train = merged[(split_name, "train")]
                    model = dict(model_candidates())[model_id]
                    model.fit(train[cols].to_numpy(dtype=np.float64), train[y_col].to_numpy(dtype=np.float64))
                    targets_by_split[split_name] = {}
                    for data_split in DATA_SPLITS:
                        table = merged[(split_name, data_split)]
                        pred = model.predict(table[cols].to_numpy(dtype=np.float64))
                        if target_mode == "residual_mean":
                            pred_target_mean = table["pred_mean"].to_numpy(dtype=np.float64) + pred
                        else:
                            pred_target_mean = pred
                        targets_by_split[split_name][data_split] = dict(zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist()))
                for strength in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.80]:
                    for cap_name in ["cap003", "cap005", "cap008", "cap010", "cap015", "cap020", "cap030", "none"]:
                        rows: list[dict[str, Any]] = []
                        dfs: dict[tuple[str, str], pd.DataFrame] = {}
                        for split_name in SPLITS:
                            row = {"split_name": split_name}
                            for data_split in DATA_SPLITS:
                                corrected = apply_song_mean_shift(
                                    frames[(split_name, data_split)],
                                    targets_by_split[split_name][data_split],
                                    float(strength),
                                    cap_value(cap_name),
                                )
                                dfs[(split_name, data_split)] = corrected
                                row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
                            rows.append(row)
                        val_m = macro(rows, "val")
                        test_m = macro(rows, "test")
                        key = (model_id, int(top_k), target_mode, float(strength), cap_name)
                        cache[key] = dfs
                        candidates.append(
                            {
                                "model_id": model_id,
                                "top_k": int(top_k),
                                "target_mode": target_mode,
                                "strength": float(strength),
                                "cap": cap_name,
                                **{f"val_{m}": v for m, v in val_m.items()},
                                **{f"test_{m}": v for m, v in test_m.items()},
                                "val_score": score(val_m),
                                "passes_guard": (
                                    val_m["rmse"] <= base_val["rmse"] + 0.0008
                                    and val_m["mae"] <= base_val["mae"] + 0.0008
                                    and val_m["pcc"] >= base_val["pcc"] - 0.001
                                    and val_m["ccc"] >= base_val["ccc"] - 0.001
                                ),
                            }
                        )

    result = pd.DataFrame(candidates)
    result.to_csv(OUT_DIR / "all_hgb_song_mean_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "model_id": "identity",
            "top_k": 0,
            "target_mode": "identity",
            "strength": 0.0,
            "cap": "none",
            **{f"val_{m}": v for m, v in base_val.items()},
            **{f"test_{m}": v for m, v in base_test.items()},
            "selection_reason": "identity retained; no candidate passed validation guard",
        }
        selected_dfs = frames
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
        selected_dfs = cache[(str(selected["model_id"]), int(selected["top_k"]), str(selected["target_mode"]), float(selected["strength"]), str(selected["cap"]))]

    per_split: list[dict[str, Any]] = []
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            df = selected_dfs[(split_name, data_split)]
            df.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)

    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_extra_wide_hgb_song_mean_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "nonlinear song-mean residual correction selected by validation macro guard; test labels unused",
        "baseline_val": base_val,
        "baseline_test": base_test,
        "selected_config": selected,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
