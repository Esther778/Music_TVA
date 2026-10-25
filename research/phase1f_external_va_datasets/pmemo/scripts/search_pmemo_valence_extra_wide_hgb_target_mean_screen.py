#!/usr/bin/env python3
"""Small target-mean HGB song-level correction screen for Valence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_correlated_song_mean import PRED_COLS, cap_value, load_static_song_table, macro, select_audio_cols
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
OUT_DIR = ROOT / "outputs" / "valence_extra_wide_hgb_target_mean_screen_20260708"
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
        (
            "hgb_l2_010_leaf7_lr002",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=140,
                max_leaf_nodes=7,
                min_samples_leaf=16,
                l2_regularization=0.10,
                random_state=23,
            ),
        ),
        (
            "hgb_l2_100_leaf5_lr002",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=140,
                max_leaf_nodes=5,
                min_samples_leaf=20,
                l2_regularization=1.00,
                random_state=23,
            ),
        ),
    ]


def eval_candidate(frames, targets_by_split, strength: float, cap_name: str) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            corrected = apply_song_mean_shift(
                frames[(split_name, data_split)],
                targets_by_split[split_name][data_split],
                strength,
                cap_value(cap_name),
            )
            row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
        rows.append(row)
    return rows, macro(rows, "val"), macro(rows, "test")


def materialize(frames, targets_by_split, strength: float, cap_name: str) -> list[dict[str, Any]]:
    per_split = []
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            corrected = apply_song_mean_shift(
                frames[(split_name, data_split)],
                targets_by_split[split_name][data_split],
                strength,
                cap_value(cap_name),
            )
            corrected.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
        per_split.append(row)
    return per_split


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(s, d): pd.read_csv(INPUT_DIR / s / f"{d}_predictions.csv") for s in SPLITS for d in DATA_SPLITS}
    song_tables = {(s, d): prediction_song_features(frames[(s, d)]) for s in SPLITS for d in DATA_SPLITS}
    all_ids = sorted({int(sid) for table in song_tables.values() for sid in table["musicId"].tolist()})
    audio = pd.DataFrame({"musicId": all_ids}).merge(load_static_song_table(), on="musicId", how="left").fillna(0.0)
    audio_cols = [c for c in audio.columns if c != "musicId"]
    merged = {(s, d): song_tables[(s, d)].merge(audio, on="musicId", how="left").fillna(0.0) for s in SPLITS for d in DATA_SPLITS}

    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates: list[dict[str, Any]] = []
    best_payload: tuple[dict[str, dict[str, dict[int, float]]], float, str] | None = None
    best_row: dict[str, Any] | None = None
    for top_k in [8, 32, 64, 96]:
        cols_by_split = {s: select_audio_cols(merged[(s, "train")], audio_cols, "target_mean", top_k) for s in SPLITS}
        for model_id, model_template in model_candidates():
            targets_by_split: dict[str, dict[str, dict[int, float]]] = {}
            for split_name in SPLITS:
                cols = PRED_COLS + cols_by_split[split_name]
                train = merged[(split_name, "train")]
                model = model_template
                model.fit(train[cols].to_numpy(dtype=np.float64), train["target_mean"].to_numpy(dtype=np.float64))
                targets_by_split[split_name] = {}
                for data_split in DATA_SPLITS:
                    table = merged[(split_name, data_split)]
                    pred_target_mean = model.predict(table[cols].to_numpy(dtype=np.float64))
                    targets_by_split[split_name][data_split] = dict(zip(table["musicId"].astype(int).tolist(), pred_target_mean.tolist()))
            for strength in [0.10, 0.20, 0.30, 0.40, 0.60, 0.80]:
                for cap_name in ["cap005", "cap010", "cap020", "cap030", "none"]:
                    _, val_m, test_m = eval_candidate(frames, targets_by_split, float(strength), cap_name)
                    row = {
                        "model_id": model_id,
                        "top_k": int(top_k),
                        "target_mode": "target_mean",
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
                    candidates.append(row)
                    if row["passes_guard"] and (
                        best_row is None
                        or (row["val_score"], row["val_rmse"], -row["val_ccc"]) < (best_row["val_score"], best_row["val_rmse"], -best_row["val_ccc"])
                    ):
                        best_row = row
                        best_payload = (targets_by_split, float(strength), cap_name)

    result = pd.DataFrame(candidates)
    result.to_csv(OUT_DIR / "all_target_mean_candidates.csv", index=False)
    if best_row is None or best_payload is None:
        selected = {
            "model_id": "identity",
            "top_k": 0,
            "target_mode": "identity",
            "strength": 0.0,
            "cap": "none",
            **{f"val_{m}": v for m, v in base_val.items()},
            **{f"test_{m}": v for m, v in base_test.items()},
            "selection_reason": "identity retained; no target_mean candidate passed validation guard",
        }
        per_split = baseline_rows
    else:
        selected = dict(best_row)
        selected["selection_reason"] = "selected by validation macro guard"
        targets_by_split, strength, cap_name = best_payload
        per_split = materialize(frames, targets_by_split, strength, cap_name)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_extra_wide_hgb_target_mean_screen_20260708",
        "input_model": CURRENT_BEST,
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
