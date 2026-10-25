#!/usr/bin/env python3
"""Tree-based song-mean correction on the current formal Valence best."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_correlated_song_mean import PRED_COLS, cap_value, load_static_song_table, macro, select_audio_cols
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708"
OUT_DIR = ROOT / "outputs" / "valence_current_best_tree_song_mean_20260709"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


def score(m: dict[str, float]) -> float:
    return (
        4.2 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.0 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 0.8 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.6 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.03 * m["rmse"]
    )


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        (
            "hgb_l2_010_leaf5",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=140,
                max_leaf_nodes=5,
                min_samples_leaf=16,
                l2_regularization=0.10,
                random_state=47,
            ),
        ),
        (
            "extratrees_d3_leaf4",
            ExtraTreesRegressor(
                n_estimators=120,
                max_depth=3,
                min_samples_leaf=4,
                max_features=0.65,
                random_state=47,
                n_jobs=1,
            ),
        ),
        (
            "extratrees_d4_leaf6",
            ExtraTreesRegressor(
                n_estimators=160,
                max_depth=4,
                min_samples_leaf=6,
                max_features=0.80,
                random_state=53,
                n_jobs=1,
            ),
        ),
        (
            "rf_d4_leaf6",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=4,
                min_samples_leaf=6,
                max_features=0.80,
                random_state=59,
                n_jobs=1,
            ),
        ),
    ]


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
        for top_k in [0, 16, 64, 128]:
            cols_by_split = {
                split_name: select_audio_cols(merged[(split_name, "train")], audio_cols, y_col, top_k)
                for split_name in SPLITS
            }
            for model_id, model_template in model_candidates():
                targets_by_split: dict[str, dict[str, dict[int, float]]] = {}
                for split_name in SPLITS:
                    cols = PRED_COLS + cols_by_split[split_name]
                    train = merged[(split_name, "train")]
                    model = model_template
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
                for strength in [0.02, 0.05, 0.10, 0.20, 0.40]:
                    for cap_name in ["cap003", "cap008", "cap015", "none"]:
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
                                    val_m["rmse"] <= base_val["rmse"] + 0.0006
                                    and val_m["mae"] <= base_val["mae"] + 0.0006
                                    and val_m["pcc"] >= base_val["pcc"] - 0.001
                                    and val_m["ccc"] >= base_val["ccc"] - 0.001
                                ),
                            }
                        )

    result = pd.DataFrame(candidates)
    result.to_csv(OUT_DIR / "all_tree_song_mean_candidates.csv", index=False)
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
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_current_best_tree_song_mean_20260709",
        "input_model": CURRENT_BEST,
        "selection_rule": "tree/ridge song-mean correction selected by validation macro guard; test labels unused",
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
