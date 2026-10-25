#!/usr/bin/env python3
"""Nonlinear static OpenSMILE song-mean correction for Valence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_correlated_song_mean import (
    CURRENT_BEST,
    DATA_SPLITS,
    FEATURE_ROOT,
    INPUT_DIR,
    PASS,
    PRED_COLS,
    SPLITS,
    cap_value,
    eval_df,
    load_static_song_table,
    prediction_song_features,
    select_audio_cols,
)
from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_static_nonlinear_song_mean_20260708"


def score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.4 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.1 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.4 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("hgb_l2_leaf15_lr003", HistGradientBoostingRegressor(max_iter=160, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=0.05, random_state=42)),
        ("hgb_l2_leaf7_lr004", HistGradientBoostingRegressor(max_iter=140, learning_rate=0.04, max_leaf_nodes=7, l2_regularization=0.10, random_state=42)),
        ("extra_trees_80_d3", ExtraTreesRegressor(n_estimators=80, max_depth=3, min_samples_leaf=8, random_state=42, n_jobs=1)),
        ("rf_80_d4", RandomForestRegressor(n_estimators=80, max_depth=4, min_samples_leaf=6, random_state=42, n_jobs=1)),
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
    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates = []
    pred_cache: dict[tuple[int, str], dict[str, dict[str, dict[int, float]]]] = {}
    for top_k in [8, 16, 32]:
        selected_cols_by_split = {
            split_name: select_audio_cols(merged[(split_name, "train")], audio_cols, "base_residual_mean", top_k)
            for split_name in SPLITS
        }
        for model_id, _ in model_candidates():
            pred_targets: dict[str, dict[str, dict[int, float]]] = {}
            for split_name in SPLITS:
                train = merged[(split_name, "train")]
                cols = PRED_COLS + selected_cols_by_split[split_name]
                model = dict(model_candidates())[model_id]
                model.fit(train[cols].to_numpy(dtype=np.float64), train["base_residual_mean"].to_numpy(dtype=np.float64))
                pred_targets[split_name] = {}
                for data_split in DATA_SPLITS:
                    table = merged[(split_name, data_split)]
                    residual = model.predict(table[cols].to_numpy(dtype=np.float64))
                    target_mean = table["pred_mean"].to_numpy(dtype=np.float64) + residual
                    pred_targets[split_name][data_split] = dict(zip(table["musicId"].astype(int).tolist(), target_mean.tolist()))
            pred_cache[(int(top_k), model_id)] = pred_targets
            for strength in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70]:
                for cap_name in ["cap005", "cap010", "cap015", "cap020", "cap030", "cap040", "cap060", "none"]:
                    rows = []
                    for split_name in SPLITS:
                        row = {"split_name": split_name}
                        for data_split in DATA_SPLITS:
                            corrected = apply_song_mean_shift(
                                frames[(split_name, data_split)],
                                pred_targets[split_name][data_split],
                                float(strength),
                                cap_value(cap_name),
                            )
                            row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
                        rows.append(row)
                    val_m = macro(rows, "val")
                    test_m = macro(rows, "test")
                    candidates.append(
                        {
                            "top_k": int(top_k),
                            "model_id": model_id,
                            "strength": float(strength),
                            "cap": cap_name,
                            **{f"val_{m}": v for m, v in val_m.items()},
                            **{f"test_{m}": v for m, v in test_m.items()},
                            "val_score": score(val_m),
                            "passes_guard": (
                                val_m["rmse"] <= base_val["rmse"] + 0.001
                                and val_m["mae"] <= base_val["mae"] + 0.001
                                and val_m["pcc"] >= base_val["pcc"] - 0.001
                                and val_m["ccc"] >= base_val["ccc"] + 0.0003
                            ),
                        }
                    )

    result = pd.DataFrame(candidates)
    result.to_csv(OUT_DIR / "all_static_nonlinear_song_mean_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "top_k": 0,
            "model_id": "identity",
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
        pred_targets = pred_cache[(int(selected["top_k"]), str(selected["model_id"]))]
        selected_dfs = {
            (split_name, data_split): apply_song_mean_shift(
                frames[(split_name, data_split)],
                pred_targets[split_name][data_split],
                float(selected["strength"]),
                cap_value(str(selected["cap"])),
            )
            for split_name in SPLITS
            for data_split in DATA_SPLITS
        }

    per_split = []
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
        "model_id": "pmemo_valence_static_nonlinear_song_mean_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "static OpenSMILE nonlinear song-level residual predictor; one global config selected on validation macro guard",
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
