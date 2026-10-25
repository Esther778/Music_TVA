#!/usr/bin/env python3
"""Nonlinear MERT song-mean correction for Valence.

This experiment tests compact nonlinear song-level regressors (KNN, Kernel
Ridge, SVR) for the remaining Valence mean/polarity bias. It is intentionally
song-level and audio-only so the resulting policy can still be used on full
unseen songs.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from search_pmemo_valence_mert_song_mean_correction import MERT_DIR, PRED_COLS, cap_value, select_cols
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_guardtopccc_temporal_shift_wide_20260708"
OUT_DIR = ROOT / "outputs" / "valence_mert_knn_song_mean_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]

CURRENT_BEST = {
    "model_id": "pmemo_valence_guardtopccc_temporal_shift_wide_20260708",
    "rmse": 0.1896185697242768,
    "mae": 0.14745982046674147,
    "pcc": 0.8105040088472681,
    "ccc": 0.797385312909266,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.8 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def mert_song_features(music_ids: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for sid in music_ids:
        emb = np.load(MERT_DIR / f"{sid}.npz")["embedding"].astype(np.float64)
        row: dict[str, float] = {"musicId": float(sid)}
        for prefix, values in {
            "mean": emb.mean(axis=0),
            "std": emb.std(axis=0),
            "q25": np.quantile(emb, 0.25, axis=0),
            "q75": np.quantile(emb, 0.75, axis=0),
        }.items():
            for i, value in enumerate(values):
                row[f"mert_{prefix}_{i}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).assign(musicId=lambda d: d["musicId"].astype(int)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("knn5_dist", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))),
        ("knn9_dist", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=9, weights="distance"))),
        ("knn15_dist", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))),
        ("knn25_dist", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="distance"))),
        ("krr_rbf_a01_g001", make_pipeline(StandardScaler(), KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01))),
        ("krr_rbf_a1_g001", make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01))),
        ("krr_rbf_a10_g001", make_pipeline(StandardScaler(), KernelRidge(alpha=10.0, kernel="rbf", gamma=0.01))),
        ("svr_c1_gscale", make_pipeline(StandardScaler(), SVR(C=1.0, gamma="scale", epsilon=0.03))),
        ("svr_c3_gscale", make_pipeline(StandardScaler(), SVR(C=3.0, gamma="scale", epsilon=0.03))),
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
    mert = mert_song_features(all_ids)
    mert_cols = [c for c in mert.columns if c != "musicId"]
    merged = {
        (split_name, data_split): song_tables[(split_name, data_split)].merge(mert, on="musicId", how="left").fillna(0.0)
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

    rows: list[dict[str, Any]] = []
    pred_cache: dict[tuple[str, int, str], dict[str, dict[str, dict[int, float]]]] = {}
    for target_mode, y_col in [("target_mean", "target_mean"), ("residual_mean", "base_residual_mean")]:
        for top_k in [8, 16, 32, 64]:
            selected_cols = {
                split_name: select_cols(merged[(split_name, "train")], mert_cols, y_col, top_k)
                for split_name in SPLITS
            }
            for model_id, _ in model_candidates():
                pred_targets: dict[str, dict[str, dict[int, float]]] = {}
                for split_name in SPLITS:
                    cols = PRED_COLS + selected_cols[split_name]
                    model = dict(model_candidates())[model_id]
                    train = merged[(split_name, "train")]
                    model.fit(train[cols].to_numpy(dtype=np.float64), train[y_col].to_numpy(dtype=np.float64))
                    pred_targets[split_name] = {}
                    for data_split in DATA_SPLITS:
                        table = merged[(split_name, data_split)]
                        pred = model.predict(table[cols].to_numpy(dtype=np.float64))
                        if target_mode == "residual_mean":
                            pred = table["pred_mean"].to_numpy(dtype=np.float64) + pred
                        pred_targets[split_name][data_split] = dict(zip(table["musicId"].astype(int).tolist(), pred.tolist()))
                pred_cache[(target_mode, int(top_k), model_id)] = pred_targets
                for strength in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70]:
                    for cap_name in ["cap003", "cap005", "cap008", "cap010", "cap015", "cap020", "cap030", "cap040", "none"]:
                        eval_rows = []
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
                            eval_rows.append(row)
                        val_m = macro(eval_rows, "val")
                        test_m = macro(eval_rows, "test")
                        rows.append(
                            {
                                "target_mode": target_mode,
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
                                    and val_m["ccc"] >= base_val["ccc"] + 0.0005
                                ),
                            }
                        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_mert_knn_song_mean_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "target_mode": "identity",
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
        pred_targets = pred_cache[(str(selected["target_mode"]), int(selected["top_k"]), str(selected["model_id"]))]
        selected_dfs = {}
        for split_name in SPLITS:
            for data_split in DATA_SPLITS:
                selected_dfs[(split_name, data_split)] = apply_song_mean_shift(
                    frames[(split_name, data_split)],
                    pred_targets[split_name][data_split],
                    float(selected["strength"]),
                    cap_value(str(selected["cap"])),
                )

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
        "model_id": "pmemo_valence_mert_knn_song_mean_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "nonlinear pooled-MERT song-level correction; train-only feature correlation; one global config selected on validation macro guard",
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
