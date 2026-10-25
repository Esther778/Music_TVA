#!/usr/bin/env python3
"""Cross-devfold Valence stacker over full-coverage prediction artifacts.

The stacker is trained on two locked development folds and evaluated on the
held-out development fold. It does not read the final locked test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_audio_ensemble_postprocess import candidate_paths, pred_col, target_col
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_locked_crossfold_stacker_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]

FULL_COVERAGE_DIRS = {
    "residual_cal": ROOT / "outputs" / "valence_locked_residual_refine_xwide_calibrated_20260713",
    "song_polarity": ROOT / "outputs" / "valence_locked_song_polarity_correction_20260713",
    "song_polarity_cal": ROOT / "outputs" / "valence_locked_song_polarity_calibrated_20260713",
}


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.04 * max(0.0, 0.80 - m["pcc"])
        + 0.09 * max(0.0, 0.80 - m["ccc"])
    )


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: float(np.mean([row[m] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def standard(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pc = pred_col(df)
    tc = target_col(df)
    return df[["musicId", "time_sec", tc, pc]].rename(columns={tc: "target_valence", pc: f"pred_{name}"})


def add_song_meta(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        means = out.groupby("musicId")[col].transform("mean")
        stds = out.groupby("musicId")[col].transform("std").fillna(0.0)
        out[f"{col}_song_mean"] = means
        out[f"{col}_song_std"] = stds
        out[f"{col}_delta"] = out.groupby("musicId")[col].diff().fillna(0.0)
    time_min = out.groupby("musicId")["time_sec"].transform("min")
    time_max = out.groupby("musicId")["time_sec"].transform("max")
    dur = (time_max - time_min).replace(0, 1.0)
    out["song_pos"] = (out["time_sec"] - time_min) / dur
    return out


def load_fold(fold: str) -> pd.DataFrame:
    paths = {
        "bestlike": candidate_paths(fold)["bestlike"],
        "ridge": candidate_paths(fold)["ridge"],
    }
    for name, root in FULL_COVERAGE_DIRS.items():
        path = root / f"{fold}_val_predictions.csv"
        if path.exists():
            paths[name] = path
    frames = [standard(path, name) for name, path in paths.items()]
    base = frames[0][["musicId", "time_sec", "target_valence"]].copy()
    for frame in frames:
        base = base.merge(frame.drop(columns=["target_valence"]), on=["musicId", "time_sec"], how="inner")
    pred_cols = [c for c in base.columns if c.startswith("pred_")]
    return add_song_meta(base, pred_cols)


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a3", make_pipeline(StandardScaler(), Ridge(alpha=3.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("elastic_a001_l101", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=20000))),
        ("elastic_a003_l101", make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.1, max_iter=20000))),
        (
            "hgb_l2_02",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.03,
                max_leaf_nodes=8,
                l2_regularization=0.2,
                max_iter=150,
                random_state=20260713,
            ),
        ),
    ]


def apply_post(df: pd.DataFrame, pred: np.ndarray, blend: float, cap: float | None) -> np.ndarray:
    base = df["pred_song_polarity_cal"].to_numpy(dtype=np.float64)
    out = (1.0 - blend) * base + blend * pred.astype(np.float64)
    if cap is not None:
        out = base + np.clip(out - base, -cap, cap)
    return np.clip(out, -1.0, 1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_frames = {fold: load_fold(fold) for fold in FOLDS}
    feature_cols = [c for c in fold_frames[FOLDS[0]].columns if c.startswith("pred_") or c.endswith("_mean") or c.endswith("_std") or c.endswith("_delta")]
    feature_cols.append("song_pos")
    baseline_rows = [
        metrics(
            fold_frames[fold]["pred_song_polarity_cal"].to_numpy(dtype=np.float64),
            fold_frames[fold]["target_valence"].to_numpy(dtype=np.float64),
        )
        for fold in FOLDS
    ]
    baseline_macro = macro(baseline_rows)

    rows = []
    pred_cache: dict[tuple[str, float, str], dict[str, np.ndarray]] = {}
    for model_id, model in model_candidates():
        raw_preds: dict[str, np.ndarray] = {}
        for held_fold in FOLDS:
            train = pd.concat([fold_frames[f] for f in FOLDS if f != held_fold], ignore_index=True)
            val = fold_frames[held_fold]
            model.fit(train[feature_cols].to_numpy(dtype=np.float64), train["target_valence"].to_numpy(dtype=np.float64))
            raw_preds[held_fold] = model.predict(val[feature_cols].to_numpy(dtype=np.float64))
        for blend in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00]:
            for cap_name, cap in [
                ("cap003", 0.003),
                ("cap005", 0.005),
                ("cap008", 0.008),
                ("cap010", 0.010),
                ("cap015", 0.015),
                ("cap020", 0.020),
                ("cap030", 0.030),
                ("cap050", 0.050),
                ("none", None),
            ]:
                per_fold = []
                preds: dict[str, np.ndarray] = {}
                for fold in FOLDS:
                    pred = apply_post(fold_frames[fold], raw_preds[fold], blend, cap)
                    preds[fold] = pred
                    per_fold.append(metrics(pred, fold_frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                m = macro(per_fold)
                row = {
                    "model_id": model_id,
                    "blend": float(blend),
                    "cap": cap_name,
                    **m,
                    "score": score(m),
                    "beats_baseline": (
                        m["rmse"] <= baseline_macro["rmse"]
                        and m["mae"] <= baseline_macro["mae"] + 0.0005
                        and m["pcc"] >= baseline_macro["pcc"] - 0.0005
                        and m["ccc"] >= baseline_macro["ccc"] - 0.0005
                    ),
                }
                rows.append(row)
                pred_cache[(model_id, float(blend), cap_name)] = preds

    candidates = pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False])
    candidates.to_csv(OUT_DIR / "all_crossfold_stacker_candidates.csv", index=False)
    guarded = candidates[candidates["beats_baseline"]].copy()
    if guarded.empty:
        selected = candidates.iloc[0].to_dict()
        selected["selection_reason"] = "diagnostic best only; no baseline-guarded improvement"
    else:
        selected = guarded.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by cross-devfold score with baseline guard"

    key = (str(selected["model_id"]), float(selected["blend"]), str(selected["cap"]))
    selected_preds = pred_cache[key]
    per_fold_rows = []
    frames_out = []
    for fold in FOLDS:
        df = fold_frames[fold][["musicId", "time_sec", "target_valence"]].copy()
        df["base_pred_valence"] = fold_frames[fold]["pred_song_polarity_cal"].to_numpy(dtype=np.float64)
        df["pred_valence"] = selected_preds[fold]
        df.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(df["pred_valence"], df["target_valence"])})
        frames_out.append(df)

    selected_macro = macro(per_fold_rows)
    merged = pd.concat(frames_out, ignore_index=True)
    global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged["target_valence"].to_numpy(dtype=np.float64))
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Cross-devfold Valence stacker over full-coverage predictors; final locked test excluded",
        "selection_boundary": "Train on two locked devfolds, evaluate held devfold; no final locked test labels used",
        "baseline_model": str(FULL_COVERAGE_DIRS["song_polarity_cal"]),
        "baseline_macro": baseline_macro,
        "selected_config": selected,
        "per_fold": per_fold_rows,
        "macro_mean": selected_macro,
        "global_frame": global_frame,
        "delta_vs_baseline_macro": {m: float(selected_macro[m] - baseline_macro[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "gap_rows_macro": threshold_rows(selected_macro),
        "gap_rows_global_frame": threshold_rows(global_frame),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
