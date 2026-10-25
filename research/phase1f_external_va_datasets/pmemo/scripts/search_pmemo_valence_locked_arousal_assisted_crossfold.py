#!/usr/bin/env python3
"""Locked-dev Valence stacker assisted by predicted Arousal.

For each held development fold, this script trains on the other two folds using
only deployable predictions: Valence predictors and predicted Arousal curves.
The final locked test is not read.
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

from search_pmemo_valence_locked_audio_ensemble_postprocess import pred_col, target_col
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_locked_arousal_assisted_crossfold_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]

VALENCE_DIRS = {
    "v_residual_cal": ROOT / "outputs" / "valence_locked_residual_refine_xwide_calibrated_20260713",
    "v_polarity_cal": ROOT / "outputs" / "valence_locked_song_polarity_calibrated_20260713",
    "v_crossfold_cal": ROOT / "outputs" / "valence_locked_crossfold_stacker_calibrated_20260713",
    "v_ccc_blend": ROOT / "outputs" / "valence_locked_ccc_blend_calibration_20260713",
}

AROUSAL_DIRS = {
    "a_mert_smile": ROOT / "outputs" / "arousal_locked_mert_smile_blend_continuous_20260713",
    "a_nineway": ROOT / "outputs" / "arousal_locked_nineway_norm_refine_20260713",
    "a_eightway": ROOT / "outputs" / "arousal_locked_eightway_h112_refine_20260713",
    "a_trainfit": ROOT / "outputs" / "arousal_locked_trainfit_stacker_20260713",
}


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.045 * max(0.0, 0.80 - m["pcc"])
        + 0.10 * max(0.0, 0.80 - m["ccc"])
    )


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: float(np.mean([row[m] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def standard(path: Path, name: str, target_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target_name == "target_arousal":
        pc = "pred_arousal" if "pred_arousal" in df.columns else pred_col(df)
        tc = "target_arousal" if "target_arousal" in df.columns else target_col(df)
    else:
        pc = pred_col(df)
        tc = target_col(df)
    return df[["musicId", "time_sec", tc, pc]].rename(columns={tc: target_name, pc: f"pred_{name}"})


def add_temporal_features(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in pred_cols:
        out[f"{col}_song_mean"] = out.groupby("musicId")[col].transform("mean")
        out[f"{col}_song_std"] = out.groupby("musicId")[col].transform("std").fillna(0.0)
        out[f"{col}_centered"] = out[col] - out[f"{col}_song_mean"]
        out[f"{col}_delta"] = out.groupby("musicId")[col].diff().fillna(0.0)
    a_cols = [c for c in pred_cols if c.startswith("pred_a_")]
    if a_cols:
        out["pred_a_mean"] = out[a_cols].mean(axis=1)
        out["pred_a_spread"] = out[a_cols].std(axis=1).fillna(0.0)
        out["pred_a_mean_song_mean"] = out.groupby("musicId")["pred_a_mean"].transform("mean")
        out["pred_a_mean_centered"] = out["pred_a_mean"] - out["pred_a_mean_song_mean"]
        out["pred_a_mean_delta"] = out.groupby("musicId")["pred_a_mean"].diff().fillna(0.0)
        out["vxa"] = out["pred_v_crossfold_cal"] * out["pred_a_mean"]
    time_min = out.groupby("musicId")["time_sec"].transform("min")
    time_max = out.groupby("musicId")["time_sec"].transform("max")
    dur = (time_max - time_min).replace(0, 1.0)
    out["song_pos"] = (out["time_sec"] - time_min) / dur
    return out


def load_fold(fold: str) -> pd.DataFrame:
    first_name, first_root = next(iter(VALENCE_DIRS.items()))
    base = standard(first_root / f"{fold}_val_predictions.csv", first_name, "target_valence")
    for name, root in list(VALENCE_DIRS.items())[1:]:
        frame = standard(root / f"{fold}_val_predictions.csv", name, "target_valence")
        base = base.merge(frame.drop(columns=["target_valence"]), on=["musicId", "time_sec"], how="inner")
    for name, root in AROUSAL_DIRS.items():
        frame = standard(root / f"{fold}_val_predictions.csv", name, "target_arousal")
        base = base.merge(frame.drop(columns=["target_arousal"]), on=["musicId", "time_sec"], how="inner")
    pred_cols = [c for c in base.columns if c.startswith("pred_")]
    return add_temporal_features(base, pred_cols)


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a3", make_pipeline(StandardScaler(), Ridge(alpha=3.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("elastic_a003_l101", make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.1, max_iter=20000))),
        (
            "hgb_l2_02",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.03,
                max_leaf_nodes=8,
                l2_regularization=0.2,
                max_iter=180,
                random_state=20260713,
            ),
        ),
        (
            "hgb_l2_08",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.025,
                max_leaf_nodes=6,
                l2_regularization=0.8,
                max_iter=220,
                random_state=20260713,
            ),
        ),
    ]


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    base = [c for c in columns if c.startswith("pred_v_")]
    arousal = [c for c in columns if c.startswith("pred_a_") or c in ["vxa"]]
    compact = [
        "pred_v_crossfold_cal",
        "pred_v_ccc_blend",
        "pred_v_crossfold_cal_song_mean",
        "pred_v_crossfold_cal_centered",
        "pred_a_mean",
        "pred_a_mean_song_mean",
        "pred_a_mean_centered",
        "pred_a_mean_delta",
        "vxa",
        "song_pos",
    ]
    compact = [c for c in compact if c in columns]
    return {
        "compact": compact,
        "valence_only": [c for c in columns if c.startswith("pred_v_") or c == "song_pos"],
        "valence_arousal_raw": base + [c for c in arousal if c.startswith("pred_a_")] + ["song_pos"],
        "all": columns,
    }


def apply_post(df: pd.DataFrame, raw: np.ndarray, blend: float, cap: float | None) -> np.ndarray:
    base = df["pred_v_crossfold_cal"].to_numpy(dtype=np.float64)
    pred = (1.0 - blend) * base + blend * raw.astype(np.float64)
    if cap is not None:
        pred = base + np.clip(pred - base, -cap, cap)
    return np.clip(pred, -1.0, 1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = {fold: load_fold(fold) for fold in FOLDS}
    all_feature_cols = [
        c
        for c in folds[FOLDS[0]].columns
        if c.startswith("pred_") or c in ["vxa", "song_pos"]
    ]
    fsets = feature_sets(all_feature_cols)
    baseline_rows = [
        metrics(
            folds[fold]["pred_v_crossfold_cal"].to_numpy(dtype=np.float64),
            folds[fold]["target_valence"].to_numpy(dtype=np.float64),
        )
        for fold in FOLDS
    ]
    baseline_macro = macro(baseline_rows)

    rows = []
    pred_cache: dict[tuple[str, str, float, str], dict[str, np.ndarray]] = {}
    for feature_set_name, cols in fsets.items():
        if not cols:
            continue
        for model_id, model in model_candidates():
            raw_preds: dict[str, np.ndarray] = {}
            for held in FOLDS:
                train = pd.concat([folds[f] for f in FOLDS if f != held], ignore_index=True)
                val = folds[held]
                model.fit(train[cols].to_numpy(dtype=np.float64), train["target_valence"].to_numpy(dtype=np.float64))
                raw_preds[held] = model.predict(val[cols].to_numpy(dtype=np.float64))
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
                        pred = apply_post(folds[fold], raw_preds[fold], blend, cap)
                        preds[fold] = pred
                        per_fold.append(metrics(pred, folds[fold]["target_valence"].to_numpy(dtype=np.float64)))
                    m = macro(per_fold)
                    row = {
                        "feature_set": feature_set_name,
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
                    pred_cache[(feature_set_name, model_id, float(blend), cap_name)] = preds

    candidates = pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False])
    candidates.to_csv(OUT_DIR / "all_arousal_assisted_candidates.csv", index=False)
    guarded = candidates[candidates["beats_baseline"]].copy()
    if guarded.empty:
        selected = candidates.iloc[0].to_dict()
        selected["selection_reason"] = "diagnostic best only; no baseline-guarded improvement"
    else:
        selected = guarded.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by cross-devfold score with baseline guard"

    key = (str(selected["feature_set"]), str(selected["model_id"]), float(selected["blend"]), str(selected["cap"]))
    selected_preds = pred_cache[key]
    per_fold_rows = []
    frames_out = []
    for fold in FOLDS:
        out = folds[fold][["musicId", "time_sec", "target_valence"]].copy()
        out["base_pred_valence"] = folds[fold]["pred_v_crossfold_cal"].to_numpy(dtype=np.float64)
        out["pred_valence"] = selected_preds[fold]
        out.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
        frames_out.append(out)

    selected_macro = macro(per_fold_rows)
    merged = pd.concat(frames_out, ignore_index=True)
    global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged["target_valence"].to_numpy(dtype=np.float64))
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Valence cross-devfold stacker assisted by predicted Arousal; final locked test excluded",
        "selection_boundary": "Train on two locked devfolds, evaluate held devfold; no final locked test labels used",
        "baseline_model": str(VALENCE_DIRS["v_crossfold_cal"]),
        "arousal_inputs": {k: str(v) for k, v in AROUSAL_DIRS.items()},
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
