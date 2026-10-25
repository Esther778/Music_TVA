#!/usr/bin/env python3
"""Train-fit Arousal stacker on final-locked development folds.

The stacker is fitted on each fold's train predictions and evaluated on the
same fold's validation predictions. The locked final test split is not read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_opensmile_arousal_trials import ROOT, metrics, threshold_rows
from search_pmemo_arousal_mert_harmonic_ensemble_devfolds import (
    calibrate,
    scale_amplitude_by_song,
    shift_by_song,
    smooth_by_song,
)


OUT_DIR = ROOT / "outputs" / "arousal_locked_trainfit_stacker_20260713"
BRANCHES = {
    "h96": lambda fold: ROOT / "outputs" / "arousal_locked_devfolds_h96_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse",
    "h64": lambda fold: ROOT / "outputs" / "arousal_locked_devfolds_h64_20260713" / fold / "a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003",
    "h128": lambda fold: ROOT / "outputs" / "arousal_locked_devfolds_h128_20260713" / fold / "a_dev_h128_l1_do020_lr6e4_mse",
    "mert_harm": lambda fold: ROOT / "outputs" / "arousal_mert_harmonic_locked_devfolds_20260713" / fold,
    "tcn": lambda fold: ROOT / "outputs" / "arousal_locked_tcn_20260713" / fold,
    "h96_s14": lambda fold: ROOT / "outputs" / "arousal_locked_devfolds_h96_seed14_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse",
    "h112": lambda fold: ROOT / "outputs" / "arousal_locked_devfolds_h112_20260713" / fold / "a_dev_h112_l1_do015_lr7e4_mse",
    "norm_lstm": lambda fold: ROOT / "outputs" / "arousal_locked_norm_lstm_20260713" / fold,
}


def pred_col(df: pd.DataFrame) -> str:
    for col in ["pred_arousal", "pred_arousal_raw"]:
        if col in df.columns:
            return col
    raise ValueError(df.columns.tolist())


def load_split(fold: str, split: str) -> pd.DataFrame:
    frames = {}
    for name, path_fn in BRANCHES.items():
        frame = pd.read_csv(path_fn(fold) / f"{split}_predictions.csv")
        pcol = pred_col(frame)
        frames[name] = frame[["musicId", "time_sec", "target_arousal", pcol]].rename(columns={pcol: f"pred_{name}"})
    base = frames["h96"][["musicId", "time_sec", "target_arousal", "pred_h96"]].copy()
    for name, frame in frames.items():
        if name == "h96":
            continue
        base = base.merge(frame[["musicId", "time_sec", f"pred_{name}"]], on=["musicId", "time_sec"], how="inner")
    base = base.sort_values(["musicId", "time_sec"]).reset_index(drop=True)
    return base


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pred_cols = [f"pred_{name}" for name in BRANCHES]
    pred_values = out[pred_cols].to_numpy(dtype=np.float64)
    out["pred_mean"] = pred_values.mean(axis=1)
    out["pred_std"] = pred_values.std(axis=1)
    out["pred_min"] = pred_values.min(axis=1)
    out["pred_max"] = pred_values.max(axis=1)
    out["pred_spread"] = out["pred_max"] - out["pred_min"]
    out["time_norm"] = out.groupby("musicId")["time_sec"].transform(lambda s: s / max(float(s.max()), 1e-6))
    for col in pred_cols + ["pred_mean"]:
        out[f"{col}_song_mean"] = out.groupby("musicId")[col].transform("mean")
        out[f"{col}_centered"] = out[col] - out[f"{col}_song_mean"]
        out[f"{col}_delta"] = out.groupby("musicId")[col].diff().fillna(0.0)
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"musicId", "time_sec", "target_arousal"}
    return [col for col in df.columns if col not in skip]


def candidate_models() -> dict[str, Any]:
    return {
        "ridge_a0.01": make_pipeline(StandardScaler(), Ridge(alpha=0.01)),
        "ridge_a0.1": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "ridge_a1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_a10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "elastic_a0.001_l1.1": make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=20000, random_state=7)),
        "elastic_a0.003_l1.1": make_pipeline(StandardScaler(), ElasticNet(alpha=0.003, l1_ratio=0.1, max_iter=20000, random_state=7)),
        "huber_e1.35": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)),
        "hgb_l2_0.03": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=0.1, random_state=7),
        "hgb_l2_0.05": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.3, random_state=7),
        "extra_trees_light": ExtraTreesRegressor(n_estimators=160, max_depth=6, min_samples_leaf=10, random_state=7, n_jobs=-1),
        "rf_light": RandomForestRegressor(n_estimators=120, max_depth=7, min_samples_leaf=10, random_state=7, n_jobs=-1),
    }


def score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.02 * max(0.0, m["mae"] - 0.15) + 0.01 * max(0.0, 0.80 - m["ccc"])


def postprocess_search(base: pd.DataFrame, pred: np.ndarray) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    target = base["target_arousal"].to_numpy(dtype=np.float64)
    rows = []
    raw_metrics = metrics(pred, target)
    best = {"postprocess": "raw", "metrics": raw_metrics, **raw_metrics, "score": score(raw_metrics)}
    best_pred = pred.copy()
    for shift_frames in range(0, 9):
        shifted = shift_by_song(base, pred, shift_frames)
        for amp in [0.45, 0.55, 0.65, 0.75, 0.9, 1.0]:
            amplified = scale_amplitude_by_song(base, shifted, amp)
            for window in [1, 3, 7, 11, 15, 21, 31]:
                smoothed = smooth_by_song(base, amplified, window)
                for method in ["identity", "bias", "linear", "mean_std"]:
                    final, params = calibrate(smoothed, target, method)
                    final = np.clip(final, -1.0, 1.0)
                    m = metrics(final, target)
                    row = {
                        "postprocess": f"shift{shift_frames}_amp{amp:g}_w{window}_{method}",
                        "shift_frames": shift_frames,
                        "shift_sec": shift_frames * 0.5,
                        "amp": amp,
                        "window": window,
                        "method": method,
                        "params": json.dumps(params, ensure_ascii=True),
                        **m,
                        "score": score(m),
                    }
                    rows.append(row)
                    if (row["score"], row["rmse"], -row["ccc"]) < (best["score"], best["rmse"], -best["ccc"]):
                        best = {"metrics": m, **row}
                        best_pred = final
    return best, best_pred, pd.DataFrame(rows)


def run_fold(fold: str) -> tuple[dict[str, Any], pd.DataFrame]:
    train = add_features(load_split(fold, "train"))
    val = add_features(load_split(fold, "val"))
    cols = feature_cols(train)
    x_train = train[cols].to_numpy(dtype=np.float64)
    y_train = train["target_arousal"].to_numpy(dtype=np.float64)
    x_val = val[cols].to_numpy(dtype=np.float64)
    y_val = val["target_arousal"].to_numpy(dtype=np.float64)
    rows = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    best_pp_rows: pd.DataFrame | None = None
    for model_name, model in candidate_models().items():
        model.fit(x_train, y_train)
        pred = np.clip(model.predict(x_val), -1.0, 1.0)
        raw_m = metrics(pred, y_val)
        pp, pp_pred, pp_rows = postprocess_search(val, pred)
        row = {"fold": fold, "model": model_name, **{f"raw_{k}": v for k, v in raw_m.items()}, **{f"post_{k}": v for k, v in pp["metrics"].items()}, "postprocess": pp["postprocess"], "post_score": pp["score"]}
        rows.append(row)
        if best is None or (pp["score"], pp["metrics"]["rmse"], -pp["metrics"]["ccc"]) < (best["score"], best["metrics"]["rmse"], -best["metrics"]["ccc"]):
            best = {"fold": fold, "model": model_name, **pp}
            best_pred = pp_pred
            best_pp_rows = pp_rows.assign(model=model_name)
    assert best is not None and best_pred is not None and best_pp_rows is not None
    pred_out = val[["musicId", "time_sec", "target_arousal"]].copy()
    pred_out["pred_arousal"] = best_pred
    pred_out.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
    pd.DataFrame(rows).sort_values(["post_score", "post_rmse"]).to_csv(OUT_DIR / f"{fold}_model_comparison.csv", index=False)
    best_pp_rows.sort_values(["score", "rmse"]).to_csv(OUT_DIR / f"{fold}_postprocess_candidates.csv", index=False)
    return best, pred_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_fold = []
    all_pred = []
    for fold in ["fold_01", "fold_02", "fold_03"]:
        best, pred = run_fold(fold)
        per_fold.append({
            "fold": fold,
            "model": best["model"],
            "postprocess": best["postprocess"],
            **best["metrics"],
            "score": best["score"],
        })
        all_pred.append(pred)
    mean = {m: float(np.mean([row[m] for row in per_fold])) for m in ["rmse", "mae", "pcc", "ccc"]}
    merged = pd.concat(all_pred, ignore_index=True)
    global_metrics = metrics(merged["pred_arousal"].to_numpy(dtype=np.float64), merged["target_arousal"].to_numpy(dtype=np.float64))
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Train-fit Arousal stacker using train predictions, evaluated on locked development folds; final test excluded",
        "selection_boundary": "Development folds only; no final locked test usage",
        "per_fold": per_fold,
        "mean": mean,
        "global_frame": global_metrics,
        "gap_rows_mean": threshold_rows(mean),
        "gap_rows_global_frame": threshold_rows(global_metrics),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(per_fold).to_csv(OUT_DIR / "comparison.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
