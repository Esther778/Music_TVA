#!/usr/bin/env python3
"""Deployable frame-level calibrators for PMEmo Valence predictions.

The calibrator uses only prediction-derived features at inference time:
frame prediction, local smoothed prediction, normalized time position, and
song-level predicted trajectory statistics. It is trained on train labels and
selected on validation metrics.
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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_song_bias_correction_rmse_aware_20260628"
OUT_DIR = ROOT / "outputs" / "valence_frame_calibrator_20260628"
PREVIOUS_BEST = {
    "model_id": "pmemo_valence_song_bias_correction_rmse_aware_20260628",
    "rmse": 0.20050966596572084,
    "mae": 0.15405092914501886,
    "pcc": 0.7845220440639044,
    "ccc": 0.7720363870353871,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.20 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.20 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.20 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.40 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    s = pd.Series(values)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.sort_values(["musicId", "time_sec"]).groupby("musicId", sort=False):
        out = g.copy()
        pred = out["pred_valence"].to_numpy(dtype=np.float64)
        n = len(pred)
        pos = np.linspace(0.0, 1.0, n, dtype=np.float64) if n > 1 else np.zeros(n, dtype=np.float64)
        out["pos"] = pos
        out["pos2"] = pos**2
        out["pred"] = pred
        out["pred2"] = pred**2
        out["pred3"] = pred**3
        out["pred_abs"] = np.abs(pred)
        out["pred_ma3"] = moving_average(pred, 3)
        out["pred_ma9"] = moving_average(pred, 9)
        out["pred_delta"] = np.concatenate([[0.0], np.diff(pred)])
        out["song_pred_mean"] = float(np.mean(pred))
        out["song_pred_std"] = float(np.std(pred))
        out["song_pred_min"] = float(np.min(pred))
        out["song_pred_max"] = float(np.max(pred))
        out["song_pred_range"] = float(np.max(pred) - np.min(pred))
        out["song_pred_q10"] = float(np.quantile(pred, 0.10))
        out["song_pred_q90"] = float(np.quantile(pred, 0.90))
        out["song_frames"] = float(n)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


FEATURE_COLS = [
    "pred",
    "pred2",
    "pred3",
    "pred_abs",
    "pred_ma3",
    "pred_ma9",
    "pred_delta",
    "pos",
    "pos2",
    "song_pred_mean",
    "song_pred_std",
    "song_pred_min",
    "song_pred_max",
    "song_pred_range",
    "song_pred_q10",
    "song_pred_q90",
    "song_frames",
]


def model_candidates() -> list[tuple[str, Any]]:
    return [
        ("ridge_a0.1_poly1", make_pipeline(StandardScaler(), Ridge(alpha=0.1))),
        ("ridge_a1_poly1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a10_poly1", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a100_poly1", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a10_poly2", make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=10.0))),
        ("ridge_a100_poly2", make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=100.0))),
        ("elastic_a0001_l1_02", make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000))),
        ("elastic_a001_l1_02", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000))),
        (
            "hgb_lr003_leaf31_l2_001",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.03,
                max_iter=160,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=42,
            ),
        ),
        (
            "hgb_lr002_leaf15_l2_01",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=220,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
    ]


def eval_frame(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def corrected_df(df: pd.DataFrame, pred: np.ndarray, blend: float) -> pd.DataFrame:
    out = df.copy()
    raw = out["pred_valence"].to_numpy(dtype=np.float64)
    out["pred_valence"] = np.clip((1.0 - blend) * raw + blend * pred, -1.0, 1.0)
    return out


def evaluate_split(split_dir: str) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: add_features(pd.read_csv(INPUT_DIR / split_dir / f"{split}_predictions.csv")) for split in ["train", "val", "test"]}
    baseline = {split: eval_frame(frames[split]) for split in ["train", "val", "test"]}
    x_train = frames["train"][FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = frames["train"]["target_valence"].to_numpy(dtype=np.float64)
    rows = []
    cache: dict[tuple[str, str, float], pd.DataFrame] = {}
    for model_id, model in model_candidates():
        model.fit(x_train, y_train)
        pred_by_split = {
            split: np.clip(model.predict(frames[split][FEATURE_COLS].to_numpy(dtype=np.float64)), -1.0, 1.0)
            for split in ["train", "val", "test"]
        }
        for blend in [0.25, 0.50, 0.75, 1.00]:
            split_metrics = {}
            for split in ["train", "val", "test"]:
                out = corrected_df(frames[split], pred_by_split[split], blend)
                split_metrics[split] = eval_frame(out)
                cache[(split, model_id, blend)] = out
            rows.append(
                {
                    "split_name": split_dir,
                    "model_id": model_id,
                    "blend": blend,
                    **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                    "val_selection_score": selection_score(split_metrics["val"]),
                }
            )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = df.iloc[0].to_dict()
    split_out = OUT_DIR / selected["split_name"]
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        cache[(split, str(selected["model_id"]), float(selected["blend"]))].loc[
            :, ["musicId", "time_sec", "target_valence", "pred_valence"]
        ].to_csv(split_out / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {
            metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, df


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "created_at": "2026-06-28",
        "previous_best": PREVIOUS_BEST,
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["baseline"]["test"][metric] for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_previous_best"] = {
        metric: float(selected_mean[metric] - PREVIOUS_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p.name for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_calibrator_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
