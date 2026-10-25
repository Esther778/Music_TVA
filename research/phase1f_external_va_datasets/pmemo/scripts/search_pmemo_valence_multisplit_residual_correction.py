#!/usr/bin/env python3
"""Residual correction evaluation for PMEmo Valence multi-split predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


BASE_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
OUT_DIR = ROOT / "outputs" / "valence_multisplit_residual_correction_20260627"

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.10 * m["rmse"]
    )


def safe_slope(time_sec: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.std(values) < 1e-8:
        return 0.0
    t = (time_sec - time_sec.min()) / max(float(time_sec.max() - time_sec.min()), 1e-6)
    return float(np.polyfit(t, values, 1)[0])


def pred_stats(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("musicId", sort=True):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        target = g["target_valence"].to_numpy(dtype=np.float64)
        time_sec = g["time_sec"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "musicId": str(sid),
                "split": split,
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "pred_range": float(np.max(pred) - np.min(pred)),
                "pred_q25": float(np.quantile(pred, 0.25)),
                "pred_q50": float(np.quantile(pred, 0.50)),
                "pred_q75": float(np.quantile(pred, 0.75)),
                "pred_first": float(pred[0]),
                "pred_last": float(pred[-1]),
                "pred_slope": safe_slope(time_sec, pred),
                "target_mean": float(np.mean(target)),
                "residual": float(np.mean(target) - np.mean(pred)),
                "frames": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def split_metrics(df: pd.DataFrame, residual_by_song: dict[str, float] | None = None, strength: float = 0.0, cap: float = 0.0) -> dict[str, float]:
    pred = df["pred_valence"].to_numpy(dtype=np.float64).copy()
    if residual_by_song is not None and strength:
        corr = df["musicId"].astype(str).map(residual_by_song).fillna(0.0).to_numpy(dtype=np.float64)
        pred = np.clip(pred + np.clip(corr, -float(cap), float(cap)) * float(strength), -1.0, 1.0)
    target = df["target_valence"].to_numpy(dtype=np.float64)
    return metrics(pred, target)


def fit_model(kind: str, alpha: float | None, x: np.ndarray, y: np.ndarray):
    if kind == "mean_bias":
        return {"bias": float(np.mean(y))}
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(x, y)
    return model


def predict_model(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full((len(x),), float(model["bias"]), dtype=np.float64)
    return model.predict(x).astype(np.float64)


def evaluate_split_dir(split_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {}
    songs = {}
    for split in ["train", "val", "test"]:
        frames[split] = pd.read_csv(split_dir / f"{split}_predictions.csv")
        frames[split]["musicId"] = frames[split]["musicId"].astype(str)
        songs[split] = pred_stats(frames[split], split)
    cols = [
        "pred_mean",
        "pred_std",
        "pred_min",
        "pred_max",
        "pred_range",
        "pred_q25",
        "pred_q50",
        "pred_q75",
        "pred_first",
        "pred_last",
        "pred_slope",
        "frames",
    ]
    candidates = []
    specs = [("mean_bias", None)] + [("ridge", alpha) for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]]
    strengths = [0.0, 0.25, 0.50, 0.75, 1.0]
    caps = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
    x_train = songs["train"][cols].to_numpy(dtype=np.float64)
    y_train = songs["train"]["residual"].to_numpy(dtype=np.float64)
    baseline = {split: split_metrics(frames[split]) for split in ["train", "val", "test"]}
    for kind, alpha in specs:
        model = fit_model(kind, alpha, x_train, y_train)
        residual_pred = {
            split: predict_model(model, songs[split][cols].to_numpy(dtype=np.float64)) for split in ["train", "val", "test"]
        }
        for strength in strengths:
            for cap in caps:
                residual_maps = {
                    split: dict(zip(songs[split]["musicId"], residual_pred[split])) for split in ["train", "val", "test"]
                }
                ms = {
                    split: split_metrics(frames[split], residual_maps[split], strength, cap)
                    for split in ["train", "val", "test"]
                }
                candidates.append(
                    {
                        "split_name": split_dir.name,
                        "model_kind": kind,
                        "alpha": "" if alpha is None else alpha,
                        "strength": strength,
                        "cap": cap,
                        **{f"{split}_{k}": v for split, m in ms.items() for k, v in m.items()},
                        "val_selection_score": selection_score(ms["val"]),
                    }
                )
    cand_df = pd.DataFrame(candidates).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = cand_df.iloc[0].to_dict()
    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_baseline_test": {
            "rmse": float(selected["test_rmse"] - baseline["test"]["rmse"]),
            "mae": float(selected["test_mae"] - baseline["test"]["mae"]),
            "pcc": float(selected["test_pcc"] - baseline["test"]["pcc"]),
            "ccc": float(selected["test_ccc"] - baseline["test"]["ccc"]),
        },
    }
    return summary, cand_df


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"created_at": "2026-06-27", "num_splits": len(rows), "per_split": rows}
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "baseline":
                vals = np.asarray([r["baseline"]["test"][metric] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: out[f"baseline_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_baseline_mean"] = {
        "rmse": float(selected_mean["rmse"] - baseline_mean["rmse"]),
        "mae": float(selected_mean["mae"] - baseline_mean["mae"]),
        "pcc": float(selected_mean["pcc"] - baseline_mean["pcc"]),
        "ccc": float(selected_mean["ccc"] - baseline_mean["ccc"]),
    }
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    all_candidates = []
    for split_dir in sorted(BASE_DIR.glob("split_*")):
        if not (split_dir / "test_predictions.csv").exists():
            continue
        summary, cand_df = evaluate_split_dir(split_dir)
        rows.append(summary)
        all_candidates.append(cand_df)
    all_candidates_df = pd.concat(all_candidates, ignore_index=True)
    all_candidates_df.to_csv(OUT_DIR / "all_multisplit_residual_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
