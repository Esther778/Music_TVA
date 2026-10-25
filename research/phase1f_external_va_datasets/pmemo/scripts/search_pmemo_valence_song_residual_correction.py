#!/usr/bin/env python3
"""Validation-selected song-level residual correction for PMEmo Valence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_harmonic_valence_fusion import selected_indices
from train_pmemo_mert_valence_baseline import metrics, threshold_rows


ROOT = Path(__file__).resolve().parents[1]
BEST_DIR = ROOT / "outputs" / "mert_hidden6_harmonic_extreme_weighted" / "h6_harm_h80_ext125_clip010_ccc005"
MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
LIBROSA_DIR = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset" / "sequence"
OUT_DIR = ROOT / "outputs" / "valence_song_residual_correction_20260627"

HISTORICAL_BEST = {
    "model_id": "pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1",
    "trial_id": "h6_harm_h80_ext125_clip010_ccc005",
    "rmse": 0.205672949552536,
    "mae": 0.16456688940525055,
    "pcc": 0.7834003513415063,
    "ccc": 0.7673044285835935,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.10 * m["rmse"]
    )


def read_predictions(split: str) -> pd.DataFrame:
    df = pd.read_csv(BEST_DIR / f"{split}_predictions.csv")
    df["musicId"] = df["musicId"].astype(str)
    df["split"] = split
    return df


def safe_slope(time_sec: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.std(values) < 1e-8:
        return 0.0
    t = (time_sec - time_sec.min()) / max(float(time_sec.max() - time_sec.min()), 1e-6)
    return float(np.polyfit(t, values, 1)[0])


def pred_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("musicId", sort=True):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        target = g["target_valence"].to_numpy(dtype=np.float64)
        time_sec = g["time_sec"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "musicId": sid,
                "split": g["split"].iloc[0],
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


def feature_summary(sid: str, harmonic_idx: np.ndarray) -> dict[str, np.ndarray]:
    mert = np.load(MERT_DIR / f"{sid}.npz")["embedding"].astype(np.float32)
    lib = np.load(LIBROSA_DIR / f"{sid}.npz")["X"].astype(np.float32)[:, harmonic_idx]
    n = min(len(mert), len(lib))
    mert = mert[:n]
    lib = lib[:n]
    return {
        "mert_meanstd": np.concatenate([mert.mean(axis=0), mert.std(axis=0)]),
        "harm_meanstd": np.concatenate([lib.mean(axis=0), lib.std(axis=0)]),
    }


def build_song_table(all_pred: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    stats = pred_stats(all_pred)
    harmonic_idx = np.asarray(selected_indices(["harmonic"]), dtype=np.int64)
    feature_rows = []
    for row in stats.to_dict("records"):
        sid = str(row["musicId"])
        arrays = feature_summary(sid, harmonic_idx)
        flat = {"musicId": sid}
        for name, arr in arrays.items():
            for i, value in enumerate(arr):
                flat[f"{name}_{i}"] = float(value)
        feature_rows.append(flat)
    features = pd.DataFrame(feature_rows)
    table = stats.merge(features, on="musicId", how="left")
    pred_cols = [
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
    feature_sets = {
        "pred_stats": pred_cols,
        "pred_harm": pred_cols + [c for c in table.columns if c.startswith("harm_meanstd_")],
        "pred_mert": pred_cols + [c for c in table.columns if c.startswith("mert_meanstd_")],
        "pred_mert_harm": pred_cols
        + [c for c in table.columns if c.startswith("mert_meanstd_")]
        + [c for c in table.columns if c.startswith("harm_meanstd_")],
    }
    return table, feature_sets


def fit_residual_model(kind: str, alpha: float | None, x_train: np.ndarray, y_train: np.ndarray):
    if kind == "mean_bias":
        return {"kind": "mean_bias", "bias": float(np.mean(y_train))}
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    if kind == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=float(alpha), max_iter=1000, epsilon=1.35))
    raise ValueError(kind)


def predict_residual(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full((len(x),), float(model["bias"]), dtype=np.float64)
    return model.predict(x).astype(np.float64)


def apply_song_correction(frame_df: pd.DataFrame, residual_by_song: dict[str, float], strength: float, cap: float) -> np.ndarray:
    pred = frame_df["pred_valence"].to_numpy(dtype=np.float64).copy()
    corrections = frame_df["musicId"].astype(str).map(residual_by_song).fillna(0.0).to_numpy(dtype=np.float64)
    corrections = np.clip(corrections, -float(cap), float(cap)) * float(strength)
    return np.clip(pred + corrections, -1.0, 1.0)


def eval_split(split_df: pd.DataFrame, song_table: pd.DataFrame, residual_pred: np.ndarray, strength: float, cap: float) -> dict[str, float]:
    residual_by_song = dict(zip(song_table["musicId"].astype(str), residual_pred))
    pred = apply_song_correction(split_df, residual_by_song, strength, cap)
    target = split_df["target_valence"].to_numpy(dtype=np.float64)
    return metrics(pred, target)


def write_corrected_predictions(split: str, split_df: pd.DataFrame, song_table: pd.DataFrame, residual_pred: np.ndarray, strength: float, cap: float) -> None:
    residual_by_song = dict(zip(song_table["musicId"].astype(str), residual_pred))
    out = split_df.copy()
    out["pred_valence_raw"] = out["pred_valence"]
    out["song_residual_pred"] = out["musicId"].astype(str).map(residual_by_song).fillna(0.0)
    out["song_residual_applied"] = np.clip(out["song_residual_pred"].to_numpy(dtype=np.float64), -float(cap), float(cap)) * float(strength)
    out["pred_valence"] = np.clip(out["pred_valence_raw"].to_numpy(dtype=np.float64) + out["song_residual_applied"].to_numpy(dtype=np.float64), -1.0, 1.0)
    out.to_csv(OUT_DIR / f"selected_{split}_predictions.csv", index=False)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_frames = {split: read_predictions(split) for split in ["train", "val", "test"]}
    all_pred = pd.concat(split_frames.values(), ignore_index=True)
    song_table, feature_sets = build_song_table(all_pred)
    song_table.to_csv(OUT_DIR / "song_residual_feature_table.csv", index=False)

    rows = []
    strengths = [0.0, 0.25, 0.50, 0.75, 1.0]
    caps = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
    model_specs = [("mean_bias", None)]
    model_specs += [("ridge", alpha) for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]]
    model_specs += [("huber", alpha) for alpha in [0.0001, 0.001, 0.01, 0.1]]

    split_song = {split: song_table[song_table["split"] == split].reset_index(drop=True) for split in ["train", "val", "test"]}
    baseline = {
        split: metrics(
            split_frames[split]["pred_valence"].to_numpy(dtype=np.float64),
            split_frames[split]["target_valence"].to_numpy(dtype=np.float64),
        )
        for split in ["train", "val", "test"]
    }

    for feature_set, cols in feature_sets.items():
        x_train = split_song["train"][cols].to_numpy(dtype=np.float64)
        y_train = split_song["train"]["residual"].to_numpy(dtype=np.float64)
        x_by_split = {split: split_song[split][cols].to_numpy(dtype=np.float64) for split in ["train", "val", "test"]}
        for model_kind, alpha in model_specs:
            model = fit_residual_model(model_kind, alpha, x_train, y_train)
            if not isinstance(model, dict):
                model.fit(x_train, y_train)
            residual_by_split = {split: predict_residual(model, x_by_split[split]) for split in ["train", "val", "test"]}
            residual_metrics = {
                split: metrics(residual_by_split[split], split_song[split]["residual"].to_numpy(dtype=np.float64))
                for split in ["train", "val", "test"]
            }
            for strength in strengths:
                for cap in caps:
                    split_metrics = {
                        split: eval_split(split_frames[split], split_song[split], residual_by_split[split], strength, cap)
                        for split in ["train", "val", "test"]
                    }
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "model_kind": model_kind,
                            "alpha": alpha if alpha is not None else "",
                            "strength": strength,
                            "cap": cap,
                            **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                            **{f"{split}_residual_{k}": v for split, m in residual_metrics.items() for k, v in m.items()},
                            "val_selection_score": selection_score(split_metrics["val"]),
                        }
                    )

    results = pd.DataFrame(rows)
    results = results.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    results.to_csv(OUT_DIR / "all_song_residual_results.csv", index=False)
    results.head(200).to_csv(OUT_DIR / "top_song_residual_results.csv", index=False)
    selected = results.iloc[0].to_dict()

    cols = feature_sets[str(selected["feature_set"])]
    x_train = split_song["train"][cols].to_numpy(dtype=np.float64)
    y_train = split_song["train"]["residual"].to_numpy(dtype=np.float64)
    model = fit_residual_model(str(selected["model_kind"]), None if selected["alpha"] == "" else float(selected["alpha"]), x_train, y_train)
    if not isinstance(model, dict):
        model.fit(x_train, y_train)
    residual_by_split = {
        split: predict_residual(model, split_song[split][cols].to_numpy(dtype=np.float64)) for split in ["train", "val", "test"]
    }
    for split in ["train", "val", "test"]:
        write_corrected_predictions(
            split,
            split_frames[split],
            split_song[split],
            residual_by_split[split],
            float(selected["strength"]),
            float(selected["cap"]),
        )

    test_metrics = {k.removeprefix("test_"): float(v) for k, v in selected.items() if k.startswith("test_") and not k.startswith("test_residual_")}
    summary = {
        "created_at": "2026-06-27",
        "selection_rule": "choose residual correction by validation pass-gap composite; test not used for selection",
        "historical_best": HISTORICAL_BEST,
        "baseline": baseline,
        "selected": selected,
        "test_delta_vs_historical_best": {
            "rmse": float(test_metrics["rmse"] - HISTORICAL_BEST["rmse"]),
            "mae": float(test_metrics["mae"] - HISTORICAL_BEST["mae"]),
            "pcc": float(test_metrics["pcc"] - HISTORICAL_BEST["pcc"]),
            "ccc": float(test_metrics["ccc"] - HISTORICAL_BEST["ccc"]),
        },
        "selected_test_gap_rows": threshold_rows(test_metrics),
        "top_test_rmse": results.sort_values("test_rmse").iloc[0].to_dict(),
        "top_test_ccc": results.sort_values("test_ccc", ascending=False).iloc[0].to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
