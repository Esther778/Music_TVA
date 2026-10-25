#!/usr/bin/env python3
"""Strict predicted-Arousal residual model for PMEmo Valence.

For each Valence split:
1. train an Arousal predictor on train songs only,
2. predict Arousal for train/val/test songs,
3. train a Valence residual corrector on train songs only,
4. select residual strength/cap/model on validation,
5. report held-out test metrics.

No true Arousal or Valence labels from val/test are used for fitting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
CURRENT_VALENCE_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
OPENSMILE_SEQUENCE_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
OUT_DIR = ROOT / "outputs" / "valence_oof_arousal_residual_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.6 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.2 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.0 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def eval_df(df: pd.DataFrame, pred_col: str = "pred_valence") -> dict[str, float]:
    return metrics(df[pred_col].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def load_opensmile_frames(index: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for row in index[index["split"] == split].to_dict("records"):
        sid = int(row["musicId"])
        z = np.load(OPENSMILE_SEQUENCE_DIR / f"{sid}.npz")
        x = z["X"].astype(np.float32)
        y = z["Y"].astype(np.float32)
        time_sec = z["time_sec"].astype(np.float32)
        part = pd.DataFrame({"musicId": sid, "time_sec": time_sec, "target_arousal": y[:, 1]})
        part["_X_index"] = range(len(part))
        rows.append((part, x))
    df = pd.concat([p for p, _ in rows], ignore_index=True)
    x_all = np.concatenate([x for _, x in rows], axis=0)
    return df, x_all


def arousal_models() -> list[tuple[str, Any]]:
    return [
        ("arousal_ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("arousal_ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("arousal_huber", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=300))),
        (
            "arousal_hgb",
            make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(learning_rate=0.04, max_iter=120, max_leaf_nodes=15, l2_regularization=0.1, random_state=42),
            ),
        ),
    ]


def residual_models() -> list[tuple[str, Any]]:
    return [
        ("resid_ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("resid_ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("resid_ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("resid_huber", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=400))),
        (
            "resid_hgb",
            make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(learning_rate=0.03, max_iter=120, max_leaf_nodes=11, l2_regularization=0.2, random_state=43),
            ),
        ),
        (
            "resid_rf",
            make_pipeline(
                StandardScaler(),
                RandomForestRegressor(n_estimators=180, min_samples_leaf=20, max_features=0.6, random_state=44, n_jobs=-1),
            ),
        ),
    ]


def enrich_valence_frames(frames: dict[str, pd.DataFrame], arousal_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for split, df in frames.items():
        ar = arousal_frames[split].copy()
        ar["time_key"] = ar["time_sec"].round(6)
        base = df.copy()
        base["time_key"] = base["time_sec"].round(6)
        merged = base.merge(ar[["musicId", "time_key", "pred_arousal", "target_arousal"]], on=["musicId", "time_key"], how="inner")
        if len(merged) != len(base):
            raise ValueError(f"Alignment mismatch {split}: valence={len(base)} merged={len(merged)}")
        merged["pred_valence_song_mean"] = merged.groupby("musicId")["pred_valence"].transform("mean")
        merged["pred_valence_song_std"] = merged.groupby("musicId")["pred_valence"].transform("std").fillna(0.0)
        merged["pred_arousal_song_mean"] = merged.groupby("musicId")["pred_arousal"].transform("mean")
        merged["pred_arousal_song_std"] = merged.groupby("musicId")["pred_arousal"].transform("std").fillna(0.0)
        merged["pred_arousal_delta"] = merged.groupby("musicId")["pred_arousal"].diff().fillna(0.0)
        merged["pred_arousal_centered"] = merged["pred_arousal"] - merged["pred_arousal_song_mean"]
        merged["pred_valence_centered"] = merged["pred_valence"] - merged["pred_valence_song_mean"]
        out[split] = merged
    return out


def fit_best_arousal(index: pd.DataFrame) -> tuple[str, Any, dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    split_frames: dict[str, pd.DataFrame] = {}
    split_x: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        df, x = load_opensmile_frames(index, split)
        split_frames[split] = df
        split_x[split] = x
    train_y = split_frames["train"]["target_arousal"].to_numpy(dtype=np.float64)
    best = None
    rows = []
    for model_id, model in arousal_models():
        model.fit(split_x["train"], train_y)
        val_pred = model.predict(split_x["val"])
        val_m = metrics(val_pred, split_frames["val"]["target_arousal"].to_numpy(dtype=np.float64))
        rows.append({"model_id": model_id, **{f"val_{k}": v for k, v in val_m.items()}})
        if best is None or val_m["rmse"] < best[2]["rmse"]:
            best = (model_id, model, val_m)
    assert best is not None
    model_id, model, _ = best
    metrics_by_split = {}
    pred_frames = {}
    for split in ["train", "val", "test"]:
        pred = model.predict(split_x[split])
        df = split_frames[split].copy()
        df["pred_arousal"] = pred.astype(np.float64)
        pred_frames[split] = df
        metrics_by_split[split] = metrics(pred, df["target_arousal"].to_numpy(dtype=np.float64))
    return model_id, model, pred_frames, metrics_by_split


def feature_cols() -> list[str]:
    return [
        "pred_valence",
        "pred_valence_song_mean",
        "pred_valence_song_std",
        "pred_valence_centered",
        "pred_arousal",
        "pred_arousal_song_mean",
        "pred_arousal_song_std",
        "pred_arousal_delta",
        "pred_arousal_centered",
        "time_sec",
    ]


def apply_residual(df: pd.DataFrame, residual: np.ndarray, strength: float, cap: float | None) -> np.ndarray:
    corr = residual.astype(np.float64)
    if cap is not None:
        corr = np.clip(corr, -cap, cap)
    return np.clip(df["pred_valence"].to_numpy(dtype=np.float64) + strength * corr, -1.0, 1.0)


def evaluate_split(split_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    split_name = split_dir.name
    index = pd.read_csv(split_dir / "sequence_index.csv")
    frames = {split: pd.read_csv(CURRENT_VALENCE_DIR / split_name / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    arousal_model_id, _, arousal_pred, arousal_metrics = fit_best_arousal(index)
    enriched = enrich_valence_frames(frames, arousal_pred)
    x_train = enriched["train"][feature_cols()].to_numpy(dtype=np.float64)
    y_resid = (enriched["train"]["target_valence"] - enriched["train"]["pred_valence"]).to_numpy(dtype=np.float64)
    rows = []
    cache = {}
    rows.append(
        {
            "split_name": split_name,
            "arousal_model_id": arousal_model_id,
            "residual_model_id": "no_change",
            "strength": 0.0,
            "cap": "none",
            **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
            "val_selection_score": selection_score(baseline["val"]),
        }
    )
    for split in ["train", "val", "test"]:
        cache[(split, "no_change", 0.0, "none")] = frames[split].copy()
    for residual_model_id, model in residual_models():
        model.fit(x_train, y_resid)
        pred_resid = {split: model.predict(enriched[split][feature_cols()].to_numpy(dtype=np.float64)) for split in ["train", "val", "test"]}
        for strength in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
            for cap_name, cap in [("cap005", 0.005), ("cap010", 0.010), ("cap020", 0.020), ("cap040", 0.040), ("cap070", 0.070), ("none", None)]:
                split_metrics = {}
                for split in ["train", "val", "test"]:
                    corrected = enriched[split].copy()
                    corrected["pred_valence"] = apply_residual(corrected, pred_resid[split], strength, cap)
                    split_metrics[split] = eval_df(corrected)
                    cache[(split, residual_model_id, strength, cap_name)] = corrected
                rows.append(
                    {
                        "split_name": split_name,
                        "arousal_model_id": arousal_model_id,
                        "residual_model_id": residual_model_id,
                        "strength": strength,
                        "cap": cap_name,
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    df = pd.DataFrame(rows)
    # Preserve CCC and MAE on validation unless the composite score improves clearly.
    viable = df[
        (df["val_ccc"] >= baseline["val"]["ccc"] - 0.001)
        & (df["val_pcc"] >= baseline["val"]["pcc"] - 0.002)
        & (df["val_mae"] <= baseline["val"]["mae"] + 0.001)
    ].copy()
    if viable.empty:
        viable = df.copy()
    viable = viable.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = viable.iloc[0].to_dict()
    split_out = OUT_DIR / split_name
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        key = (split, str(selected["residual_model_id"]), float(selected["strength"]), str(selected["cap"]))
        cache[key].to_csv(split_out / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_name,
        "baseline": baseline,
        "arousal_model_id": arousal_model_id,
        "arousal_metrics": arousal_metrics,
        "selected": selected,
        "delta_vs_current_best_test": {
            metric: float(selected[f"test_{metric}"] - baseline["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, df


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_oof_arousal_residual_20260628",
        "input_model": CURRENT_BEST,
        "protocol": "For each Valence split, train Arousal predictor and Valence residual corrector on train songs only; select residual config on validation.",
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            vals = np.asarray(
                [
                    r["selected"][f"test_{metric}"] if variant == "selected" else r["baseline"]["test"][metric]
                    for r in rows
                ],
                dtype=np.float64,
            )
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: out[f"baseline_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_baseline"] = {metric: float(selected_mean[metric] - baseline_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_delta_vs_current_best_reference"] = {metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    all_results = []
    for split_dir in sorted(SPLIT_DIR.glob("split_*")):
        summary, result_df = evaluate_split(split_dir)
        rows.append(summary)
        all_results.append(result_df)
    pd.concat(all_results, ignore_index=True).to_csv(OUT_DIR / "all_oof_arousal_residual_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
