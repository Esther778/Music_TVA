#!/usr/bin/env python3
"""Song-level bias correction for PMEmo Valence predictions.

This is a deployable postprocess: it uses only statistics of the predicted
Valence trajectory to estimate a per-song bias. It does not use target labels at
inference time.
"""

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


INPUT_DIR = ROOT / "outputs" / "valence_shift_postprocess_20260628"
OUT_DIR = ROOT / "outputs" / "valence_song_bias_correction_20260628"
PREVIOUS_BEST = {
    "model_id": "pmemo_valence_shift_postprocess_20260628",
    "rmse": 0.20098223017249192,
    "mae": 0.1548021408458333,
    "pcc": 0.782259890296069,
    "ccc": 0.7671011152983341,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        0.80 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 0.80 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.50 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.50 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def song_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.sort_values(["musicId", "time_sec"]).groupby("musicId", sort=False):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        target = g["target_valence"].to_numpy(dtype=np.float64)
        n = len(pred)
        head = pred[: max(1, n // 5)]
        tail = pred[-max(1, n // 5) :]
        rows.append(
            {
                "musicId": sid,
                "frames": n,
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "pred_range": float(np.max(pred) - np.min(pred)),
                "pred_q10": float(np.quantile(pred, 0.10)),
                "pred_q50": float(np.quantile(pred, 0.50)),
                "pred_q90": float(np.quantile(pred, 0.90)),
                "pred_head_mean": float(np.mean(head)),
                "pred_tail_mean": float(np.mean(tail)),
                "pred_tail_minus_head": float(np.mean(tail) - np.mean(head)),
                "target_mean": float(np.mean(target)),
                "residual_mean": float(np.mean(target) - np.mean(pred)),
            }
        )
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "frames",
    "pred_mean",
    "pred_std",
    "pred_min",
    "pred_max",
    "pred_range",
    "pred_q10",
    "pred_q50",
    "pred_q90",
    "pred_head_mean",
    "pred_tail_mean",
    "pred_tail_minus_head",
]


def apply_bias(df: pd.DataFrame, bias_by_song: dict[Any, float], strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    bias = out["musicId"].map(bias_by_song).fillna(0.0).to_numpy(dtype=np.float64)
    if cap is not None:
        bias = np.clip(bias, -cap, cap)
    out["song_bias"] = strength * bias
    out["pred_valence"] = np.clip(out["pred_valence"].to_numpy(dtype=np.float64) + out["song_bias"].to_numpy(dtype=np.float64), -1.0, 1.0)
    return out


def fit_and_predict(train_sf: pd.DataFrame, apply_sf: pd.DataFrame, alpha: float) -> dict[Any, float]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_sf[FEATURE_COLS].to_numpy(dtype=np.float64), train_sf["residual_mean"].to_numpy(dtype=np.float64))
    pred = model.predict(apply_sf[FEATURE_COLS].to_numpy(dtype=np.float64))
    return dict(zip(apply_sf["musicId"], pred))


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def evaluate_split(split_dir: str) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: pd.read_csv(INPUT_DIR / split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    song_stats = {split: song_features(frames[split]) for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache: dict[tuple[str, float, float, str], pd.DataFrame] = {}
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        pred_bias = {split: fit_and_predict(song_stats["train"], song_stats[split], alpha) for split in ["train", "val", "test"]}
        for strength in [0.25, 0.50, 0.75, 1.00]:
            for cap_name, cap in [("cap004", 0.04), ("cap008", 0.08), ("cap012", 0.12), ("cap018", 0.18), ("none", None)]:
                split_metrics = {}
                for split in ["train", "val", "test"]:
                    corrected = apply_bias(frames[split], pred_bias[split], strength, cap)
                    split_metrics[split] = eval_df(corrected)
                    cache[(split, alpha, strength, cap_name)] = corrected
                rows.append(
                    {
                        "split_name": split_dir,
                        "alpha": alpha,
                        "strength": strength,
                        "cap": cap_name,
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_ccc"], ascending=[True, True, False])
    selected = df.iloc[0].to_dict()
    split_out = OUT_DIR / selected["split_name"]
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        cache[(split, float(selected["alpha"]), float(selected["strength"]), str(selected["cap"]))].to_csv(split_out / f"{split}_predictions.csv", index=False)
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
    out: dict[str, Any] = {"created_at": "2026-06-28", "previous_best": PREVIOUS_BEST, "num_splits": len(rows), "per_split": rows}
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
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_bias_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
