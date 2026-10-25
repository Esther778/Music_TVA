#!/usr/bin/env python3
"""Song-level residual bias correction after prediction-only Valence rescale."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from search_pmemo_valence_song_bias_correction import apply_bias, eval_df, song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
OUT_DIR = ROOT / "outputs" / "valence_rescale_song_bias_ccc_preserving_20260628"
CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.1953163033728265,
    "mae": 0.15179290177231072,
    "pcc": 0.7977655929469954,
    "ccc": 0.7848342582143512,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CCC_EPS = 0.001


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


def selection_score(m: dict[str, float]) -> float:
    return (
        2.20 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.20 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.25 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.50 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def fit_and_predict(train_sf: pd.DataFrame, apply_sf: pd.DataFrame, alpha: float, degree: int) -> dict[Any, float]:
    if degree == 1:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    else:
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=alpha))
    model.fit(train_sf[FEATURE_COLS].to_numpy(dtype=np.float64), train_sf["residual_mean"].to_numpy(dtype=np.float64))
    pred = model.predict(apply_sf[FEATURE_COLS].to_numpy(dtype=np.float64))
    return dict(zip(apply_sf["musicId"], pred))


def evaluate_split(split_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    song_stats = {split: song_features(frames[split]) for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache: dict[tuple[str, int, float, float, str], pd.DataFrame] = {}
    for degree in [1, 2]:
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            pred_bias = {split: fit_and_predict(song_stats["train"], song_stats[split], alpha, degree) for split in ["train", "val", "test"]}
            for strength in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
                for cap_name, cap in [("cap005", 0.005), ("cap010", 0.010), ("cap020", 0.020), ("cap040", 0.040), ("cap080", 0.080), ("none", None)]:
                    split_metrics = {}
                    for split in ["train", "val", "test"]:
                        corrected = apply_bias(frames[split], pred_bias[split], strength, cap)
                        split_metrics[split] = eval_df(corrected)
                        cache[(split, degree, alpha, strength, cap_name)] = corrected
                    rows.append(
                        {
                            "split_name": split_dir.name,
                            "degree": degree,
                            "alpha": alpha,
                            "strength": strength,
                            "cap": cap_name,
                            **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                            "val_selection_score": selection_score(split_metrics["val"]),
                        }
                    )

    df = pd.DataFrame(rows)
    allowed = df[df["val_ccc"] >= baseline["val"]["ccc"] - CCC_EPS].copy()
    if allowed.empty:
        selected = {
            "split_name": split_dir.name,
            "source": "baseline",
            "degree": 0,
            "alpha": 0.0,
            "strength": 0.0,
            "cap": "none",
            **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
            "val_selection_score": None,
        }
    else:
        selected = allowed.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False]).iloc[0].to_dict()
        selected["source"] = "bias"

    out_split = OUT_DIR / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        if selected["source"] == "baseline":
            frames[split].to_csv(out_split / f"{split}_predictions.csv", index=False)
        else:
            cache[(split, int(selected["degree"]), float(selected["alpha"]), float(selected["strength"]), str(selected["cap"]))].to_csv(
                out_split / f"{split}_predictions.csv", index=False
            )
    summary = {
        "split_name": split_dir.name,
        "ccc_eps": CCC_EPS,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    return summary, df


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_rescale_song_bias_ccc_preserving_20260628",
        "input_model": CURRENT_BEST,
        "selection_rule": f"song bias selected only if val_ccc >= baseline_val_ccc - {CCC_EPS}",
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
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_input_model"] = {metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_bias_results.csv", index=False)
    pd.DataFrame([r["selected"] for r in rows]).to_csv(OUT_DIR / "selected_by_split.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
