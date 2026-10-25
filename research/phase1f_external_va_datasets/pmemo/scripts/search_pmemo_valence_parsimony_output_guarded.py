#!/usr/bin/env python3
"""Guarded output-level search for the current Valence parsimony best.

This script does not retrain the audio model. It starts from
`pmemo_valence_per_split_direct_multilayer_parsimony_20260629`, fits only
lightweight prediction transforms on train predictions, chooses transforms on
validation predictions, and reports test metrics after selection.

The guard is intentionally conservative:
- identity/no-change is always a candidate;
- a transform must improve validation selection score enough to be selected;
- it must not materially worsen validation RMSE or CCC.

This keeps the experiment deployable and avoids using test labels for tuning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_per_split_direct_multilayer_parsimony_20260629"
OUT_DIR = ROOT / "outputs" / "valence_parsimony_output_guarded_20260707"

CURRENT_BEST = {
    "model_id": "pmemo_valence_per_split_direct_multilayer_parsimony_20260629",
    "rmse": 0.19130205118779167,
    "mae": 0.14893115017555528,
    "pcc": 0.8059953934483183,
    "ccc": 0.7914302668500032,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
GOOD = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
EXCELLENT = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}


@dataclass
class Transform:
    name: str
    apply: Callable[[np.ndarray], np.ndarray]


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column found in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    if "target_valence" in df.columns:
        return "target_valence"
    if "target" in df.columns:
        return "target"
    raise ValueError(f"No target column found in {list(df.columns)}")


def eval_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64))


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return eval_arrays(df[pred_col(df)].to_numpy(), df[target_col(df)].to_numpy())


def selection_score(m: dict[str, float]) -> float:
    # Heavier pressure on remaining bottlenecks, while keeping MAE/PCC visible.
    return (
        3.00 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.25 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.25 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.50 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def moving_average_by_song(df: pd.DataFrame, window: int) -> np.ndarray:
    pcol = pred_col(df)
    if window <= 1:
        return df[pcol].to_numpy(dtype=np.float64)
    out = np.empty(len(df), dtype=np.float64)
    kernel = np.ones(int(window), dtype=np.float64)
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        vals = df.loc[idx, pcol].to_numpy(dtype=np.float64)
        start = (int(window) - 1) // 2
        summed = np.convolve(vals, kernel, mode="full")[start : start + len(vals)]
        counts = np.convolve(np.ones_like(vals, dtype=np.float64), kernel, mode="full")[start : start + len(vals)]
        smoothed = summed / np.maximum(counts, 1.0)
        out[np.asarray(idx)] = smoothed
    return out


def fit_transforms(x_train: np.ndarray, y_train: np.ndarray) -> list[Transform]:
    transforms: list[Transform] = [Transform("identity", lambda x: x)]

    # Affine least-squares calibration.
    x_mean = float(np.mean(x_train))
    y_mean = float(np.mean(y_train))
    x_var = float(np.var(x_train) + 1e-8)
    slope = float(np.mean((x_train - x_mean) * (y_train - y_mean)) / x_var)
    intercept = y_mean - slope * x_mean
    transforms.append(Transform("affine", lambda x, a=slope, b=intercept: a * x + b))

    # Shrunk affine variants are often safer for small validation splits.
    for strength in [0.15, 0.25, 0.35, 0.50, 0.75]:
        transforms.append(
            Transform(
                f"affine_blend_train_s{strength:.2f}",
                lambda x, a=slope, b=intercept, s=float(strength): (1.0 - s) * x + s * (a * x + b),
            )
        )

    # Mean-only and std-only train distribution matching.
    target_std = float(np.std(y_train) + 1e-8)
    pred_std = float(np.std(x_train) + 1e-8)
    transforms.append(Transform("mean_match", lambda x, xm=x_mean, ym=y_mean: x - xm + ym))
    transforms.append(Transform("std_match", lambda x, xm=x_mean, ym=y_mean, xs=pred_std, ys=target_std: (x - xm) * (ys / xs) + ym))

    # Conservative std expansion/compression around train prediction mean.
    for gain in [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]:
        transforms.append(Transform(f"std_gain_{gain:.2f}", lambda x, xm=x_mean, g=float(gain): xm + (x - xm) * g))
    return transforms


def apply_prediction(df: pd.DataFrame, pred: np.ndarray, tag: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    out[pred_col(out)] = pred
    for key, value in tag.items():
        out[key] = value
    return out


def score_candidate(
    frames: dict[str, pd.DataFrame],
    smooth_window: int,
    transform_name: str,
    transform: Transform,
    clip: float | None,
    smooth_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    split_preds: dict[str, np.ndarray] = {}
    split_metrics: dict[str, dict[str, float]] = {}
    for split, df in frames.items():
        raw = smooth_cache[(split, smooth_window)]
        pred = transform.apply(raw)
        if clip is not None:
            pred = np.clip(pred, -float(clip), float(clip))
        split_preds[split] = pred
        split_metrics[split] = eval_arrays(pred, df[target_col(df)].to_numpy(dtype=np.float64))
    return split_metrics, split_preds


def passes_guard(candidate: dict[str, float], baseline: dict[str, float], base_score: float, candidate_score: float) -> bool:
    if candidate_score > base_score - 0.0005:
        return False
    if candidate["rmse"] > baseline["rmse"] + 0.001:
        return False
    if candidate["ccc"] < baseline["ccc"] - 0.001:
        return False
    return True


def evaluate_split(split_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    base_score = selection_score(baseline["val"])
    windows = [1, 3, 5, 7, 9, 11, 15, 21]
    smooth_cache = {
        (split, window): moving_average_by_song(df, window)
        for split, df in frames.items()
        for window in windows
    }

    rows = []
    pred_cache: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    for smooth_window in windows:
        train_smoothed = smooth_cache[("train", smooth_window)]
        y_train = frames["train"][target_col(frames["train"])].to_numpy(dtype=np.float64)
        for transform in fit_transforms(train_smoothed, y_train):
            for clip_name, clip in [("none", None), ("clip095", 0.95), ("clip090", 0.90)]:
                split_metrics, split_preds = score_candidate(frames, smooth_window, transform.name, transform, clip, smooth_cache)
                val_score = selection_score(split_metrics["val"])
                key = (smooth_window, transform.name, clip_name)
                pred_cache[key] = split_preds
                rows.append(
                    {
                        "split_name": split_dir.name,
                        "smooth_window": int(smooth_window),
                        "transform": transform.name,
                        "clip": clip_name,
                        **{f"{split}_{metric}": float(value) for split, vals in split_metrics.items() for metric, value in vals.items()},
                        "val_selection_score": float(val_score),
                        "passes_guard": passes_guard(split_metrics["val"], baseline["val"], base_score, val_score),
                    }
                )

    df = pd.DataFrame(rows)
    guarded = df[df["passes_guard"]].sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    if guarded.empty:
        selected = {
            "split_name": split_dir.name,
            "smooth_window": 1,
            "transform": "identity",
            "clip": "none",
            **{f"{split}_{metric}": float(value) for split, vals in baseline.items() for metric, value in vals.items()},
            "val_selection_score": float(base_score),
            "passes_guard": True,
            "selected_reason": "identity retained because no candidate passed validation guard",
        }
        selected_preds = {split: frames[split][pred_col(frames[split])].to_numpy(dtype=np.float64) for split in ["train", "val", "test"]}
    else:
        selected = guarded.iloc[0].to_dict()
        selected["selected_reason"] = "validation guard passed"
        key = (int(selected["smooth_window"]), str(selected["transform"]), str(selected["clip"]))
        selected_preds = pred_cache[key]

    out_split = OUT_DIR / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)
    tag = {
        "guarded_output_smooth_window": selected["smooth_window"],
        "guarded_output_transform": selected["transform"],
        "guarded_output_clip": selected["clip"],
    }
    for split in ["train", "val", "test"]:
        apply_prediction(frames[split], selected_preds[split], tag).to_csv(out_split / f"{split}_predictions.csv", index=False)

    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {
            metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (out_split / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, df


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
        "purpose": "guarded validation-selected output-level search for current Valence parsimony best",
        "input_model": CURRENT_BEST,
        "selection_rule": "fit transforms on train predictions; select by validation score only; retain identity unless candidate improves validation score and does not materially worsen validation RMSE/CCC",
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([float(r["selected"][f"test_{metric}"]) for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([float(r["baseline"]["test"][metric]) for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: float(out[f"selected_test_{metric}_mean"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: float(out[f"baseline_test_{metric}_mean"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_input_model"] = {metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_delta_vs_baseline_recomputed"] = {metric: float(selected_mean[metric] - baseline_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*_seed_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_guarded_output_candidates.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
