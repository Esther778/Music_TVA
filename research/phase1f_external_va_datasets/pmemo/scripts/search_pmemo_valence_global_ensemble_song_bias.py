#!/usr/bin/env python3
"""Validation-safe song-bias correction for the global Valence ensemble.

For each split, the residual-bias model is fit on train songs only. Hyperparams
are selected once using macro validation metrics across all three splits, then
reported on held-out test splits. No test labels are used for selection.
"""

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


INPUT_DIR = ROOT / "outputs" / "valence_multilayer_global_ensemble_20260629"
OUT_DIR = ROOT / "outputs" / "valence_global_ensemble_song_bias_20260629"
CURRENT_STRICTISH = {
    "model_id": "pmemo_valence_multilayer_global_ensemble_20260629",
    "rmse": 0.19411262662053563,
    "mae": 0.15142593782241862,
    "pcc": 0.7998908886855055,
    "ccc": 0.7851966731300748,
}
EXPLORATORY_BEST = {
    "model_id": "pmemo_valence_universal_metric_optimized_exploratory_20260628",
    "rmse": 0.19182218120147795,
    "mae": 0.14919783550199786,
    "pcc": 0.8044833131619238,
    "ccc": 0.7881895323604796,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
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
        2.50 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.25 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.50 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.75 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
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


def macro_from_metrics(split_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {k: float(np.mean([m[k] for m in split_metrics.values()])) for k in ["rmse", "mae", "pcc", "ccc"]}


def evaluate_candidate(
    frames: dict[str, dict[str, pd.DataFrame]],
    song_stats: dict[str, dict[str, pd.DataFrame]],
    degree: int,
    alpha: float,
    strength: float,
    cap: float | None,
) -> tuple[dict[str, Any], dict[tuple[str, str], pd.DataFrame]]:
    corrected_cache: dict[tuple[str, str], pd.DataFrame] = {}
    split_metrics: dict[str, dict[str, dict[str, float]]] = {split: {} for split in ["train", "val", "test"]}
    for split_name in sorted(frames["train"]):
        pred_bias = {
            split: fit_and_predict(song_stats["train"][split_name], song_stats[split][split_name], alpha, degree)
            for split in ["train", "val", "test"]
        }
        for split in ["train", "val", "test"]:
            corrected = apply_bias(frames[split][split_name], pred_bias[split], strength, cap)
            corrected_cache[(split_name, split)] = corrected
            split_metrics[split][split_name] = eval_df(corrected)
    result: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        macro = macro_from_metrics(split_metrics[split])
        result.update({f"{split}_macro_{k}": v for k, v in macro.items()})
        result[f"{split}_per_split"] = [{"split_name": name, **m} for name, m in split_metrics[split].items()]
    result["val_selection_score"] = selection_score({k: result[f"val_macro_{k}"] for k in ["rmse", "mae", "pcc", "ccc"]})
    return result, corrected_cache


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_names = sorted(p.name for p in INPUT_DIR.glob("split_*") if p.is_dir())
    frames = {
        split: {split_name: pd.read_csv(INPUT_DIR / split_name / f"{split}_predictions.csv") for split_name in split_names}
        for split in ["train", "val", "test"]
    }
    song_stats = {
        split: {split_name: song_features(frames[split][split_name]) for split_name in split_names}
        for split in ["train", "val", "test"]
    }
    baseline_split_metrics = {
        split: {split_name: eval_df(frames[split][split_name]) for split_name in split_names}
        for split in ["train", "val", "test"]
    }
    baseline = {split: macro_from_metrics(baseline_split_metrics[split]) for split in ["train", "val", "test"]}
    candidates = []
    cache_by_key: dict[tuple[int, float, float, str], dict[tuple[str, str], pd.DataFrame]] = {}
    for degree in [1, 2]:
        for alpha in [1.0, 10.0, 100.0, 1000.0, 3000.0]:
            for strength in [0.025, 0.05, 0.10, 0.20, 0.35, 0.50]:
                for cap_name, cap in [("cap0025", 0.0025), ("cap005", 0.005), ("cap010", 0.010), ("cap020", 0.020), ("cap040", 0.040), ("none", None)]:
                    result, corrected = evaluate_candidate(frames, song_stats, degree, alpha, strength, cap)
                    row = {
                        "degree": degree,
                        "alpha": alpha,
                        "strength": strength,
                        "cap": cap_name,
                        **{k: v for k, v in result.items() if not k.endswith("_per_split")},
                    }
                    candidates.append(row)
                    cache_by_key[(degree, alpha, strength, cap_name)] = corrected
    cand_df = pd.DataFrame(candidates)
    baseline_val = baseline["val"]
    allowed = cand_df[
        (cand_df["val_macro_pcc"] >= baseline_val["pcc"] - 0.001)
        & (cand_df["val_macro_ccc"] >= baseline_val["ccc"] - 0.001)
    ].copy()
    if allowed.empty:
        allowed = cand_df.copy()
    selected = allowed.sort_values(
        ["val_selection_score", "val_macro_rmse", "val_macro_mae", "val_macro_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    selected_key = (int(selected["degree"]), float(selected["alpha"]), float(selected["strength"]), str(selected["cap"]))
    for (split_name, split), df in cache_by_key[selected_key].items():
        out_dir = OUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    selected_result, _ = evaluate_candidate(
        frames,
        song_stats,
        int(selected["degree"]),
        float(selected["alpha"]),
        float(selected["strength"]),
        None if selected["cap"] == "none" else {
            "cap0025": 0.0025,
            "cap005": 0.005,
            "cap010": 0.010,
            "cap020": 0.020,
            "cap040": 0.040,
        }[str(selected["cap"])],
    )
    test_macro = {k: selected_result[f"test_macro_{k}"] for k in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_global_ensemble_song_bias_20260629",
        "purpose": "validation-safe song-level residual bias correction for the global Valence ensemble",
        "selection_rule": "bias model fit on train songs per split; one hyperparameter set selected on macro validation; test labels not used",
        "input_model": CURRENT_STRICTISH,
        "exploratory_best_reference": EXPLORATORY_BEST,
        "baseline_macro": baseline,
        "selected": selected,
        "selected_macro": {
            "val": {k: selected_result[f"val_macro_{k}"] for k in ["rmse", "mae", "pcc", "ccc"]},
            "test": test_macro,
            "test_per_split": selected_result["test_per_split"],
        },
        "test_gap_rows": threshold_rows(test_macro),
        "delta_vs_input_model": {k: float(test_macro[k] - CURRENT_STRICTISH[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
        "delta_vs_exploratory_best": {k: float(test_macro[k] - EXPLORATORY_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
    }
    cand_df.to_csv(OUT_DIR / "all_song_bias_candidates.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
