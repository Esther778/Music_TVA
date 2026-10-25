#!/usr/bin/env python3
"""Cross-fold extreme-weighted MusiCNN song head for locked Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_musicnn_embedding_correction import (
    DEFAULT_FEATURES, FOLDS, feature_matrix, load_song_table, macro,
)
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_BASE = ROOT / "outputs" / "valence_locked_monotonic_song_calibration_20260714"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_extreme_weighted_song_head_20260714"


def sample_weights(target: np.ndarray, mode: str, amount: float) -> np.ndarray:
    if mode == "none":
        return np.ones_like(target)
    if mode == "distance":
        center = float(np.median(target))
        scale = float(np.std(target)) + 1e-8
        return 1.0 + amount * np.abs(target - center) / scale
    if mode == "tails":
        low, high = np.quantile(target, [0.20, 0.80])
        return 1.0 + amount * ((target <= low) | (target >= high)).astype(np.float64)
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    songs = {fold: load_song_table(frames[fold], fold, args.feature_dir) for fold in FOLDS}
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    weight_specs = [("none", 0.0)] + [("distance", x) for x in [0.25, 0.5, 1.0, 2.0, 3.0]] + [("tails", x) for x in [0.5, 1.0, 2.0, 3.0]]
    raw_cache: dict[tuple[str, float, float, str], dict[str, np.ndarray]] = {}
    candidates: list[dict[str, Any]] = [{"weight_mode": "identity", "weight_amount": 0.0, "alpha": 0.0, "target_mode": "identity", "strength": 0.0, "cap": 0.0, **base_macro, "guarded": True}]
    for weight_mode, weight_amount in weight_specs:
        for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
            for target_mode in ["direct", "residual"]:
                shifts: dict[str, np.ndarray] = {}
                for held in FOLDS:
                    train = pd.concat([songs[f] for f in FOLDS if f != held], ignore_index=True)
                    test = songs[held]
                    target = train["target_song"].to_numpy(dtype=np.float64)
                    weights = sample_weights(target, weight_mode, weight_amount)
                    if target_mode == "residual":
                        target = target - train["base_mean"].to_numpy(dtype=np.float64)
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(feature_matrix(train, "mean_std_base"), target, ridge__sample_weight=weights)
                    estimate = model.predict(feature_matrix(test, "mean_std_base"))
                    if target_mode == "direct":
                        estimate = estimate - test["base_mean"].to_numpy(dtype=np.float64)
                    shifts[held] = estimate
                raw_cache[(weight_mode, weight_amount, alpha, target_mode)] = shifts
                for strength in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
                    for cap in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080]:
                        per_fold = []
                        for fold in FOLDS:
                            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                            shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
                            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                            pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                            per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                        result = macro(per_fold)
                        guarded = (
                            result["mae"] <= base_macro["mae"] + 0.0005
                            and result["pcc"] >= base_macro["pcc"] - 0.0005
                            and result["ccc"] >= 0.80
                        )
                        candidates.append({
                            "weight_mode": weight_mode, "weight_amount": weight_amount,
                            "alpha": alpha, "target_mode": target_mode,
                            "strength": strength, "cap": cap, **result, "guarded": guarded,
                        })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    if selected["weight_mode"] == "identity":
        selected_preds = {fold: frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in FOLDS}
    else:
        shifts = raw_cache[(str(selected["weight_mode"]), float(selected["weight_amount"]), float(selected["alpha"]), str(selected["target_mode"]))]
        selected_preds = {}
        for fold in FOLDS:
            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
            shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
            selected_preds[fold] = np.clip(base + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])), -1.0, 1.0)
    per_fold_rows = []
    for fold in FOLDS:
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_extreme_head_pred_valence"})
        out["pred_valence"] = selected_preds[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Extreme weights and song head fit on the other two development folds only; final locked test excluded",
        "base_dir": str(args.base_dir), "base_macro": base_macro,
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
