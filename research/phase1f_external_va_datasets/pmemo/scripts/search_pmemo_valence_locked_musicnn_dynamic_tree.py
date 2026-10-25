#!/usr/bin/env python3
"""Cross-fold shallow nonlinear heads over past-context MusiCNN tags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from search_pmemo_valence_locked_musicnn_dynamic_tags import FOLDS, feature_matrix, interpolate_tags, macro
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_BASE = ROOT / "outputs/valence_locked_musicnn_tag_monotonic_20260714"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_musicnn_tag_sequences_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_musicnn_dynamic_tree_20260714"


def model_specs() -> list[tuple[str, Any]]:
    return [
        ("hgb_leaf7", HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=160, max_leaf_nodes=7, min_samples_leaf=60,
            l2_regularization=10.0, random_state=20260714,
        )),
        ("hgb_leaf15", HistGradientBoostingRegressor(
            learning_rate=0.025, max_iter=180, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=15.0, random_state=20260714,
        )),
        ("extra_leaf20", ExtraTreesRegressor(
            n_estimators=120, min_samples_leaf=20, max_features=0.5,
            random_state=20260714, n_jobs=-1,
        )),
        ("extra_leaf50", ExtraTreesRegressor(
            n_estimators=120, min_samples_leaf=50, max_features=0.7,
            random_state=20260714, n_jobs=-1,
        )),
        ("rf_leaf20", RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20, max_features=0.5,
            random_state=20260714, n_jobs=-1,
        )),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    features: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for fold in FOLDS:
        tags, position, weights[fold] = interpolate_tags(frames[fold], args.tag_dir)
        features[fold] = feature_matrix(frames[fold], tags, position, "tags_context_past_dense")

    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_rows)
    candidates: list[dict[str, Any]] = [{
        "model": "identity", "target_mode": "identity", "weight_mode": "identity",
        "strength": 0.0, "cap": 0.0, **base_macro, "guarded": True,
    }]
    raw_cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for model_name, model in model_specs():
        for target_mode in ["direct", "residual"]:
            for weight_mode in ["frame", "song"]:
                shifts: dict[str, np.ndarray] = {}
                for held in FOLDS:
                    train_folds = [fold for fold in FOLDS if fold != held]
                    x_train = np.concatenate([features[fold] for fold in train_folds])
                    target = np.concatenate([
                        frames[fold]["target_valence"].to_numpy(dtype=np.float64) for fold in train_folds
                    ])
                    base_train = np.concatenate([
                        frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in train_folds
                    ])
                    if target_mode == "residual":
                        target = target - base_train
                    fit_kwargs = {}
                    if weight_mode == "song":
                        fit_kwargs["sample_weight"] = np.concatenate([weights[fold] for fold in train_folds])
                    model.fit(x_train, target, **fit_kwargs)
                    estimate = model.predict(features[held])
                    if target_mode == "direct":
                        estimate = estimate - frames[held]["pred_valence"].to_numpy(dtype=np.float64)
                    shifts[held] = estimate
                raw_cache[(model_name, target_mode, weight_mode)] = shifts
                for strength in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
                    for cap in [0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040]:
                        per_fold = []
                        for fold in FOLDS:
                            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                            pred = np.clip(base + np.clip(strength * shifts[fold], -cap, cap), -1.0, 1.0)
                            per_fold.append(metrics(pred, frames[fold]["target_valence"]))
                        result = macro(per_fold)
                        guarded = (
                            result["mae"] <= base_macro["mae"] + 0.0005
                            and result["pcc"] >= base_macro["pcc"] - 0.0005
                            and result["ccc"] >= 0.80
                        )
                        candidates.append({
                            "model": model_name, "target_mode": target_mode, "weight_mode": weight_mode,
                            "strength": strength, "cap": cap, **result, "guarded": guarded,
                        })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    if selected["model"] == "identity":
        predictions = {fold: frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in FOLDS}
    else:
        shifts = raw_cache[(str(selected["model"]), str(selected["target_mode"]), str(selected["weight_mode"]))]
        predictions = {}
        for fold in FOLDS:
            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
            predictions[fold] = np.clip(
                base + np.clip(float(selected["strength"]) * shifts[fold], -float(selected["cap"]), float(selected["cap"])),
                -1.0, 1.0,
            )

    per_fold = []
    for fold in FOLDS:
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_dynamic_tree_pred_valence"})
        out["pred_valence"] = predictions[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Nonlinear head fit on other two locked development folds; final locked test excluded",
        "base_dir": str(args.base_dir), "tag_dir": str(args.tag_dir),
        "base_macro": base_macro, "selected": selected, "macro": selected_macro,
        "per_fold": per_fold,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
