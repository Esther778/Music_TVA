#!/usr/bin/env python3
"""Supervised top-k MAEST song-mean correction for locked Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_maest_songmean import (
    DEFAULT_BASE,
    DEFAULT_FEATURES,
    FOLDS,
    load_fold,
    macro,
)
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_OUT = ROOT / "outputs/valence_locked_maest_supervised_songmean_20260714"
BASE_COLUMNS = ["base_mean", "base_std", "base_min", "base_max", "base_range", "frames"]


def select_indices(x: np.ndarray, y: np.ndarray, top_k: int) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered, axis=0) * np.linalg.norm(y_centered)
    correlation = np.divide(
        x_centered.T @ y_centered,
        denominator,
        out=np.zeros(x.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.argsort(np.abs(correlation))[-top_k:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    songs = {fold: load_fold(frames[fold], args.feature_dir) for fold in FOLDS}
    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_rows)
    candidates = [{
        "target_mode": "identity", "top_k": 0, "alpha": 0.0, "strength": 0.0, "cap": 0.0,
        **base_macro, "guarded": True,
    }]
    shift_cache = {}
    for target_mode in ["residual", "direct"]:
        for top_k in [16, 32, 64, 128]:
            for alpha in [1.0, 10.0, 100.0, 1000.0]:
                shifts = {}
                for held in FOLDS:
                    train = pd.concat([songs[fold] for fold in FOLDS if fold != held], ignore_index=True)
                    test = songs[held]
                    x_train = np.stack(train["maest_mean"].to_list()).astype(np.float64)
                    x_test = np.stack(test["maest_mean"].to_list()).astype(np.float64)
                    target = train["target_song"].to_numpy(dtype=np.float64)
                    if target_mode == "residual":
                        target -= train["base_mean"].to_numpy(dtype=np.float64)
                    indices = select_indices(x_train, target, top_k)
                    x_train = np.column_stack([x_train[:, indices], train[BASE_COLUMNS]])
                    x_test = np.column_stack([x_test[:, indices], test[BASE_COLUMNS]])
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(x_train, target)
                    estimate = model.predict(x_test)
                    if target_mode == "direct":
                        estimate -= test["base_mean"].to_numpy(dtype=np.float64)
                    shifts[held] = estimate
                shift_cache[(target_mode, top_k, alpha)] = shifts
                for strength in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
                    for cap in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040]:
                        per_fold = []
                        for fold in FOLDS:
                            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                            frame_shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
                            prediction = np.clip(
                                frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                                + np.clip(strength * frame_shift, -cap, cap), -1.0, 1.0,
                            )
                            per_fold.append(metrics(prediction, frames[fold]["target_valence"]))
                        result = macro(per_fold)
                        guarded = (
                            result["mae"] <= base_macro["mae"]
                            and result["pcc"] >= base_macro["pcc"]
                            and result["ccc"] >= base_macro["ccc"]
                        )
                        candidates.append({
                            "target_mode": target_mode, "top_k": top_k, "alpha": alpha,
                            "strength": strength, "cap": cap, **result, "guarded": guarded,
                        })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    per_fold_rows = []
    for fold in FOLDS:
        base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
        if selected["target_mode"] == "identity":
            frame_shift = np.zeros(len(base), dtype=np.float64)
            prediction = base
        else:
            key = (str(selected["target_mode"]), int(selected["top_k"]), float(selected["alpha"]))
            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shift_cache[key][fold]))
            raw_shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
            frame_shift = np.clip(float(selected["strength"]) * raw_shift, -float(selected["cap"]), float(selected["cap"]))
            prediction = np.clip(base + frame_shift, -1.0, 1.0)
        output = frames[fold].copy()
        output["pre_maest_supervised_pred_valence"] = base
        output["maest_supervised_song_shift"] = frame_shift
        output["pred_valence"] = prediction
        output.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(prediction, output["target_valence"])})

    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Top-k feature selection and Ridge fit on other two song folds; final locked test excluded",
        "feature": "Supervised top-k dimensions from MAEST layer 7 full-window mean representation",
        "base_dir": str(args.base_dir),
        "base_macro": base_macro,
        "selected": selected,
        "macro": selected_macro,
        "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
