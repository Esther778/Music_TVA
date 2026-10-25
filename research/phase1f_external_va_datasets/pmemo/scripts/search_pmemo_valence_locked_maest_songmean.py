#!/usr/bin/env python3
"""Cross-fold MAEST song-mean correction for the locked Valence candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_FEATURES = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_maest_30s_layer7_20260714/per_song"
DEFAULT_OUT = ROOT / "outputs/valence_locked_maest_songmean_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def load_fold(frame: pd.DataFrame, feature_dir: Path) -> pd.DataFrame:
    songs = frame.groupby("musicId", sort=False).agg(
        target_song=("target_valence", "mean"),
        base_mean=("pred_valence", "mean"),
        base_std=("pred_valence", "std"),
        base_min=("pred_valence", "min"),
        base_max=("pred_valence", "max"),
        frames=("pred_valence", "size"),
    ).reset_index()
    songs["base_std"] = songs["base_std"].fillna(0.0)
    songs["base_range"] = songs["base_max"] - songs["base_min"]
    means, mean_stds = [], []
    for music_id in songs["musicId"].astype(int):
        embedding = np.load(feature_dir / f"{music_id}.npz")["embedding"].astype(np.float64)
        mean = embedding.mean(axis=0)
        means.append(mean)
        mean_stds.append(np.concatenate([mean, embedding.std(axis=0)]))
    songs["maest_mean"] = means
    songs["maest_mean_std"] = mean_stds
    return songs


def block(table: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(table[column].to_list()).astype(np.float64)


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
    base_columns = ["base_mean", "base_std", "base_min", "base_max", "base_range", "frames"]
    reduced: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]] = {}
    for held in FOLDS:
        train = pd.concat([songs[fold] for fold in FOLDS if fold != held], ignore_index=True)
        test = songs[held]
        for mode in ["maest_mean", "maest_mean_std"]:
            for components in [8, 16, 32]:
                reducer = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=components, svd_solver="randomized", random_state=20260714),
                )
                x_train = reducer.fit_transform(block(train, mode))
                x_test = reducer.transform(block(test, mode))
                reduced[(held, mode, components)] = (x_train, x_test, train, test)

    candidates: list[dict[str, Any]] = [{
        "mode": "identity", "components": 0, "target_mode": "identity", "alpha": 0.0,
        "strength": 0.0, "cap": 0.0, **base_macro, "passes_thresholds": True,
    }]
    shift_cache: dict[tuple[str, int, str, float], dict[str, np.ndarray]] = {}
    for mode in ["maest_mean", "maest_mean_std"]:
        for components in [8, 16, 32]:
            for target_mode in ["residual", "direct"]:
                for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                    shifts = {}
                    for held in FOLDS:
                        x_train, x_test, train, test = reduced[(held, mode, components)]
                        train_stats = train[base_columns].to_numpy(dtype=np.float64)
                        test_stats = test[base_columns].to_numpy(dtype=np.float64)
                        x_train_full = np.column_stack([x_train, train_stats])
                        x_test_full = np.column_stack([x_test, test_stats])
                        target = train["target_song"].to_numpy(dtype=np.float64)
                        if target_mode == "residual":
                            target -= train["base_mean"].to_numpy(dtype=np.float64)
                        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                        model.fit(x_train_full, target)
                        estimate = model.predict(x_test_full)
                        if target_mode == "direct":
                            estimate -= test["base_mean"].to_numpy(dtype=np.float64)
                        shifts[held] = estimate
                    shift_cache[(mode, components, target_mode, alpha)] = shifts
                    for strength in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]:
                        for cap in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080, 0.120]:
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
                            candidates.append({
                                "mode": mode, "components": components, "target_mode": target_mode,
                                "alpha": alpha, "strength": strength, "cap": cap, **result,
                                "passes_thresholds": result["mae"] <= 0.15 and result["pcc"] >= 0.80 and result["ccc"] >= 0.80,
                            })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["passes_thresholds"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    per_fold_rows = []
    for fold in FOLDS:
        base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
        if selected["mode"] == "identity":
            frame_shift = np.zeros(len(base), dtype=np.float64)
            prediction = base
        else:
            key = (
                str(selected["mode"]), int(selected["components"]),
                str(selected["target_mode"]), float(selected["alpha"]),
            )
            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shift_cache[key][fold]))
            raw_shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
            frame_shift = np.clip(float(selected["strength"]) * raw_shift, -float(selected["cap"]), float(selected["cap"]))
            prediction = np.clip(base + frame_shift, -1.0, 1.0)
        output = frames[fold].copy()
        output["pre_maest_songmean_pred_valence"] = base
        output["maest_songmean_shift"] = frame_shift
        output["pred_valence"] = prediction
        output.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(prediction, output["target_valence"])})

    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Other-two-song-fold PCA/Ridge fit; final locked test excluded",
        "feature": "MAEST layer 7 CLS + DIST + mean patch tokens over full 30 s windows",
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
