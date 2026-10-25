#!/usr/bin/env python3
"""Cross-devfold MusiCNN mood correction for the strict Valence candidate."""

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

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_muq_arousal_stacker_fine_20260714"
DEFAULT_MOOD = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_musicnn_mood_semantics_val207_20260714/song_features.csv"
DEFAULT_OUT = ROOT / "outputs/valence_locked_musicnn_mood_correction_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def load_fold(base_dir: Path, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(base_dir / f"{fold}_val_predictions.csv")
    song = frame.groupby("musicId", sort=False).agg(
        target_song=("target_valence", "mean"), base_mean=("pred_valence", "mean"),
        base_std=("pred_valence", "std"), base_min=("pred_valence", "min"),
        base_max=("pred_valence", "max"), frames=("pred_valence", "size"),
    ).reset_index()
    song["base_std"] = song["base_std"].fillna(0.0)
    song["base_range"] = song["base_max"] - song["base_min"]
    song["fold"] = fold
    return frame, song


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mood-csv", type=Path, default=DEFAULT_MOOD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mood = pd.read_csv(args.mood_csv)
    frames, songs = {}, {}
    for fold in FOLDS:
        frames[fold], song = load_fold(args.base_dir, fold)
        songs[fold] = song.merge(mood, on="musicId", how="inner", validate="one_to_one")
        if len(songs[fold]) != song["musicId"].nunique():
            raise ValueError(f"Missing mood features for {fold}: {len(songs[fold])}/{len(song)}")

    mood_cols = [
        col for col in mood.columns
        if col.startswith("happy_") or col.startswith("sad_")
    ]
    feature_sets = {
        "happy": ["happy_mean", "happy_std", "happy_min", "happy_max", "happy_q25", "happy_q75"],
        "mood": mood_cols,
        "mood_base": mood_cols + ["base_mean", "base_std", "base_min", "base_max", "base_range", "frames"],
    }
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    raw_cache: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    candidates: list[dict[str, Any]] = []
    for feature_name, cols in feature_sets.items():
        for target_mode in ["direct", "residual"]:
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                raw_shift: dict[str, np.ndarray] = {}
                for held in FOLDS:
                    train = pd.concat([songs[f] for f in FOLDS if f != held], ignore_index=True)
                    test = songs[held]
                    target = train["target_song"] if target_mode == "direct" else train["target_song"] - train["base_mean"]
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(train[cols].to_numpy(dtype=np.float64), target.to_numpy(dtype=np.float64))
                    estimate = model.predict(test[cols].to_numpy(dtype=np.float64))
                    raw_shift[held] = estimate - test["base_mean"].to_numpy() if target_mode == "direct" else estimate
                raw_cache[(feature_name, target_mode, alpha)] = raw_shift
                for strength in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
                    for cap in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.20]:
                        per_fold = []
                        for fold in FOLDS:
                            song_shift = dict(zip(songs[fold]["musicId"].astype(int), raw_shift[fold]))
                            shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy()
                            pred = np.clip(frames[fold]["pred_valence"].to_numpy() + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                            per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy()))
                        result = macro(per_fold)
                        guarded = result["mae"] <= base_macro["mae"] + 0.001 and result["pcc"] >= base_macro["pcc"] - 0.001 and result["ccc"] >= base_macro["ccc"] - 0.001
                        candidates.append({"feature_set": feature_name, "target_mode": target_mode, "alpha": alpha, "strength": strength, "cap": cap, **result, "guarded": guarded})

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    key = (str(selected["feature_set"]), str(selected["target_mode"]), float(selected["alpha"]))
    raw_shift = raw_cache[key]
    per_fold_rows = []
    for fold in FOLDS:
        song_shift = dict(zip(songs[fold]["musicId"].astype(int), raw_shift[fold]))
        shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy()
        out = frames[fold].copy()
        out["pre_musicnn_mood_pred_valence"] = out["pred_valence"].to_numpy(dtype=np.float64)
        out["musicnn_song_shift"] = shift
        out["pred_valence"] = np.clip(out["pre_musicnn_mood_pred_valence"] + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])), -1.0, 1.0)
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Cross-devfold song-level fit; final locked test excluded",
        "base": str(args.base_dir), "base_macro": base_macro,
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
