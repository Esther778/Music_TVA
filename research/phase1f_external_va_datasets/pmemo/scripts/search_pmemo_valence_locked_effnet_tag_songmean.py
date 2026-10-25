#!/usr/bin/env python3
"""Song-level Valence mean correction from EffNet mood/theme tag statistics."""

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
DEFAULT_BASE = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_effnet_moodtheme_tags_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_effnet_tag_songmean_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def tag_stats(tag_dir: Path, music_ids: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for music_id in music_ids:
        tags = np.load(tag_dir / "per_song" / f"{music_id}.npz")["tags"].astype(np.float64)
        row: dict[str, float] = {"musicId": float(music_id)}
        for prefix, values in [
            ("mean", tags.mean(axis=0)),
            ("std", tags.std(axis=0)),
            ("q10", np.quantile(tags, 0.10, axis=0)),
            ("q90", np.quantile(tags, 0.90, axis=0)),
        ]:
            for column, value in enumerate(values):
                row[f"tag_{prefix}_{column:02d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).assign(musicId=lambda frame: frame["musicId"].astype(int))


def song_table(frame: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    songs = frame.groupby("musicId", sort=False).agg(
        target_mean=("target_valence", "mean"),
        pred_mean=("pred_valence", "mean"),
        pred_std=("pred_valence", "std"),
        pred_min=("pred_valence", "min"),
        pred_max=("pred_valence", "max"),
        pred_q10=("pred_valence", lambda values: values.quantile(0.10)),
        pred_q90=("pred_valence", lambda values: values.quantile(0.90)),
        frames=("pred_valence", "size"),
    ).reset_index()
    songs["pred_std"] = songs["pred_std"].fillna(0.0)
    songs["pred_range"] = songs["pred_max"] - songs["pred_min"]
    songs["residual_mean"] = songs["target_mean"] - songs["pred_mean"]
    return songs.merge(tags, on="musicId", how="left", validate="one_to_one")


def select_features(train: pd.DataFrame, columns: list[str], target: str, top_k: int) -> list[str]:
    y = train[target].to_numpy(dtype=np.float64)
    y -= y.mean()
    y_norm = np.linalg.norm(y)
    scored = []
    for column in columns:
        x = train[column].to_numpy(dtype=np.float64)
        x -= x.mean()
        denom = np.linalg.norm(x) * y_norm
        if denom > 1e-12:
            score = abs(float(np.dot(x, y) / denom))
            if np.isfinite(score):
                scored.append((score, column))
    scored.sort(reverse=True)
    return [column for _, column in scored[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    music_ids = sorted({int(value) for frame in frames.values() for value in frame["musicId"].unique()})
    tags = tag_stats(args.tag_dir, music_ids)
    songs = {fold: song_table(frames[fold], tags) for fold in FOLDS}
    tag_columns = [column for column in tags.columns if column != "musicId"]
    base_columns = [
        "pred_mean", "pred_std", "pred_min", "pred_max", "pred_q10", "pred_q90",
        "pred_range", "frames",
    ]
    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_rows)

    candidates: list[dict[str, Any]] = [{
        "target_mode": "identity", "top_k": 0, "alpha": 0.0, "strength": 0.0, "cap": 0.0,
        **base_macro, "passes_thresholds": base_macro["mae"] <= 0.15 and base_macro["pcc"] >= 0.80 and base_macro["ccc"] >= 0.80,
    }]
    shift_cache: dict[tuple[str, int, float], dict[str, np.ndarray]] = {}
    for target_mode in ["residual_mean", "target_mean"]:
        for top_k in [8, 16, 32, 64, 112, 224]:
            for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0]:
                shifts: dict[str, np.ndarray] = {}
                for held in FOLDS:
                    train = pd.concat([songs[fold] for fold in FOLDS if fold != held], ignore_index=True)
                    test = songs[held]
                    selected = select_features(train, tag_columns, target_mode, top_k)
                    columns = base_columns + selected
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(train[columns], train[target_mode])
                    estimate = model.predict(test[columns])
                    if target_mode == "target_mean":
                        estimate -= test["pred_mean"].to_numpy(dtype=np.float64)
                    shifts[held] = estimate
                shift_cache[(target_mode, top_k, alpha)] = shifts
                for strength in [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75]:
                    for cap in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060]:
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
                            "target_mode": target_mode, "top_k": top_k, "alpha": alpha,
                            "strength": strength, "cap": cap, **result,
                            "passes_thresholds": result["mae"] <= 0.15 and result["pcc"] >= 0.80 and result["ccc"] >= 0.80,
                        })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["passes_thresholds"]]
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
        output["pre_effnet_tag_songmean_pred_valence"] = base
        output["effnet_tag_songmean_shift"] = frame_shift
        output["pred_valence"] = prediction
        output.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(prediction, output["target_valence"])})

    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Other-two-song-fold fit; final locked test excluded",
        "feature": "Per-song mean/std/q10/q90 of 56 Discogs-EffNet MTG mood/theme tags",
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
