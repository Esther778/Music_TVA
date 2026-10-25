#!/usr/bin/env python3
"""Nested song-level validation of EffNet mood-tag Valence correction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_effnet_tag_songmean import (
    DEFAULT_BASE,
    DEFAULT_TAGS,
    FOLDS,
    macro,
    select_features,
    song_table,
    tag_stats,
)
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_OUT = ROOT / "outputs/valence_locked_effnet_tag_songmean_nested_20260714"
BASE_COLUMNS = [
    "pred_mean", "pred_std", "pred_min", "pred_max", "pred_q10", "pred_q90",
    "pred_range", "frames",
]


@dataclass(frozen=True)
class Config:
    top_k: int
    alpha: float
    strength: float
    cap: float


def fit_model(train: pd.DataFrame, tag_columns: list[str], config: Config):
    selected = select_features(train, tag_columns, "residual_mean", config.top_k)
    columns = BASE_COLUMNS + selected
    model = make_pipeline(StandardScaler(), Ridge(alpha=config.alpha))
    model.fit(train[columns], train["residual_mean"])
    return model, columns


def predict_fold(
    frame: pd.DataFrame,
    songs: pd.DataFrame,
    model,
    columns: list[str],
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    raw_shift = model.predict(songs[columns])
    song_shift = np.clip(config.strength * raw_shift, -config.cap, config.cap)
    shift_map = dict(zip(songs["musicId"].astype(int), song_shift))
    frame_shift = frame["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
    prediction = np.clip(frame["pred_valence"].to_numpy(dtype=np.float64) + frame_shift, -1.0, 1.0)
    return prediction, frame_shift


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
    configs = [
        Config(top_k, alpha, strength, cap)
        for top_k in [64, 112, 224]
        for alpha in [0.03, 0.10, 0.30, 1.0]
        for strength in [0.025, 0.05, 0.075, 0.10, 0.15]
        for cap in [0.010, 0.015, 0.020, 0.030]
    ]

    base_per_fold = [
        {"fold": fold, **metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"])}
        for fold in FOLDS
    ]
    selected_per_fold = []
    selection_rows = []
    selected_configs = []

    for held in FOLDS:
        inner_folds = [fold for fold in FOLDS if fold != held]
        fit_cache = {}
        for top_k in [64, 112, 224]:
            for alpha in [0.03, 0.10, 0.30, 1.0]:
                for inner_val in inner_folds:
                    inner_train = next(fold for fold in inner_folds if fold != inner_val)
                    template = Config(top_k, alpha, 1.0, 1.0)
                    model, columns = fit_model(songs[inner_train], tag_columns, template)
                    raw = model.predict(songs[inner_val][columns])
                    fit_cache[(top_k, alpha, inner_val)] = raw

        inner_base = macro([
            metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"])
            for fold in inner_folds
        ])
        config_rows = []
        for config in configs:
            inner_results = []
            for inner_val in inner_folds:
                raw = fit_cache[(config.top_k, config.alpha, inner_val)]
                shift_map = dict(zip(
                    songs[inner_val]["musicId"].astype(int),
                    np.clip(config.strength * raw, -config.cap, config.cap),
                ))
                frame_shift = frames[inner_val]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
                prediction = np.clip(frames[inner_val]["pred_valence"] + frame_shift, -1.0, 1.0)
                inner_results.append(metrics(prediction, frames[inner_val]["target_valence"]))
            result = macro(inner_results)
            guarded = (
                result["mae"] <= inner_base["mae"] + 0.0005
                and result["pcc"] >= inner_base["pcc"] - 0.001
                and result["ccc"] >= inner_base["ccc"] - 0.001
            )
            row = {
                "outer_held_fold": held,
                **config.__dict__,
                **{f"inner_{key}": value for key, value in result.items()},
                **{f"inner_base_{key}": value for key, value in inner_base.items()},
                "guarded": guarded,
            }
            config_rows.append(row)
            selection_rows.append(row)

        eligible = [row for row in config_rows if row["guarded"]]
        pool = eligible if eligible else config_rows
        chosen = min(pool, key=lambda row: (row["inner_rmse"], row["inner_mae"], -row["inner_ccc"]))
        selected = Config(
            int(chosen["top_k"]), float(chosen["alpha"]),
            float(chosen["strength"]), float(chosen["cap"]),
        )
        selected_configs.append(chosen)
        outer_train = pd.concat([songs[fold] for fold in inner_folds], ignore_index=True)
        model, columns = fit_model(outer_train, tag_columns, selected)
        prediction, frame_shift = predict_fold(frames[held], songs[held], model, columns, selected)
        result = {"fold": held, **metrics(prediction, frames[held]["target_valence"])}
        selected_per_fold.append(result)

        output = frames[held].copy()
        output["pre_nested_effnet_tag_songmean_pred_valence"] = output["pred_valence"]
        output["nested_effnet_tag_songmean_shift"] = frame_shift
        output["pred_valence"] = prediction
        output.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)

    pd.DataFrame(selection_rows).to_csv(args.out_dir / "inner_candidates.csv", index=False)
    pd.DataFrame(selected_configs).to_csv(args.out_dir / "selected_configs.csv", index=False)
    base_macro = macro(base_per_fold)
    selected_macro = macro(selected_per_fold)
    summary = {
        "created_at": "2026-07-14",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Nested song-level cross-fold selection; final locked test excluded",
        "base_dir": str(args.base_dir),
        "base_macro": base_macro,
        "macro": selected_macro,
        "per_fold": selected_per_fold,
        "selected_configs": selected_configs,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
