#!/usr/bin/env python3
"""Nested cross-fold song-polarity correction from audio-model disagreement.

The outer held fold is never used to choose features or hyperparameters. For
each outer fold, the other two development folds form a two-way inner
cross-validation. The final locked PMEmo test is never loaded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_musicnn_dynamic_tag_past_dense_monotonic_20260714"
DEFAULT_REFERENCES = {
    "muq_arousal": ROOT / "outputs/valence_locked_muq_arousal_stacker_fine_20260714",
    "muq_threeway": ROOT / "outputs/valence_locked_muq_threeway_fusion_20260713",
    "mert_smile": ROOT / "outputs/valence_locked_audio_ensemble_postprocess_mertsmile_20260713",
}
DEFAULT_OUT = ROOT / "outputs/valence_locked_nested_backbone_polarity_20260714"


@dataclass(frozen=True)
class Config:
    feature_mode: str
    alpha: float
    strength: float
    cap: float


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def load_aligned(base_dir: Path, references: dict[str, Path], fold: str) -> pd.DataFrame:
    base = pd.read_csv(base_dir / f"{fold}_val_predictions.csv")
    required = {"musicId", "time_sec", "target_valence", "pred_valence"}
    if not required.issubset(base.columns):
        raise ValueError(f"Missing columns in base {fold}: {required - set(base.columns)}")
    out = base.copy().rename(columns={"pred_valence": "pred_base"})
    for name, directory in references.items():
        ref = pd.read_csv(directory / f"{fold}_val_predictions.csv")
        keep = ref[["musicId", "time_sec", "pred_valence"]].rename(
            columns={"pred_valence": f"pred_{name}"}
        )
        out = out.merge(keep, on=["musicId", "time_sec"], how="inner", validate="one_to_one")
    if len(out) != len(base):
        raise ValueError(f"Incomplete alignment for {fold}: {len(out)} != {len(base)}")
    return out


def song_table(frame: pd.DataFrame, reference_names: list[str]) -> pd.DataFrame:
    prediction_columns = ["pred_base", *[f"pred_{name}" for name in reference_names]]
    means = frame.groupby("musicId", sort=False)[["target_valence", *prediction_columns]].mean()
    stds = frame.groupby("musicId", sort=False)[prediction_columns].std(ddof=0).add_suffix("_std")
    return means.join(stds).reset_index()


def feature_matrix(songs: pd.DataFrame, reference_names: list[str], mode: str) -> np.ndarray:
    base_mean = songs["pred_base"].to_numpy(dtype=np.float64)
    base_std = songs["pred_base_std"].to_numpy(dtype=np.float64)
    ref_means = np.column_stack([
        songs[f"pred_{name}"].to_numpy(dtype=np.float64) for name in reference_names
    ])
    ref_stds = np.column_stack([
        songs[f"pred_{name}_std"].to_numpy(dtype=np.float64) for name in reference_names
    ])
    deltas = ref_means - base_mean[:, None]
    std_deltas = ref_stds - base_std[:, None]
    if mode == "mean_deltas":
        return deltas
    if mode == "mean_std_deltas":
        return np.column_stack([deltas, std_deltas])
    if mode == "all_stats":
        return np.column_stack([base_mean, base_std, ref_means, ref_stds, deltas, std_deltas])
    raise ValueError(mode)


def fit_residual_model(
    songs: pd.DataFrame, reference_names: list[str], config: Config
) -> Any:
    x = feature_matrix(songs, reference_names, config.feature_mode)
    target = songs["target_valence"].to_numpy(dtype=np.float64) - songs["pred_base"].to_numpy(dtype=np.float64)
    model = make_pipeline(StandardScaler(), Ridge(alpha=config.alpha))
    model.fit(x, target)
    return model


def apply_model(
    frame: pd.DataFrame,
    songs: pd.DataFrame,
    reference_names: list[str],
    config: Config,
    model: Any,
) -> tuple[np.ndarray, pd.DataFrame]:
    x = feature_matrix(songs, reference_names, config.feature_mode)
    raw_shift = model.predict(x)
    shift = np.clip(config.strength * raw_shift, -config.cap, config.cap)
    shift_map = dict(zip(songs["musicId"].astype(int), shift))
    frame_shift = frame["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
    prediction = np.clip(frame["pred_base"].to_numpy(dtype=np.float64) + frame_shift, -1.0, 1.0)
    diagnostics = songs[["musicId", "target_valence", "pred_base"]].copy()
    diagnostics["raw_shift"] = raw_shift
    diagnostics["applied_shift"] = shift
    return prediction, diagnostics


def validation_score(result: dict[str, float], base: dict[str, float]) -> tuple[float, ...]:
    guarded = (
        result["mae"] <= base["mae"] + 0.001
        and result["pcc"] >= base["pcc"] - 0.003
        and result["ccc"] >= base["ccc"] - 0.003
    )
    return (0.0 if guarded else 1.0, result["rmse"], result["mae"], -result["ccc"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    references = DEFAULT_REFERENCES
    reference_names = list(references)
    frames = {fold: load_aligned(args.base_dir, references, fold) for fold in FOLDS}
    songs = {fold: song_table(frames[fold], reference_names) for fold in FOLDS}
    configs = [
        Config(feature_mode, alpha, strength, cap)
        for feature_mode in ["mean_deltas", "mean_std_deltas", "all_stats"]
        for alpha in [1.0, 10.0, 100.0, 1000.0]
        for strength in [0.25, 0.5, 0.75, 1.0]
        for cap in [0.005, 0.010, 0.015, 0.020]
    ]

    base_per_fold = [
        {"fold": fold, **metrics(frames[fold]["pred_base"], frames[fold]["target_valence"])}
        for fold in FOLDS
    ]
    selected_per_fold: list[dict[str, float | str]] = []
    selection_rows: list[dict[str, Any]] = []

    for held in FOLDS:
        inner_folds = [fold for fold in FOLDS if fold != held]
        config_rows: list[dict[str, Any]] = []
        for config in configs:
            inner_results: list[dict[str, float]] = []
            inner_base_results: list[dict[str, float]] = []
            for inner_val in inner_folds:
                inner_train = next(fold for fold in inner_folds if fold != inner_val)
                model = fit_residual_model(songs[inner_train], reference_names, config)
                prediction, _ = apply_model(
                    frames[inner_val], songs[inner_val], reference_names, config, model
                )
                inner_results.append(metrics(prediction, frames[inner_val]["target_valence"]))
                inner_base_results.append(metrics(frames[inner_val]["pred_base"], frames[inner_val]["target_valence"]))
            result = macro(inner_results)
            base_result = macro(inner_base_results)
            rank = validation_score(result, base_result)
            config_rows.append({
                "outer_held_fold": held,
                **config.__dict__,
                **{f"inner_{key}": value for key, value in result.items()},
                **{f"inner_base_{key}": value for key, value in base_result.items()},
                "guarded": rank[0] == 0.0,
                "rank_key": json.dumps(rank),
            })
        selected_row = min(
            config_rows,
            key=lambda row: (
                0 if row["guarded"] else 1,
                row["inner_rmse"],
                row["inner_mae"],
                -row["inner_ccc"],
            ),
        )
        selection_rows.extend(config_rows)
        selected = Config(
            str(selected_row["feature_mode"]),
            float(selected_row["alpha"]),
            float(selected_row["strength"]),
            float(selected_row["cap"]),
        )
        train_songs = pd.concat([songs[fold] for fold in inner_folds], ignore_index=True)
        model = fit_residual_model(train_songs, reference_names, selected)
        prediction, diagnostics = apply_model(frames[held], songs[held], reference_names, selected, model)
        result = {"fold": held, **metrics(prediction, frames[held]["target_valence"])}
        selected_per_fold.append(result)

        out = frames[held].copy()
        out["pre_nested_polarity_pred_valence"] = out["pred_base"]
        shift_map = dict(zip(diagnostics["musicId"].astype(int), diagnostics["applied_shift"]))
        out["nested_song_shift"] = out["musicId"].astype(int).map(shift_map)
        out["pred_valence"] = prediction
        out.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        diagnostics.to_csv(args.out_dir / f"{held}_song_diagnostics.csv", index=False)
        selected_row.update({f"outer_{key}": value for key, value in result.items() if key != "fold"})

    pd.DataFrame(selection_rows).to_csv(args.out_dir / "inner_candidates.csv", index=False)
    # Reconstruct a compact selection table from the nested-validation rows.
    compact_rows = []
    for held in FOLDS:
        held_rows = [row for row in selection_rows if row["outer_held_fold"] == held]
        chosen = min(
            held_rows,
            key=lambda row: (
                0 if row["guarded"] else 1,
                row["inner_rmse"],
                row["inner_mae"],
                -row["inner_ccc"],
            ),
        )
        compact_rows.append(chosen)
    pd.DataFrame(compact_rows).to_csv(args.out_dir / "selected_configs.csv", index=False)

    base_macro = macro(base_per_fold)  # type: ignore[arg-type]
    selected_macro = macro(selected_per_fold)  # type: ignore[arg-type]
    summary = {
        "created_at": "2026-07-14",
        "model_id": "pmemo_valence_locked_nested_backbone_polarity_20260714",
        "selection_boundary": (
            "Nested cross-fold selection: each outer held fold excluded from feature/hyperparameter selection; "
            "final locked test excluded"
        ),
        "base_dir": str(args.base_dir),
        "reference_dirs": {name: str(path) for name, path in references.items()},
        "base_macro": base_macro,
        "macro": selected_macro,
        "per_fold": selected_per_fold,
        "selected_configs": compact_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
