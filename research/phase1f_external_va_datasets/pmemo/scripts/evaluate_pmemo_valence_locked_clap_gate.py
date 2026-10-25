#!/usr/bin/env python3
"""Nested song-level gate for music-CLAP global Valence information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_FEATURES = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_clap_music_10s_20260714/per_song"
DEFAULT_FINAL = ROOT / "data/splits/pmemo_final_locked_20260713/final_locked_test_index.csv"
DEFAULT_OUT = ROOT / "outputs/valence_locked_clap_music_gate_20260714"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def song_table(frame: pd.DataFrame, feature_dir: Path, fold: str) -> pd.DataFrame:
    songs = frame.groupby("musicId", sort=False).agg(
        target_song=("target_valence", "mean"),
        base_song=("pred_valence", "mean"),
        frames=("pred_valence", "size"),
    ).reset_index()
    features = []
    for music_id in songs["musicId"].astype(int):
        embedding = np.load(feature_dir / f"{music_id}.npz")["embedding"].astype(np.float64)
        features.append(np.concatenate([embedding.mean(axis=0), embedding.std(axis=0)]))
    songs["feature"] = features
    songs["fold"] = fold
    return songs


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(np.stack(train["feature"]), train["target_song"].to_numpy(dtype=np.float64))
    return model.predict(np.stack(test["feature"]))


def corrected_frame_metrics(
    frame: pd.DataFrame, song: pd.DataFrame, clap_prediction: np.ndarray, blend: float
) -> dict[str, float]:
    shifts = dict(zip(
        song["musicId"].astype(int),
        blend * (clap_prediction - song["base_song"].to_numpy(dtype=np.float64)),
    ))
    prediction = np.clip(
        frame["pred_valence"].to_numpy(dtype=np.float64)
        + frame["musicId"].astype(int).map(shifts).to_numpy(dtype=np.float64),
        -1.0,
        1.0,
    )
    return metrics(prediction, frame["target_valence"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    songs = {fold: song_table(frames[fold], args.feature_dir, fold) for fold in FOLDS}
    development_ids = set().union(*(set(table["musicId"].astype(int)) for table in songs.values()))
    final_ids = set(pd.read_csv(args.final_index)["musicId"].astype(int))
    overlap = development_ids.intersection(final_ids)
    if overlap:
        raise ValueError(f"Final locked leakage in CLAP gate: {sorted(overlap)[:20]}")

    base_per_fold = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_per_fold)
    rows = []
    output_frames = []
    output_songs = []
    for held in FOLDS:
        inner = [fold for fold in FOLDS if fold != held]
        alpha_rows = []
        predictions_by_alpha = {}
        for alpha in ALPHAS:
            inner_predictions = {}
            for validation, training in [(inner[0], inner[1]), (inner[1], inner[0])]:
                inner_predictions[validation] = fit_predict(songs[training], songs[validation], alpha)
            predictions_by_alpha[alpha] = inner_predictions
            direct_rmse = float(np.mean([
                np.sqrt(np.mean((inner_predictions[fold] - songs[fold]["target_song"].to_numpy()) ** 2))
                for fold in inner
            ]))
            alpha_rows.append({"alpha": alpha, "direct_song_rmse": direct_rmse})
        alpha_table = pd.DataFrame(alpha_rows).sort_values(["direct_song_rmse", "alpha"])
        selected_alpha = float(alpha_table.iloc[0]["alpha"])
        inner_predictions = predictions_by_alpha[selected_alpha]
        base_inner = macro([
            metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"])
            for fold in inner
        ])
        inner_candidates = []
        for blend in BLENDS:
            result_rows = [
                corrected_frame_metrics(frames[fold], songs[fold], inner_predictions[fold], blend)
                for fold in inner
            ]
            result = macro(result_rows)
            guarded = (
                result["mae"] <= base_inner["mae"]
                and result["pcc"] >= base_inner["pcc"]
                and result["ccc"] >= base_inner["ccc"]
            )
            inner_candidates.append({
                "alpha": selected_alpha,
                "blend": blend,
                "direct_song_rmse": float(alpha_table.iloc[0]["direct_song_rmse"]),
                "guarded": guarded,
                **result,
            })
        candidate_table = pd.DataFrame(inner_candidates)
        eligible = candidate_table[candidate_table["guarded"]]
        selected = (eligible if not eligible.empty else candidate_table).sort_values(
            ["rmse", "mae", "ccc"], ascending=[True, True, False]
        ).iloc[0]
        alpha_table.to_csv(args.out_dir / f"{held}_inner_alpha_candidates.csv", index=False)
        candidate_table.to_csv(args.out_dir / f"{held}_inner_candidates.csv", index=False)

        train = pd.concat([songs[fold] for fold in inner], ignore_index=True)
        clap_prediction = fit_predict(train, songs[held], float(selected["alpha"]))
        song = songs[held].copy()
        song["clap_pred_song_valence"] = clap_prediction
        song["selected_alpha"] = float(selected["alpha"])
        song["selected_blend"] = float(selected["blend"])
        output_songs.append(song.drop(columns=["feature"]))

        shift_map = dict(zip(
            song["musicId"].astype(int),
            float(selected["blend"]) * (clap_prediction - song["base_song"].to_numpy(dtype=np.float64)),
        ))
        frame = frames[held].copy()
        frame["pre_clap_pred_valence"] = frame["pred_valence"]
        frame["clap_song_shift"] = frame["musicId"].astype(int).map(shift_map)
        frame["pred_valence"] = np.clip(frame["pred_valence"] + frame["clap_song_shift"], -1.0, 1.0)
        output_frames.append(frame.assign(fold=held))
        frame.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        rows.append({
            "fold": held,
            "selected_alpha": float(selected["alpha"]),
            "selected_blend": float(selected["blend"]),
            **metrics(frame["pred_valence"], frame["target_valence"]),
        })

    song_output = pd.concat(output_songs, ignore_index=True)
    song_output.to_csv(args.out_dir / "oof_song_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_fold_metrics.csv", index=False)
    selected_macro = macro(rows)
    direct_song = metrics(song_output["clap_pred_song_valence"], song_output["target_song"])
    base_song = metrics(song_output["base_song"], song_output["target_song"])
    summary = {
        "created_at": "2026-07-14",
        "model_id": "pmemo_valence_locked_clap_music_gate_20260714",
        "selection_boundary": "Nested other-two-fold alpha/blend selection; final locked 115 songs excluded",
        "feature": "laion/larger_clap_music 10 s non-overlapping 512-D audio windows, song mean+std",
        "development_songs": len(development_ids),
        "final_locked_overlap": 0,
        "base_frame_macro": base_macro,
        "clap_direct_song_metrics": direct_song,
        "base_song_metrics": base_song,
        "selected_frame_macro": selected_macro,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "per_fold": rows,
        "gap_rows": threshold_rows(selected_macro),
        "decision_rule": "Advance only for guarded cross-fold frame gain and useful direct song-mean information",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
