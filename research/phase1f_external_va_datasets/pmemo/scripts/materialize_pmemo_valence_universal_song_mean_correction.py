#!/usr/bin/env python3
"""Materialize a universal deployable-style Valence song-mean correction.

This uses one fixed configuration across all CV splits. Unlike targeted split
correction, this is aligned with arbitrary-song deployment because the same
feature set, regressor family, and correction strength are used everywhere.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_song_mean_fusion import (
    CURRENT_BEST,
    INPUT_DIR,
    ROOT,
    apply_song_mean_shift,
    audio_song_features,
    eval_df,
    model_candidates,
    prediction_song_features,
)
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LIBROSA_DATASET
from train_pmemo_mert_valence_baseline import threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR


DEFAULT_OUT_DIR = ROOT / "outputs" / "valence_universal_song_mean_correction_20260628"


def select_model(train_rows: int, feature_dim: int, target_model_id: str):
    for model_id, model in model_candidates(train_rows, feature_dim):
        if model_id == target_model_id:
            return model
    raise ValueError(f"Model candidate not found: {target_model_id}")


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([row[variant]["test"][metric] for row in rows], dtype=np.float64)
        out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
        out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def feature_groups(feature_set: str) -> list[str]:
    if feature_set == "mert_harmonic":
        return ["harmonic"]
    if feature_set == "mert_all_librosa":
        return ["mfcc", "spectral", "harmonic", "energy", "tempo"]
    raise ValueError(f"Unsupported feature_set for universal correction: {feature_set}")


def materialize_split(
    split_dir: Path,
    out_dir: Path,
    feature_set: str,
    model_id: str,
    strength: float,
    cap: float | None,
    cap_name: str,
) -> dict[str, Any]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    song_tables = {split: prediction_song_features(frames[split]) for split in ["train", "val", "test"]}
    all_ids = sorted({int(x) for split in ["train", "val", "test"] for x in song_tables[split]["musicId"].tolist()})

    audio = audio_song_features(
        all_ids,
        DEFAULT_MERT_DIR,
        DEFAULT_LIBROSA_DATASET,
        feature_groups(feature_set),
    )
    pred_feature_cols = [
        "frames",
        "duration_proxy",
        "pred_mean",
        "pred_std",
        "pred_min",
        "pred_max",
        "pred_range",
        "pred_q10",
        "pred_q90",
        "pred_head_mean",
        "pred_tail_mean",
    ]
    merged = {split: song_tables[split].merge(audio, on="musicId", how="left") for split in ["train", "val", "test"]}
    feature_cols = pred_feature_cols + [c for c in audio.columns if c != "musicId"]
    x_train = merged["train"][feature_cols].to_numpy(dtype=np.float64)
    y_train = merged["train"]["target_mean"].to_numpy(dtype=np.float64)
    model = select_model(len(x_train), x_train.shape[1], model_id)
    model.fit(x_train, y_train)
    pred_means = {
        split: dict(
            zip(
                merged[split]["musicId"].astype(int).tolist(),
                model.predict(merged[split][feature_cols].to_numpy(dtype=np.float64)),
            )
        )
        for split in ["train", "val", "test"]
    }

    corrected = {
        split: apply_song_mean_shift(frames[split], pred_means[split], strength, cap)
        for split in ["train", "val", "test"]
    }
    selected = {split: eval_df(corrected[split]) for split in ["train", "val", "test"]}
    split_out = out_dir / split_dir.name
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        corrected[split]["universal_correction_feature_set"] = feature_set
        corrected[split]["universal_correction_model_id"] = model_id
        corrected[split]["universal_correction_strength"] = strength
        corrected[split]["universal_correction_cap"] = cap_name
        corrected[split].to_csv(split_out / f"{split}_predictions.csv", index=False)

    return {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected,
        "selected_config": {
            "feature_set": feature_set,
            "model_id": model_id,
            "strength": strength,
            "cap": cap_name,
        },
        "delta_vs_input_test": {
            metric: float(selected["test"][metric] - baseline["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "delta_vs_input_val": {
            metric: float(selected["val"][metric] - baseline["val"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-id", default="ridge_pca64_a30")
    parser.add_argument("--feature-set", choices=["mert_harmonic", "mert_all_librosa"], default="mert_all_librosa")
    parser.add_argument("--strength", type=float, default=0.10)
    parser.add_argument("--cap", default="none")
    parser.add_argument("--result-id", default="pmemo_valence_universal_song_mean_correction_20260628")
    args = parser.parse_args()

    cap = None if args.cap == "none" else float(args.cap)
    cap_name = "none" if cap is None else f"cap{int(round(cap * 1000)):03d}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        materialize_split(
            split_dir,
            args.out_dir,
            args.feature_set,
            args.model_id,
            float(args.strength),
            cap,
            cap_name,
        )
        for split_dir in sorted(INPUT_DIR.glob("split_*_seed_*"))
    ]
    baseline_mean = aggregate(rows, "baseline")
    selected_mean = aggregate(rows, "selected")
    selected_metrics = {
        "rmse": selected_mean["selected_test_rmse_mean"],
        "mae": selected_mean["selected_test_mae_mean"],
        "pcc": selected_mean["selected_test_pcc_mean"],
        "ccc": selected_mean["selected_test_ccc_mean"],
    }
    current_best = {
        "rmse": CURRENT_BEST["rmse"],
        "mae": CURRENT_BEST["mae"],
        "pcc": CURRENT_BEST["pcc"],
        "ccc": CURRENT_BEST["ccc"],
    }
    summary = {
        "created_at": "2026-06-28",
        "model_id": args.result_id,
        "input_model": CURRENT_BEST,
        "purpose": "Universal deployable-style song-mean correction using one fixed configuration across all CV splits.",
        "config": {
            "feature_set": args.feature_set,
            "model_id": args.model_id,
            "strength": float(args.strength),
            "cap": cap_name,
        },
        "num_splits": len(rows),
        "per_split": rows,
        **baseline_mean,
        **selected_mean,
        "selected_mean_test_gap_rows": threshold_rows(selected_metrics),
        "delta_vs_input_test_mean": {
            metric: float(selected_metrics[metric] - current_best[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
