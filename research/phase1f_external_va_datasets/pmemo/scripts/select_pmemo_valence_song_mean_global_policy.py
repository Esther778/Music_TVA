#!/usr/bin/env python3
"""Global song-level Valence mean correction policy.

Unlike per-split selection, this script chooses one shared correction
configuration across all PMEmo splits using validation macro metrics. A shared
policy is closer to the eventual arbitrary-song teacher because inference on a
new MP3 cannot know a split-specific postprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import search_pmemo_valence_song_mean_fusion as song_mean
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LIBROSA_DATASET
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR


INPUT_DIR = ROOT / "outputs" / "valence_parsimony_output_guarded_20260707"
SOURCE_CANDIDATES = ROOT / "outputs" / "valence_current_best_song_mean_fusion_20260707" / "all_song_mean_fusion_results.csv"
OUT_DIR = ROOT / "outputs" / "valence_current_best_song_mean_global_policy_20260707"

CURRENT_BEST = {
    "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
    "rmse": 0.19127265472577384,
    "mae": 0.14892050958155043,
    "pcc": 0.8060728304873086,
    "ccc": 0.7915010047864408,
}


def cap_value(cap_name: str) -> float | None:
    if cap_name == "none":
        return None
    if cap_name.startswith("cap"):
        return float(cap_name.replace("cap", "")) / 1000.0
    raise ValueError(f"Unknown cap name: {cap_name}")


def macro_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["feature_set", "model_id", "strength", "cap"]
    agg = (
        df.groupby(keys, dropna=False)
        .agg({c: "mean" for c in ["val_rmse", "val_mae", "val_pcc", "val_ccc", "test_rmse", "test_mae", "test_pcc", "test_ccc"]})
        .reset_index()
    )
    return agg


def select_global_config(agg: pd.DataFrame) -> dict[str, Any]:
    base = agg[agg["model_id"].eq("no_change")].iloc[0]
    eligible = agg[
        (agg["val_rmse"] <= base["val_rmse"] + 0.001)
        & (agg["val_mae"] <= base["val_mae"] + 0.001)
        & (agg["val_ccc"] >= base["val_ccc"] + 0.001)
    ].copy()
    if eligible.empty:
        selected = base.copy()
        reason = "identity retained; no global candidate passed validation guard"
    else:
        selected = eligible.sort_values(
            ["val_ccc", "val_rmse", "val_mae", "val_pcc"],
            ascending=[False, True, True, False],
        ).iloc[0]
        reason = "global candidate selected by validation CCC guard"
    out = selected.to_dict()
    out["selection_reason"] = reason
    out["baseline_val_rmse"] = float(base["val_rmse"])
    out["baseline_val_mae"] = float(base["val_mae"])
    out["baseline_val_pcc"] = float(base["val_pcc"])
    out["baseline_val_ccc"] = float(base["val_ccc"])
    out["baseline_test_rmse"] = float(base["test_rmse"])
    out["baseline_test_mae"] = float(base["test_mae"])
    out["baseline_test_pcc"] = float(base["test_pcc"])
    out["baseline_test_ccc"] = float(base["test_ccc"])
    return out


def selected_model(model_id: str, train_rows: int, feature_dim: int):
    for name, model in song_mean.model_candidates(train_rows, feature_dim):
        if name == model_id:
            return model
    raise ValueError(f"Model id not available: {model_id}")


def audio_table_for(feature_set: str, music_ids: list[int]) -> pd.DataFrame:
    if feature_set == "pred_only":
        return pd.DataFrame({"musicId": music_ids})
    if feature_set == "mert_harmonic":
        return song_mean.audio_song_features(music_ids, DEFAULT_MERT_DIR, DEFAULT_LIBROSA_DATASET, ["harmonic"])
    if feature_set == "mert_all_librosa":
        return song_mean.audio_song_features(
            music_ids,
            DEFAULT_MERT_DIR,
            DEFAULT_LIBROSA_DATASET,
            ["mfcc", "spectral", "harmonic", "energy", "tempo"],
        )
    if feature_set == "baseline":
        return pd.DataFrame({"musicId": music_ids})
    raise ValueError(f"Unknown feature_set: {feature_set}")


def materialize_split(split_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: song_mean.eval_df(frames[split]) for split in ["train", "val", "test"]}
    if str(config["model_id"]) == "no_change":
        corrected = {split: frames[split].copy() for split in ["train", "val", "test"]}
    else:
        song_tables = {split: song_mean.prediction_song_features(frames[split]) for split in ["train", "val", "test"]}
        all_ids = sorted({int(x) for split in ["train", "val", "test"] for x in song_tables[split]["musicId"].tolist()})
        audio = audio_table_for(str(config["feature_set"]), all_ids)
        merged = {split: song_tables[split].merge(audio, on="musicId", how="left") for split in ["train", "val", "test"]}
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
        feature_cols = pred_feature_cols + [c for c in audio.columns if c != "musicId"]
        x_train = merged["train"][feature_cols].to_numpy(dtype=np.float64)
        y_train = merged["train"]["target_mean"].to_numpy(dtype=np.float64)
        model = selected_model(str(config["model_id"]), len(x_train), x_train.shape[1])
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
            split: song_mean.apply_song_mean_shift(
                frames[split],
                pred_means[split],
                float(config["strength"]),
                cap_value(str(config["cap"])),
            )
            for split in ["train", "val", "test"]
        }

    split_out = OUT_DIR / split_dir.name
    split_out.mkdir(parents=True, exist_ok=True)
    selected_metrics = {}
    for split in ["train", "val", "test"]:
        corrected[split].to_csv(split_out / f"{split}_predictions.csv", index=False)
        selected_metrics[split] = song_mean.eval_df(corrected[split])
    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected_metrics,
        "delta_vs_current_best_test": {
            metric: float(selected_metrics["test"][metric] - baseline["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary


def aggregate_metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([r[variant]["test"][metric] for r in rows], dtype=np.float64)
        out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
        out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(SOURCE_CANDIDATES)
    agg = macro_rows(candidates)
    selected = select_global_config(agg)
    agg.to_csv(OUT_DIR / "all_global_policy_candidates.csv", index=False)

    rows = [materialize_split(split_dir, selected) for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir())]
    result = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_current_best_song_mean_global_policy_20260707",
        "input_model": CURRENT_BEST,
        "source_candidates": str(SOURCE_CANDIDATES.relative_to(ROOT)),
        "selection_rule": "one shared song-mean correction config across all splits; validation guard allows RMSE/MAE <= baseline +0.001 and requires CCC >= baseline +0.001; select highest validation CCC",
        "selected_config": selected,
        "num_splits": len(rows),
        "per_split": rows,
        **aggregate_metrics(rows, "baseline"),
        **aggregate_metrics(rows, "selected"),
    }
    selected_mean = {metric: result[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: result[f"baseline_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    result["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    result["delta_vs_input_test_mean"] = {
        metric: float(selected_mean[metric] - baseline_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    result["delta_vs_previous_current_best"] = {
        metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
