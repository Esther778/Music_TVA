#!/usr/bin/env python3
"""Validation-safe song-mean fusion selection for current best Valence output."""

from __future__ import annotations

import json
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


OUT_DIR = ROOT / "outputs" / "valence_song_mean_valsafe_selection_20260628"


def pass_val_filter(candidate: dict[str, float], baseline: dict[str, float], eps: float) -> bool:
    return (
        candidate["rmse"] <= baseline["rmse"] + eps
        and candidate["mae"] <= baseline["mae"] + eps
        and candidate["pcc"] >= baseline["pcc"] - eps
        and candidate["ccc"] >= baseline["ccc"] - eps
    )


def selection_score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.15 * m["mae"]
        + 0.08 * max(0.0, 0.80 - m["ccc"])
        + 0.04 * max(0.0, 0.80 - m["pcc"])
        - 0.01 * m["ccc"]
    )


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([r[variant]["test"][metric] for r in rows], dtype=np.float64)
        out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
        out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def evaluate_split(split_dir: Path, eps: float) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    song_tables = {split: prediction_song_features(frames[split]) for split in ["train", "val", "test"]}
    all_ids = sorted({int(x) for split in ["train", "val", "test"] for x in song_tables[split]["musicId"].tolist()})

    audio_tables = {
        "pred_only": pd.DataFrame({"musicId": all_ids}),
        "mert_harmonic": audio_song_features(all_ids, DEFAULT_MERT_DIR, DEFAULT_LIBROSA_DATASET, ["harmonic"]),
        "mert_all_librosa": audio_song_features(
            all_ids,
            DEFAULT_MERT_DIR,
            DEFAULT_LIBROSA_DATASET,
            ["mfcc", "spectral", "harmonic", "energy", "tempo"],
        ),
    }
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
    rows = []
    cache: dict[tuple[str, str, str, float, str], pd.DataFrame] = {}
    baseline_row = {
        "split_name": split_dir.name,
        "feature_set": "baseline",
        "model_id": "no_change",
        "strength": 0.0,
        "cap": "none",
        **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
        "val_selection_score": selection_score(baseline["val"]),
        "passes_val_filter": True,
    }
    rows.append(baseline_row)
    for split in ["train", "val", "test"]:
        cache[(split, "baseline", "no_change", 0.0, "none")] = frames[split].copy()

    for feature_set, audio in audio_tables.items():
        merged = {split: song_tables[split].merge(audio, on="musicId", how="left") for split in ["train", "val", "test"]}
        feature_cols = pred_feature_cols + [c for c in audio.columns if c != "musicId"]
        x_train = merged["train"][feature_cols].to_numpy(dtype=np.float64)
        y_train = merged["train"]["target_mean"].to_numpy(dtype=np.float64)
        for model_id, model in model_candidates(len(x_train), x_train.shape[1]):
            try:
                model.fit(x_train, y_train)
            except Exception as exc:
                rows.append({"split_name": split_dir.name, "feature_set": feature_set, "model_id": model_id, "error": repr(exc)})
                continue
            pred_means = {
                split: dict(
                    zip(
                        merged[split]["musicId"].astype(int).tolist(),
                        model.predict(merged[split][feature_cols].to_numpy(dtype=np.float64)),
                    )
                )
                for split in ["train", "val", "test"]
            }
            for strength in [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.00]:
                for cap_name, cap in [
                    ("cap003", 0.003),
                    ("cap005", 0.005),
                    ("cap010", 0.010),
                    ("cap020", 0.020),
                    ("cap040", 0.040),
                    ("cap070", 0.070),
                    ("cap100", 0.100),
                    ("none", None),
                ]:
                    split_metrics = {}
                    for split in ["train", "val", "test"]:
                        corrected = apply_song_mean_shift(frames[split], pred_means[split], strength, cap)
                        split_metrics[split] = eval_df(corrected)
                        cache[(split, feature_set, model_id, strength, cap_name)] = corrected
                    val_m = split_metrics["val"]
                    rows.append(
                        {
                            "split_name": split_dir.name,
                            "feature_set": feature_set,
                            "model_id": model_id,
                            "strength": strength,
                            "cap": cap_name,
                            **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                            "val_selection_score": selection_score(val_m),
                            "passes_val_filter": pass_val_filter(val_m, baseline["val"], eps),
                        }
                    )

    result_df = pd.DataFrame(rows)
    valid = result_df[result_df["passes_val_filter"] == True].copy()
    valid = valid.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = valid.iloc[0].to_dict()
    split_out = OUT_DIR / split_dir.name
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        key = (split, str(selected["feature_set"]), str(selected["model_id"]), float(selected["strength"]), str(selected["cap"]))
        cache[key].to_csv(split_out / f"{split}_predictions.csv", index=False)
    selected_metrics = {
        split: {metric: float(selected[f"{split}_{metric}"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
        for split in ["train", "val", "test"]
    }
    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "eps": eps,
        "valid_candidates": int(len(valid)),
        "selected_config": {
            "feature_set": str(selected["feature_set"]),
            "model_id": str(selected["model_id"]),
            "strength": float(selected["strength"]),
            "cap": str(selected["cap"]),
            "val_selection_score": float(selected["val_selection_score"]),
        },
        "selected": selected_metrics,
        "delta_vs_current_best_test": {
            metric: float(selected_metrics["test"][metric] - baseline["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "delta_vs_current_best_val": {
            metric: float(selected_metrics["val"][metric] - baseline["val"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, result_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eps = 0.0
    rows = []
    all_results = []
    for split_dir in sorted(INPUT_DIR.glob("split_*_seed_*")):
        summary, result_df = evaluate_split(split_dir, eps)
        rows.append(summary)
        all_results.append(result_df)
    pd.concat(all_results, ignore_index=True).to_csv(OUT_DIR / "all_valsafe_song_mean_results.csv", index=False)
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
    aggregate_summary = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_song_mean_valsafe_selection_20260628",
        "input_model": CURRENT_BEST,
        "selection_rule": "Per split, candidate must not degrade validation RMSE/MAE/PCC/CCC versus current best; then select by validation score.",
        "eps": eps,
        "num_splits": len(rows),
        "per_split": rows,
        **baseline_mean,
        **selected_mean,
        "selected_mean_test_gap_rows": threshold_rows(selected_metrics),
        "delta_vs_current_best_test_mean": {
            metric: float(selected_metrics[metric] - current_best[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
