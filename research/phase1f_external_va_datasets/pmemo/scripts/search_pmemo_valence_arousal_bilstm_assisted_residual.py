#!/usr/bin/env python3
"""Arousal-assisted Valence residual search with split-matched Arousal BiLSTM.

This experiment keeps the arbitrary-song requirement: Valence is corrected only
with audio-derived predictions, not with true Arousal labels. For each Valence
song-level split, it trains an Arousal BiLSTM on the same train songs, predicts
Arousal for train/val/test, optionally postprocesses Arousal using validation
only, then trains a small Valence residual corrector selected on validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_opensmile_arousal_trials import (
    DEFAULT_TRIALS,
    ROOT,
    TrialConfig,
    fit_linear_calibration,
    metrics,
    moving_average_by_song,
    resolve_device,
    run_trial,
    threshold_rows,
)


SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
CURRENT_VALENCE_DIR = ROOT / "outputs" / "valence_multilayer_global_ensemble_20260629"
OPENSMILE_DATASET_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset"
OUT_DIR = ROOT / "outputs" / "valence_arousal_bilstm_assisted_residual_20260629"

CURRENT_BEST = {
    "model_id": "pmemo_valence_multilayer_global_ensemble_20260629",
    "rmse": 0.19411262662053563,
    "mae": 0.15142593782241862,
    "pcc": 0.7998908886855055,
    "ccc": 0.7851966731300748,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
AROUSAL_TRIAL_ID = "a_only_h96_l1_do015_mse"


def selection_score(m: dict[str, float]) -> float:
    """Prefer passing all metrics, with RMSE and CCC as the largest penalties."""
    return (
        2.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.2 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.2 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.0 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def eval_valence_df(df: pd.DataFrame, pred_col: str = "pred_valence") -> dict[str, float]:
    return metrics(
        df[pred_col].to_numpy(dtype=np.float64),
        df["target_valence"].to_numpy(dtype=np.float64),
    )


def eval_arousal_df(df: pd.DataFrame, pred_col: str = "pred_arousal") -> dict[str, float]:
    return metrics(
        df[pred_col].to_numpy(dtype=np.float64),
        df["target_arousal"].to_numpy(dtype=np.float64),
    )


def arousal_trial_config() -> TrialConfig:
    for cfg in DEFAULT_TRIALS:
        if cfg.trial_id == AROUSAL_TRIAL_ID:
            return cfg
    raise ValueError(f"Missing Arousal trial config: {AROUSAL_TRIAL_ID}")


def materialize_arousal_dataset(split_dir: Path, split_name: str) -> Path:
    """Create a sequence_index with split labels from Valence and OpenSMILE paths."""
    out = OUT_DIR / "arousal_by_split" / split_name / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    val_split = pd.read_csv(split_dir / "sequence_index.csv")[["musicId", "split", "frames"]]
    base = pd.read_csv(OPENSMILE_DATASET_DIR / "sequence_index.csv")[["musicId", "path"]]
    merged = val_split.merge(base, on="musicId", how="left", validate="one_to_one")
    if merged["path"].isna().any():
        missing = merged.loc[merged["path"].isna(), "musicId"].tolist()[:10]
        raise ValueError(f"Missing OpenSMILE paths for {split_name}: {missing}")
    merged.to_csv(out / "sequence_index.csv", index=False)
    return out


def load_or_train_arousal(split_dir: Path, split_name: str, device: torch.device, seed: int, force: bool) -> Path:
    arousal_out = OUT_DIR / "arousal_by_split" / split_name
    trial_dir = arousal_out / AROUSAL_TRIAL_ID
    required = [trial_dir / f"{split}_predictions.csv" for split in ["train", "val", "test"]]
    if not force and all(p.exists() for p in required) and (trial_dir / "summary.json").exists():
        return trial_dir
    arousal_dataset = materialize_arousal_dataset(split_dir, split_name)
    cfg = arousal_trial_config()
    run_trial(cfg, arousal_dataset, arousal_out, device, seed)
    return trial_dir


def postprocess_arousal(arousal_dir: Path, split_out: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames = {split: pd.read_csv(arousal_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    base = {split: eval_arousal_df(df) for split, df in frames.items()}
    rows = []
    cache: dict[tuple[int, str], dict[str, pd.DataFrame]] = {}
    for window in [1, 3, 5, 7, 9, 11]:
        val = frames["val"]
        val_sm = moving_average_by_song(
            val["pred_arousal"].to_numpy(dtype=np.float64),
            val["musicId"].to_numpy(dtype=np.int64),
            window,
        )
        a, b = fit_linear_calibration(val_sm, val["target_arousal"].to_numpy(dtype=np.float64))
        for mode in ["smooth", "smooth_linear"]:
            corrected = {}
            for split, df in frames.items():
                pred = moving_average_by_song(
                    df["pred_arousal"].to_numpy(dtype=np.float64),
                    df["musicId"].to_numpy(dtype=np.int64),
                    window,
                )
                if mode == "smooth_linear":
                    pred = a * pred + b
                out = df.copy()
                out["pred_arousal"] = pred
                out["arousal_postprocess"] = mode
                out["arousal_smooth_window"] = window
                corrected[split] = out
            metrics_by_split = {split: eval_arousal_df(df) for split, df in corrected.items()}
            row = {
                "window": window,
                "mode": mode,
                "calibration_a": a,
                "calibration_b": b,
                **{f"{split}_{k}": v for split, m in metrics_by_split.items() for k, v in m.items()},
            }
            rows.append(row)
            cache[(window, mode)] = corrected
    cand = pd.DataFrame(rows)
    allowed = cand[(cand["val_pcc"] >= 0.80) & (cand["val_ccc"] >= 0.80)].copy()
    if allowed.empty:
        allowed = cand.copy()
    selected = allowed.sort_values(["val_rmse", "val_mae", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
    selected_frames = cache[(int(selected["window"]), str(selected["mode"]))]
    pp_out = split_out / "arousal_postprocess"
    pp_out.mkdir(parents=True, exist_ok=True)
    cand.to_csv(pp_out / "all_arousal_postprocess_candidates.csv", index=False)
    for split, df in selected_frames.items():
        df.to_csv(pp_out / f"{split}_predictions.csv", index=False)
    summary = {
        "raw": base,
        "selected": selected,
        "selected_metrics": {split: eval_arousal_df(df) for split, df in selected_frames.items()},
    }
    (pp_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected_frames, summary


def merge_valence_arousal(split_name: str, arousal_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for split in ["train", "val", "test"]:
        val = pd.read_csv(CURRENT_VALENCE_DIR / split_name / f"{split}_predictions.csv")
        ar = arousal_frames[split].copy()
        val["time_key"] = val["time_sec"].round(6)
        ar["time_key"] = ar["time_sec"].round(6)
        merged = val.merge(
            ar[["musicId", "time_key", "pred_arousal", "target_arousal"]],
            on=["musicId", "time_key"],
            how="inner",
        )
        if len(merged) != len(val):
            raise ValueError(f"{split_name}/{split} alignment mismatch: valence={len(val)} merged={len(merged)}")
        merged = merged.sort_values(["musicId", "time_sec"]).reset_index(drop=True)
        for col in ["pred_valence", "pred_arousal"]:
            merged[f"{col}_song_mean"] = merged.groupby("musicId")[col].transform("mean")
            merged[f"{col}_song_std"] = merged.groupby("musicId")[col].transform("std").fillna(0.0)
            merged[f"{col}_centered"] = merged[col] - merged[f"{col}_song_mean"]
            merged[f"{col}_delta"] = merged.groupby("musicId")[col].diff().fillna(0.0)
        merged["time_norm"] = merged.groupby("musicId")["time_sec"].transform(
            lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
        )
        out[split] = merged
    return out


def feature_cols() -> list[str]:
    return [
        "pred_valence",
        "pred_valence_song_mean",
        "pred_valence_song_std",
        "pred_valence_centered",
        "pred_valence_delta",
        "pred_arousal",
        "pred_arousal_song_mean",
        "pred_arousal_song_std",
        "pred_arousal_centered",
        "pred_arousal_delta",
        "time_sec",
        "time_norm",
    ]


def residual_models() -> list[tuple[str, Any]]:
    return [
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a1000", make_pipeline(StandardScaler(), Ridge(alpha=1000.0))),
        ("huber", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500))),
    ]


def corrected_predictions(base: pd.DataFrame, residual: np.ndarray, strength: float, cap: float | None) -> np.ndarray:
    corr = residual.astype(np.float64)
    if cap is not None:
        corr = np.clip(corr, -cap, cap)
    return np.clip(base["pred_valence"].to_numpy(dtype=np.float64) + strength * corr, -1.0, 1.0)


def search_residual(split_name: str, enriched: dict[str, pd.DataFrame], split_out: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline = {split: eval_valence_df(df) for split, df in enriched.items()}
    rows = []
    cache: dict[tuple[str, float, str, str], dict[str, pd.DataFrame]] = {}
    for split, df in enriched.items():
        cache[(split, "no_change", 0.0, "none")] = df.copy()
    rows.append(
        {
            "split_name": split_name,
            "residual_model_id": "no_change",
            "strength": 0.0,
            "cap": "none",
            **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
            "val_selection_score": selection_score(baseline["val"]),
        }
    )
    x_train = enriched["train"][feature_cols()].to_numpy(dtype=np.float64)
    y_train = (
        enriched["train"]["target_valence"].to_numpy(dtype=np.float64)
        - enriched["train"]["pred_valence"].to_numpy(dtype=np.float64)
    )
    for model_id, model in residual_models():
        model.fit(x_train, y_train)
        residuals = {
            split: model.predict(enriched[split][feature_cols()].to_numpy(dtype=np.float64))
            for split in ["train", "val", "test"]
        }
        for strength in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
            for cap_name, cap in [
                ("cap0025", 0.0025),
                ("cap005", 0.005),
                ("cap010", 0.010),
                ("cap020", 0.020),
                ("cap040", 0.040),
                ("none", None),
            ]:
                metrics_by_split = {}
                for split in ["train", "val", "test"]:
                    df = enriched[split].copy()
                    df["pred_valence"] = corrected_predictions(df, residuals[split], strength, cap)
                    df["residual_model_id"] = model_id
                    df["residual_strength"] = strength
                    df["residual_cap"] = cap_name
                    metrics_by_split[split] = eval_valence_df(df)
                    cache[(split, model_id, float(strength), cap_name)] = df
                rows.append(
                    {
                        "split_name": split_name,
                        "residual_model_id": model_id,
                        "strength": strength,
                        "cap": cap_name,
                        **{f"{split}_{k}": v for split, m in metrics_by_split.items() for k, v in m.items()},
                        "val_selection_score": selection_score(metrics_by_split["val"]),
                    }
                )
    result = pd.DataFrame(rows)
    base_val = baseline["val"]
    viable = result[
        (result["val_rmse"] <= base_val["rmse"] + 0.0008)
        & (result["val_mae"] <= base_val["mae"] + 0.0008)
        & (result["val_pcc"] >= base_val["pcc"] - 0.0015)
        & (result["val_ccc"] >= base_val["ccc"] - 0.0015)
    ].copy()
    if viable.empty:
        viable = result.copy()
    selected = viable.sort_values(
        ["val_selection_score", "val_rmse", "val_mae", "val_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    pred_out = split_out / "valence_predictions"
    pred_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        key = (split, str(selected["residual_model_id"]), float(selected["strength"]), str(selected["cap"]))
        cache[key].to_csv(pred_out / f"{split}_predictions.csv", index=False)
    result.to_csv(split_out / "all_residual_candidates.csv", index=False)
    summary = {
        "split_name": split_name,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_baseline_test": {
            metric: float(selected[f"test_{metric}"] - baseline["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, result


def aggregate(per_split: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_arousal_bilstm_assisted_residual_20260629",
        "purpose": "Use split-matched predicted Arousal from audio as auxiliary information for Valence residual correction.",
        "selection_rule": "Arousal postprocess and Valence residual config selected on validation inside each song-level split; test labels are not used for selection.",
        "input_valence_model": CURRENT_BEST,
        "arousal_trial_id": AROUSAL_TRIAL_ID,
        "num_splits": len(per_split),
        "per_split": per_split,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            vals = np.asarray(
                [
                    s["selected"][f"test_{metric}"] if variant == "selected" else s["baseline"]["test"][metric]
                    for s in per_split
                ],
                dtype=np.float64,
            )
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: out[f"baseline_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_baseline"] = {
        metric: float(selected_mean[metric] - baseline_mean[metric])
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    out["selected_delta_vs_current_best_reference"] = {
        metric: float(selected_mean[metric] - CURRENT_BEST[metric])
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-arousal", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    per_split = []
    all_candidates = []
    for split_dir in sorted(SPLIT_DIR.glob("split_*")):
        split_name = split_dir.name
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        print(f"=== {split_name}: train/load split-matched Arousal ===", flush=True)
        arousal_dir = load_or_train_arousal(split_dir, split_name, device, args.seed, args.force_arousal)
        arousal_frames, arousal_summary = postprocess_arousal(arousal_dir, split_out)
        print(f"=== {split_name}: search Valence residual ===", flush=True)
        enriched = merge_valence_arousal(split_name, arousal_frames)
        split_summary, candidates = search_residual(split_name, enriched, split_out)
        split_summary["arousal_summary"] = arousal_summary
        per_split.append(split_summary)
        all_candidates.append(candidates)
        (split_out / "summary.json").write_text(json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(split_summary["selected"], ensure_ascii=False, indent=2), flush=True)
    if all_candidates:
        pd.concat(all_candidates, ignore_index=True).to_csv(OUT_DIR / "all_residual_candidates.csv", index=False)
    summary = aggregate(per_split)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
