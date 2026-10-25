#!/usr/bin/env python3
"""Diagnose PMEmo Valence split distribution and shared test failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence_index.csv"
LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
OUT_DIR = ROOT / "outputs" / "valence_split_diagnostics_20260626"

REPRESENTATIVE_MODELS = {
    "current_best_extreme": ROOT
    / "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext125_clip010_ccc005",
    "previous_clip_aux": ROOT
    / "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse010_ccc005",
    "best_refine_rmse": ROOT
    / "outputs/mert_hidden6_harmonic_extreme_weighted_refine/h6_harm_h80_ext100_clip008_ccc005",
    "best_refine_ccc": ROOT
    / "outputs/mert_hidden6_harmonic_extreme_weighted_refine/h6_harm_h80_ext150_clip010_ccc008",
    "global_aux_blend": ROOT
    / "outputs/mert_hidden6_harmonic_global_aux/h6_harm_h80_globalmse010_blend010",
    "song_balanced_ccc": ROOT
    / "outputs/mert_hidden6_harmonic_song_balanced/h6_harm_h80_songbal_ext125_clip010_ccc008",
    "mert_opensmile_fusion": ROOT
    / "outputs/mert_hidden6_opensmile_valence_fusion_fixed/h6_smile_h80_proj320_do030_ccc005",
    "opensmile_harmonic": ROOT
    / "outputs/valence_opensmile_harmonic_focus/v_harm_h96_l1_do025_lr6e4_ccc005",
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov = float(np.mean((x - mean_x) * (y - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 0.0


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pcc": pearson(pred, target),
        "ccc": ccc(pred, target),
    }


def load_song_targets(index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in index.to_dict("records"):
        sid = int(row["musicId"])
        data = np.load(LABEL_DIR / f"{sid}.npz")
        y = data["Y"].astype(np.float32)[:, 0]
        rows.append(
            {
                "musicId": str(sid),
                "split": row["split"],
                "frames": int(len(y)),
                "duration_sec": float(len(y) * 0.5),
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
                "target_min": float(np.min(y)),
                "target_max": float(np.max(y)),
                "target_range": float(np.max(y) - np.min(y)),
                "target_abs_mean": float(abs(np.mean(y))),
            }
        )
    return pd.DataFrame(rows)


def split_summary(song_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, df in song_df.groupby("split"):
        rows.append(
            {
                "split": split,
                "songs": int(len(df)),
                "frames": int(df["frames"].sum()),
                "target_mean_mean": float(df["target_mean"].mean()),
                "target_mean_std": float(df["target_mean"].std(ddof=0)),
                "target_mean_min": float(df["target_mean"].min()),
                "target_mean_q10": float(df["target_mean"].quantile(0.10)),
                "target_mean_q25": float(df["target_mean"].quantile(0.25)),
                "target_mean_median": float(df["target_mean"].median()),
                "target_mean_q75": float(df["target_mean"].quantile(0.75)),
                "target_mean_q90": float(df["target_mean"].quantile(0.90)),
                "target_mean_max": float(df["target_mean"].max()),
                "target_abs_mean_mean": float(df["target_abs_mean"].mean()),
                "frame_weighted_target_mean": float(np.average(df["target_mean"], weights=df["frames"])),
                "mean_frames_per_song": float(df["frames"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("split")


def read_predictions(model_id: str, model_dir: Path, split: str) -> pd.DataFrame:
    path = model_dir / f"{split}_predictions.csv"
    df = pd.read_csv(path)
    pred_col = "pred_valence"
    if pred_col not in df.columns and "pred_valence_raw" in df.columns:
        pred_col = "pred_valence_raw"
    required = {"musicId", "time_sec", "target_valence", pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    out = df.loc[:, ["musicId", "time_sec", "target_valence", pred_col]].copy()
    out["musicId"] = out["musicId"].astype(str)
    out["time_sec"] = out["time_sec"].astype(float).round(3)
    out = out.rename(columns={pred_col: "pred_valence"})
    out["model_id"] = model_id
    return out


def per_song_prediction_metrics(model_id: str, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, df in pred_df.groupby("musicId"):
        pred = df["pred_valence"].to_numpy(dtype=np.float64)
        target = df["target_valence"].to_numpy(dtype=np.float64)
        m = metrics(pred, target)
        rows.append(
            {
                "model_id": model_id,
                "musicId": str(sid),
                "frames": int(len(df)),
                "target_mean": float(target.mean()),
                "pred_mean": float(pred.mean()),
                "mean_error": float(pred.mean() - target.mean()),
                "abs_mean_error": float(abs(pred.mean() - target.mean())),
                "target_std": float(target.std()),
                "pred_std": float(pred.std()),
                "std_ratio": float(pred.std() / (target.std() + 1e-12)),
                **m,
            }
        )
    return pd.DataFrame(rows)


def model_summary(model_id: str, pred_df: pd.DataFrame, per_song: pd.DataFrame) -> dict[str, Any]:
    pred = pred_df["pred_valence"].to_numpy(dtype=np.float64)
    target = pred_df["target_valence"].to_numpy(dtype=np.float64)
    frame_m = metrics(pred, target)
    song_m = metrics(per_song["pred_mean"].to_numpy(dtype=np.float64), per_song["target_mean"].to_numpy(dtype=np.float64))
    return {
        "model_id": model_id,
        **{f"frame_{k}": v for k, v in frame_m.items()},
        **{f"song_mean_{k}": v for k, v in song_m.items()},
        "pred_mean": float(pred.mean()),
        "target_mean": float(target.mean()),
        "pred_std": float(pred.std()),
        "target_std": float(target.std()),
        "mean_error": float(pred.mean() - target.mean()),
        "median_song_rmse": float(per_song["rmse"].median()),
        "p90_song_rmse": float(per_song["rmse"].quantile(0.90)),
        "worst_song_rmse": float(per_song["rmse"].max()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(INDEX)
    song_df = load_song_targets(index)
    song_df.to_csv(OUT_DIR / "song_level_target_distribution.csv", index=False)
    split_df = split_summary(song_df)
    split_df.to_csv(OUT_DIR / "split_target_summary.csv", index=False)

    per_song_tables = []
    model_rows = []
    for model_id, model_dir in REPRESENTATIVE_MODELS.items():
        if not (model_dir / "test_predictions.csv").exists():
            continue
        pred_df = read_predictions(model_id, model_dir, "test")
        per_song = per_song_prediction_metrics(model_id, pred_df)
        per_song_tables.append(per_song)
        model_rows.append(model_summary(model_id, pred_df, per_song))

    all_per_song = pd.concat(per_song_tables, ignore_index=True)
    all_per_song.to_csv(OUT_DIR / "test_per_song_model_metrics.csv", index=False)
    model_summary_df = pd.DataFrame(model_rows).sort_values(["frame_rmse", "frame_mae"])
    model_summary_df.to_csv(OUT_DIR / "test_model_summary.csv", index=False)

    current = all_per_song[all_per_song["model_id"] == "current_best_extreme"].copy()
    current = current.merge(song_df[["musicId", "split", "target_abs_mean", "target_range"]], on="musicId", how="left")
    current.sort_values("rmse", ascending=False).to_csv(OUT_DIR / "current_best_worst_test_songs.csv", index=False)

    pivot_rmse = all_per_song.pivot(index="musicId", columns="model_id", values="rmse")
    pivot_abs_mean = all_per_song.pivot(index="musicId", columns="model_id", values="abs_mean_error")
    failure = pd.DataFrame(index=pivot_rmse.index)
    failure["models_count"] = pivot_rmse.notna().sum(axis=1)
    failure["models_rmse_over_025"] = (pivot_rmse > 0.25).sum(axis=1)
    failure["models_rmse_over_030"] = (pivot_rmse > 0.30).sum(axis=1)
    failure["mean_rmse_across_models"] = pivot_rmse.mean(axis=1)
    failure["max_rmse_across_models"] = pivot_rmse.max(axis=1)
    failure["mean_abs_mean_error_across_models"] = pivot_abs_mean.mean(axis=1)
    failure = failure.reset_index().merge(song_df, on="musicId", how="left")
    failure = failure.sort_values(["models_rmse_over_025", "mean_rmse_across_models"], ascending=False)
    failure.to_csv(OUT_DIR / "shared_test_failures.csv", index=False)

    correlations = []
    current_for_corr = current.copy()
    for col in ["target_mean", "target_abs_mean", "target_std", "target_range", "frames"]:
        correlations.append(
            {
                "metric": col,
                "corr_with_current_rmse": pearson(
                    current_for_corr[col].to_numpy(dtype=np.float64),
                    current_for_corr["rmse"].to_numpy(dtype=np.float64),
                ),
                "corr_with_abs_mean_error": pearson(
                    current_for_corr[col].to_numpy(dtype=np.float64),
                    current_for_corr["abs_mean_error"].to_numpy(dtype=np.float64),
                ),
            }
        )
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(OUT_DIR / "current_error_correlations.csv", index=False)

    summary = {
        "created_at": "2026-06-26",
        "index": str(INDEX),
        "songs_total": int(len(song_df)),
        "split_summary": split_df.to_dict("records"),
        "model_summary": model_summary_df.to_dict("records"),
        "top_current_best_worst_songs": current.sort_values("rmse", ascending=False).head(10).to_dict("records"),
        "top_shared_failures": failure.head(10).to_dict("records"),
        "current_error_correlations": corr_df.to_dict("records"),
    }
    (OUT_DIR / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
