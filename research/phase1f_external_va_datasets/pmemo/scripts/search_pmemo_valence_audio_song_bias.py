#!/usr/bin/env python3
"""Audio-feature song-level residual correction for PMEmo Valence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LIBROSA_DATASET, selected_indices
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR


INPUT_DIR = ROOT / "outputs" / "valence_frame_calibrator_song_bias_20260628"
OUT_DIR = ROOT / "outputs" / "valence_audio_song_bias_20260628"
PREVIOUS_BEST = {
    "model_id": "pmemo_valence_frame_calibrator_song_bias_20260628",
    "rmse": 0.19748015854593182,
    "mae": 0.15352887995061668,
    "pcc": 0.7932350928356282,
    "ccc": 0.7800520380365018,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.20 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.20 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.20 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.40 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def song_target_residuals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("musicId", sort=False):
        rows.append(
            {
                "musicId": int(sid),
                "target_mean": float(g["target_valence"].mean()),
                "pred_mean": float(g["pred_valence"].mean()),
                "residual_mean": float(g["target_valence"].mean() - g["pred_valence"].mean()),
            }
        )
    return pd.DataFrame(rows)


def stats_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = x.astype(np.float32)
    stats = {}
    for name, arr in [
        ("mean", np.mean(x, axis=0)),
        ("std", np.std(x, axis=0)),
        ("min", np.min(x, axis=0)),
        ("max", np.max(x, axis=0)),
    ]:
        for i, v in enumerate(arr):
            stats[f"{prefix}_{name}_{i}"] = float(v)
    return stats


def build_audio_feature_table(music_ids: list[int], mert_dir: Path, librosa_dataset: Path) -> pd.DataFrame:
    feature_idx = np.asarray(selected_indices(["harmonic"]), dtype=np.int64)
    rows = []
    for sid in music_ids:
        mert = np.load(mert_dir / f"{sid}.npz")["embedding"].astype(np.float32)
        lib = np.load(librosa_dataset / "sequence" / f"{sid}.npz")["X"].astype(np.float32)[:, feature_idx]
        row = {"musicId": int(sid), "frames": int(min(len(mert), len(lib)))}
        row.update(stats_features(mert, "mert"))
        row.update(stats_features(lib, "harm"))
        rows.append(row)
    return pd.DataFrame(rows)


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_pca64_a10", make_pipeline(StandardScaler(), PCA(n_components=64, random_state=42), Ridge(alpha=10.0))),
        ("ridge_pca128_a10", make_pipeline(StandardScaler(), PCA(n_components=128, random_state=42), Ridge(alpha=10.0))),
        ("elastic_a001_l102", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000))),
        (
            "hgb_pca64",
            make_pipeline(
                StandardScaler(),
                PCA(n_components=64, random_state=42),
                HistGradientBoostingRegressor(learning_rate=0.03, max_iter=160, max_leaf_nodes=15, l2_regularization=0.1, random_state=42),
            ),
        ),
    ]


def apply_bias(df: pd.DataFrame, bias_by_song: dict[int, float], strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    bias = out["musicId"].astype(int).map(bias_by_song).fillna(0.0).to_numpy(dtype=np.float64)
    if cap is not None:
        bias = np.clip(bias, -cap, cap)
    out["audio_song_bias"] = strength * bias
    out["pred_valence"] = np.clip(out["pred_valence"].to_numpy(dtype=np.float64) + out["audio_song_bias"].to_numpy(dtype=np.float64), -1.0, 1.0)
    return out


def evaluate_split(split_dir: str, mert_dir: Path, librosa_dataset: Path) -> tuple[dict, pd.DataFrame]:
    frames = {split: pd.read_csv(INPUT_DIR / split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    residuals = {split: song_target_residuals(frames[split]) for split in ["train", "val", "test"]}
    all_ids = sorted(set(int(x) for split in ["train", "val", "test"] for x in residuals[split]["musicId"].tolist()))
    audio = build_audio_feature_table(all_ids, mert_dir, librosa_dataset)
    feature_cols = [c for c in audio.columns if c != "musicId"]
    merged = {split: residuals[split].merge(audio, on="musicId", how="left") for split in ["train", "val", "test"]}
    rows = []
    cache = {}
    for model_id, model in model_candidates():
        model.fit(merged["train"][feature_cols].to_numpy(dtype=np.float64), merged["train"]["residual_mean"].to_numpy(dtype=np.float64))
        pred_bias = {
            split: dict(
                zip(
                    merged[split]["musicId"].astype(int).tolist(),
                    model.predict(merged[split][feature_cols].to_numpy(dtype=np.float64)),
                )
            )
            for split in ["train", "val", "test"]
        }
        for strength in [0.10, 0.25, 0.50, 0.75, 1.00]:
            for cap_name, cap in [("cap002", 0.02), ("cap004", 0.04), ("cap008", 0.08), ("cap012", 0.12), ("cap018", 0.18), ("none", None)]:
                split_metrics = {}
                for split in ["train", "val", "test"]:
                    corrected = apply_bias(frames[split], pred_bias[split], strength, cap)
                    split_metrics[split] = eval_df(corrected)
                    cache[(split, model_id, strength, cap_name)] = corrected
                rows.append(
                    {
                        "split_name": split_dir,
                        "model_id": model_id,
                        "strength": strength,
                        "cap": cap_name,
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = df.iloc[0].to_dict()
    split_out = OUT_DIR / selected["split_name"]
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        cache[(split, str(selected["model_id"]), float(selected["strength"]), str(selected["cap"]))].to_csv(split_out / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    return summary, df


def aggregate(rows: list[dict]) -> dict:
    out = {"created_at": "2026-06-28", "previous_best": PREVIOUS_BEST, "num_splits": len(rows), "per_split": rows}
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["baseline"]["test"][metric] for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_previous_best"] = {metric: float(selected_mean[metric] - PREVIOUS_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p.name for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir, DEFAULT_MERT_DIR, DEFAULT_LIBROSA_DATASET)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_audio_bias_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
