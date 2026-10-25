#!/usr/bin/env python3
"""Song-level Valence mean fusion on top of the current best PMEmo predictions.

This is a conservative post-model experiment: keep the current frame-level
trajectory, predict the song-level Valence mean from audio/prediction features,
then shift the whole curve by a small validation-selected amount.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LIBROSA_DATASET, selected_indices
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR


INPUT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
OUT_DIR = ROOT / "outputs" / "valence_song_mean_fusion_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.195316,
    "mae": 0.151793,
    "pcc": 0.797766,
    "ccc": 0.784834,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    """Validation score: prioritize pass-threshold gaps, especially RMSE/CCC."""
    return (
        2.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.8 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.3 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.8 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.08 * m["rmse"]
        - 0.02 * m["ccc"]
    )


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def stats_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = x.astype(np.float32)
    arrays = {
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
        "min": np.min(x, axis=0),
        "max": np.max(x, axis=0),
        "q10": np.quantile(x, 0.10, axis=0),
        "q90": np.quantile(x, 0.90, axis=0),
    }
    return {f"{prefix}_{stat}_{i}": float(v) for stat, arr in arrays.items() for i, v in enumerate(arr)}


def prediction_song_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid, g in df.groupby("musicId", sort=False):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        target = g["target_valence"].to_numpy(dtype=np.float64)
        time = g["time_sec"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "musicId": int(sid),
                "frames": int(len(g)),
                "duration_proxy": float(time.max() - time.min() + 0.5) if len(time) else 0.0,
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "pred_range": float(np.max(pred) - np.min(pred)),
                "pred_q10": float(np.quantile(pred, 0.10)),
                "pred_q90": float(np.quantile(pred, 0.90)),
                "pred_head_mean": float(np.mean(pred[: min(6, len(pred))])),
                "pred_tail_mean": float(np.mean(pred[-min(6, len(pred)) :])),
                "target_mean": float(np.mean(target)),
                "base_residual_mean": float(np.mean(target) - np.mean(pred)),
            }
        )
    return pd.DataFrame(rows)


def audio_song_features(music_ids: list[int], mert_dir: Path, librosa_dataset: Path, feature_groups: list[str]) -> pd.DataFrame:
    feature_idx = np.asarray(selected_indices(feature_groups), dtype=np.int64)
    rows = []
    for sid in music_ids:
        mert = np.load(mert_dir / f"{sid}.npz")["embedding"].astype(np.float32)
        lib = np.load(librosa_dataset / "sequence" / f"{sid}.npz")["X"].astype(np.float32)[:, feature_idx]
        n = min(len(mert), len(lib))
        row: dict[str, Any] = {"musicId": int(sid), "audio_frames": int(n)}
        row.update(stats_features(mert[:n], "mert"))
        row.update(stats_features(lib[:n], "lib"))
        rows.append(row)
    return pd.DataFrame(rows)


def pca_steps(n_components: int, train_rows: int, feature_dim: int):
    n = min(n_components, max(2, train_rows - 2), max(2, feature_dim - 1))
    return PCA(n_components=n, random_state=42)


def model_candidates(train_rows: int, feature_dim: int) -> list[tuple[str, object]]:
    return [
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        ("huber_a0001", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500))),
        ("elastic_a001_l102", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=8000))),
        ("ridge_pca32_a10", make_pipeline(StandardScaler(), pca_steps(32, train_rows, feature_dim), Ridge(alpha=10.0))),
        ("ridge_pca64_a30", make_pipeline(StandardScaler(), pca_steps(64, train_rows, feature_dim), Ridge(alpha=30.0))),
        ("ridge_pca96_a100", make_pipeline(StandardScaler(), pca_steps(96, train_rows, feature_dim), Ridge(alpha=100.0))),
        (
            "hgb_pca32",
            make_pipeline(
                StandardScaler(),
                pca_steps(32, train_rows, feature_dim),
                HistGradientBoostingRegressor(
                    learning_rate=0.025,
                    max_iter=120,
                    max_leaf_nodes=11,
                    l2_regularization=0.2,
                    random_state=42,
                ),
            ),
        ),
        (
            "rf_pca32",
            make_pipeline(
                StandardScaler(),
                pca_steps(32, train_rows, feature_dim),
                RandomForestRegressor(n_estimators=220, min_samples_leaf=8, max_features=0.5, random_state=42, n_jobs=-1),
            ),
        ),
    ]


def apply_song_mean_shift(df: pd.DataFrame, target_mean_by_song: dict[int, float], strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    pred_mean = out.groupby("musicId")["pred_valence"].transform("mean").to_numpy(dtype=np.float64)
    target_mean_pred = out["musicId"].astype(int).map(target_mean_by_song).to_numpy(dtype=np.float64)
    shift = target_mean_pred - pred_mean
    if cap is not None:
        shift = np.clip(shift, -cap, cap)
    out["base_pred_valence"] = out["pred_valence"].to_numpy(dtype=np.float64)
    out["song_mean_shift"] = strength * shift
    out["pred_valence"] = np.clip(out["base_pred_valence"].to_numpy(dtype=np.float64) + out["song_mean_shift"].to_numpy(dtype=np.float64), -1.0, 1.0)
    return out


def aggregate_metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([r[variant]["test"][metric] for r in rows], dtype=np.float64)
        out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
        out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def evaluate_split(split_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    song_tables = {split: prediction_song_features(frames[split]) for split in ["train", "val", "test"]}
    all_ids = sorted({int(x) for split in ["train", "val", "test"] for x in song_tables[split]["musicId"].tolist()})

    audio_tables = {
        "pred_only": pd.DataFrame({"musicId": all_ids}),
        "mert_harmonic": audio_song_features(all_ids, DEFAULT_MERT_DIR, DEFAULT_LIBROSA_DATASET, ["harmonic"]),
        "mert_all_librosa": audio_song_features(all_ids, DEFAULT_MERT_DIR, DEFAULT_LIBROSA_DATASET, ["mfcc", "spectral", "harmonic", "energy", "tempo"]),
    }

    results = []
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
    cache: dict[tuple[str, str, str, float, str], pd.DataFrame] = {}

    # Always include no-change baseline so validation selection cannot degrade silently.
    base_row = {
        "split_name": split_dir.name,
        "feature_set": "baseline",
        "model_id": "no_change",
        "strength": 0.0,
        "cap": "none",
        **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
        "val_selection_score": selection_score(baseline["val"]),
    }
    results.append(base_row)
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
            except Exception as exc:  # keep the search robust across small/ill-conditioned feature sets
                results.append({"split_name": split_dir.name, "feature_set": feature_set, "model_id": model_id, "error": repr(exc)})
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
            for strength in [0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.00]:
                for cap_name, cap in [("cap005", 0.005), ("cap010", 0.010), ("cap020", 0.020), ("cap040", 0.040), ("cap070", 0.070), ("cap100", 0.100), ("none", None)]:
                    split_metrics = {}
                    for split in ["train", "val", "test"]:
                        corrected = apply_song_mean_shift(frames[split], pred_means[split], strength, cap)
                        split_metrics[split] = eval_df(corrected)
                        cache[(split, feature_set, model_id, strength, cap_name)] = corrected
                    results.append(
                        {
                            "split_name": split_dir.name,
                            "feature_set": feature_set,
                            "model_id": model_id,
                            "strength": strength,
                            "cap": cap_name,
                            **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                            "val_selection_score": selection_score(split_metrics["val"]),
                        }
                    )

    result_df = pd.DataFrame(results)
    valid_df = result_df[result_df["val_selection_score"].notna()].copy()
    valid_df = valid_df.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = valid_df.iloc[0].to_dict()

    split_out = OUT_DIR / split_dir.name
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        key = (split, str(selected["feature_set"]), str(selected["model_id"]), float(selected["strength"]), str(selected["cap"]))
        cache[key].to_csv(split_out / f"{split}_predictions.csv", index=False)

    selected_metrics = {
        split: {k: float(selected[f"{split}_{k}"]) for k in ["rmse", "mae", "pcc", "ccc"]}
        for split in ["train", "val", "test"]
    }
    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
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
    }
    return summary, result_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    all_results = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, result_df = evaluate_split(split_dir)
        rows.append(summary)
        all_results.append(result_df)
    pd.concat(all_results, ignore_index=True).to_csv(OUT_DIR / "all_song_mean_fusion_results.csv", index=False)
    aggregate = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_song_mean_fusion_20260628",
        "input_model": CURRENT_BEST,
        "purpose": "Validation-selected song-level Valence mean correction for the current best frame-level model.",
        "selection_rule": "per split, select by validation composite pass-gap score; baseline/no-change is included as a candidate.",
        "num_splits": len(rows),
        "per_split": rows,
        **aggregate_metrics(rows, "baseline"),
        **aggregate_metrics(rows, "selected"),
    }
    selected_mean = {metric: aggregate[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: aggregate[f"baseline_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    aggregate["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    aggregate["delta_vs_input_test_mean"] = {
        metric: float(selected_mean[metric] - baseline_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
