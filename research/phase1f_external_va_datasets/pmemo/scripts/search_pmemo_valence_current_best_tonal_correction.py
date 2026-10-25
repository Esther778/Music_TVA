#!/usr/bin/env python3
"""Global tonal song-mean correction for the current best Valence teacher.

The current Valence error is dominated by song-level mean/polarity bias. This
experiment tests whether compact tonal descriptors derived from MP3-extractable
Librosa chroma/tonnetz features can improve that mean correction without using
test labels or split-specific postprocess policies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_song_mean_global_policy_20260707"
LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset" / "sequence"
OUT_DIR = ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"

CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_song_mean_global_policy_20260707",
    "rmse": 0.19078885880411342,
    "mae": 0.1486967345561368,
    "pcc": 0.8074956914596779,
    "ccc": 0.7951366852830063,
}

PASS_TARGETS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def cap_value(name: str) -> float | None:
    if name == "none":
        return None
    if name.startswith("cap"):
        return float(name.replace("cap", "")) / 1000.0
    raise ValueError(f"Unknown cap: {name}")


def prediction_song_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid, g in df.groupby("musicId", sort=False):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        target = g["target_valence"].to_numpy(dtype=np.float64)
        time = g["time_sec"].to_numpy(dtype=np.float64)
        delta = np.diff(pred) if len(pred) > 1 else np.asarray([0.0])
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
                "pred_q25": float(np.quantile(pred, 0.25)),
                "pred_q75": float(np.quantile(pred, 0.75)),
                "pred_q90": float(np.quantile(pred, 0.90)),
                "pred_head_mean": float(np.mean(pred[: min(8, len(pred))])),
                "pred_tail_mean": float(np.mean(pred[-min(8, len(pred)) :])),
                "pred_slope_proxy": float(pred[-1] - pred[0]) if len(pred) else 0.0,
                "pred_abs_delta_mean": float(np.mean(np.abs(delta))),
                "target_mean": float(np.mean(target)),
            }
        )
    return pd.DataFrame(rows)


def _corr_template(chroma: np.ndarray, template: np.ndarray) -> float:
    x = chroma.astype(np.float64)
    t = template.astype(np.float64)
    x = x - x.mean()
    t = t - t.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(t))
    return float(np.dot(x, t) / denom) if denom > 1e-12 else 0.0


def tonal_features_for_song(music_id: int) -> dict[str, float]:
    data = np.load(LIBROSA_DATASET / f"{music_id}.npz")
    x = data["X"].astype(np.float64)
    chroma = np.maximum(x[:, 31:43], 0.0)
    tonnetz = x[:, 43:49]
    chroma_std_block = x[:, 82:94]
    tonnetz_std_block = x[:, 94:100]

    eps = 1e-9
    chroma_sum = chroma.sum(axis=1, keepdims=True) + eps
    chroma_norm = chroma / chroma_sum
    mean_chroma = chroma_norm.mean(axis=0)
    std_chroma = chroma_norm.std(axis=0)
    sorted_mean = np.sort(mean_chroma)[::-1]
    frame_top = np.max(chroma_norm, axis=1)
    frame_sorted = np.sort(chroma_norm, axis=1)[:, ::-1]
    entropy = -np.sum(chroma_norm * np.log(chroma_norm + eps), axis=1) / np.log(12.0)

    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_corrs = np.asarray([_corr_template(mean_chroma, np.roll(major, k)) for k in range(12)])
    minor_corrs = np.asarray([_corr_template(mean_chroma, np.roll(minor, k)) for k in range(12)])
    best_major_idx = int(np.argmax(major_corrs))
    best_minor_idx = int(np.argmax(minor_corrs))
    best_major = float(major_corrs[best_major_idx])
    best_minor = float(minor_corrs[best_minor_idx])

    tonal_delta = np.diff(tonnetz, axis=0) if len(tonnetz) > 1 else np.zeros((1, tonnetz.shape[1]))
    chroma_delta = np.diff(chroma_norm, axis=0) if len(chroma_norm) > 1 else np.zeros((1, chroma_norm.shape[1]))

    row: dict[str, float] = {
        "musicId": float(music_id),
        "chroma_entropy_mean": float(np.mean(entropy)),
        "chroma_entropy_std": float(np.std(entropy)),
        "chroma_peak_mean": float(np.mean(frame_top)),
        "chroma_peak_std": float(np.std(frame_top)),
        "chroma_concentration": float(sorted_mean[0]),
        "chroma_top_gap": float(sorted_mean[0] - sorted_mean[1]),
        "chroma_top3_sum": float(sorted_mean[:3].sum()),
        "major_corr": best_major,
        "minor_corr": best_minor,
        "mode_score_major_minus_minor": float(best_major - best_minor),
        "mode_abs_confidence": float(max(best_major, best_minor)),
        "key_sin": float(np.sin(2.0 * np.pi * best_major_idx / 12.0)),
        "key_cos": float(np.cos(2.0 * np.pi * best_major_idx / 12.0)),
        "minor_key_sin": float(np.sin(2.0 * np.pi * best_minor_idx / 12.0)),
        "minor_key_cos": float(np.cos(2.0 * np.pi * best_minor_idx / 12.0)),
        "tonnetz_move_mean": float(np.mean(np.linalg.norm(tonal_delta, axis=1))),
        "chroma_move_mean": float(np.mean(np.linalg.norm(chroma_delta, axis=1))),
    }
    for i, v in enumerate(mean_chroma):
        row[f"chroma_mean_{i}"] = float(v)
    for i, v in enumerate(std_chroma):
        row[f"chroma_std_{i}"] = float(v)
    for i, v in enumerate(np.mean(tonnetz, axis=0)):
        row[f"tonnetz_mean_{i}"] = float(v)
    for i, v in enumerate(np.std(tonnetz, axis=0)):
        row[f"tonnetz_std_{i}"] = float(v)
    for i, v in enumerate(np.mean(chroma_std_block, axis=0)):
        row[f"chroma_block_std_mean_{i}"] = float(v)
    for i, v in enumerate(np.mean(tonnetz_std_block, axis=0)):
        row[f"tonnetz_block_std_mean_{i}"] = float(v)
    return row


def tonal_table(music_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame([tonal_features_for_song(sid) for sid in music_ids]).assign(musicId=lambda d: d["musicId"].astype(int))


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a3", make_pipeline(StandardScaler(), Ridge(alpha=3.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("huber_a0001", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=1000))),
        ("elastic_a001_l102", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000))),
        ("quad_ridge_a30", make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=30.0))),
        ("quad_ridge_a100", make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=100.0))),
    ]


def apply_song_mean_shift(df: pd.DataFrame, target_mean_by_song: dict[int, float], strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    pred_mean = out.groupby("musicId")["pred_valence"].transform("mean").to_numpy(dtype=np.float64)
    target_mean_pred = out["musicId"].astype(int).map(target_mean_by_song).to_numpy(dtype=np.float64)
    shift = target_mean_pred - pred_mean
    if cap is not None:
        shift = np.clip(shift, -cap, cap)
    out["tonal_base_pred_valence"] = out["pred_valence"].to_numpy(dtype=np.float64)
    out["tonal_song_mean_shift"] = strength * shift
    out["pred_valence"] = np.clip(out["tonal_base_pred_valence"].to_numpy(dtype=np.float64) + out["tonal_song_mean_shift"].to_numpy(dtype=np.float64), -1.0, 1.0)
    return out


def build_candidate_rows() -> tuple[pd.DataFrame, dict[tuple[str, str, str, float, str], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    materialized: dict[tuple[str, str, str, float, str], pd.DataFrame] = {}
    split_dirs = sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir())
    all_ids = sorted(
        {
            int(sid)
            for split_dir in split_dirs
            for part in ["train", "val", "test"]
            for sid in pd.read_csv(split_dir / f"{part}_predictions.csv")["musicId"].unique()
        }
    )
    tonal = tonal_table(all_ids)
    pred_cols = [
        "frames",
        "duration_proxy",
        "pred_mean",
        "pred_std",
        "pred_min",
        "pred_max",
        "pred_range",
        "pred_q10",
        "pred_q25",
        "pred_q75",
        "pred_q90",
        "pred_head_mean",
        "pred_tail_mean",
        "pred_slope_proxy",
        "pred_abs_delta_mean",
    ]
    feature_sets = {
        "pred_only": [],
        "tonal_only": [c for c in tonal.columns if c != "musicId"],
        "pred_tonal": [c for c in tonal.columns if c != "musicId"],
        "pred_compact_tonal": [
            "chroma_entropy_mean",
            "chroma_peak_mean",
            "chroma_concentration",
            "chroma_top_gap",
            "chroma_top3_sum",
            "major_corr",
            "minor_corr",
            "mode_score_major_minus_minor",
            "mode_abs_confidence",
            "tonnetz_move_mean",
            "chroma_move_mean",
            "key_sin",
            "key_cos",
        ],
    }

    for split_dir in split_dirs:
        frames = {part: pd.read_csv(split_dir / f"{part}_predictions.csv") for part in ["train", "val", "test"]}
        song_tables = {part: prediction_song_features(frames[part]) for part in ["train", "val", "test"]}
        baseline = {part: eval_df(frames[part]) for part in ["train", "val", "test"]}
        base_key = (split_dir.name, "baseline", "no_change", 0.0, "none")
        for part in ["train", "val", "test"]:
            materialized[(part, *base_key)] = frames[part].copy()
        rows.append(
            {
                "split_name": split_dir.name,
                "feature_set": "baseline",
                "model_id": "no_change",
                "strength": 0.0,
                "cap": "none",
                **{f"{part}_{k}": v for part, m in baseline.items() for k, v in m.items()},
            }
        )
        merged = {part: song_tables[part].merge(tonal, on="musicId", how="left") for part in ["train", "val", "test"]}
        for feature_set, tonal_cols in feature_sets.items():
            cols = tonal_cols if feature_set == "tonal_only" else pred_cols + tonal_cols
            x_train = merged["train"][cols].to_numpy(dtype=np.float64)
            y_train = merged["train"]["target_mean"].to_numpy(dtype=np.float64)
            for model_id, model in model_candidates():
                try:
                    model.fit(x_train, y_train)
                except Exception as exc:
                    rows.append({"split_name": split_dir.name, "feature_set": feature_set, "model_id": model_id, "error": repr(exc)})
                    continue
                pred_means = {
                    part: dict(
                        zip(
                            merged[part]["musicId"].astype(int).tolist(),
                            model.predict(merged[part][cols].to_numpy(dtype=np.float64)),
                        )
                    )
                    for part in ["train", "val", "test"]
                }
                for strength in [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]:
                    for cap_name in ["cap010", "cap020", "cap040", "cap060", "cap080", "none"]:
                        cap = cap_value(cap_name)
                        split_metrics = {}
                        key = (split_dir.name, feature_set, model_id, strength, cap_name)
                        for part in ["train", "val", "test"]:
                            corrected = apply_song_mean_shift(frames[part], pred_means[part], strength, cap)
                            split_metrics[part] = eval_df(corrected)
                            materialized[(part, *key)] = corrected
                        rows.append(
                            {
                                "split_name": split_dir.name,
                                "feature_set": feature_set,
                                "model_id": model_id,
                                "strength": strength,
                                "cap": cap_name,
                                **{f"{part}_{k}": v for part, m in split_metrics.items() for k, v in m.items()},
                            }
                        )
    return pd.DataFrame(rows), materialized


def aggregate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["feature_set", "model_id", "strength", "cap"]
    metric_cols = [f"{part}_{metric}" for part in ["val", "test"] for metric in ["rmse", "mae", "pcc", "ccc"]]
    return df.dropna(subset=["val_rmse"]).groupby(keys, dropna=False)[metric_cols].mean().reset_index()


def select_global(agg: pd.DataFrame) -> dict[str, Any]:
    baseline = agg[agg["model_id"].eq("no_change")].iloc[0]
    # Main guard: do not buy test-looking gains with a validation collapse.
    guarded = agg[
        (agg["val_rmse"] <= baseline["val_rmse"] + 0.0015)
        & (agg["val_mae"] <= baseline["val_mae"] + 0.0015)
        & (agg["val_pcc"] >= baseline["val_pcc"] - 0.0015)
        & (agg["val_ccc"] >= baseline["val_ccc"] + 0.0005)
    ].copy()
    if guarded.empty:
        selected = baseline.copy()
        reason = "identity retained; no tonal correction passed validation guard"
    else:
        guarded["selection_score"] = (
            3.0 * np.maximum(0.0, guarded["val_rmse"] - PASS_TARGETS["rmse"]) / PASS_TARGETS["rmse"]
            + 1.6 * np.maximum(0.0, guarded["val_mae"] - PASS_TARGETS["mae"]) / PASS_TARGETS["mae"]
            + 1.8 * np.maximum(0.0, PASS_TARGETS["ccc"] - guarded["val_ccc"]) / PASS_TARGETS["ccc"]
            + 1.2 * np.maximum(0.0, PASS_TARGETS["pcc"] - guarded["val_pcc"]) / PASS_TARGETS["pcc"]
        )
        selected = guarded.sort_values(["selection_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0]
        reason = "global tonal correction selected by validation guard"
    out = selected.to_dict()
    out["selection_reason"] = reason
    out["baseline_val_rmse"] = float(baseline["val_rmse"])
    out["baseline_val_mae"] = float(baseline["val_mae"])
    out["baseline_val_pcc"] = float(baseline["val_pcc"])
    out["baseline_val_ccc"] = float(baseline["val_ccc"])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates, materialized = build_candidate_rows()
    candidates.to_csv(OUT_DIR / "all_tonal_correction_candidates_by_split.csv", index=False)
    agg = aggregate_candidates(candidates)
    agg.to_csv(OUT_DIR / "all_tonal_correction_candidates_global.csv", index=False)
    selected = select_global(agg)

    split_dirs = sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir())
    selected_rows: list[dict[str, Any]] = []
    for split_dir in split_dirs:
        key = (
            split_dir.name,
            str(selected["feature_set"]),
            str(selected["model_id"]),
            float(selected["strength"]),
            str(selected["cap"]),
        )
        split_out = OUT_DIR / split_dir.name
        split_out.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {"split_name": split_dir.name}
        for part in ["train", "val", "test"]:
            df = materialized[(part, *key)].copy()
            df.to_csv(split_out / f"{part}_predictions.csv", index=False)
            m = eval_df(df)
            row.update({f"{part}_{metric}": value for metric, value in m.items()})
        selected_rows.append(row)
    selected_df = pd.DataFrame(selected_rows)

    selected_mean = {
        metric: float(selected_df[f"test_{metric}"].mean())
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    result = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_current_best_tonal_correction_20260707",
        "input_model": CURRENT_BEST,
        "selection_rule": "one shared global song-mean correction config; validation guard vs current best; no test-label tuning",
        "selected_config": selected,
        "num_splits": len(split_dirs),
        "selected_test_rmse_mean": selected_mean["rmse"],
        "selected_test_mae_mean": selected_mean["mae"],
        "selected_test_pcc_mean": selected_mean["pcc"],
        "selected_test_ccc_mean": selected_mean["ccc"],
        "selected_test_rmse_std": float(selected_df["test_rmse"].std(ddof=0)),
        "selected_test_mae_std": float(selected_df["test_mae"].std(ddof=0)),
        "selected_test_pcc_std": float(selected_df["test_pcc"].std(ddof=0)),
        "selected_test_ccc_std": float(selected_df["test_ccc"].std(ddof=0)),
        "delta_vs_current_best": {
            metric: float(selected_mean[metric] - CURRENT_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "threshold_gaps": threshold_rows(selected_mean),
        "per_split": selected_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
