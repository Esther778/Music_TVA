#!/usr/bin/env python3
"""Locked-dev song-level Valence polarity correction.

This development-only experiment treats the current frame-level Valence curve
as the primary predictor, then learns a small song-level mean/polarity
correction from held-in development folds. The final locked test is not read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_current_best_tonal_correction import (
    apply_song_mean_shift,
    eval_df,
    prediction_song_features,
)
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_locked_residual_refine_xwide_calibrated_20260713"
MERT_DIR = ROOT / "outputs" / "mert_pmemo" / "features"
OUT_DIR = ROOT / "outputs" / "valence_locked_song_polarity_correction_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}

PRED_COLS = [
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


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - PASS["mae"])
        + 0.04 * max(0.0, PASS["pcc"] - m["pcc"])
        + 0.09 * max(0.0, PASS["ccc"] - m["ccc"])
    )


def cap_value(name: str) -> float | None:
    if name == "none":
        return None
    return float(name.replace("cap", "")) / 1000.0


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: float(np.mean([row[m] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def load_frames() -> dict[str, pd.DataFrame]:
    return {fold: pd.read_csv(INPUT_DIR / f"{fold}_val_predictions.csv") for fold in FOLDS}


def mert_song_features(music_ids: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for sid in music_ids:
        z = np.load(MERT_DIR / f"{sid}.npz")
        emb = z["embedding"].astype(np.float64)
        blocks = {
            "mean": emb.mean(axis=0),
            "std": emb.std(axis=0),
            "q10": np.quantile(emb, 0.10, axis=0),
            "q90": np.quantile(emb, 0.90, axis=0),
        }
        row: dict[str, float] = {"musicId": float(sid)}
        for prefix, values in blocks.items():
            for i, value in enumerate(values):
                row[f"mert_{prefix}_{i}"] = float(value)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .assign(musicId=lambda d: d["musicId"].astype(int))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def select_cols(train: pd.DataFrame, candidate_cols: list[str], y_col: str, top_k: int) -> list[str]:
    if top_k <= 0:
        return []
    y = train[y_col].to_numpy(dtype=np.float64)
    y = y - y.mean()
    y_norm = float(np.linalg.norm(y))
    if y_norm < 1e-12:
        return []
    scored: list[tuple[float, str]] = []
    for col in candidate_cols:
        x = train[col].to_numpy(dtype=np.float64)
        x = x - x.mean()
        denom = float(np.linalg.norm(x) * y_norm)
        if denom < 1e-12:
            continue
        corr = abs(float(np.dot(x, y) / denom))
        if np.isfinite(corr):
            scored.append((corr, col))
    scored.sort(reverse=True)
    return [col for _, col in scored[: min(top_k, len(scored))]]


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a3", make_pipeline(StandardScaler(), Ridge(alpha=3.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        ("elastic_a003_l102", make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=20000))),
        ("elastic_a010_l102", make_pipeline(StandardScaler(), ElasticNet(alpha=0.10, l1_ratio=0.2, max_iter=20000))),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    baseline_rows = [eval_df(frames[fold]) for fold in FOLDS]
    baseline_macro = macro(baseline_rows)
    all_ids = sorted({int(sid) for df in frames.values() for sid in df["musicId"].tolist()})
    mert = mert_song_features(all_ids)
    mert_cols = [c for c in mert.columns if c != "musicId"]

    song_tables = {}
    for fold, df in frames.items():
        table = prediction_song_features(df).merge(mert, on="musicId", how="left").fillna(0.0)
        table["residual_mean"] = table["target_mean"].to_numpy(dtype=np.float64) - table["pred_mean"].to_numpy(dtype=np.float64)
        song_tables[fold] = table

    rows = []
    pred_cache: dict[tuple[str, int, str], dict[str, dict[int, float]]] = {}
    for target_mode, y_col in [("target_mean", "target_mean"), ("residual_mean", "residual_mean")]:
        for top_k in [0, 4, 8, 16, 32, 64, 96, 128]:
            for model_id, model in model_candidates():
                fold_target_preds: dict[str, dict[int, float]] = {}
                per_fold_model_rows = []
                for held_fold in FOLDS:
                    train = pd.concat([song_tables[f] for f in FOLDS if f != held_fold], ignore_index=True)
                    val = song_tables[held_fold]
                    selected = select_cols(train, mert_cols, y_col, top_k)
                    cols = PRED_COLS + selected
                    model.fit(train[cols].to_numpy(dtype=np.float64), train[y_col].to_numpy(dtype=np.float64))
                    pred = model.predict(val[cols].to_numpy(dtype=np.float64))
                    if target_mode == "residual_mean":
                        pred = val["pred_mean"].to_numpy(dtype=np.float64) + pred
                    fold_target_preds[held_fold] = dict(zip(val["musicId"].astype(int).tolist(), pred.tolist()))
                    per_fold_model_rows.append({"fold": held_fold, "selected_mert_cols": selected})
                pred_cache[(target_mode, top_k, model_id)] = fold_target_preds

                for strength in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0]:
                    for cap in ["cap003", "cap005", "cap008", "cap010", "cap015", "cap020", "cap030", "cap040", "cap060", "none"]:
                        eval_rows = []
                        for fold in FOLDS:
                            corrected = apply_song_mean_shift(
                                frames[fold],
                                fold_target_preds[fold],
                                strength=float(strength),
                                cap=cap_value(cap),
                            )
                            eval_rows.append(eval_df(corrected))
                        m = macro(eval_rows)
                        row = {
                            "target_mode": target_mode,
                            "top_k": int(top_k),
                            "model_id": model_id,
                            "strength": float(strength),
                            "cap": cap,
                            **m,
                            "score": score(m),
                            "beats_baseline": (
                                m["rmse"] <= baseline_macro["rmse"]
                                and m["mae"] <= baseline_macro["mae"] + 0.0005
                                and m["pcc"] >= baseline_macro["pcc"] - 0.0005
                                and m["ccc"] >= baseline_macro["ccc"] - 0.0005
                            ),
                        }
                        rows.append(row)

    candidates = pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False])
    candidates.to_csv(OUT_DIR / "all_song_polarity_candidates.csv", index=False)
    if candidates.empty:
        raise RuntimeError("No candidates generated")

    guarded = candidates[candidates["beats_baseline"]].copy()
    if guarded.empty:
        selected = candidates.iloc[0].to_dict()
        selected["selection_reason"] = "diagnostic best only; did not pass baseline guard"
    else:
        selected = guarded.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by locked-dev macro score with baseline guard"

    selected_key = (str(selected["target_mode"]), int(selected["top_k"]), str(selected["model_id"]))
    target_preds = pred_cache[selected_key]
    out_frames = []
    per_fold = []
    for fold in FOLDS:
        corrected = apply_song_mean_shift(
            frames[fold],
            target_preds[fold],
            strength=float(selected["strength"]),
            cap=cap_value(str(selected["cap"])),
        )
        corrected["base_locked_best_pred_valence"] = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
        corrected.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
        fold_m = eval_df(corrected)
        per_fold.append({"fold": fold, **fold_m})
        out_frames.append(corrected)

    selected_macro = macro(per_fold)
    merged = pd.concat(out_frames, ignore_index=True)
    global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged["target_valence"].to_numpy(dtype=np.float64))
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Locked-dev song-level Valence polarity correction; final locked test excluded",
        "selection_boundary": "Leave-one-devfold-out over current locked-dev predictions; no final locked test labels used",
        "input_dir": str(INPUT_DIR),
        "baseline_macro": baseline_macro,
        "selected_config": selected,
        "per_fold": per_fold,
        "macro_mean": selected_macro,
        "global_frame": global_frame,
        "delta_vs_baseline_macro": {m: float(selected_macro[m] - baseline_macro[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "gap_rows_macro": threshold_rows(selected_macro),
        "gap_rows_global_frame": threshold_rows(global_frame),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
