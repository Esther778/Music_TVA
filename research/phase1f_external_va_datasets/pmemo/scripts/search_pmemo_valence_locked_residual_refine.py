#!/usr/bin/env python3
"""Locked-dev Valence residual refinement from audio features.

This is an exploratory development-fold search. It uses train-fold residuals
and validation-fold selection only; the final locked test is not read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_audio_ensemble_postprocess import candidate_paths, pred_col, target_col
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_BASE = ROOT / "outputs" / "valence_locked_audio_ensemble_postprocess_tflstm_allfolds_20260713"
DEFAULT_SUMMARY = DEFAULT_BASE / "summary.json"
DEFAULT_OPENSMILE = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_STATIC = ROOT / "data" / "raw" / "PMEmo_2019_updated" / "extracted" / "PMEmo2019" / "features" / "static_features.csv"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_residual_refine_20260713"


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - m["pcc"])
        + 0.05 * max(0.0, 0.80 - m["ccc"])
    )


def standard(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pc = pred_col(df)
    tc = target_col(df)
    return df[["musicId", "time_sec", tc, pc]].rename(columns={tc: "target_valence", pc: f"pred_{name}"})


def train_path_from_val(path: Path) -> Path | None:
    train = path.with_name("train_predictions.csv")
    return train if train.exists() else None


def reconstruct_train_base(fold: str, fold_row: dict[str, Any]) -> pd.DataFrame:
    paths = candidate_paths(fold)
    frames = []
    weights = []
    names = []
    for name, val_path in paths.items():
        train_path = train_path_from_val(val_path)
        weight = float(fold_row.get(f"w_{name}", 0.0))
        if train_path and weight > 0:
            frames.append(standard(train_path, name))
            weights.append(weight)
            names.append(name)
    if not frames:
        raise RuntimeError(f"No train prediction candidates for {fold}")
    base = frames[0][["musicId", "time_sec", "target_valence"]].copy()
    for frame in frames:
        base = base.merge(frame.drop(columns=["target_valence"]), on=["musicId", "time_sec"], how="inner")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    pred = np.zeros(len(base), dtype=np.float64)
    for name, weight in zip(names, w):
        pred += weight * base[f"pred_{name}"].to_numpy(dtype=np.float64)
    out = base[["musicId", "time_sec", "target_valence"]].copy()
    out["pred_valence"] = pred
    return out


def load_val_base(base_dir: Path, fold: str) -> pd.DataFrame:
    return pd.read_csv(base_dir / f"{fold}_val_predictions.csv")


def load_smile_for_df(df: pd.DataFrame, opensmile_dir: Path) -> np.ndarray:
    chunks = []
    for music_id, idx in df.groupby("musicId", sort=False).groups.items():
        ii = np.asarray(idx)
        g = df.loc[ii]
        data = np.load(opensmile_dir / f"{int(music_id)}.npz")
        times = np.round(data["time_sec"].astype(np.float64), 3)
        lookup = {float(t): j for j, t in enumerate(times)}
        ids = [lookup[float(round(t, 3))] for t in g["time_sec"].to_numpy(dtype=np.float64)]
        chunks.append(data["X"].astype(np.float64)[np.asarray(ids, dtype=np.int64)])
    return np.vstack(chunks)


def meta_features(df: pd.DataFrame) -> np.ndarray:
    chunks = []
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        ii = np.asarray(idx)
        g = df.loc[ii]
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        time = g["time_sec"].to_numpy(dtype=np.float64)
        dur = max(float(time.max() - time.min()), 1e-6)
        pos = (time - time.min()) / dur
        delta = np.concatenate([[0.0], np.diff(pred)])
        chunks.append(
            np.column_stack(
                [
                    pred,
                    pred**2,
                    np.abs(pred),
                    delta,
                    pos,
                    pos**2,
                    np.sin(2.0 * np.pi * pos),
                    np.cos(2.0 * np.pi * pos),
                    np.full_like(pred, pred.mean()),
                    np.full_like(pred, pred.std()),
                ]
            )
        )
    return np.vstack(chunks)


def topk_indices(x: np.ndarray, residual: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    xc = x - x.mean(axis=0, keepdims=True)
    yc = residual - residual.mean()
    denom = np.sqrt(np.sum(xc**2, axis=0) * np.sum(yc**2)) + 1e-12
    corr = np.abs((xc.T @ yc) / denom)
    return np.argsort(corr)[-k:]


def load_static_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "musicId"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.set_index("musicId")


def static_matrix_for_df(df: pd.DataFrame, static_df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(df), 0), dtype=np.float64)
    values = static_df.loc[df["musicId"].astype(int).to_numpy(), columns].to_numpy(dtype=np.float64)
    return values


def topk_static_columns(train: pd.DataFrame, residual: np.ndarray, static_df: pd.DataFrame, k: int) -> list[str]:
    if k <= 0:
        return []
    tmp = train[["musicId"]].copy()
    tmp["residual"] = residual
    song_resid = tmp.groupby("musicId")["residual"].mean()
    song_ids = song_resid.index.astype(int).to_numpy()
    cols = list(static_df.columns)
    x = static_df.loc[song_ids, cols].to_numpy(dtype=np.float64)
    y = song_resid.to_numpy(dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc**2, axis=0) * np.sum(yc**2)) + 1e-12
    corr = np.abs((xc.T @ yc) / denom)
    return [cols[i] for i in np.argsort(corr)[-k:]]


def corrected(base: np.ndarray, residual: np.ndarray, strength: float, cap: float | None) -> np.ndarray:
    corr = residual.copy()
    if cap is not None:
        corr = np.clip(corr, -cap, cap)
    return np.clip(base + strength * corr, -1.0, 1.0)


def model_candidates():
    return [
        ("ridge30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        ("ridge1000", make_pipeline(StandardScaler(), Ridge(alpha=1000.0))),
    ]


def search_fold(
    base_dir: Path,
    fold_row: dict[str, Any],
    opensmile_dir: Path,
    static_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold = str(fold_row["fold"])
    train = reconstruct_train_base(fold, fold_row)
    val = load_val_base(base_dir, fold)
    train_smile = load_smile_for_df(train, opensmile_dir)
    val_smile = load_smile_for_df(val, opensmile_dir)
    train_residual = train["target_valence"].to_numpy(dtype=np.float64) - train["pred_valence"].to_numpy(dtype=np.float64)
    base_val_pred = val["pred_valence"].to_numpy(dtype=np.float64)
    target_val = val["target_valence"].to_numpy(dtype=np.float64)
    rows = []
    best = None
    best_pred = None
    for top_k in [0, 8, 16, 32, 64]:
        idx = topk_indices(train_smile, train_residual, top_k)
        for static_top_k in [0, 16, 32, 64, 128]:
            static_cols = topk_static_columns(train, train_residual, static_df, static_top_k)
            x_train = np.concatenate(
                [meta_features(train), train_smile[:, idx], static_matrix_for_df(train, static_df, static_cols)],
                axis=1,
            )
            x_val = np.concatenate(
                [meta_features(val), val_smile[:, idx], static_matrix_for_df(val, static_df, static_cols)],
                axis=1,
            )
            for model_id, model in model_candidates():
                model.fit(x_train, train_residual)
                val_resid = model.predict(x_val)
                for strength in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50, 2.00]:
                    for cap in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.040, 0.080, 0.120, 0.160, None]:
                        pred = corrected(base_val_pred, val_resid, strength, cap)
                        m = metrics(pred, target_val)
                        row = {
                            "fold": fold,
                            "top_k": top_k,
                            "static_top_k": static_top_k,
                            "model_id": model_id,
                            "strength": strength,
                            "cap": "none" if cap is None else cap,
                            **m,
                        }
                        row["score"] = score(m)
                        rows.append(row)
                        key = (row["score"], row["rmse"], -row["ccc"])
                        if best is None or key < (best["score"], best["rmse"], -best["ccc"]):
                            best = row
                            best_pred = pred
    assert best is not None and best_pred is not None
    val_out = val.copy()
    val_out["base_pred_valence"] = base_val_pred
    val_out["pred_valence"] = best_pred
    pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(out_dir / f"{fold}_residual_candidates.csv", index=False)
    val_out.to_csv(out_dir / f"{fold}_val_predictions.csv", index=False)
    return best, val_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--opensmile-dir", type=Path, default=DEFAULT_OPENSMILE)
    parser.add_argument("--static-features", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.base_summary.read_text(encoding="utf-8"))
    static_df = load_static_features(args.static_features)
    per_fold = []
    frames = []
    for fold_row in summary["per_fold"]:
        best, pred = search_fold(args.base_dir, fold_row, args.opensmile_dir, static_df, args.out_dir)
        per_fold.append(best)
        frames.append(pred)
    macro = {m: float(np.mean([row[m] for row in per_fold])) for m in ["rmse", "mae", "pcc", "ccc"]}
    merged = pd.concat(frames, ignore_index=True)
    global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged["target_valence"].to_numpy(dtype=np.float64))
    out = {
        "created_at": "2026-07-13",
        "purpose": "Locked-dev Valence residual refinement from train-fold audio features; final locked test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "base_summary": str(args.base_summary),
        "per_fold": per_fold,
        "macro_mean": macro,
        "global_frame": global_frame,
        "gap_rows_macro": threshold_rows(macro),
        "gap_rows_global_frame": threshold_rows(global_frame),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
