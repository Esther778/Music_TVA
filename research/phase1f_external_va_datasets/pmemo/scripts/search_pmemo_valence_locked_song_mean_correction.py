#!/usr/bin/env python3
"""Song-level mean correction for locked-dev Valence models.

The correction is trained on train-fold songs only and selected on validation
folds. It uses audio-derived MERT pooled features plus statistics of the base
Valence prediction curve, so it remains compatible with arbitrary full-song
prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_song_mean_correction_20260713"


def base_pred_path(fold: str, split: str, base: str) -> Path:
    if base == "bestlike":
        return ROOT / "outputs" / "valence_locked_devfolds_bestlike_20260713" / fold / "v_dev_best_like_h80_do030_lr4e4" / f"{split}_predictions.csv"
    if base == "conservative":
        return ROOT / "outputs" / "valence_locked_fold03_config_screen_20260713" / "conservative" / "v_dev_conservative_h80_do035_lr25e4" / f"{split}_predictions.csv"
    raise ValueError(base)


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def song_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    pcol = pred_col(df)
    tcol = target_col(df)
    rows: list[dict[str, float]] = []
    for sid, g in df.sort_values(["musicId", "time_sec"]).groupby("musicId", sort=True):
        pred = g[pcol].to_numpy(dtype=np.float64)
        target = g[tcol].to_numpy(dtype=np.float64)
        time = g["time_sec"].to_numpy(dtype=np.float64)
        q = np.quantile(pred, [0.1, 0.25, 0.75, 0.9])
        rows.append(
            {
                "musicId": int(sid),
                "frames": float(len(pred)),
                "duration_proxy": float(time.max() - time.min()) if len(time) else 0.0,
                "pred_mean": float(pred.mean()),
                "pred_std": float(pred.std()),
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
                "pred_range": float(pred.max() - pred.min()),
                "pred_q10": float(q[0]),
                "pred_q25": float(q[1]),
                "pred_q75": float(q[2]),
                "pred_q90": float(q[3]),
                "pred_head_mean": float(pred[: max(1, len(pred) // 5)].mean()),
                "pred_tail_mean": float(pred[-max(1, len(pred) // 5) :].mean()),
                "pred_slope_proxy": float(pred[-1] - pred[0]) if len(pred) > 1 else 0.0,
                "pred_abs_delta_mean": float(np.mean(np.abs(np.diff(pred)))) if len(pred) > 1 else 0.0,
                "target_mean": float(target.mean()),
                "residual_mean": float(target.mean() - pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def mert_song_features(music_ids: list[int], mert_dir: Path) -> pd.DataFrame:
    rows = []
    for sid in music_ids:
        emb = np.load(mert_dir / f"{sid}.npz")["embedding"].astype(np.float64)
        row: dict[str, float] = {"musicId": int(sid)}
        for prefix, values in {"mean": emb.mean(axis=0), "std": emb.std(axis=0)}.items():
            for i, value in enumerate(values):
                row[f"mert_{prefix}_{i}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def select_cols(train: pd.DataFrame, cols: list[str], y_col: str, k: int) -> list[str]:
    if k <= 0:
        return []
    y = train[y_col].to_numpy(dtype=np.float64)
    y = y - y.mean()
    y_std = float(np.sqrt(np.mean(y * y)))
    scored = []
    for col in cols:
        x = train[col].to_numpy(dtype=np.float64)
        x = x - x.mean()
        x_std = float(np.sqrt(np.mean(x * x)))
        if x_std < 1e-12 or y_std < 1e-12:
            continue
        corr = abs(float(np.mean(x * y) / (x_std * y_std)))
        if np.isfinite(corr):
            scored.append((corr, col))
    scored.sort(reverse=True)
    return [col for _, col in scored[: min(k, len(scored))]]


def apply_song_mean_shift(df: pd.DataFrame, target_mean_pred: dict[int, float], strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    pcol = pred_col(out)
    song_mean = out.groupby("musicId", sort=False)[pcol].transform("mean").to_numpy(dtype=np.float64)
    desired = out["musicId"].astype(int).map(target_mean_pred).to_numpy(dtype=np.float64)
    delta = strength * (desired - song_mean)
    if cap is not None:
        delta = np.clip(delta, -cap, cap)
    pred = out[pcol].to_numpy(dtype=np.float64)
    out["base_pred_valence"] = pred
    out["pred_valence"] = np.clip(pred + delta, -1.0, 1.0)
    out["song_mean_delta"] = delta
    return out


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df[target_col(df)].to_numpy(dtype=np.float64))


def models(seed: int) -> dict[str, Any]:
    return {
        "ridge30": make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
        "ridge100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "ridge300": make_pipeline(StandardScaler(), Ridge(alpha=300.0)),
        "elastic003": make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=20000)),
        "hgb15": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.2, min_samples_leaf=12, random_state=seed),
        "hgb7": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.04, max_leaf_nodes=7, l2_regularization=0.4, min_samples_leaf=18, random_state=seed + 1),
    }


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - m["pcc"])
        + 0.05 * max(0.0, 0.80 - m["ccc"])
    )


def run_fold(fold: str, base: str, mert_dir: Path, seed: int, fast: bool) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(base_pred_path(fold, "train", base))
    val_df = pd.read_csv(base_pred_path(fold, "val", base))
    train_song = song_prediction_features(train_df)
    val_song = song_prediction_features(val_df)
    all_ids = sorted(set(train_song["musicId"].astype(int)).union(set(val_song["musicId"].astype(int))))
    mert = mert_song_features(all_ids, mert_dir)
    train = train_song.merge(mert, on="musicId", how="left").fillna(0.0)
    val = val_song.merge(mert, on="musicId", how="left").fillna(0.0)
    pred_cols = [c for c in train_song.columns if c not in {"musicId", "target_mean", "residual_mean"}]
    mert_cols = [c for c in mert.columns if c != "musicId"]
    rows = []
    best = None
    best_df = None
    top_k_choices = [0, 16, 64] if fast else [0, 8, 16, 32, 64, 128]
    model_items = list(models(seed).items())
    if fast:
        model_items = [(name, model) for name, model in model_items if name in {"ridge30", "ridge100", "elastic003"}]
    strength_choices = [0.05, 0.10, 0.20, 0.45, 0.80] if fast else [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00]
    cap_choices = [0.02, 0.05, None] if fast else [0.01, 0.02, 0.03, 0.05, 0.08, None]
    for target_mode, y_col in [("target_mean", "target_mean"), ("residual_mean", "residual_mean")]:
        for top_k in top_k_choices:
            selected = select_cols(train, mert_cols, y_col, top_k)
            cols = pred_cols + selected
            for model_id, model in model_items:
                model.fit(train[cols].to_numpy(dtype=np.float64), train[y_col].to_numpy(dtype=np.float64))
                train_pred = model.predict(train[cols].to_numpy(dtype=np.float64))
                val_pred = model.predict(val[cols].to_numpy(dtype=np.float64))
                if target_mode == "residual_mean":
                    train_target_mean_pred = train["pred_mean"].to_numpy(dtype=np.float64) + train_pred
                    val_target_mean_pred = val["pred_mean"].to_numpy(dtype=np.float64) + val_pred
                else:
                    train_target_mean_pred = train_pred
                    val_target_mean_pred = val_pred
                val_target_map = dict(zip(val["musicId"].astype(int).tolist(), val_target_mean_pred.tolist()))
                for strength in strength_choices:
                    for cap in cap_choices:
                        corrected = apply_song_mean_shift(val_df, val_target_map, strength, cap)
                        m = eval_df(corrected)
                        row = {
                            "fold": fold,
                            "base": base,
                            "target_mode": target_mode,
                            "top_k": top_k,
                            "model_id": model_id,
                            "strength": strength,
                            "cap": "none" if cap is None else cap,
                            "selected_mert_cols": len(selected),
                            **m,
                        }
                        row["score"] = score(m)
                        rows.append(row)
                        if best is None or (row["score"], row["rmse"], -row["ccc"]) < (best["score"], best["rmse"], -best["ccc"]):
                            best = row
                            best_df = corrected
    assert best is not None and best_df is not None
    return best, pd.DataFrame(rows), best_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = []
    frames = []
    for fold in ["fold_01", "fold_02", "fold_03"]:
        base_choices = ["bestlike"]
        if fold == "fold_03" and base_pred_path(fold, "val", "conservative").exists():
            base_choices.append("conservative")
        fold_rows = []
        fold_best = None
        fold_best_df = None
        for base in base_choices:
            best, rows, corrected = run_fold(fold, base, args.mert_dir, args.seed, args.fast)
            fold_rows.append(rows)
            if fold_best is None or (best["score"], best["rmse"], -best["ccc"]) < (fold_best["score"], fold_best["rmse"], -fold_best["ccc"]):
                fold_best = best
                fold_best_df = corrected
        assert fold_best is not None and fold_best_df is not None
        pd.concat(fold_rows, ignore_index=True).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(args.out_dir / f"{fold}_candidates.csv", index=False)
        fold_best_df.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold.append(fold_best)
        frames.append(fold_best_df)
    macro = {m: float(np.mean([row[m] for row in per_fold])) for m in ["rmse", "mae", "pcc", "ccc"]}
    merged = pd.concat(frames, ignore_index=True)
    global_frame = eval_df(merged)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Valence locked-dev song-level mean correction; final locked test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "per_fold": per_fold,
        "macro_mean": macro,
        "global_frame": global_frame,
        "gap_rows_macro": threshold_rows(macro),
        "gap_rows_global_frame": threshold_rows(global_frame),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
