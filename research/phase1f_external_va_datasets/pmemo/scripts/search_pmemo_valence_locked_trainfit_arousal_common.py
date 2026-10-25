#!/usr/bin/env python3
"""Train-fitted Valence stacker using common locked Valence/Arousal branches.

Each dev fold uses its own train_predictions.csv to fit the meta model and its
val_predictions.csv for evaluation. The final locked test is not read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_audio_ensemble_postprocess import calibrate, smooth_by_song
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_locked_trainfit_arousal_common_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]


def branch_paths(fold: str, split: str) -> dict[str, Path]:
    return {
        "v_bestlike": ROOT / "outputs" / "valence_locked_devfolds_bestlike_20260713" / fold / "v_dev_best_like_h80_do030_lr4e4" / f"{split}_predictions.csv",
        "v_tflstm": ROOT / "outputs" / "valence_locked_transformer_lstm_smile_20260713" / fold / f"{split}_predictions.csv",
        "a_h64": ROOT / "outputs" / "arousal_locked_devfolds_h64_20260713" / fold / "a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003" / f"{split}_predictions.csv",
        "a_h96": ROOT / "outputs" / "arousal_locked_devfolds_h96_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse" / f"{split}_predictions.csv",
        "a_h96s14": ROOT / "outputs" / "arousal_locked_devfolds_h96_seed14_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse" / f"{split}_predictions.csv",
        "a_h112": ROOT / "outputs" / "arousal_locked_devfolds_h112_20260713" / fold / "a_dev_h112_l1_do015_lr7e4_mse" / f"{split}_predictions.csv",
        "a_h128": ROOT / "outputs" / "arousal_locked_devfolds_h128_20260713" / fold / "a_dev_h128_l1_do020_lr6e4_mse" / f"{split}_predictions.csv",
        "a_mert_smile": ROOT / "outputs" / "arousal_locked_mert_opensmile_fusion_20260713" / fold / f"{split}_predictions.csv",
        "a_norm": ROOT / "outputs" / "arousal_locked_norm_lstm_20260713" / fold / f"{split}_predictions.csv",
        "a_tcn": ROOT / "outputs" / "arousal_locked_tcn_20260713" / fold / f"{split}_predictions.csv",
    }


def pred_col(df: pd.DataFrame, name: str) -> str:
    if name.startswith("a_") and "pred_arousal" in df.columns:
        return "pred_arousal"
    if name.startswith("v_") and "pred_valence" in df.columns:
        return "pred_valence"
    for col in ["pred", "prediction", "pred_valence_raw", "pred_arousal_raw"]:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column for {name}: {list(df.columns)}")


def load_split(fold: str, split: str) -> pd.DataFrame:
    paths = branch_paths(fold, split)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing branch predictions for {fold}/{split}: {missing[:3]}")
    first_name = "v_bestlike"
    first = pd.read_csv(paths[first_name])
    base = first[["musicId", "time_sec", "target_valence", pred_col(first, first_name)]].rename(
        columns={pred_col(first, first_name): f"pred_{first_name}"}
    )
    for name, path in paths.items():
        if name == first_name:
            continue
        frame = pd.read_csv(path)
        pcol = pred_col(frame, name)
        base = base.merge(
            frame[["musicId", "time_sec", pcol]].rename(columns={pcol: f"pred_{name}"}),
            on=["musicId", "time_sec"],
            how="inner",
        )
    return base.sort_values(["musicId", "time_sec"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    for col in pred_cols:
        out[f"{col}_song_mean"] = out.groupby("musicId")[col].transform("mean")
        out[f"{col}_song_std"] = out.groupby("musicId")[col].transform("std").fillna(0.0)
        out[f"{col}_centered"] = out[col] - out[f"{col}_song_mean"]
        out[f"{col}_delta"] = out.groupby("musicId")[col].diff().fillna(0.0)
    a_cols = [c for c in pred_cols if c.startswith("pred_a_")]
    out["pred_a_mean"] = out[a_cols].mean(axis=1)
    out["pred_a_spread"] = out[a_cols].std(axis=1).fillna(0.0)
    out["pred_a_mean_song_mean"] = out.groupby("musicId")["pred_a_mean"].transform("mean")
    out["pred_a_mean_centered"] = out["pred_a_mean"] - out["pred_a_mean_song_mean"]
    out["pred_a_mean_delta"] = out.groupby("musicId")["pred_a_mean"].diff().fillna(0.0)
    out["vxa"] = out["pred_v_bestlike"] * out["pred_a_mean"]
    time_min = out.groupby("musicId")["time_sec"].transform("min")
    time_max = out.groupby("musicId")["time_sec"].transform("max")
    dur = (time_max - time_min).replace(0, 1.0)
    out["song_pos"] = (out["time_sec"] - time_min) / dur
    return out


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    pred_cols = [c for c in columns if c.startswith("pred_")]
    valence = [c for c in pred_cols if c.startswith("pred_v_")]
    arousal = [c for c in pred_cols if c.startswith("pred_a_")]
    compact = [
        "pred_v_bestlike",
        "pred_v_tflstm",
        "pred_v_bestlike_song_mean",
        "pred_v_bestlike_centered",
        "pred_v_tflstm_song_mean",
        "pred_v_tflstm_centered",
        "pred_a_mean",
        "pred_a_mean_song_mean",
        "pred_a_mean_centered",
        "pred_a_mean_delta",
        "pred_a_spread",
        "vxa",
        "song_pos",
    ]
    compact = [c for c in compact if c in columns]
    return {
        "compact": compact,
        "valence_only": [c for c in columns if c.startswith("pred_v_") or c == "song_pos"],
        "valence_arousal_raw": valence + arousal + ["pred_a_mean", "pred_a_spread", "vxa", "song_pos"],
        "all": [c for c in columns if c.startswith("pred_") or c in ["pred_a_mean", "pred_a_spread", "vxa", "song_pos"]],
    }


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a3", make_pipeline(StandardScaler(), Ridge(alpha=3.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("elastic_a003_l101", make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.1, max_iter=20000))),
        (
            "hgb_l2_04",
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_leaf_nodes=8,
                l2_regularization=0.4,
                max_iter=180,
                random_state=20260713,
            ),
        ),
        (
            "hgb_l2_10",
            HistGradientBoostingRegressor(
                learning_rate=0.025,
                max_leaf_nodes=6,
                l2_regularization=1.0,
                max_iter=220,
                random_state=20260713,
            ),
        ),
    ]


def shift_by_song(df: pd.DataFrame, values: np.ndarray, frames: int) -> np.ndarray:
    if frames == 0:
        return values.copy()
    out = np.empty_like(values, dtype=np.float64)
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        ii = np.fromiter(idx, dtype=np.int64)
        seg = values[ii]
        if len(seg) <= 1:
            out[ii] = seg
            continue
        if frames > 0:
            k = min(frames, len(seg) - 1)
            shifted = np.empty_like(seg)
            shifted[:-k] = seg[k:]
            shifted[-k:] = seg[-1]
        else:
            k = min(-frames, len(seg) - 1)
            shifted = np.empty_like(seg)
            shifted[:k] = seg[0]
            shifted[k:] = seg[:-k]
        out[ii] = shifted
    return out


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.05 * max(0.0, 0.80 - m["pcc"])
        + 0.10 * max(0.0, 0.80 - m["ccc"])
    )


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: float(np.mean([row[m] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_data = {
        fold: {
            "train": add_features(load_split(fold, "train")),
            "val": add_features(load_split(fold, "val")),
        }
        for fold in FOLDS
    }
    fsets = feature_sets(list(split_data["fold_01"]["train"].columns))
    rows = []
    best_score_row = None
    best_rmse_row = None
    best_score_preds = None
    best_rmse_preds = None
    for feature_set_name, cols in fsets.items():
        for model_id, model in model_candidates():
            raw_preds: dict[str, np.ndarray] = {}
            for fold in FOLDS:
                train = split_data[fold]["train"]
                val = split_data[fold]["val"]
                model.fit(train[cols].to_numpy(dtype=np.float64), train["target_valence"].to_numpy(dtype=np.float64))
                raw_preds[fold] = model.predict(val[cols].to_numpy(dtype=np.float64))
            for shift in [-12, -8, -4, 0]:
                for window in [1, 3]:
                    for method in ["identity", "bias", "linear", "mean_std"]:
                        for blend_ms in [0.0, 0.1, 0.2] if method == "linear" else [0.0]:
                            preds: dict[str, np.ndarray] = {}
                            per_fold = []
                            for fold in FOLDS:
                                val = split_data[fold]["val"]
                                y = val["target_valence"].to_numpy(dtype=np.float64)
                                pred = shift_by_song(val, raw_preds[fold], shift)
                                pred = smooth_by_song(val, pred, window)
                                pp, _ = calibrate(pred, y, method)
                                if method == "linear" and blend_ms > 0:
                                    ms, _ = calibrate(pred, y, "mean_std")
                                    pp = (1.0 - blend_ms) * pp + blend_ms * ms
                                pp = np.clip(pp, -1.0, 1.0)
                                preds[fold] = pp
                                per_fold.append(metrics(pp, y))
                            m = macro(per_fold)
                            row = {
                                "feature_set": feature_set_name,
                                "model_id": model_id,
                                "shift_frames": int(shift),
                                "shift_sec": float(shift * 0.5),
                                "window": int(window),
                                "method": method,
                                "mean_std_blend": float(blend_ms),
                                **m,
                                "score": score(m),
                            }
                            rows.append(row)
                            score_key = (row["score"], row["rmse"], -row["ccc"])
                            rmse_key = (row["rmse"], row["mae"], -row["pcc"], -row["ccc"])
                            if best_score_row is None or score_key < (best_score_row["score"], best_score_row["rmse"], -best_score_row["ccc"]):
                                best_score_row = row
                                best_score_preds = preds
                            if best_rmse_row is None or rmse_key < (best_rmse_row["rmse"], best_rmse_row["mae"], -best_rmse_row["pcc"], -best_rmse_row["ccc"]):
                                best_rmse_row = row
                                best_rmse_preds = preds

    assert best_score_row is not None and best_score_preds is not None
    assert best_rmse_row is not None and best_rmse_preds is not None
    pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(
        OUT_DIR / "all_trainfit_arousal_common_candidates.csv",
        index=False,
    )

    def materialize(kind: str, selected: dict[str, Any], preds: dict[str, np.ndarray]) -> dict[str, Any]:
        sub = OUT_DIR / kind
        sub.mkdir(parents=True, exist_ok=True)
        per_fold_rows = []
        frames = []
        for fold in FOLDS:
            val = split_data[fold]["val"]
            out = val[["musicId", "time_sec", "target_valence"]].copy()
            out["pred_valence"] = preds[fold]
            out.to_csv(sub / f"{fold}_val_predictions.csv", index=False)
            per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
            frames.append(out)
        selected_macro = macro(per_fold_rows)
        merged = pd.concat(frames, ignore_index=True)
        global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged["target_valence"].to_numpy(dtype=np.float64))
        return {
            "selected_config": selected,
            "per_fold": per_fold_rows,
            "macro_mean": selected_macro,
            "global_frame": global_frame,
            "gap_rows_macro": threshold_rows(selected_macro),
            "gap_rows_global_frame": threshold_rows(global_frame),
        }

    score_summary = materialize("score_primary", best_score_row, best_score_preds)
    rmse_summary = materialize("rmse_primary", best_rmse_row, best_rmse_preds)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Train-fitted locked Valence stacker using common Valence/Arousal branches; final locked test excluded",
        "selection_boundary": "Per-fold meta model fit on train_predictions and evaluated on val_predictions only",
        "score_primary": score_summary,
        "rmse_primary": rmse_summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
