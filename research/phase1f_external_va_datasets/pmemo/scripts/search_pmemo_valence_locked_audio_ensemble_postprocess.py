#!/usr/bin/env python3
"""Valence audio-model ensemble/postprocess search on locked development folds.

This is a development-fold selection script. It does not read the final locked
test split. Candidate inputs are predictions from audio-only models or
audio-derived baselines, so the resulting direction remains compatible with
arbitrary full-song prediction once the same feature extractors are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_OUT = ROOT / "outputs" / "valence_locked_audio_ensemble_postprocess_20260713"


def pred_col(df: pd.DataFrame) -> str:
    for col in ["pred_valence", "pred_valence_raw", "pred", "prediction"]:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    for col in ["target_valence", "target", "valence"]:
        if col in df.columns:
            return col
    raise ValueError(f"No target column in {list(df.columns)}")


def candidate_paths(fold: str) -> dict[str, Path]:
    paths = {
        "bestlike": ROOT / "outputs" / "valence_locked_devfolds_bestlike_20260713" / fold / "v_dev_best_like_h80_do030_lr4e4" / "val_predictions.csv",
        "ridge": ROOT / "outputs" / "opensmile_ridge_locked_devfolds_20260713" / "valence" / fold / "val_predictions.csv",
    }
    mert_smile = ROOT / "outputs" / "valence_locked_mert_opensmile_fusion_20260713" / fold / "val_predictions.csv"
    if mert_smile.exists():
        paths["mert_smile"] = mert_smile
    if fold == "fold_01":
        fold01_screen = ROOT / "outputs" / "valence_locked_fold01_config_screen_20260713"
        for name, rel in {
            "fold01_wider": "v_dev_wider_h96_do035_lr25e4/val_predictions.csv",
            "fold01_lowccc": "v_dev_lowccc_h96_do030_lr3e4/val_predictions.csv",
            "fold01_conservative": "v_dev_conservative_h80_do035_lr25e4/val_predictions.csv",
        }.items():
            path = fold01_screen / rel
            if path.exists():
                paths[name] = path
        path = ROOT / "outputs" / "valence_locked_transformer_lstm_smile_20260713" / "fold_01" / "val_predictions.csv"
        if path.exists():
            paths["fold01_transformer_lstm_smile"] = path
        path = ROOT / "outputs" / "valence_locked_fold01_bestlike_seed707_20260713" / "v_dev_best_like_h80_do030_lr4e4" / "val_predictions.csv"
        if path.exists():
            paths["fold01_bestlike_seed707"] = path
    if fold == "fold_02":
        path = ROOT / "outputs" / "valence_locked_fold02_wider_20260713" / "v_dev_wider_h96_do035_lr25e4" / "val_predictions.csv"
        if path.exists():
            paths["fold02_wider"] = path
        path = ROOT / "outputs" / "valence_locked_transformer_lstm_smile_20260713" / "fold_02" / "val_predictions.csv"
        if path.exists():
            paths["fold02_transformer_lstm_smile"] = path
    if fold == "fold_03":
        paths.update(
            {
                "conservative": ROOT / "outputs" / "valence_locked_fold03_config_screen_20260713" / "conservative" / "v_dev_conservative_h80_do035_lr25e4" / "val_predictions.csv",
                "wider": ROOT / "outputs" / "valence_locked_fold03_config_screen_20260713" / "wider" / "v_dev_wider_h96_do035_lr25e4" / "val_predictions.csv",
                "lowccc": ROOT / "outputs" / "valence_locked_fold03_config_screen_20260713" / "lowccc" / "v_dev_lowccc_h96_do030_lr3e4" / "val_predictions.csv",
            }
        )
        path = ROOT / "outputs" / "valence_locked_fold03_wider_20260713" / "v_dev_wider_h96_do035_lr25e4" / "val_predictions.csv"
        if path.exists():
            paths["fold03_wider_new"] = path
        path = ROOT / "outputs" / "valence_locked_fold03_wider_seed909_20260713" / "v_dev_wider_h96_do035_lr25e4" / "val_predictions.csv"
        if path.exists():
            paths["fold03_wider_seed909"] = path
        path = ROOT / "outputs" / "valence_locked_fold03_bestlike_seed707_20260713" / "v_dev_best_like_h80_do030_lr4e4" / "val_predictions.csv"
        if path.exists():
            paths["fold03_bestlike_seed707"] = path
        path = ROOT / "outputs" / "valence_locked_fold03_hidden8_screen_20260713" / "v_dev_wider_h96_do035_lr25e4" / "val_predictions.csv"
        if path.exists():
            paths["fold03_hidden8"] = path
        path = ROOT / "outputs" / "valence_locked_transformer_smile_20260713" / "fold_03" / "val_predictions.csv"
        if path.exists():
            paths["fold03_transformer_smile"] = path
        path = ROOT / "outputs" / "valence_locked_transformer_lstm_smile_20260713" / "fold_03" / "val_predictions.csv"
        if path.exists():
            paths["fold03_transformer_lstm_smile"] = path
        path = ROOT / "outputs" / "valence_locked_transformer_lstm_smile_cap192_fold03_20260713" / "val_predictions.csv"
        if path.exists():
            paths["fold03_transformer_lstm_cap192"] = path
        path = (
            ROOT
            / "outputs"
            / "valence_locked_layer_gated_mert_20260713"
            / "h468_gate_l128_harm64_bilstm_h80_do035"
            / "split_03_seed_20260713"
            / "val_predictions.csv"
        )
        if path.exists():
            paths["fold03_layer_gate"] = path
    return {name: path for name, path in paths.items() if path.exists()}


def load_fold(fold: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = {name: pd.read_csv(path) for name, path in candidate_paths(fold).items()}
    if "bestlike" not in frames:
        raise FileNotFoundError(f"Missing bestlike predictions for {fold}")
    base_frame = frames["bestlike"]
    base = base_frame[["musicId", "time_sec", target_col(base_frame)]].rename(columns={target_col(base_frame): "target_valence"})
    for name, frame in frames.items():
        pcol = pred_col(frame)
        base = base.merge(
            frame[["musicId", "time_sec", pcol]].rename(columns={pcol: f"pred_{name}"}),
            on=["musicId", "time_sec"],
            how="inner",
        )
    names = list(frames.keys())
    return base[["musicId", "time_sec", "target_valence"]].copy(), {
        name: base[f"pred_{name}"].to_numpy(dtype=np.float64) for name in names
    }


def shift_by_song(df: pd.DataFrame, values: np.ndarray, frames: int) -> np.ndarray:
    if frames == 0:
        return values.copy()
    out = np.empty_like(values, dtype=np.float64)
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        ii = np.fromiter(idx, dtype=np.int64)
        seg = values[ii]
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


def smooth_by_song(df: pd.DataFrame, values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.empty_like(values, dtype=np.float64)
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        ii = np.fromiter(idx, dtype=np.int64)
        seg = values[ii]
        k = min(window, len(seg))
        if k <= 1:
            out[ii] = seg
        else:
            kernel = np.ones(k, dtype=np.float64) / float(k)
            out[ii] = np.convolve(seg, kernel, mode="same")
    return out


def scale_amplitude_by_song(df: pd.DataFrame, values: np.ndarray, amp: float) -> np.ndarray:
    if abs(amp - 1.0) < 1e-12:
        return values.copy()
    out = np.empty_like(values, dtype=np.float64)
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        ii = np.fromiter(idx, dtype=np.int64)
        seg = values[ii]
        out[ii] = seg.mean() + amp * (seg - seg.mean())
    return out


def calibrate(pred: np.ndarray, target: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, float | str]]:
    if method == "identity":
        return pred.copy(), {"method": method}
    if method == "bias":
        b = float(np.mean(target - pred))
        return pred + b, {"method": method, "b": b}
    if method == "linear":
        var = float(np.var(pred))
        if var < 1e-12:
            return pred.copy(), {"method": method, "a": 1.0, "b": 0.0}
        a = float(np.mean((pred - pred.mean()) * (target - target.mean())) / var)
        b = float(target.mean() - a * pred.mean())
        return a * pred + b, {"method": method, "a": a, "b": b}
    if method == "mean_std":
        pp = (pred - pred.mean()) / (pred.std() + 1e-12) * target.std() + target.mean()
        return pp, {"method": method}
    raise ValueError(method)


def weight_candidates(names: list[str], fold: str, samples: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for name in names:
        rows.append({n: float(n == name) for n in names})
    if "bestlike" in names and "ridge" in names:
        for w in [0.05, 0.10, 0.20, 0.35]:
            rows.append({n: (1.0 - w if n == "bestlike" else w if n == "ridge" else 0.0) for n in names})
    if fold == "fold_03" and "conservative" in names:
        for w in [0.25, 0.50, 0.75]:
            rows.append({n: (1.0 - w if n == "bestlike" else w if n == "conservative" else 0.0) for n in names})
    rng = np.random.default_rng(20260713)
    alpha = np.ones(len(names), dtype=np.float64)
    for idx, name in enumerate(names):
        if name == "bestlike":
            alpha[idx] = 5.0
        elif name in {"conservative", "mert_smile"}:
            alpha[idx] = 2.0
        else:
            alpha[idx] = 0.8
    for _ in range(samples):
        vec = rng.dirichlet(alpha)
        rows.append({name: float(value) for name, value in zip(names, vec)})
    return rows


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - m["pcc"])
        + 0.05 * max(0.0, 0.80 - m["ccc"])
    )


def search_fold(fold: str, samples: int, fast: bool) -> tuple[dict[str, float | str | int], pd.DataFrame, pd.DataFrame]:
    base, preds = load_fold(fold)
    names = list(preds.keys())
    target = base["target_valence"].to_numpy(dtype=np.float64)
    rows = []
    best = None
    best_pred = None
    shift_choices = range(-16, 17, 4) if fast else range(-16, 17, 2)
    amp_choices = [0.75, 1.00] if fast else [0.60, 0.75, 0.90, 1.00, 1.10]
    window_choices = [1, 21, 41] if fast else [1, 11, 21, 31, 41, 61]
    method_choices = ["linear", "mean_std"] if fast else ["identity", "bias", "linear", "mean_std"]
    for weights in weight_candidates(names, fold, samples):
        raw = sum(weights[name] * preds[name] for name in names)
        for shift_frames in shift_choices:
            shifted = shift_by_song(base, raw, shift_frames)
            for amp in amp_choices:
                amplified = scale_amplitude_by_song(base, shifted, amp)
                for window in window_choices:
                    smoothed = smooth_by_song(base, amplified, window)
                    for method in method_choices:
                        final, params = calibrate(smoothed, target, method)
                        final = np.clip(final, -1.0, 1.0)
                        m = metrics(final, target)
                        row = {
                            "fold": fold,
                            "shift_frames": shift_frames,
                            "shift_sec": shift_frames * 0.5,
                            "amp": amp,
                            "window": window,
                            "method": method,
                            "params": json.dumps(params, ensure_ascii=True),
                            **{f"w_{name}": weights[name] for name in names},
                            **m,
                        }
                        row["score"] = score(m)
                        rows.append(row)
                        key = (row["score"], row["rmse"], -row["ccc"])
                        if best is None or key < (best["score"], best["rmse"], -best["ccc"]):
                            best = row
                            best_pred = final
    assert best is not None and best_pred is not None
    pred_out = base.copy()
    pred_out["pred_valence"] = best_pred
    return best, pd.DataFrame(rows), pred_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--folds", nargs="*", default=["fold_01", "fold_02", "fold_03"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = []
    frames = []
    for fold in args.folds:
        best, rows, pred = search_fold(fold, args.samples, args.fast)
        rows.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(args.out_dir / f"{fold}_candidates.csv", index=False)
        pred.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold.append(best)
        frames.append(pred)
    macro = {m: float(np.mean([row[m] for row in per_fold])) for m in ["rmse", "mae", "pcc", "ccc"]}
    merged = pd.concat(frames, ignore_index=True)
    global_frame = metrics(
        merged["pred_valence"].to_numpy(dtype=np.float64),
        merged["target_valence"].to_numpy(dtype=np.float64),
    )
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Valence locked-dev audio-model ensemble and postprocess search; final locked test excluded",
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
