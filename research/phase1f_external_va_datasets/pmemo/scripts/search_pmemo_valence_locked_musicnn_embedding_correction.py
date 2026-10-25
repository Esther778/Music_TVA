#!/usr/bin/env python3
"""Cross-fold MusiCNN embedding correction for locked-development Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs" / "valence_locked_musicnn_mood_correction_20260714"
DEFAULT_FEATURES = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_musicnn_song_embeddings_20260714"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_musicnn_embedding_correction_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def load_song_table(frame: pd.DataFrame, fold: str, feature_dir: Path) -> pd.DataFrame:
    table = frame.groupby("musicId", sort=False).agg(
        target_song=("target_valence", "mean"), base_mean=("pred_valence", "mean"),
        base_std=("pred_valence", "std"), base_min=("pred_valence", "min"),
        base_max=("pred_valence", "max"), frames=("pred_valence", "size"),
    ).reset_index()
    table["base_std"] = table["base_std"].fillna(0.0)
    table["base_range"] = table["base_max"] - table["base_min"]
    table["fold"] = fold
    table["features"] = [np.load(feature_dir / "per_song" / f"{int(mid)}.npz")["pooled"] for mid in table["musicId"]]
    return table


def feature_matrix(table: pd.DataFrame, feature_set: str) -> np.ndarray:
    matrix = np.stack(table["features"].to_list()).astype(np.float64)
    base = table[["base_mean", "base_std", "base_min", "base_max", "base_range", "frames"]].to_numpy(dtype=np.float64)
    if feature_set == "mean":
        return matrix[:, :200]
    if feature_set == "mean_std":
        return matrix[:, :400]
    if feature_set == "all":
        return matrix
    if feature_set == "mean_base":
        return np.concatenate([matrix[:, :200], base], axis=1)
    if feature_set == "mean_std_base":
        return np.concatenate([matrix[:, :400], base], axis=1)
    raise ValueError(feature_set)


def build_model(alpha: float, pca_dim: int):
    steps: list[Any] = [StandardScaler()]
    if pca_dim > 0:
        steps.append(PCA(n_components=pca_dim, whiten=False, random_state=0))
    steps.append(Ridge(alpha=alpha))
    return make_pipeline(*steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-sets", nargs="+", default=["mean", "mean_std", "all", "mean_base", "mean_std_base"])
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[0, 16, 32, 64])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0])
    parser.add_argument("--caps", nargs="+", type=float, default=[0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080, 0.120])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    songs = {fold: load_song_table(frames[fold], fold, args.feature_dir) for fold in FOLDS}
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)

    candidates: list[dict[str, Any]] = []
    raw_cache: dict[tuple[str, str, int, float], dict[str, np.ndarray]] = {}
    for feature_set in args.feature_sets:
        for target_mode in ["residual", "direct"]:
            for pca_dim in args.pca_dims:
                for alpha in args.alphas:
                    shifts: dict[str, np.ndarray] = {}
                    for held in FOLDS:
                        train = pd.concat([songs[f] for f in FOLDS if f != held], ignore_index=True)
                        test = songs[held]
                        model = build_model(alpha, pca_dim)
                        target = train["target_song"].to_numpy(dtype=np.float64)
                        if target_mode == "residual":
                            target = target - train["base_mean"].to_numpy(dtype=np.float64)
                        model.fit(feature_matrix(train, feature_set), target)
                        estimate = model.predict(feature_matrix(test, feature_set))
                        if target_mode == "direct":
                            estimate = estimate - test["base_mean"].to_numpy(dtype=np.float64)
                        shifts[held] = estimate
                    raw_cache[(feature_set, target_mode, pca_dim, alpha)] = shifts
                    for strength in args.strengths:
                        for cap in args.caps:
                            per_fold = []
                            for fold in FOLDS:
                                song_shift = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                                shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy(dtype=np.float64)
                                base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                                pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                                per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                            result = macro(per_fold)
                            guarded = (
                                result["mae"] <= base_macro["mae"] + 0.001
                                and result["pcc"] >= base_macro["pcc"] - 0.001
                                and result["ccc"] >= base_macro["ccc"] - 0.001
                            )
                            candidates.append({
                                "feature_set": feature_set, "target_mode": target_mode,
                                "pca_dim": pca_dim, "alpha": alpha, "strength": strength,
                                "cap": cap, **result, "guarded": guarded,
                            })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    key = (str(selected["feature_set"]), str(selected["target_mode"]), int(selected["pca_dim"]), float(selected["alpha"]))
    shifts = raw_cache[key]
    per_fold_rows = []
    for fold in FOLDS:
        song_shift = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
        shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy(dtype=np.float64)
        out = frames[fold].copy()
        input_pred = out["pred_valence"].to_numpy(dtype=np.float64).copy()
        out["embedding_input_pred_valence"] = input_pred
        out["musicnn_embedding_song_shift"] = shift
        out["pred_valence"] = np.clip(
            input_pred
            + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])),
            -1.0, 1.0,
        )
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Cross-devfold song-level fit; final locked test excluded",
        "base_dir": str(args.base_dir), "feature_dir": str(args.feature_dir),
        "base_macro": base_macro, "selected": selected, "macro": selected_macro,
        "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
