#!/usr/bin/env python3
"""Cross-fold Discogs-EffNet song-polarity correction for locked Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_musicnn_dynamic_tag_past_dense_monotonic_20260714"
DEFAULT_FEATURES = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_discogs_effnet_features_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_discogs_effnet_refine_20260714"
BLOCK_DIM = 1280


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def load_song_table(frame: pd.DataFrame, feature_dir: Path) -> pd.DataFrame:
    table = frame.groupby("musicId", sort=False).agg(
        target_song=("target_valence", "mean"),
        base_mean=("pred_valence", "mean"),
    ).reset_index()
    table["features"] = [
        np.load(feature_dir / "per_song" / f"{int(music_id)}.npz")["pooled"]
        for music_id in table["musicId"]
    ]
    return table


def feature_matrix(table: pd.DataFrame, feature_set: str) -> np.ndarray:
    matrix = np.stack(table["features"].to_list()).astype(np.float64)
    if matrix.shape[1] != 4 * BLOCK_DIM:
        raise ValueError(f"Expected {4 * BLOCK_DIM} pooled features, got {matrix.shape[1]}")
    if feature_set == "mean":
        return matrix[:, :BLOCK_DIM]
    if feature_set == "mean_std":
        return matrix[:, : 2 * BLOCK_DIM]
    if feature_set == "mean_quantiles":
        return np.concatenate([matrix[:, :BLOCK_DIM], matrix[:, 2 * BLOCK_DIM :]], axis=1)
    if feature_set == "all":
        return matrix
    raise ValueError(feature_set)


def transformed(
    train: np.ndarray, test: np.ndarray, pca_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    pca = PCA(n_components=pca_dim, whiten=False, svd_solver="randomized", random_state=0)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-sets", nargs="+", default=["mean", "mean_std", "mean_quantiles", "all"])
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--caps", nargs="+", type=float, default=[0.005, 0.0075, 0.010, 0.0125, 0.015])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    songs = {fold: load_song_table(frames[fold], args.feature_dir) for fold in FOLDS}
    matrices = {
        feature_set: {fold: feature_matrix(songs[fold], feature_set) for fold in FOLDS}
        for feature_set in args.feature_sets
    }
    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_rows)

    candidates: list[dict[str, Any]] = []
    raw_cache: dict[tuple[str, str, int, float], dict[str, np.ndarray]] = {}
    for feature_set in args.feature_sets:
        for pca_dim in args.pca_dims:
            transformed_folds: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
            for held in FOLDS:
                train_folds = [fold for fold in FOLDS if fold != held]
                train_songs = pd.concat([songs[fold] for fold in train_folds], ignore_index=True)
                x_train = np.concatenate([matrices[feature_set][fold] for fold in train_folds])
                x_test = matrices[feature_set][held]
                x_train_pca, x_test_pca = transformed(x_train, x_test, pca_dim)
                transformed_folds[held] = (x_train_pca, x_test_pca, train_songs)
            for target_mode in ["residual", "direct"]:
                for alpha in args.alphas:
                    shifts: dict[str, np.ndarray] = {}
                    for held in FOLDS:
                        x_train, x_test, train_songs = transformed_folds[held]
                        target = train_songs["target_song"].to_numpy(dtype=np.float64)
                        if target_mode == "residual":
                            target -= train_songs["base_mean"].to_numpy(dtype=np.float64)
                        model = Ridge(alpha=alpha).fit(x_train, target)
                        estimate = model.predict(x_test)
                        if target_mode == "direct":
                            estimate -= songs[held]["base_mean"].to_numpy(dtype=np.float64)
                        shifts[held] = estimate
                    raw_cache[(feature_set, target_mode, pca_dim, alpha)] = shifts
                    for strength in args.strengths:
                        for cap in args.caps:
                            per_fold = []
                            for fold in FOLDS:
                                shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                                shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
                                base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                                pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                                per_fold.append(metrics(pred, frames[fold]["target_valence"]))
                            result = macro(per_fold)
                            guarded = (
                                result["mae"] <= base_macro["mae"] + 0.001
                                and result["pcc"] >= base_macro["pcc"] - 0.001
                                and result["ccc"] >= base_macro["ccc"] - 0.001
                            )
                            candidates.append({
                                "feature_set": feature_set,
                                "target_mode": target_mode,
                                "pca_dim": pca_dim,
                                "alpha": alpha,
                                "strength": strength,
                                "cap": cap,
                                **result,
                                "guarded": guarded,
                            })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    key = (
        str(selected["feature_set"]), str(selected["target_mode"]),
        int(selected["pca_dim"]), float(selected["alpha"]),
    )
    per_fold_rows = []
    for fold in FOLDS:
        shift_map = dict(zip(songs[fold]["musicId"].astype(int), raw_cache[key][fold]))
        shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
        out = frames[fold].copy()
        out["effnet_input_pred_valence"] = out["pred_valence"]
        out["effnet_song_shift"] = shift
        out["pred_valence"] = np.clip(
            out["effnet_input_pred_valence"].to_numpy(dtype=np.float64)
            + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])),
            -1.0, 1.0,
        )
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})

    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "model_id": "pmemo_valence_locked_discogs_effnet_refine_20260714",
        "selection_boundary": "Three song-level development folds; final locked test excluded",
        "base_dir": str(args.base_dir),
        "feature_dir": str(args.feature_dir),
        "base_macro": base_macro,
        "selected": selected,
        "macro": selected_macro,
        "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
