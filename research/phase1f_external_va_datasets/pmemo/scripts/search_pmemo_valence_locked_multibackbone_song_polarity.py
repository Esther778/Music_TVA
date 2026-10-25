#!/usr/bin/env python3
"""Cross-fold multi-backbone song-polarity correction for Valence."""

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
DEFAULT_BASE = ROOT / "outputs/valence_locked_musicnn_pair_blend_calibrated_20260714"
DEFAULT_MUSICNN = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_musicnn_song_embeddings_20260714/per_song"
DEFAULT_MUQ = ROOT / "outputs/muq_pmemo_locked_development/layer_06/features"
DEFAULT_MERT = ROOT / "outputs/mert_pmemo_h468_concat/features"
DEFAULT_OUT = ROOT / "outputs/valence_locked_multibackbone_song_polarity_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def pooled(path: Path, key: str, mode: str) -> np.ndarray:
    values = np.load(path)[key].astype(np.float64)
    if values.ndim == 1:
        return values
    mean = values.mean(axis=0)
    if mode == "mean":
        return mean
    return np.concatenate([mean, values.std(axis=0)])


def load_songs(frame: pd.DataFrame, fold: str, musicnn: Path, muq: Path, mert: Path) -> pd.DataFrame:
    song = frame.groupby("musicId", sort=False).agg(
        target_song=("target_valence", "mean"), base_mean=("pred_valence", "mean"),
        base_std=("pred_valence", "std"), base_min=("pred_valence", "min"),
        base_max=("pred_valence", "max"), frames=("pred_valence", "size"),
    ).reset_index()
    song["base_std"] = song["base_std"].fillna(0.0)
    song["base_range"] = song["base_max"] - song["base_min"]
    song["fold"] = fold
    song["musicnn"] = [pooled(musicnn / f"{int(sid)}.npz", "pooled", "mean")[:400] for sid in song["musicId"]]
    song["muq"] = [pooled(muq / f"{int(sid)}.npz", "embedding", "mean_std") for sid in song["musicId"]]
    song["mert"] = [pooled(mert / f"{int(sid)}.npz", "embedding", "mean") for sid in song["musicId"]]
    return song


def block_matrix(table: pd.DataFrame, block: str) -> np.ndarray:
    return np.stack(table[block].to_list()).astype(np.float64)


def reduce_block(train: pd.DataFrame, test: pd.DataFrame, block: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(block_matrix(train, block))
    test_scaled = scaler.transform(block_matrix(test, block))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=20260714)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--musicnn-dir", type=Path, default=DEFAULT_MUSICNN)
    parser.add_argument("--muq-dir", type=Path, default=DEFAULT_MUQ)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    songs = {fold: load_songs(frames[fold], fold, args.musicnn_dir, args.muq_dir, args.mert_dir) for fold in FOLDS}
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    combos = {
        "musicnn_muq": ["musicnn", "muq"],
        "musicnn_mert": ["musicnn", "mert"],
        "muq_mert": ["muq", "mert"],
        "all": ["musicnn", "muq", "mert"],
    }
    base_cols = ["base_mean", "base_std", "base_min", "base_max", "base_range", "frames"]
    transformed: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]] = {}
    for held in FOLDS:
        train = pd.concat([songs[f] for f in FOLDS if f != held], ignore_index=True)
        test = songs[held]
        for dim in [8, 16, 32]:
            for block in ["musicnn", "muq", "mert"]:
                x_train, x_test = reduce_block(train, test, block, dim)
                transformed[(held, dim, block)] = (x_train, x_test, train, test)

    candidates: list[dict[str, Any]] = []
    raw_cache: dict[tuple[str, int, str, float], dict[str, np.ndarray]] = {}
    for combo_name, blocks in combos.items():
        for dim in [8, 16, 32]:
            for target_mode in ["direct", "residual"]:
                for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                    shifts: dict[str, np.ndarray] = {}
                    for held in FOLDS:
                        train = transformed[(held, dim, blocks[0])][2]
                        test = transformed[(held, dim, blocks[0])][3]
                        train_parts = [transformed[(held, dim, block)][0] for block in blocks]
                        test_parts = [transformed[(held, dim, block)][1] for block in blocks]
                        train_parts.append(train[base_cols].to_numpy(dtype=np.float64))
                        test_parts.append(test[base_cols].to_numpy(dtype=np.float64))
                        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                        target = train["target_song"].to_numpy(dtype=np.float64)
                        if target_mode == "residual":
                            target = target - train["base_mean"].to_numpy(dtype=np.float64)
                        model.fit(np.concatenate(train_parts, axis=1), target)
                        estimate = model.predict(np.concatenate(test_parts, axis=1))
                        if target_mode == "direct":
                            estimate = estimate - test["base_mean"].to_numpy(dtype=np.float64)
                        shifts[held] = estimate
                    raw_cache[(combo_name, dim, target_mode, alpha)] = shifts
                    for strength in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
                        for cap in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060]:
                            per_fold = []
                            for fold in FOLDS:
                                song_shift = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                                shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy(dtype=np.float64)
                                base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                                pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                                per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                            result = macro(per_fold)
                            guarded = (
                                result["mae"] <= base_macro["mae"] + 0.0005
                                and result["pcc"] >= base_macro["pcc"] - 0.0005
                                and result["ccc"] >= base_macro["ccc"] - 0.0005
                            )
                            candidates.append({
                                "combo": combo_name, "pca_dim": dim, "target_mode": target_mode,
                                "alpha": alpha, "strength": strength, "cap": cap,
                                **result, "guarded": guarded,
                            })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    key = (str(selected["combo"]), int(selected["pca_dim"]), str(selected["target_mode"]), float(selected["alpha"]))
    shifts = raw_cache[key]
    per_fold_rows = []
    for fold in FOLDS:
        song_shift = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
        shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy(dtype=np.float64)
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_multibackbone_pred_valence"})
        out["multibackbone_song_shift"] = shift
        out["pred_valence"] = np.clip(
            out["pre_multibackbone_pred_valence"].to_numpy(dtype=np.float64)
            + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])), -1.0, 1.0,
        )
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Each held development fold uses PCA and Ridge fit only on the other two folds; final locked test excluded",
        "base_dir": str(args.base_dir), "base_macro": base_macro,
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
