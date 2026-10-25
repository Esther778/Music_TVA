#!/usr/bin/env python3
"""Valence tree baselines from audio-derived OpenSMILE + Librosa features.

The script uses only locked development folds and does not read the final
locked test split. Features are computable from arbitrary audio files once the
same extraction pipeline is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows
from train_pmemo_valence_dev_only_trials import postprocess_search, selection_score


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "devfolds_3fold_20260713" / "fold_03" / "sequence_index.csv"
DEFAULT_OPENSMILE = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset" / "sequence"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_tree_audio_features_20260713" / "fold_03"


def load_song_features(music_id: int, opensmile_dir: Path, librosa_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smile = np.load(opensmile_dir / f"{music_id}.npz")
    lib = np.load(librosa_dir / f"{music_id}.npz")
    x1 = smile["X"].astype(np.float32)
    x2 = lib["X"].astype(np.float32)
    y = smile["Y"].astype(np.float32)[:, 0]
    time_sec = smile["time_sec"].astype(np.float32)
    n = min(len(x1), len(x2), len(y), len(time_sec))
    x = np.concatenate([x1[:n], x2[:n]], axis=1)
    dx = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), np.diff(x, axis=0)])
    pos = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    x = np.concatenate([x, dx, pos], axis=1)
    return x, y[:n], time_sec[:n]


def load_split(index: pd.DataFrame, split: str, opensmile_dir: Path, librosa_dir: Path):
    xs, ys, songs, times = [], [], [], []
    for music_id in index.loc[index["split"] == split, "musicId"].astype(int):
        x, y, time_sec = load_song_features(music_id, opensmile_dir, librosa_dir)
        xs.append(x)
        ys.append(y)
        songs.append(np.full(len(y), music_id, dtype=np.int64))
        times.append(time_sec)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(songs), np.concatenate(times)


def build_model(model_name: str, seed: int):
    if model_name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=360,
            max_depth=24,
            min_samples_leaf=4,
            max_features=0.45,
            bootstrap=False,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=260,
            max_depth=22,
            min_samples_leaf=5,
            max_features=0.45,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "hgb":
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=450,
                learning_rate=0.035,
                max_leaf_nodes=31,
                l2_regularization=0.04,
                min_samples_leaf=24,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--opensmile-dir", type=Path, default=DEFAULT_OPENSMILE)
    parser.add_argument("--librosa-dir", type=Path, default=DEFAULT_LIBROSA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", choices=["extratrees", "rf", "hgb"], default="extratrees")
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    train_x, train_y, train_song, train_time = load_split(index, "train", args.opensmile_dir, args.librosa_dir)
    val_x, val_y, val_song, val_time = load_split(index, "val", args.opensmile_dir, args.librosa_dir)

    model = build_model(args.model, args.seed)
    model.fit(train_x, train_y)
    train_pred = np.clip(model.predict(train_x), -1.0, 1.0)
    val_pred = np.clip(model.predict(val_x), -1.0, 1.0)
    train_m = metrics(train_pred, train_y)
    val_m = metrics(val_pred, val_y)
    val_eval = {"pred": val_pred, "target": val_y, "song_id": val_song, "time_sec": val_time, "metrics": val_m}
    best_pp, pp_rows = postprocess_search(val_eval)
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)

    pd.DataFrame({"musicId": train_song, "time_sec": train_time, "target_valence": train_y, "pred_valence": train_pred}).to_csv(
        args.out_dir / "train_predictions.csv", index=False
    )
    pd.DataFrame({"musicId": val_song, "time_sec": val_time, "target_valence": val_y, "pred_valence": val_pred}).to_csv(
        args.out_dir / "val_predictions.csv", index=False
    )
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Audio-derived OpenSMILE+Librosa tree Valence baseline on locked development fold; final locked test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "model": args.model,
        "seed": args.seed,
        "feature_dim": int(train_x.shape[1]),
        "train_metrics": train_m,
        "val_metrics": val_m,
        "best_val_postprocess": best_pp,
        "gap_rows_raw": threshold_rows(val_m),
        "gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
        "selection_score_raw": selection_score(val_m),
        "selection_score_post": selection_score(best_pp["metrics"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
