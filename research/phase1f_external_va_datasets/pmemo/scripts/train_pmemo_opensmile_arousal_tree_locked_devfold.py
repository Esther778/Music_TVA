#!/usr/bin/env python3
"""Train lightweight tree Arousal branches on locked development folds.

The inputs are still audio-derived OpenSMILE frame features. Time and first
difference features are computed from each song sequence and can be reproduced
for arbitrary full-song prediction once the same OpenSMILE extractor is used.
The final locked PMEmo test split is intentionally not read.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from train_pmemo_opensmile_arousal_trials import ROOT, metrics, threshold_rows


DEFAULT_DATA_ROOT = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "devfolds_3fold_20260713"
DEFAULT_OUT = ROOT / "outputs" / "arousal_locked_tree_branches_20260713"


def sequence_features(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    x = data["X"].astype(np.float32)
    y = data["Y"].astype(np.float32)[:, 1]
    time_sec = data["time_sec"].astype(np.float32)
    n = min(len(x), len(y), len(time_sec))
    x = x[:n]
    y = y[:n]
    time_sec = time_sec[:n]
    if n > 1:
        delta = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), np.diff(x, axis=0)])
        pos = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    else:
        delta = np.zeros_like(x, dtype=np.float32)
        pos = np.zeros((n, 1), dtype=np.float32)
    time_norm = ((time_sec - time_sec.min()) / max(float(time_sec.max() - time_sec.min()), 1e-6)).reshape(-1, 1)
    feat = np.concatenate([x, delta, pos, time_norm.astype(np.float32)], axis=1)
    return feat.astype(np.float32), y.astype(np.float32), time_sec.astype(np.float32), np.full(n, int(data["music_id"]), dtype=np.int32)


def load_split(index: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, times, songs = [], [], [], []
    for row in index[index["split"] == split].sort_values("musicId").to_dict("records"):
        x, y, time_sec, song = sequence_features(row["path"])
        xs.append(x)
        ys.append(y)
        times.append(time_sec)
        songs.append(song)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(times), np.concatenate(songs)


def candidate_models(seed: int) -> dict[str, Any]:
    return {
        "hgb_leaf15_lr003_l2_01": HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            min_samples_leaf=20,
            random_state=seed,
        ),
        "hgb_leaf31_lr002_l2_03": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.02,
            max_leaf_nodes=31,
            l2_regularization=0.30,
            min_samples_leaf=20,
            random_state=seed + 1,
        ),
        "hgb_leaf7_lr004_l2_05": HistGradientBoostingRegressor(
            max_iter=140,
            learning_rate=0.04,
            max_leaf_nodes=7,
            l2_regularization=0.50,
            min_samples_leaf=30,
            random_state=seed + 2,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_01", "fold_02", "fold_03"], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    fold_dir = args.data_root / args.fold
    out_dir = args.out_dir / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(fold_dir / "sequence_index.csv")

    x_train, y_train, _, _ = load_split(index, "train")
    x_val, y_val, time_val, song_val = load_split(index, "val")

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    best_model: Any | None = None
    for name, model in candidate_models(args.seed).items():
        model.fit(x_train, y_train)
        pred = np.clip(model.predict(x_val).astype(np.float64), -1.0, 1.0)
        m = metrics(pred, y_val.astype(np.float64))
        row = {"fold": args.fold, "model": name, **m}
        rows.append(row)
        if best is None or (row["rmse"], -row["ccc"]) < (best["rmse"], -best["ccc"]):
            best = row
            best_pred = pred
            best_model = model

    assert best is not None and best_pred is not None and best_model is not None
    pd.DataFrame(rows).sort_values(["rmse", "ccc"], ascending=[True, False]).to_csv(out_dir / "comparison.csv", index=False)
    pd.DataFrame(
        {
            "musicId": song_val,
            "time_sec": time_val,
            "target_arousal": y_val,
            "pred_arousal": best_pred,
        }
    ).to_csv(out_dir / "val_predictions.csv", index=False)
    with (out_dir / "best_model.pkl").open("wb") as handle:
        pickle.dump(best_model, handle)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Arousal tree branch on locked development fold; final test excluded",
        "feature_type": "PMEmo OpenSMILE frame features + frame deltas + normalized time position",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "best": best,
        "gap_rows": threshold_rows({k: float(best[k]) for k in ["rmse", "mae", "pcc", "ccc"]}),
        "candidates": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
