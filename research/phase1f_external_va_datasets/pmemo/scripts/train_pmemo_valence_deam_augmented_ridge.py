#!/usr/bin/env python3
"""Fast DEAM-augmented PMEmo Valence Ridge diagnostic.

This is a shallow audio-only diagnostic, not the final teacher architecture.
For each fixed PMEmo split, train on PMEmo train frames plus DEAM train frames,
select alpha and DEAM sample weight on PMEmo validation, and report PMEmo test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


PMEMO_ROOT = ROOT
RESEARCH_ROOT = ROOT.parents[1]
PMEMO_MERT_DIR = PMEMO_ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
PMEMO_LABEL_DIR = PMEMO_ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
PMEMO_SPLIT_DIR = PMEMO_ROOT / "outputs" / "valence_multisplit_best_20260627"
DEAM_SEQUENCE_DIR = RESEARCH_ROOT / "phase1b_bert_deam_va_training" / "data" / "datasets" / "mert_va" / "sequence"
OUT_DIR = PMEMO_ROOT / "outputs" / "valence_deam_augmented_ridge_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


def score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def macro(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{prefix}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def load_pmemo_song(music_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mert = np.load(PMEMO_MERT_DIR / f"{int(music_id)}.npz")
    lab = np.load(PMEMO_LABEL_DIR / f"{int(music_id)}.npz")
    x = mert["embedding"].astype(np.float32)
    y = lab["Y"].astype(np.float32)[:, 0]
    t = lab["time_sec"].astype(np.float32)
    n = min(len(x), len(y), len(t))
    return x[:n], y[:n], t[:n]


def load_pmemo_split(index: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, rows = [], [], []
    for row in index[index["split"] == split].to_dict("records"):
        sid = int(row["musicId"])
        x, y, t = load_pmemo_song(sid)
        xs.append(x)
        ys.append(y)
        rows.append(pd.DataFrame({"musicId": sid, "time_sec": t, "target_valence": y}))
    return np.vstack(xs), np.concatenate(ys), pd.concat(rows, ignore_index=True)


def load_deam_train() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for path in sorted(DEAM_SEQUENCE_DIR.glob("*.npz")):
        data = np.load(path)
        if str(data["split"]) != "train":
            continue
        xs.append(data["X"].astype(np.float32))
        ys.append(data["Y"].astype(np.float32)[:, 0])
    return np.vstack(xs), np.concatenate(ys)


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_deam: np.ndarray,
    y_deam: np.ndarray,
    data: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    alpha: float,
    deam_weight: float,
) -> dict[str, pd.DataFrame]:
    x_all = np.vstack([x_train, x_deam])
    y_all = np.concatenate([y_train, y_deam])
    weights = np.concatenate([
        np.ones(len(y_train), dtype=np.float32),
        np.full(len(y_deam), float(deam_weight), dtype=np.float32),
    ])
    mean = x_all.mean(axis=0, keepdims=True)
    std = x_all.std(axis=0, keepdims=True) + 1e-6
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=1000, tol=1e-3)
    model.fit((x_all - mean) / std, y_all, sample_weight=weights)
    out = {}
    for split, (x, _y, frame) in data.items():
        pred = model.predict((x - mean) / std)
        df = frame.copy()
        df["pred_valence"] = np.clip(pred.astype(np.float64), -1.0, 1.0)
        out[split] = df
    return out


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_deam, y_deam = load_deam_train()
    all_rows = []
    selected_rows = []
    selected_frames: dict[tuple[str, str], pd.DataFrame] = {}
    alphas = [100.0, 1000.0, 10000.0]
    deam_weights = [0.0, 0.10, 0.35, 0.75]
    for split_name in SPLITS:
        index = pd.read_csv(PMEMO_SPLIT_DIR / split_name / "sequence_index.csv")
        data = {split: load_pmemo_split(index, split) for split in DATA_SPLITS}
        x_train, y_train, _ = data["train"]
        cache: dict[tuple[float, float], dict[str, pd.DataFrame]] = {}
        for alpha in alphas:
            for deam_weight in deam_weights:
                pred_frames = fit_predict(x_train, y_train, x_deam, y_deam, data, alpha, deam_weight)
                cache[(alpha, deam_weight)] = pred_frames
                row = {"split_name": split_name, "alpha": alpha, "deam_weight": deam_weight}
                for data_split in DATA_SPLITS:
                    row.update({f"{data_split}_{m}": v for m, v in eval_df(pred_frames[data_split]).items()})
                row["val_score"] = score({m: row[f"val_{m}"] for m in ["rmse", "mae", "pcc", "ccc"]})
                all_rows.append(row)
        split_rows = [r for r in all_rows if r["split_name"] == split_name]
        selected = sorted(split_rows, key=lambda r: (r["val_score"], r["val_rmse"], -r["val_ccc"]))[0]
        selected_rows.append(selected)
        key = (float(selected["alpha"]), float(selected["deam_weight"]))
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            df = cache[key][data_split]
            df.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            selected_frames[(split_name, data_split)] = df

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "all_deam_augmented_ridge_candidates.csv", index=False)
    selected_test = macro(selected_rows, "test")
    selected_val = macro(selected_rows, "val")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_deam_augmented_ridge_20260708",
        "purpose": "fast diagnostic for whether DEAM train frames help PMEmo Valence generalization",
        "feature_type": "MERT hidden6 frame embeddings only",
        "selection_rule": "per PMEmo split, train on PMEmo train + DEAM train; select alpha/deam_weight on PMEmo validation; test held out",
        "deam_train_frames": int(len(y_deam)),
        "selected_per_split": selected_rows,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "current_best_reference": CURRENT_BEST,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
