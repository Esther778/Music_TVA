#!/usr/bin/env python3
"""Fast top-k OpenSMILE residual calibration for current best Valence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
OPENSMILE_SEQUENCE_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
OUT_DIR = ROOT / "outputs" / "valence_best_after_smoothing_opensmile_topk_residual_20260708"
MODEL_ID = "pmemo_valence_best_after_smoothing_opensmile_topk_residual_20260708"
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


@dataclass
class SplitArrays:
    df: pd.DataFrame
    target: np.ndarray
    base: np.ndarray
    meta: np.ndarray
    smile: np.ndarray


def eval_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(pred.astype(np.float64), target.astype(np.float64))


def score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def load_song_smile(music_id: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(OPENSMILE_SEQUENCE_DIR / f"{int(music_id)}.npz")
    return np.round(data["time_sec"].astype(np.float64), 3), data["X"].astype(np.float64)


def make_arrays(path: str) -> SplitArrays:
    df = pd.read_csv(path)
    keep = df.copy()
    target = keep["target_valence"].to_numpy(dtype=np.float64)
    base = keep["pred_valence"].to_numpy(dtype=np.float64)
    smile_chunks = []
    meta_chunks = []
    for music_id, idx in keep.groupby("musicId", sort=False).groups.items():
        idx_arr = np.asarray(idx)
        g = keep.loc[idx_arr]
        song_times, song_x = load_song_smile(int(music_id))
        lookup = {float(t): i for i, t in enumerate(song_times)}
        row_ids = [lookup[float(round(t, 3))] for t in g["time_sec"].to_numpy(dtype=np.float64)]
        smile_chunks.append(song_x[np.asarray(row_ids, dtype=np.int64)])

        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        time = g["time_sec"].to_numpy(dtype=np.float64)
        duration = max(float(time.max() - time.min() + 0.5), 0.5)
        pos = (time - time.min()) / duration
        meta_chunks.append(
            np.column_stack(
                [
                    pred,
                    pred**2,
                    np.abs(pred),
                    pos,
                    pos**2,
                    np.sin(2.0 * np.pi * pos),
                    np.cos(2.0 * np.pi * pos),
                    np.full_like(pred, np.mean(pred)),
                    np.full_like(pred, np.std(pred)),
                    np.full_like(pred, np.max(pred) - np.min(pred)),
                ]
            )
        )
    return SplitArrays(
        df=keep,
        target=target,
        base=base,
        meta=np.vstack(meta_chunks).astype(np.float64),
        smile=np.vstack(smile_chunks).astype(np.float64),
    )


def topk_indices(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    x_center = x - x.mean(axis=0, keepdims=True)
    y_center = y - y.mean()
    denom = np.sqrt(np.sum(x_center**2, axis=0) * np.sum(y_center**2)) + 1e-12
    corr = np.abs((x_center.T @ y_center) / denom)
    return np.argsort(corr)[-k:]


def build_x(arr: SplitArrays, idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return arr.meta
    return np.concatenate([arr.meta, arr.smile[:, idx]], axis=1)


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        ("ridge_a1000", make_pipeline(StandardScaler(), Ridge(alpha=1000.0))),
        ("ridge_a3000", make_pipeline(StandardScaler(), Ridge(alpha=3000.0))),
        ("elastic_a002", make_pipeline(StandardScaler(), ElasticNet(alpha=0.02, l1_ratio=0.15, max_iter=20000))),
        ("elastic_a005", make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.15, max_iter=20000))),
    ]


def corrected(base: np.ndarray, residual: np.ndarray, strength: float, cap: str) -> np.ndarray:
    corr = residual.copy()
    if cap != "none":
        c = float(cap)
        corr = np.clip(corr, -c, c)
    return np.clip(base + strength * corr, -1.0, 1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {
        (split_name, data_split): make_arrays(str(INPUT_DIR / split_name / f"{data_split}_predictions.csv"))
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            arr = arrays[(split_name, data_split)]
            row.update({f"{data_split}_{m}": v for m, v in eval_arrays(arr.base, arr.target).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_score = score(base_val)

    residual_cache: dict[tuple[int, str, str, str], np.ndarray] = {}
    topk_by_split: dict[tuple[int, str], np.ndarray] = {}
    for top_k in [0, 16, 32, 64, 96]:
        for split_name in SPLITS:
            train = arrays[(split_name, "train")]
            residual_target = train.target - train.base
            idx = topk_indices(train.smile, residual_target, top_k)
            topk_by_split[(top_k, split_name)] = idx
            x_train = build_x(train, idx)
            for model_id, model in model_candidates():
                model.fit(x_train, residual_target)
                for data_split in DATA_SPLITS:
                    residual_cache[(top_k, model_id, split_name, data_split)] = np.asarray(
                        model.predict(build_x(arrays[(split_name, data_split)], idx)),
                        dtype=np.float64,
                    )

    rows = []
    caps = ["0.003", "0.005", "0.008", "0.010", "0.015", "0.020", "none"]
    strengths = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    for top_k in [0, 16, 32, 64, 96]:
        for model_id, _ in model_candidates():
            for strength in strengths:
                for cap in caps:
                    per_split = []
                    for split_name in SPLITS:
                        row = {"split_name": split_name}
                        for data_split in DATA_SPLITS:
                            arr = arrays[(split_name, data_split)]
                            pred = corrected(arr.base, residual_cache[(top_k, model_id, split_name, data_split)], float(strength), cap)
                            row.update({f"{data_split}_{m}": v for m, v in eval_arrays(pred, arr.target).items()})
                        per_split.append(row)
                    val_m = macro(per_split, "val")
                    test_m = macro(per_split, "test")
                    val_score = score(val_m)
                    rows.append(
                        {
                            "top_k": int(top_k),
                            "model_id": model_id,
                            "strength": float(strength),
                            "cap": cap,
                            **{f"val_{m}": v for m, v in val_m.items()},
                            **{f"test_{m}": v for m, v in test_m.items()},
                            "val_score": val_score,
                            "passes_guard": (
                                val_score <= base_score - 0.00005
                                and val_m["rmse"] <= base_val["rmse"] + 0.0015
                                and val_m["mae"] <= base_val["mae"] + 0.0015
                                and val_m["pcc"] >= base_val["pcc"] - 0.0025
                                and val_m["ccc"] >= base_val["ccc"] - 0.0015
                            ),
                        }
                    )

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_opensmile_topk_residual_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "top_k": 0,
            "model_id": "identity",
            "strength": 0.0,
            "cap": "none",
            **{f"val_{m}": v for m, v in base_val.items()},
            "selection_reason": "identity retained; no candidate passed validation guard",
        }
        selected_key = None
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
        selected_key = (int(selected["top_k"]), str(selected["model_id"]), float(selected["strength"]), str(selected["cap"]))

    per_split = []
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            arr = arrays[(split_name, data_split)]
            out = arr.df.copy()
            if selected_key is None:
                pred = arr.base
            else:
                top_k, model_id, strength, cap = selected_key
                pred = corrected(arr.base, residual_cache[(top_k, model_id, split_name, data_split)], strength, cap)
                out["base_pred_valence"] = arr.base
                out["opensmile_topk_residual_correction"] = pred - arr.base
                out["opensmile_top_k"] = top_k
            out["pred_valence"] = pred
            out.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_arrays(pred, arr.target).items()})
        per_split.append(row)

    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": MODEL_ID,
        "input_model": CURRENT_BEST,
        "feature_source": "top-k PMEmo official dynamic OpenSMILE sequence features plus current prediction/time features",
        "selection_rule": "residual models fit on train only; top-k selected from train correlation; one global config selected by validation pass-gap guard",
        "selected_config": selected,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
