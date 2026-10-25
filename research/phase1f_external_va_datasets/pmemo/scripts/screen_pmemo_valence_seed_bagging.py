#!/usr/bin/env python3
"""Screen same-split multi-seed bagging for PMEmo Valence.

The experiment keeps the deployable MERT hidden_6 + harmonic BiLSTM family and
only changes random initialization. Bagging is suitable for arbitrary-song
prediction because inference only averages audio-model predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET
from train_pmemo_mert_valence_baseline import ROOT, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import (
    DEFAULT_MERT_DIR,
    BestConfig,
    train_one_split,
)


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_seed_bagging_screen_20260628"

REFERENCE_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
BASE_RAW_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"

CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}

SPLIT_REFERENCES = {
    1: {
        "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
        "split_id": 1,
        "test_rmse": 0.1815586949125916,
        "test_mae": 0.1419195070813537,
        "test_pcc": 0.8449925768337622,
        "test_ccc": 0.835335379286839,
    },
    2: {
        "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
        "split_id": 2,
        "test_rmse": 0.2093862469953028,
        "test_mae": 0.1624014423698784,
        "test_pcc": 0.7622912308938166,
        "test_ccc": 0.7429679473144445,
    },
    3: {
        "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
        "split_id": 3,
        "test_rmse": 0.1950039682105851,
        "test_mae": 0.1510577558657,
        "test_pcc": 0.7860129711134074,
        "test_ccc": 0.77619944804177,
    },
}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def read_prediction(path: Path, split: str, col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path / f"{split}_predictions.csv")
    return pd.DataFrame(
        {
            "musicId": df["musicId"].astype(str),
            "time_key": df["time_sec"].round(6),
            "time_sec": df["time_sec"],
            "target_valence": df[target_col(df)],
            col_name: df[pred_col(df)],
        }
    )


def merge_predictions(paths: list[Path], split: str) -> pd.DataFrame:
    merged = read_prediction(paths[0], split, "pred_0")
    for i, path in enumerate(paths[1:], start=1):
        cur = read_prediction(path, split, f"pred_{i}")
        merged = merged.merge(
            cur.loc[:, ["musicId", "time_key", f"pred_{i}"]],
            on=["musicId", "time_key"],
            how="inner",
        )
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged["pred_valence"] = merged[pred_cols].mean(axis=1)
    return merged.loc[:, ["musicId", "time_sec", "target_valence", "pred_valence", *pred_cols]]


def eval_frame(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def run_ensemble(paths: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = {}
    for split in ["train", "val", "test"]:
        df = merge_predictions(paths, split)
        df.to_csv(out_dir / f"{split}_predictions.csv", index=False)
        split_metrics[split] = eval_frame(df)
    return {
        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
        "test_gap_rows": threshold_rows(split_metrics["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-id", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[404, 505])
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = BestConfig()
    split_dir_in = BASE_SPLIT_DIR / f"split_{args.split_id:02d}_seed_{[101, 202, 303][args.split_id - 1]}"
    index = pd.read_csv(split_dir_in / "sequence_index.csv")

    trained_rows = []
    if args.train:
        for seed in args.seeds:
            trained_rows.append(train_one_split(args.split_id, seed, index, args, device, config))

    seed_paths = [BASE_RAW_DIR / f"split_{args.split_id:02d}_seed_{[101, 202, 303][args.split_id - 1]}"]
    seed_paths.extend(args.out_dir / f"split_{args.split_id:02d}_seed_{seed}" for seed in args.seeds)
    available_paths = [p for p in seed_paths if (p / "test_predictions.csv").exists()]

    ensembles = {}
    if len(available_paths) >= 2:
        ens_dir = args.out_dir / f"split_{args.split_id:02d}_ensemble_{len(available_paths)}seeds"
        ensembles["raw_equal_average"] = run_ensemble(available_paths, ens_dir)

    summary = {
        "created_at": "2026-06-28",
        "purpose": "screen same-architecture seed bagging for deployable Valence prediction",
        "universal_prediction_note": "all ensemble members are audio-only Valence predictors; arbitrary-song inference averages their frame predictions",
        "current_best_reference": CURRENT_BEST,
        "split_reference": SPLIT_REFERENCES.get(args.split_id),
        "split_id": args.split_id,
        "trained_seeds": args.seeds if args.train else [],
        "available_prediction_dirs": [str(p) for p in available_paths],
        "trained_rows": trained_rows,
        "ensembles": ensembles,
    }
    (args.out_dir / "screen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
