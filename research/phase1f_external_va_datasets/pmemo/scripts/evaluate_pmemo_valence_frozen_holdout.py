#!/usr/bin/env python3
"""Evaluate the frozen development Valence candidate on holdout rows.

This is a recheck utility. The current PMEmo blind split has already been
inspected, so outputs from this script must not be described as fresh blind
results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from search_pmemo_valence_dev_val_postprocess import moving_average_by_song
from train_pmemo_mert_harmonic_valence_fusion import FusionBiLSTM, FusionDataset, collate, evaluate
from train_pmemo_mert_valence_baseline import ROOT, metrics, resolve_device, threshold_rows


DEFAULT_TRIAL_DIR = ROOT / "outputs" / "valence_dev_only_trials_20260709" / "v_dev_best_like_h80_do030_lr4e4"
DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_blind_20260709" / "pmemo_blind_trainvaltest_index.csv"
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "valence_frozen_holdout_recheck_20260709"


@dataclass
class PostConfig:
    window: int
    smooth_blend: float
    a: float
    b: float
    cal_blend: float


def load_post_config(summary: dict[str, Any]) -> PostConfig:
    pp = summary["best_val_postprocess"]
    params = json.loads(pp["params"])
    return PostConfig(
        window=int(pp["window"]),
        smooth_blend=float(pp["smooth_blend"]),
        a=float(params.get("a", 1.0)),
        b=float(params.get("b", 0.0)),
        cal_blend=float(pp["cal_blend"]),
    )


def apply_postprocess(pred: np.ndarray, song: np.ndarray, cfg: PostConfig) -> np.ndarray:
    smoothed = moving_average_by_song(pred, song, cfg.window)
    base = (1.0 - cfg.smooth_blend) * pred + cfg.smooth_blend * smoothed
    calibrated = cfg.a * base + cfg.b
    return (1.0 - cfg.cal_blend) * base + cfg.cal_blend * calibrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.trial_dir / "summary.json").read_text(encoding="utf-8"))
    ckpt = torch.load(args.trial_dir / "best.pt", map_location="cpu")
    post_cfg = load_post_config(summary)

    index = pd.read_csv(args.index)
    split_index = index[index["split"] == args.eval_split].copy()
    if split_index.empty:
        raise ValueError(f"No rows for split {args.eval_split!r}")

    mert_norm = np.load(args.trial_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(args.trial_dir / "dataset" / "harmonic_norm.npz")
    loader = DataLoader(
        FusionDataset(
            index,
            args.eval_split,
            args.mert_dir,
            args.label_dir,
            args.librosa_dataset,
            mert_norm["mean"],
            mert_norm["std"],
            harm_norm["mean"],
            harm_norm["std"],
            harm_norm["feature_idx"],
        ),
        batch_size=int(summary["config"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
    )

    device = resolve_device(args.device)
    model = FusionBiLSTM(
        int(ckpt["input_dim"]),
        int(summary["config"]["proj_dim"]),
        int(summary["config"]["hidden_dim"]),
        float(summary["config"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    raw_eval = evaluate(model, loader, device)
    post_pred = apply_postprocess(raw_eval["pred"], raw_eval["song_id"], post_cfg)
    post_m = metrics(post_pred, raw_eval["target"])

    pred_df = pd.DataFrame(
        {
            "musicId": raw_eval["song_id"],
            "time_sec": raw_eval["time_sec"],
            "target_valence": raw_eval["target"],
            "pred_valence_raw": raw_eval["pred"],
            "pred_valence_post": post_pred,
        }
    )
    pred_df.to_csv(args.out_dir / f"{args.eval_split}_predictions.csv", index=False)
    result = {
        "created_at": "2026-07-09",
        "purpose": "Frozen Valence holdout recheck; not fresh blind because this split has been inspected before",
        "trial_dir": str(args.trial_dir),
        "index": str(args.index),
        "eval_split": args.eval_split,
        "song_count": int(split_index["musicId"].nunique()),
        "frame_count": int(split_index["frames"].sum()),
        "raw_metrics": raw_eval["metrics"],
        "postprocess": post_cfg.__dict__,
        "post_metrics": post_m,
        "raw_gap_rows": threshold_rows(raw_eval["metrics"]),
        "post_gap_rows": threshold_rows(post_m),
        "validity_note": "Use as holdout recheck only, not as final blind result.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
