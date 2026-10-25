#!/usr/bin/env python3
"""Train-label smoothing trials for PMEmo Valence on locked dev folds.

Only training targets are smoothed. Validation targets remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    FusionBiLSTM,
    FusionDataset,
    build_harmonic_norm,
    collate,
    evaluate,
    selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, resolve_device, threshold_rows
from train_pmemo_valence_dev_only_trials import postprocess_search, selection_score
from train_pmemo_valence_multisplit_best import build_mert_norm


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "devfolds_3fold_20260713" / "fold_03" / "sequence_index.csv"
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_label_smooth_trials_20260713"


@dataclass
class TrialConfig:
    trial_id: str
    smooth_window: int
    smooth_blend: float
    proj_dim: int = 384
    hidden_dim: int = 96
    dropout: float = 0.35
    learning_rate: float = 2.5e-4
    weight_decay: float = 4e-4
    batch_size: int = 18
    frame_ccc_weight: float = 0.04
    clip_mse_weight: float = 0.08
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.10
    patience: int = 10
    epochs: int = 80


TRIALS = [
    TrialConfig("smooth_w05_b050_h96", 5, 0.50),
    TrialConfig("smooth_w09_b050_h96", 9, 0.50),
    TrialConfig("smooth_w09_b075_h96", 9, 0.75),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) <= 2:
        return y.astype(np.float32)
    k = min(int(window), len(y))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(y.astype(np.float32), kernel, mode="same").astype(np.float32)


class TrainSmoothedFusionDataset(FusionDataset):
    def __init__(self, *args, smooth_window: int, smooth_blend: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.smooth_window = smooth_window
        self.smooth_blend = float(smooth_blend)

    def __getitem__(self, idx: int):
        x, y, song, time_sec = super().__getitem__(idx)
        y_np = y.numpy()
        smoothed = smooth_1d(y_np, self.smooth_window)
        y_mix = (1.0 - self.smooth_blend) * y_np + self.smooth_blend * smoothed
        return x, torch.from_numpy(y_mix.astype(np.float32)), song, time_sec


def build_loaders(index: pd.DataFrame, args: argparse.Namespace, trial_dir: Path, cfg: TrialConfig) -> dict[str, DataLoader]:
    build_mert_norm(index, args.mert_dir, trial_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), trial_dir / "dataset", force=True)
    mert_norm = np.load(trial_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(trial_dir / "dataset" / "harmonic_norm.npz")

    def make_dataset(split: str):
        cls = TrainSmoothedFusionDataset if split == "train" else FusionDataset
        kwargs: dict[str, Any] = {}
        if split == "train":
            kwargs = {"smooth_window": cfg.smooth_window, "smooth_blend": cfg.smooth_blend}
        return cls(
            index,
            split,
            args.mert_dir,
            args.label_dir,
            args.librosa_dataset,
            mert_norm["mean"],
            mert_norm["std"],
            harm_norm["mean"],
            harm_norm["std"],
            harm_norm["feature_idx"],
            **kwargs,
        )

    return {
        split: DataLoader(make_dataset(split), batch_size=cfg.batch_size, shuffle=(split == "train"), collate_fn=collate)
        for split in ["train", "val"]
    }


def run_trial(cfg: TrialConfig, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = args.out_dir / cfg.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(index, args, trial_dir, cfg)
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = FusionBiLSTM(int(sample_x.shape[1]), cfg.proj_dim, cfg.hidden_dim, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ckpt = trial_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _, mask in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, mask)
            pred_valid = pred[mask]
            y_valid = y[mask]
            loss = weighted_frame_mse(pred_valid, y_valid, cfg.extreme_weight)
            loss = loss + cfg.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, cfg.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, cfg.clip_std_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        score = selection_score(val_m)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"{cfg.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_mae={val_m['mae']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "seed": seed, "config": asdict(cfg), "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val"]}
    for split, data in evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(trial_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Valence train-label smoothing on final-locked development folds; validation labels unchanged; final locked test excluded",
        "trial_id": cfg.trial_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "config": asdict(cfg),
        **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()},
        "best_val_postprocess": best_pp,
        "val_gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "val_gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--trial", default="all")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    trials = TRIALS if args.trial == "all" else [trial for trial in TRIALS if trial.trial_id == args.trial]
    if not trials:
        raise ValueError(f"No trial matched {args.trial}")
    device = resolve_device(args.device)
    summaries = [run_trial(trial, index, args, device, args.seed) for trial in trials]
    rows = [
        {
            "trial_id": s["trial_id"],
            "best_epoch": s["best_epoch"],
            "val_rmse": s["val_rmse"],
            "val_mae": s["val_mae"],
            "val_pcc": s["val_pcc"],
            "val_ccc": s["val_ccc"],
            "post_val_rmse": s["best_val_postprocess"]["rmse"],
            "post_val_mae": s["best_val_postprocess"]["mae"],
            "post_val_pcc": s["best_val_postprocess"]["pcc"],
            "post_val_ccc": s["best_val_postprocess"]["ccc"],
        }
        for s in summaries
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
