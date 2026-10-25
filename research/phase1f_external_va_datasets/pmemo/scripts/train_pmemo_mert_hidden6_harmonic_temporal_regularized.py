#!/usr/bin/env python3
"""Train hidden_6 MERT + harmonic Valence models with temporal regularization."""

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
    DEFAULT_INDEX,
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    FusionBiLSTM,
    FusionDataset,
    build_harmonic_norm,
    collate,
    evaluate,
    selected_indices,
)
from train_pmemo_mert_valence_baseline import ccc_loss, resolve_device, threshold_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_HIDDEN6_NORM = ROOT / "outputs" / "mert_hidden6_valence_baseline" / "dataset" / "mert_norm.npz"
DEFAULT_OUT = ROOT / "outputs" / "mert_hidden6_harmonic_temporal_regularized"

HISTORICAL_BEST = {
    "model_id": "pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1",
    "trial_id": "h6_harm_h80_ext125_clip010_ccc005",
    "rmse": 0.205672949552536,
    "mae": 0.16456688940525055,
    "pcc": 0.7834003513415063,
    "ccc": 0.7673044285835935,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrialConfig:
    trial_id: str
    proj_dim: int = 320
    hidden_dim: int = 80
    dropout: float = 0.30
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.05
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.25
    pred_smooth_weight: float = 0.0
    delta_match_weight: float = 0.0
    patience: int = 10
    epochs: int = 80


TRIALS = [
    TrialConfig(
        "h6_harm_h80_ext125_clip010_ccc005_smooth002",
        pred_smooth_weight=0.002,
    ),
    TrialConfig(
        "h6_harm_h80_ext125_clip010_ccc005_delta020",
        delta_match_weight=0.020,
    ),
    TrialConfig(
        "h6_harm_h80_ext125_clip010_std003_delta020",
        clip_std_weight=0.030,
        delta_match_weight=0.020,
    ),
    TrialConfig(
        "h6_harm_h64_ext125_clip010_ccc005_do035_wd0005",
        proj_dim=256,
        hidden_dim=64,
        dropout=0.35,
        weight_decay=5e-4,
        batch_size=24,
    ),
    TrialConfig(
        "h6_harm_h80_ext125_clip010_ccc005_do040_wd0005",
        dropout=0.40,
        weight_decay=5e-4,
    ),
]


def clip_mean_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if not weight:
        return pred.new_tensor(0.0)
    mask_f = mask.float()
    lengths = mask_f.sum(dim=1).clamp_min(1.0)
    pred_mean = (pred * mask_f).sum(dim=1) / lengths
    target_mean = (target * mask_f).sum(dim=1) / lengths
    return float(weight) * ((pred_mean - target_mean) ** 2).mean()


def clip_std_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if not weight:
        return pred.new_tensor(0.0)
    mask_f = mask.float()
    lengths = mask_f.sum(dim=1).clamp_min(2.0)
    pred_mean = (pred * mask_f).sum(dim=1, keepdim=True) / lengths.unsqueeze(1)
    target_mean = (target * mask_f).sum(dim=1, keepdim=True) / lengths.unsqueeze(1)
    pred_var = (((pred - pred_mean) * mask_f) ** 2).sum(dim=1) / lengths
    target_var = (((target - target_mean) * mask_f) ** 2).sum(dim=1) / lengths
    return float(weight) * ((torch.sqrt(pred_var + 1e-8) - torch.sqrt(target_var + 1e-8)) ** 2).mean()


def weighted_frame_mse(pred: torch.Tensor, target: torch.Tensor, extreme_weight: float) -> torch.Tensor:
    weights = 1.0 + float(extreme_weight) * target.abs()
    return (weights * (pred - target) ** 2).mean()


def temporal_smooth_loss(pred: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if not weight:
        return pred.new_tensor(0.0)
    delta_mask = mask[:, 1:] & mask[:, :-1]
    if not bool(delta_mask.any()):
        return pred.new_tensor(0.0)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    return float(weight) * (pred_delta[delta_mask] ** 2).mean()


def temporal_delta_match_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if not weight:
        return pred.new_tensor(0.0)
    delta_mask = mask[:, 1:] & mask[:, :-1]
    if not bool(delta_mask.any()):
        return pred.new_tensor(0.0)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return float(weight) * ((pred_delta[delta_mask] - target_delta[delta_mask]) ** 2).mean()


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["rmse"]
        + 0.05 * max(0.0, 0.80 - metrics["pcc"])
        + 0.05 * max(0.0, 0.80 - metrics["ccc"])
        + 0.01 * metrics["mae"]
    )


def run_trial(config: TrialConfig, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = args.out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    mert_norm = np.load(args.mert_norm)
    harm_norm = np.load(args.out_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            FusionDataset(
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
            ),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = trial_dir / "best.pt"
    history = []
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, config.epochs + 1):
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
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + float(config.frame_ccc_weight) * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            loss = loss + temporal_smooth_loss(pred, mask, config.pred_smooth_weight)
            loss = loss + temporal_delta_match_loss(pred, y, mask, config.delta_match_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        val_score = selection_score(val_m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}, "val_select_score": val_score}
        history.append(row)
        print(
            f"{config.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_mae={val_m['mae']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
            flush=True,
        )
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "seed": seed, "trial_config": asdict(config), "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val", "test"]}
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    test_m = split_evals["test"]["metrics"]
    summary = {
        "trial_id": config.trial_id,
        "seed": seed,
        "feature_type": "pmemo_mert_hidden6_768_plus_librosa_harmonic_36",
        "label_scale": "minus1_1",
        "target": "valence",
        "best_epoch": best_epoch,
        "selection_rule": "validation selection_score = rmse + pcc/ccc pass penalties + 0.01*mae",
        "checkpoint": str(ckpt),
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "delta_vs_historical_best": {
            "rmse": float(test_m["rmse"] - HISTORICAL_BEST["rmse"]),
            "mae": float(test_m["mae"] - HISTORICAL_BEST["mae"]),
            "pcc": float(test_m["pcc"] - HISTORICAL_BEST["pcc"]),
            "ccc": float(test_m["ccc"] - HISTORICAL_BEST["ccc"]),
        },
        "test_gap_rows": threshold_rows(test_m),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--mert-norm", type=Path, default=DEFAULT_HIDDEN6_NORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-norm", action="store_true")
    parser.add_argument("--trial-ids", nargs="*", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    dataset_summary = build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), args.out_dir / "dataset", args.force_norm)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    device = resolve_device(args.device)
    trials = TRIALS
    if args.trial_ids:
        wanted = set(args.trial_ids)
        trials = [trial for trial in trials if trial.trial_id in wanted]
        missing = sorted(wanted - {trial.trial_id for trial in trials})
        if missing:
            raise ValueError(f"Unknown trial ids: {missing}")
    summaries = [run_trial(trial, index, args, device, args.seed) for trial in trials]
    all_summaries = []
    for path in sorted(args.out_dir.glob("*/summary.json")):
        all_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rows = [
        {
            "trial_id": s["trial_id"],
            "best_epoch": s["best_epoch"],
            "val_rmse": s["val_rmse"],
            "val_mae": s["val_mae"],
            "val_pcc": s["val_pcc"],
            "val_ccc": s["val_ccc"],
            "test_rmse": s["test_rmse"],
            "test_mae": s["test_mae"],
            "test_pcc": s["test_pcc"],
            "test_ccc": s["test_ccc"],
            "delta_rmse_vs_best": s["delta_vs_historical_best"]["rmse"],
            "delta_mae_vs_best": s["delta_vs_historical_best"]["mae"],
            "delta_pcc_vs_best": s["delta_vs_historical_best"]["pcc"],
            "delta_ccc_vs_best": s["delta_vs_historical_best"]["ccc"],
        }
        for s in all_summaries
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
