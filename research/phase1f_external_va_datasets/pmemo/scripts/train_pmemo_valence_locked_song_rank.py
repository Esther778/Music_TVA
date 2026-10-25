#!/usr/bin/env python3
"""Screen song-level pairwise ranking loss for Valence on locked dev folds.

The script reads train/val rows only. It never reads the final locked test.
Inputs remain reproducible from arbitrary audio: MERT hidden6 plus Librosa
harmonic features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_pmemo_mert_harmonic_valence_fusion import FusionBiLSTM, evaluate
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import (
    clip_mean_loss,
    weighted_frame_mse,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, resolve_device, threshold_rows
from train_pmemo_valence_dev_only_trials import (
    build_loaders,
    postprocess_search,
    selection_score,
    set_seed,
)


DEFAULT_INDEX = (
    ROOT
    / "data"
    / "splits"
    / "pmemo_final_locked_20260713"
    / "devfolds_3fold_20260713"
    / "fold_01"
    / "sequence_index.csv"
)
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_song_rank_fold01_20260713"

LOCKED_DEV_REFERENCE = {
    "output": "valence_locked_arousal_assisted_shift_wide_rmse_20260713",
    "macro_rmse": 0.193121,
    "macro_mae": 0.151375,
    "macro_pcc": 0.796623,
    "macro_ccc": 0.776367,
    "fold_01_rmse": 0.190867,
    "fold_01_mae": 0.151228,
    "fold_01_pcc": 0.774868,
    "fold_01_ccc": 0.750329,
}


@dataclass
class RankTrialConfig:
    trial_id: str
    rank_weight: float
    rank_temperature: float = 0.10
    rank_min_gap: float = 0.05
    proj_dim: int = 384
    hidden_dim: int = 96
    dropout: float = 0.35
    learning_rate: float = 2.5e-4
    weight_decay: float = 4e-4
    batch_size: int = 18
    frame_ccc_weight: float = 0.04
    clip_mse_weight: float = 0.08
    extreme_weight: float = 1.15
    patience: int = 10
    epochs: int = 80


TRIALS = [
    RankTrialConfig("songrank_w002_t010_gap005", 0.02),
    RankTrialConfig("songrank_w005_t010_gap005", 0.05),
    RankTrialConfig("songrank_w010_t010_gap005", 0.10),
]


def masked_song_means(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    lengths = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / lengths


def pairwise_song_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    min_gap: float,
) -> tuple[torch.Tensor, int]:
    """Logistic pairwise loss over song-mean Valence within a batch."""
    pred_mean = masked_song_means(pred, mask)
    target_mean = masked_song_means(target, mask)
    pair_idx = torch.triu_indices(pred_mean.numel(), pred_mean.numel(), offset=1, device=pred.device)
    target_delta = target_mean[pair_idx[0]] - target_mean[pair_idx[1]]
    keep = target_delta.abs() >= float(min_gap)
    if not torch.any(keep):
        return pred.new_tensor(0.0), 0
    pred_delta = pred_mean[pair_idx[0]][keep] - pred_mean[pair_idx[1]][keep]
    direction = target_delta[keep].sign()
    temp = max(float(temperature), 1e-4)
    loss = F.softplus(-direction * pred_delta / temp).mean() * temp
    return loss, int(keep.sum().item())


def song_rank_metrics(eval_data: dict[str, Any], min_gap: float) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {
            "song_id": eval_data["song_id"],
            "target": eval_data["target"],
            "pred": eval_data["pred"],
        }
    )
    song = frame.groupby("song_id", sort=False)[["target", "pred"]].mean().reset_index(drop=True)
    target = song["target"].to_numpy(dtype=np.float64)
    pred = song["pred"].to_numpy(dtype=np.float64)
    left, right = np.triu_indices(len(song), k=1)
    target_delta = target[left] - target[right]
    keep = np.abs(target_delta) >= float(min_gap)
    pair_count = int(keep.sum())
    if pair_count == 0:
        return {"song_rank_pair_count": 0, "song_rank_accuracy": float("nan")}
    pred_delta = pred[left][keep] - pred[right][keep]
    accuracy = float(np.mean(np.sign(pred_delta) == np.sign(target_delta[keep])))
    return {"song_rank_pair_count": pair_count, "song_rank_accuracy": accuracy}


def run_trial(
    config: RankTrialConfig,
    index: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    trial_dir = args.out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(trial_dir / "sequence_index_train_val_only.csv", index=False)
    loaders = build_loaders(index, args, trial_dir, config)
    sample_x, _, _, _ = loaders["train"].dataset[0]
    input_dim = int(sample_x.shape[1])
    model = FusionBiLSTM(input_dim, config.proj_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint = trial_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        rank_losses: list[float] = []
        rank_pairs = 0
        for x, y, _, _, mask in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, mask)
            pred_valid = pred[mask]
            y_valid = y[mask]
            frame_loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            frame_loss = frame_loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            mean_loss = clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            rank_loss, pair_count = pairwise_song_rank_loss(
                pred,
                y,
                mask,
                config.rank_temperature,
                config.rank_min_gap,
            )
            loss = frame_loss + mean_loss + config.rank_weight * rank_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            rank_losses.append(float(rank_loss.detach().cpu()))
            rank_pairs += pair_count

        val_eval = evaluate(model, loaders["val"], device)
        val_metrics = val_eval["metrics"]
        rank_metrics = song_rank_metrics(val_eval, config.rank_min_gap)
        score = selection_score(val_metrics)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_rank_loss": float(np.mean(rank_losses)),
            "train_rank_pairs": rank_pairs,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"val_{key}": value for key, value in rank_metrics.items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"{config.trial_id} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"rank={row['train_rank_loss']:.5f} val_rmse={val_metrics['rmse']:.5f} "
            f"val_mae={val_metrics['mae']:.5f} val_pcc={val_metrics['pcc']:.3f} "
            f"val_ccc={val_metrics['ccc']:.3f} song_rank_acc={rank_metrics['song_rank_accuracy']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": args.seed,
                    "config": asdict(config),
                    "input_dim": input_dim,
                },
                checkpoint,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
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
    best_postprocess, postprocess_rows = postprocess_search(evals["val"])
    postprocess_rows.to_csv(trial_dir / "val_postprocess_search.csv", index=False)
    rank_metrics = {
        split: song_rank_metrics(data, config.rank_min_gap)
        for split, data in evals.items()
    }
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Fold_01 screen of song-mean pairwise ranking loss for Valence",
        "selection_boundary": "Final locked development train/val only; final locked test is not read.",
        "trial_id": config.trial_id,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint),
        "feature_type": "MERT hidden6 plus Librosa harmonic; raw-audio reproducible",
        "config": asdict(config),
        "locked_dev_reference": LOCKED_DEV_REFERENCE,
        "split_song_counts": {
            split: int((index["split"] == split).sum()) for split in ["train", "val"]
        },
        **{
            f"{split}_{key}": value
            for split, data in evals.items()
            for key, value in data["metrics"].items()
        },
        **{
            f"{split}_{key}": value
            for split, values in rank_metrics.items()
            for key, value in values.items()
        },
        "best_val_postprocess": best_postprocess,
        "val_gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "val_gap_rows_best_postprocess": threshold_rows(best_postprocess["metrics"]),
    }
    (trial_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--trial", default="all")
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    if set(index["split"].unique()) != {"train", "val"}:
        raise ValueError("Expected train and val rows only.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out_dir / "dev_only_sequence_index.csv", index=False)
    trials = TRIALS if args.trial == "all" else [item for item in TRIALS if item.trial_id == args.trial]
    if not trials:
        raise ValueError(f"Unknown trial: {args.trial}")
    device = resolve_device(args.device)
    summaries = [run_trial(config, index, args, device) for config in trials]
    comparison = []
    for summary in summaries:
        post = summary["best_val_postprocess"]
        comparison.append(
            {
                "trial_id": summary["trial_id"],
                "best_epoch": summary["best_epoch"],
                "rank_weight": summary["config"]["rank_weight"],
                "val_rmse": summary["val_rmse"],
                "val_mae": summary["val_mae"],
                "val_pcc": summary["val_pcc"],
                "val_ccc": summary["val_ccc"],
                "val_song_rank_accuracy": summary["val_song_rank_accuracy"],
                "postprocess": post["postprocess"],
                "post_val_rmse": post["metrics"]["rmse"],
                "post_val_mae": post["metrics"]["mae"],
                "post_val_pcc": post["metrics"]["pcc"],
                "post_val_ccc": post["metrics"]["ccc"],
            }
        )
    pd.DataFrame(comparison).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"comparison": comparison}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
