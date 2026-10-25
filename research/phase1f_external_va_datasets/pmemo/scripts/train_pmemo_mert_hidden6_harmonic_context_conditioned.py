#!/usr/bin/env python3
"""Train context-conditioned hidden_6 MERT + harmonic Valence models."""

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
    FusionDataset,
    build_harmonic_norm,
    collate,
    selected_indices,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows


DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_HIDDEN6_NORM = ROOT / "outputs" / "mert_hidden6_valence_baseline" / "dataset" / "mert_norm.npz"
DEFAULT_OUT = ROOT / "outputs" / "mert_hidden6_harmonic_context_conditioned"

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
    context_dim: int = 96
    dropout: float = 0.30
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.05
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.25
    context_mse_weight: float = 0.0
    use_film: bool = True
    patience: int = 10
    epochs: int = 80


TRIALS = [
    TrialConfig("h6_harm_ctx96_film_ext125_clip010_ccc005"),
    TrialConfig("h6_harm_ctx64_concat_ext125_clip010_ccc005", context_dim=64, use_film=False),
    TrialConfig("h6_harm_ctx96_film_ext125_clip010_std003", clip_std_weight=0.03),
    TrialConfig("h6_harm_ctx96_film_ext125_clip010_ctxmse005", context_mse_weight=0.05),
]


class ContextConditionedBiLSTM(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, hidden: int, context_dim: int, dropout: float, use_film: bool):
        super().__init__()
        self.use_film = use_film
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.context_net = nn.Sequential(
            nn.LayerNorm(input_dim * 4),
            nn.Linear(input_dim * 4, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
        )
        if use_film:
            self.film = nn.Linear(context_dim, proj_dim * 2)
            lstm_input = proj_dim
        else:
            self.context_to_frame = nn.Linear(context_dim, context_dim)
            lstm_input = proj_dim + context_dim
        self.lstm = nn.LSTM(lstm_input, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.frame_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2 + context_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.context_head = nn.Sequential(nn.LayerNorm(context_dim), nn.Dropout(dropout), nn.Linear(context_dim, 1))

    @staticmethod
    def masked_context_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        lengths = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (x * mask_f).sum(dim=1) / lengths
        centered = (x - mean.unsqueeze(1)) * mask_f
        std = torch.sqrt((centered**2).sum(dim=1) / lengths + 1e-6)
        x_masked_min = x.masked_fill(~mask.unsqueeze(-1), 1e9).amin(dim=1)
        x_masked_max = x.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)
        return torch.cat([mean, std, x_masked_min, x_masked_max], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x)
        context = self.context_net(self.masked_context_stats(x, mask))
        if self.use_film:
            gamma_beta = self.film(context).unsqueeze(1)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            z = z * (1.0 + 0.10 * torch.tanh(gamma)) + 0.10 * torch.tanh(beta)
            lstm_in = z
        else:
            ctx_frame = self.context_to_frame(context).unsqueeze(1).expand(-1, z.shape[1], -1)
            lstm_in = torch.cat([z, ctx_frame], dim=-1)
        y, _ = self.lstm(lstm_in)
        ctx_frame = context.unsqueeze(1).expand(-1, y.shape[1], -1)
        frame = self.frame_head(torch.cat([y, ctx_frame], dim=-1)).squeeze(-1)
        context_pred = self.context_head(context).squeeze(-1)
        return frame, context_pred


def weighted_frame_mse(pred: torch.Tensor, target: torch.Tensor, extreme_weight: float) -> torch.Tensor:
    weights = 1.0 + float(extreme_weight) * target.abs()
    return (weights * (pred - target) ** 2).mean()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    lengths = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / lengths


def clip_mean_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if not weight:
        return pred.new_tensor(0.0)
    return float(weight) * ((masked_mean(pred, mask) - masked_mean(target, mask)) ** 2).mean()


def clip_std_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if not weight:
        return pred.new_tensor(0.0)
    mask_f = mask.float()
    lengths = mask_f.sum(dim=1).clamp_min(2.0)
    pred_mean = masked_mean(pred, mask).unsqueeze(1)
    target_mean = masked_mean(target, mask).unsqueeze(1)
    pred_std = torch.sqrt((((pred - pred_mean) * mask_f) ** 2).sum(dim=1) / lengths + 1e-8)
    target_std = torch.sqrt((((target - target_mean) * mask_f) ** 2).sum(dim=1) / lengths + 1e-8)
    return float(weight) * ((pred_std - target_std) ** 2).mean()


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.05 * max(0.0, 0.80 - m["ccc"]) + 0.03 * max(0.0, 0.80 - m["pcc"]) + 0.01 * m["mae"]


@torch.no_grad()
def evaluate_context(model: ContextConditionedBiLSTM, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times, context_preds, context_targets = [], [], [], [], [], []
    for x, y, song, time_sec, mask in loader:
        x = x.to(device)
        mask_dev = mask.to(device)
        frame_pred, context_pred = model(x, mask_dev)
        pred = frame_pred.detach().cpu()
        context_pred_cpu = context_pred.detach().cpu()
        preds.append(pred[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
        context_preds.append(context_pred_cpu.numpy())
        context_targets.append(masked_mean(y, mask).numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    context_pred_np = np.concatenate(context_preds)
    context_target_np = np.concatenate(context_targets)
    return {
        "metrics": metrics(pred_np, target_np),
        "context_metrics": metrics(context_pred_np, context_target_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
        "context_pred": context_pred_np,
        "context_target": context_target_np,
    }


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
    model = ContextConditionedBiLSTM(
        int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.context_dim, config.dropout, config.use_film
    ).to(device)
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
            pred, context_pred = model(x, mask)
            pred_valid = pred[mask]
            y_valid = y[mask]
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + float(config.frame_ccc_weight) * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            if config.context_mse_weight:
                loss = loss + float(config.context_mse_weight) * ((context_pred - masked_mean(y, mask)) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate_context(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        val_ctx = val_eval["context_metrics"]
        score = selection_score(val_m)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"val_context_{k}": v for k, v in val_ctx.items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"{config.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_mae={val_m['mae']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f} "
            f"val_ctx_rmse={val_ctx['rmse']:.5f} val_ctx_pcc={val_ctx['pcc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "seed": seed, "trial_config": asdict(config), "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate_context(model, loaders[split], device) for split in ["train", "val", "test"]}
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
        **{f"{split}_context_{k}": v for split, e in split_evals.items() for k, v in e["context_metrics"].items()},
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


def write_comparison(out_dir: Path) -> list[dict[str, Any]]:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(out_dir.glob("*/summary.json"))]
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
            "test_context_rmse": s["test_context_rmse"],
            "test_context_pcc": s["test_context_pcc"],
            "test_context_ccc": s["test_context_ccc"],
            "delta_rmse_vs_best": s["delta_vs_historical_best"]["rmse"],
            "delta_mae_vs_best": s["delta_vs_historical_best"]["mae"],
            "delta_pcc_vs_best": s["delta_vs_historical_best"]["pcc"],
            "delta_ccc_vs_best": s["delta_vs_historical_best"]["ccc"],
        }
        for s in summaries
    ]
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


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
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
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
        for trial in trials:
            run_trial(trial, index, args, device, args.seed)
    rows = write_comparison(args.out_dir)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
