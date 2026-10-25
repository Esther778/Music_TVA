#!/usr/bin/env python3
"""Screen multi-layer MERT context-conditioned Valence models.

This is a bottleneck-split experiment. It keeps the model deployable for
arbitrary songs by using only audio-derived features:

- MERT hidden_4, hidden_6, hidden_8 frame embeddings
- harmonic Librosa frame features
- song-level context statistics computed from the same input features

The goal is to improve Valence song polarity/general bias without using test
labels, static PMEmo labels, lyrics, or section annotations.
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

from screen_pmemo_valence_multilayer_mert import (
    BASE_SPLIT_DIR,
    DEFAULT_LAYER_DIRS,
    DEFAULT_OUT as MULTILAYER_BASE_OUT,
    MultiLayerDataset,
    build_harmonic_norm,
    build_layer_norm,
)
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET, collate, selected_indices
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import selection_score


DEFAULT_OUT = ROOT / "outputs" / "valence_multilayer_context_conditioned_20260707"

CURRENT_BEST = {
    "model_id": "pmemo_valence_per_split_direct_multilayer_parsimony_20260629",
    "test_rmse": 0.19130205118779167,
    "test_mae": 0.14893115017555528,
    "test_pcc": 0.8059953934483183,
    "test_ccc": 0.7914302668500032,
}

SPLIT02_REFERENCE = {
    "source": "current best parsimony selected split_02",
    "test_rmse": 0.1965745846688062,
    "test_mae": 0.1521854575118867,
    "test_pcc": 0.788642983812088,
    "test_ccc": 0.7663425632606845,
}


@dataclass
class ContextConfig:
    trial_id: str = "h468_harm_ctx96_film_ext125_clip010_ccc008"
    proj_dim: int = 384
    hidden_dim: int = 80
    context_dim: int = 96
    dropout: float = 0.35
    learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    batch_size: int = 18
    frame_ccc_weight: float = 0.08
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.25
    context_mse_weight: float = 0.0
    use_film: bool = True
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    lengths = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / lengths


class MultiLayerContextBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: ContextConfig):
        super().__init__()
        self.use_film = config.use_film
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.context_net = nn.Sequential(
            nn.LayerNorm(input_dim * 4),
            nn.Linear(input_dim * 4, config.context_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.context_dim, config.context_dim),
            nn.GELU(),
        )
        if config.use_film:
            self.film = nn.Linear(config.context_dim, config.proj_dim * 2)
            lstm_input_dim = config.proj_dim
        else:
            self.context_to_frame = nn.Linear(config.context_dim, config.context_dim)
            lstm_input_dim = config.proj_dim + config.context_dim
        self.lstm = nn.LSTM(lstm_input_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.frame_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2 + config.context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.context_head = nn.Sequential(nn.LayerNorm(config.context_dim), nn.Dropout(config.dropout), nn.Linear(config.context_dim, 1))

    @staticmethod
    def masked_context_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        lengths = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (x * mask_f).sum(dim=1) / lengths
        centered = (x - mean.unsqueeze(1)) * mask_f
        std = torch.sqrt((centered**2).sum(dim=1) / lengths + 1e-6)
        min_v = x.masked_fill(~mask.unsqueeze(-1), 1e9).amin(dim=1)
        max_v = x.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)
        return torch.cat([mean, std, min_v, max_v], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x)
        context = self.context_net(self.masked_context_stats(x, mask))
        if self.use_film:
            gamma, beta = self.film(context).unsqueeze(1).chunk(2, dim=-1)
            z = z * (1.0 + 0.10 * torch.tanh(gamma)) + 0.10 * torch.tanh(beta)
            lstm_input = z
        else:
            ctx = self.context_to_frame(context).unsqueeze(1).expand(-1, z.shape[1], -1)
            lstm_input = torch.cat([z, ctx], dim=-1)
        y, _ = self.lstm(lstm_input)
        ctx = context.unsqueeze(1).expand(-1, y.shape[1], -1)
        frame_pred = self.frame_head(torch.cat([y, ctx], dim=-1)).squeeze(-1)
        context_pred = self.context_head(context).squeeze(-1)
        return frame_pred, context_pred


@torch.no_grad()
def evaluate_context(model: MultiLayerContextBiLSTM, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times = [], [], [], []
    ctx_preds, ctx_targets = [], []
    for x, y, song, time_sec, mask in loader:
        x = x.to(device)
        mask_dev = mask.to(device)
        frame_pred, context_pred = model(x, mask_dev)
        pred_cpu = frame_pred.detach().cpu()
        preds.append(pred_cpu[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
        ctx_preds.append(context_pred.detach().cpu().numpy())
        ctx_targets.append(masked_mean(y, mask).numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    ctx_pred_np = np.concatenate(ctx_preds)
    ctx_target_np = np.concatenate(ctx_targets)
    return {
        "metrics": metrics(pred_np, target_np),
        "context_metrics": metrics(ctx_pred_np, ctx_target_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
        "context_pred": ctx_pred_np,
        "context_target": ctx_target_np,
    }


def build_loaders(index: pd.DataFrame, args: argparse.Namespace, split_dir: Path, config: ContextConfig) -> tuple[dict[str, DataLoader], dict[str, Any], dict[str, Any]]:
    layer_summary = build_layer_norm(index, args.layer_dirs, split_dir / "dataset")
    harm_summary = build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), split_dir / "dataset")
    layer_norms = {}
    for name in args.layer_dirs:
        n = np.load(split_dir / "dataset" / f"{name}_norm.npz")
        layer_norms[name] = (n["mean"], n["std"])
    hn = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            MultiLayerDataset(
                index,
                split,
                args.layer_dirs,
                layer_norms,
                args.label_dir,
                args.librosa_dataset,
                hn["mean"],
                hn["std"],
                hn["feature_idx"],
            ),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    return loaders, layer_summary, harm_summary


def train_one_split(split_id: int, seed: int, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, config: ContextConfig) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / config.trial_id / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    loaders, layer_summary, harm_summary = build_loaders(index, args, split_dir, config)
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = MultiLayerContextBiLSTM(int(sample_x.shape[1]), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = split_dir / "best.pt"
    history: list[dict[str, float]] = []
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
            **{f"val_{k}": float(v) for k, v in val_m.items()},
            **{f"val_context_{k}": float(v) for k, v in val_ctx.items()},
            "val_select_score": float(score),
        }
        history.append(row)
        print(
            f"context split={split_id:02d} seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_mae={val_m['mae']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f} "
            f"val_ctx_rmse={val_ctx['rmse']:.5f} val_ctx_ccc={val_ctx['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "seed": seed, "config": asdict(config), "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(split_dir / "history.csv", index=False)
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
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    test_m = split_evals["test"]["metrics"]
    summary = {
        "created_at": "2026-07-07",
        "purpose": "bottleneck split_02 screen for multi-layer MERT context-conditioned Valence",
        "split_id": split_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_family": "MERT hidden_4+hidden_6+hidden_8 + harmonic + song-context-conditioned BiLSTM",
        "deployability": "audio-only features; arbitrary-song prediction requires extracting the same MERT layers and harmonic features",
        "validation_protocol": "song-level split inherited from valence_multisplit_best; checkpoint and hyperparameters selected by validation selection_score only",
        "config": asdict(config),
        "feature_dim": int(sample_x.shape[1]),
        "layer_summary": layer_summary,
        "harmonic_summary": harm_summary,
        "current_macro_best_reference": CURRENT_BEST,
        "split02_reference": SPLIT02_REFERENCE,
        **{f"{split}_{k}": float(v) for split, e in split_evals.items() for k, v in e["metrics"].items()},
        **{f"{split}_context_{k}": float(v) for split, e in split_evals.items() for k, v in e["context_metrics"].items()},
        "delta_vs_split02_reference": {
            "rmse": float(test_m["rmse"] - SPLIT02_REFERENCE["test_rmse"]),
            "mae": float(test_m["mae"] - SPLIT02_REFERENCE["test_mae"]),
            "pcc": float(test_m["pcc"] - SPLIT02_REFERENCE["test_pcc"]),
            "ccc": float(test_m["ccc"] - SPLIT02_REFERENCE["test_ccc"]),
        },
        "test_gap_rows": threshold_rows(test_m),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--trial-id", default=ContextConfig.trial_id)
    parser.add_argument("--proj-dim", type=int, default=ContextConfig.proj_dim)
    parser.add_argument("--hidden-dim", type=int, default=ContextConfig.hidden_dim)
    parser.add_argument("--context-dim", type=int, default=ContextConfig.context_dim)
    parser.add_argument("--dropout", type=float, default=ContextConfig.dropout)
    parser.add_argument("--learning-rate", type=float, default=ContextConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ContextConfig.weight_decay)
    parser.add_argument("--batch-size", type=int, default=ContextConfig.batch_size)
    parser.add_argument("--frame-ccc-weight", type=float, default=ContextConfig.frame_ccc_weight)
    parser.add_argument("--clip-mse-weight", type=float, default=ContextConfig.clip_mse_weight)
    parser.add_argument("--clip-std-weight", type=float, default=ContextConfig.clip_std_weight)
    parser.add_argument("--extreme-weight", type=float, default=ContextConfig.extreme_weight)
    parser.add_argument("--context-mse-weight", type=float, default=ContextConfig.context_mse_weight)
    parser.add_argument("--no-film", action="store_true")
    parser.add_argument("--patience", type=int, default=ContextConfig.patience)
    parser.add_argument("--epochs", type=int, default=ContextConfig.epochs)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layer_dirs = DEFAULT_LAYER_DIRS
    source_seed = {1: 101, 2: 202, 3: 303}[args.split_id]
    index = pd.read_csv(BASE_SPLIT_DIR / f"split_{args.split_id:02d}_seed_{source_seed}" / "sequence_index.csv")
    device = resolve_device(args.device)
    config = ContextConfig(
        trial_id=args.trial_id,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        frame_ccc_weight=args.frame_ccc_weight,
        clip_mse_weight=args.clip_mse_weight,
        clip_std_weight=args.clip_std_weight,
        extreme_weight=args.extreme_weight,
        context_mse_weight=args.context_mse_weight,
        use_film=not args.no_film,
        patience=args.patience,
        epochs=args.epochs,
    )
    result = train_one_split(args.split_id, args.seed, index, args, device, config)
    out_path = args.out_dir / config.trial_id / "screen_summary.json"
    out_path.write_text(json.dumps({"created_at": "2026-07-07", "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
