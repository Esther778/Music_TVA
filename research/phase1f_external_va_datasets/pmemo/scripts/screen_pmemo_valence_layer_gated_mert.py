#!/usr/bin/env python3
"""Screen low-dimensional layer-gated MERT fusion for PMEmo Valence.

The previous hidden_4+6+8 concatenation improved the difficult split_02 but
hurt split_01, likely because the 2340-dim input overfits. This version projects
each MERT layer to a small shared dimension and learns global layer gates before
the BiLSTM head.
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
    CURRENT_STRICTISH,
    DEFAULT_LAYER_DIRS,
    DEFAULT_OUT as _MULTILAYER_OUT,
    SPLIT_REFERENCES,
    MultiLayerDataset,
    build_harmonic_norm,
    build_layer_norm,
)
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET, collate, selected_indices
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import selection_score


_ = _MULTILAYER_OUT
DEFAULT_OUT = ROOT / "outputs" / "valence_layer_gated_mert_screen_20260629"


@dataclass
class LayerGatedConfig:
    trial_id: str = "h468_gate_l128_harm64_bilstm_h80_do035"
    layer_dim: int = 128
    harmonic_dim: int = 64
    hidden_dim: int = 80
    dropout: float = 0.35
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 18
    frame_ccc_weight: float = 0.08
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.25
    gate_entropy_weight: float = 0.001
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LayerGatedMERTBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: LayerGatedConfig, num_layers: int = 3, layer_input_dim: int = 768):
        super().__init__()
        self.num_layers = num_layers
        self.layer_input_dim = layer_input_dim
        self.harmonic_input_dim = input_dim - num_layers * layer_input_dim
        if self.harmonic_input_dim <= 0:
            raise ValueError(f"Unexpected input_dim={input_dim}; harmonic_input_dim={self.harmonic_input_dim}")
        self.layer_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(layer_input_dim),
                    nn.Linear(layer_input_dim, config.layer_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.harmonic_proj = nn.Sequential(
            nn.LayerNorm(self.harmonic_input_dim),
            nn.Linear(self.harmonic_input_dim, config.harmonic_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        fusion_dim = config.layer_dim + config.harmonic_dim
        self.lstm = nn.LSTM(fusion_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def gate_entropy_loss(self) -> torch.Tensor:
        w = self.layer_weights()
        entropy = -(w * torch.log(w + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(self.num_layers), device=w.device))
        return (max_entropy - entropy) / max_entropy

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        layer_chunks = [
            x[:, :, i * self.layer_input_dim : (i + 1) * self.layer_input_dim]
            for i in range(self.num_layers)
        ]
        harmonic = x[:, :, self.num_layers * self.layer_input_dim :]
        projected = torch.stack([proj(chunk) for proj, chunk in zip(self.layer_projs, layer_chunks)], dim=0)
        weights = self.layer_weights().view(self.num_layers, 1, 1, 1)
        fused_layer = (weights * projected).sum(dim=0)
        fused = torch.cat([fused_layer, self.harmonic_proj(harmonic)], dim=-1)
        y, _ = self.lstm(fused * mask.float().unsqueeze(-1))
        return self.out(y).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times = [], [], [], []
    for x, y, song, time_sec, mask in loader:
        pred = model(x.to(device), mask.to(device)).detach().cpu()
        preds.append(pred[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    return {
        "metrics": metrics(pred_np, target_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
    }


def train_one_split(split_id: int, seed: int, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, config: LayerGatedConfig) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / config.trial_id / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    layer_summary = build_layer_norm(index, args.layer_dirs, split_dir / "dataset")
    harm_summary = build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), split_dir / "dataset")
    layer_norms = {}
    for name in args.layer_dirs:
        n = np.load(split_dir / "dataset" / f"{name}_norm.npz")
        layer_norms[name] = (n["mean"], n["std"])
    hn = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    split_names = [split for split in ["train", "val", "test"] if split in set(index["split"])]
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
        for split in split_names
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = LayerGatedMERTBiLSTM(int(sample_x.shape[1]), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = split_dir / "best.pt"
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
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            loss = loss + config.gate_entropy_weight * model.gate_entropy_loss()
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
            "gate_h4": float(model.layer_weights().detach().cpu()[0]),
            "gate_h6": float(model.layer_weights().detach().cpu()[1]),
            "gate_h8": float(model.layer_weights().detach().cpu()[2]),
        }
        history.append(row)
        print(
            f"layer_gate split={split_id:02d} seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f} "
            f"gates=({row['gate_h4']:.2f},{row['gate_h6']:.2f},{row['gate_h8']:.2f})",
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
    split_evals = {split: evaluate(model, loaders[split], device) for split in split_names}
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    gate = model.layer_weights().detach().cpu().numpy().astype(float).tolist()
    summary = {
        "split_id": split_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_family": "Layer-gated MERT hidden_4+hidden_6+hidden_8 + harmonic BiLSTM",
        "deployability": "audio-only features; usable for arbitrary songs after extracting the same MERT layers and harmonic features",
        "config": asdict(config),
        "layer_summary": layer_summary,
        "harmonic_summary": harm_summary,
        "feature_dim": int(sample_x.shape[1]),
        "effective_sequence_dim": int(config.layer_dim + config.harmonic_dim),
        "final_layer_gate": {"h4": gate[0], "h6": gate[1], "h8": gate[2]},
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in split_names},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in split_names},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "val_gap_rows": threshold_rows(split_evals["val"]["metrics"]),
    }
    if "test" in split_evals:
        summary["test_gap_rows"] = threshold_rows(split_evals["test"]["metrics"])
        summary["delta_vs_split_reference"] = {
            k: float(split_evals["test"]["metrics"][k] - SPLIT_REFERENCES[split_id][k]) for k in ["rmse", "mae", "pcc", "ccc"]
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
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--index", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layer_dirs = DEFAULT_LAYER_DIRS
    if args.index is None:
        source_seed = {1: 101, 2: 202, 3: 303}[args.split_id]
        index = pd.read_csv(BASE_SPLIT_DIR / f"split_{args.split_id:02d}_seed_{source_seed}" / "sequence_index.csv")
    else:
        index = pd.read_csv(args.index)
    device = resolve_device(args.device)
    config = LayerGatedConfig()
    row = train_one_split(args.split_id, args.seed, index, args, device, config)
    summary = {
        "created_at": "2026-06-29",
        "purpose": "screen low-dimensional layer-gated MERT fusion for Valence",
        "current_strictish_reference": CURRENT_STRICTISH,
        "split_reference": SPLIT_REFERENCES[args.split_id],
        "result": row,
    }
    out_path = args.out_dir / config.trial_id / "screen_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
