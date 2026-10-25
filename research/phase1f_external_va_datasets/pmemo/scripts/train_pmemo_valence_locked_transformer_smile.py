#!/usr/bin/env python3
"""Lightweight Transformer Valence model on locked development folds.

This is a train/validation-only screen. It uses only audio-derived MERT hidden6
and OpenSMILE features, so it remains deployable for arbitrary songs after the
same feature extraction pipeline is available.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_mert_opensmile_arousal_fusion_locked_devfold import (
    DEFAULT_MERT_DIR,
    DEFAULT_OPENSMILE_DATASET,
    build_mert_norm,
    set_seed,
)
from train_pmemo_mert_opensmile_valence_locked_devfold import MertOpenSmileValenceDataset, evaluate
from train_pmemo_opensmile_arousal_trials import ROOT, ccc_loss, collate, resolve_device, smoothness_loss
from train_pmemo_mert_valence_baseline import threshold_rows
from train_pmemo_valence_dev_only_trials import postprocess_search, selection_score


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "devfolds_3fold_20260713" / "fold_03" / "sequence_index.csv"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_transformer_smile_20260713" / "fold_03"


@dataclass
class Config:
    trial_id: str = "v_mert_smile_tf_d192_l2_h4_do030_lr25e4"
    d_model: int = 192
    num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int = 384
    dropout: float = 0.30
    learning_rate: float = 2.5e-4
    weight_decay: float = 5e-4
    batch_size: int = 18
    ccc_weight: float = 0.04
    smooth_weight: float = 0.002
    clip_mean_weight: float = 0.06
    epochs: int = 80
    patience: int = 10


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class FusionTransformer(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.pos = SinusoidalPositionalEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.pos(self.proj(x))
        y = self.encoder(y, src_key_padding_mask=~mask)
        return self.head(y).squeeze(-1)


def clip_mean_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if not weight:
        return pred.new_tensor(0.0)
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    pred_mean = (pred * mask_f).sum(dim=1) / denom
    target_mean = (target * mask_f).sum(dim=1) / denom
    return float(weight) * ((pred_mean - target_mean) ** 2).mean()


def run(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    mert_mean, mert_std = build_mert_norm(index, "train", args.mert_dir, args.out_dir)
    loaders = {
        split: DataLoader(
            MertOpenSmileValenceDataset(index, split, args.mert_dir, args.opensmile_dataset, mert_mean, mert_std),
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    device = resolve_device(args.device)
    model = FusionTransformer(int(sample_x.shape[1]), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ckpt = args.out_dir / "best.pt"
    history = []
    best_score = float("inf")
    best_epoch = 0
    bad = 0
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
            loss = ((pred_valid - y_valid) ** 2).mean()
            loss = loss + cfg.ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + cfg.smooth_weight * smoothness_loss(pred, mask)
            loss = loss + clip_mean_loss(pred, y, mask, cfg.clip_mean_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        m = val_eval["metrics"]
        score = selection_score(m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}, "score": score}
        history.append(row)
        print(
            f"{cfg.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={m['rmse']:.5f} val_mae={m['mae']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": int(sample_x.shape[1]), "config": asdict(cfg)}, ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
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
        ).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "MERT hidden6 + OpenSMILE lightweight Transformer Valence on locked devfold; final locked test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "trial_id": cfg.trial_id,
        "best_epoch": best_epoch,
        "config": asdict(cfg),
        "val_metrics": evals["val"]["metrics"],
        "best_val_postprocess": best_pp,
        "gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--opensmile-dataset", type=Path, default=DEFAULT_OPENSMILE_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args, Config())


if __name__ == "__main__":
    main()
