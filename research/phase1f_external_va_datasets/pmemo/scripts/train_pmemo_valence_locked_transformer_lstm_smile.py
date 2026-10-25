#!/usr/bin/env python3
"""Transformer+BiLSTM Valence model on locked development folds."""

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
from train_pmemo_valence_locked_transformer_smile import SinusoidalPositionalEncoding, clip_mean_loss


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "devfolds_3fold_20260713" / "fold_03" / "sequence_index.csv"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_transformer_lstm_smile_20260713" / "fold_03"


@dataclass
class Config:
    trial_id: str = "v_mert_smile_tf_lstm_d160_l1_h64_do035_lr2e4"
    d_model: int = 160
    num_layers: int = 1
    nhead: int = 4
    dim_feedforward: int = 320
    hidden_dim: int = 64
    dropout: float = 0.35
    learning_rate: float = 2e-4
    weight_decay: float = 6e-4
    batch_size: int = 18
    ccc_weight: float = 0.04
    smooth_weight: float = 0.002
    clip_mean_weight: float = 0.06
    epochs: int = 80
    patience: int = 10


class TransformerLSTM(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout))
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
        self.lstm = nn.LSTM(cfg.d_model, cfg.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.pos(self.proj(x))
        y = self.encoder(y, src_key_padding_mask=~mask)
        y, _ = self.lstm(y * mask.float().unsqueeze(-1))
        return self.head(y).squeeze(-1)


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
    model = TransformerLSTM(int(sample_x.shape[1]), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ckpt = args.out_dir / "best.pt"
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
        pd.DataFrame({"musicId": data["song_id"], "time_sec": data["time_sec"], "target_valence": data["target"], "pred_valence": data["pred"]}).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "MERT hidden6 + OpenSMILE Transformer+BiLSTM Valence on locked devfold; final locked test excluded",
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
    parser.add_argument("--trial-id", default=Config.trial_id)
    parser.add_argument("--d-model", type=int, default=Config.d_model)
    parser.add_argument("--num-layers", type=int, default=Config.num_layers)
    parser.add_argument("--nhead", type=int, default=Config.nhead)
    parser.add_argument("--dim-feedforward", type=int, default=Config.dim_feedforward)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--ccc-weight", type=float, default=Config.ccc_weight)
    parser.add_argument("--smooth-weight", type=float, default=Config.smooth_weight)
    parser.add_argument("--clip-mean-weight", type=float, default=Config.clip_mean_weight)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    args = parser.parse_args()
    cfg = Config(
        trial_id=args.trial_id,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        ccc_weight=args.ccc_weight,
        smooth_weight=args.smooth_weight,
        clip_mean_weight=args.clip_mean_weight,
        epochs=args.epochs,
        patience=args.patience,
    )
    run(args, cfg)


if __name__ == "__main__":
    main()
