#!/usr/bin/env python3
"""Arousal-only OpenSMILE TCN development-fold trial."""

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

from train_pmemo_opensmile_arousal_trials import (
    ROOT,
    SequenceDataset,
    collate,
    ccc_loss,
    metrics,
    moving_average_by_song,
    resolve_device,
    smoothness_loss,
    threshold_rows,
)


DEFAULT_DATA_DIR = ROOT / "data" / "splits" / "pmemo_trainval_devfolds_20260709" / "fold_03"
DEFAULT_OUT = ROOT / "outputs" / "arousal_tcn_devfolds_20260709" / "fold_03"


@dataclass
class TCNConfig:
    trial_id: str = "a_tcn_c96_k5_dil1248_do020"
    channels: int = 96
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.20
    head_hidden: int = 64
    learning_rate: float = 6e-4
    weight_decay: float = 4e-4
    batch_size: int = 32
    ccc_weight: float = 0.03
    smooth_weight: float = 0.01
    epochs: int = 90
    patience: int = 12


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class ArousalTCN(nn.Module):
    def __init__(self, input_dim: int, config: TCNConfig):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, config.channels), nn.GELU())
        self.blocks = nn.ModuleList(
            [TCNBlock(config.channels, config.kernel_size, d, config.dropout) for d in config.dilations]
        )
        self.out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.channels, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.input(x)
        for block in self.blocks:
            y = block(y)
        return self.out(y).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.02 * max(0.0, m["mae"] - 0.15) + 0.04 * max(0.0, 0.80 - m["ccc"])


def postprocess_search(val_eval: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = val_eval["pred"]
    target = val_eval["target"]
    song = val_eval["song_id"]
    rows = []
    best = {"postprocess": "raw", "metrics": val_eval["metrics"], "score": selection_score(val_eval["metrics"])}
    for window in [1, 7, 11, 15, 21, 31, 41, 51, 71]:
        sm = moving_average_by_song(pred, song, window)
        for blend in [0.25, 0.50, 0.75, 1.0]:
            pp = (1.0 - blend) * pred + blend * sm
            m = metrics(pp, target)
            score = selection_score(m)
            row = {"postprocess": f"smooth_w{window}_b{blend:g}", "window": window, "blend": blend, **m, "score": score}
            rows.append(row)
            if score < best["score"]:
                best = {"postprocess": row["postprocess"], "metrics": m, "score": score, "window": window, "blend": blend}
    return best, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = TCNConfig()
    set_seed(args.seed)
    device = resolve_device(args.device)
    index = pd.read_csv(args.data_dir / "sequence_index.csv")
    loaders = {
        split: DataLoader(
            SequenceDataset(index, split),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    input_dim = int(np.load(index.iloc[0]["path"])["X"].shape[1])
    model = ArousalTCN(input_dim, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = args.out_dir / "best.pt"
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
            loss = ((pred_valid - y_valid) ** 2).mean()
            loss = loss + config.ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + config.smooth_weight * smoothness_loss(pred, mask)
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
            f"{config.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={m['rmse']:.5f} val_mae={m['mae']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "config": asdict(config)}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val"]}
    for split, data in evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_arousal": data["target"],
                "pred_arousal": data["pred"],
            }
        ).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Arousal TCN development fold trial; holdout/test excluded",
        "best_epoch": best_epoch,
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()},
        "best_val_postprocess": best_pp,
        "val_gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "val_gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
