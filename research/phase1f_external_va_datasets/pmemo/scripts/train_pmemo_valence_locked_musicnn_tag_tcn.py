#!/usr/bin/env python3
"""Train a causal MusiCNN-tag residual TCN on locked Valence dev folds."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_musicnn_tag_monotonic_20260714"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_musicnn_tag_sequences_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_musicnn_tag_tcn_20260714"


@dataclass
class Config:
    channels: int = 32
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.20
    correction_cap: float = 0.04
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    batch_size: int = 32
    ccc_weight: float = 0.02
    song_mean_weight: float = 0.05
    correction_weight: float = 0.005
    epochs: int = 50
    patience: int = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def selection_score(result: dict[str, float]) -> float:
    return result["rmse"] + 0.02 * max(0.0, result["mae"] - 0.15) + 0.04 * max(0.0, 0.80 - result["ccc"])


def interpolate(tags: np.ndarray, length: int) -> np.ndarray:
    source_x = np.linspace(0.0, 1.0, len(tags), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.column_stack([np.interp(target_x, source_x, tags[:, column]) for column in range(tags.shape[1])]).astype(np.float32)


def build_records(frame: pd.DataFrame, tag_dir: Path) -> list[dict[str, Any]]:
    records = []
    for music_id, song in frame.groupby("musicId", sort=False):
        song = song.reset_index(drop=True)
        tags = np.load(tag_dir / "per_song" / f"{int(music_id)}.npz")["tags"].astype(np.float32)
        tag_frames = interpolate(tags, len(song))
        base = song["pred_valence"].to_numpy(dtype=np.float32)
        base_delta = np.gradient(base).astype(np.float32)
        position = np.linspace(0.0, 1.0, len(song), dtype=np.float32)
        x = np.concatenate([tag_frames, base[:, None], base_delta[:, None], position[:, None]], axis=1)
        records.append({
            "musicId": int(music_id),
            "X": x,
            "base": base,
            "target": song["target_valence"].to_numpy(dtype=np.float32),
            "time_sec": song["time_sec"].to_numpy(dtype=np.float32),
        })
    return records


def fit_norm(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([record["X"] for record in records], axis=0).astype(np.float64)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


class SongDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray):
        self.records = records
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        x = (record["X"] - self.mean) / self.std
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(record["base"]),
            torch.from_numpy(record["target"]),
            torch.full((len(record["target"]),), int(record["musicId"]), dtype=torch.long),
            torch.from_numpy(record["time_sec"]),
        )


def collate(batch):
    length = max(item[0].shape[0] for item in batch)
    feature_dim = batch[0][0].shape[1]
    x = torch.zeros(len(batch), length, feature_dim)
    base = torch.zeros(len(batch), length)
    target = torch.zeros(len(batch), length)
    song = torch.zeros(len(batch), length, dtype=torch.long)
    time_sec = torch.zeros(len(batch), length)
    mask = torch.zeros(len(batch), length, dtype=torch.bool)
    for row, item in enumerate(batch):
        size = item[0].shape[0]
        x[row, :size], base[row, :size], target[row, :size] = item[0], item[1], item[2]
        song[row, :size], time_sec[row, :size], mask[row, :size] = item[3], item[4], True
    return x, base, target, song, time_sec, mask


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        y = self.dropout(F.gelu(y))
        return self.norm(x + y) * mask.float().unsqueeze(-1)


class TagTCN(nn.Module):
    def __init__(self, input_dim: int, config: Config):
        super().__init__()
        self.cap = config.correction_cap
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, config.channels), nn.GELU())
        self.blocks = nn.ModuleList([
            CausalBlock(config.channels, config.kernel_size, dilation, config.dropout)
            for dilation in config.dilations
        ])
        self.out = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(config.channels, config.channels // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, base: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(x) * mask.float().unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, mask)
        correction = self.cap * torch.tanh(self.out(hidden).squeeze(-1)) * mask.float()
        return base + correction, correction


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions, targets, songs, times = [], [], [], []
    for x, base, target, song, time_sec, mask in loader:
        pred, _ = model(x.to(device), base.to(device), mask.to(device))
        predictions.append(pred.detach().cpu()[mask].numpy())
        targets.append(target[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    return {
        "metrics": metrics(pred, target), "pred": pred, "target": target,
        "song": np.concatenate(songs), "time_sec": np.concatenate(times),
    }


def train_epochs(
    train_records: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, config: Config,
    device: torch.device, seed: int, epochs: int, val_records: list[dict[str, Any]] | None = None,
) -> tuple[TagTCN, list[dict[str, float]], int]:
    set_seed(seed)
    model = TagTCN(train_records[0]["X"].shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = DataLoader(SongDataset(train_records, mean, std), batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = None if val_records is None else DataLoader(SongDataset(val_records, mean, std), batch_size=config.batch_size, collate_fn=collate)
    best_state = None
    best_score = float("inf")
    best_epoch = epochs
    bad = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, base, target, _, _, mask in train_loader:
            x, base, target, mask = x.to(device), base.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad()
            pred, correction = model(x, base, mask)
            pv, tv = pred[mask], target[mask]
            loss = ((pv - tv) ** 2).mean() + config.ccc_weight * ccc_loss(pv, tv)
            song_mean = (((pred * mask).sum(1) / mask.sum(1)) - ((target * mask).sum(1) / mask.sum(1))) ** 2
            loss = loss + config.song_mean_weight * song_mean.mean()
            loss = loss + config.correction_weight * (correction[mask] ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row: dict[str, float] = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            result = evaluate(model, val_loader, device)["metrics"]
            row.update({f"val_{key}": float(value) for key, value in result.items()})
            score = selection_score(result)
            if score < best_score:
                best_score, best_epoch, bad = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                bad += 1
            print(
                f"epoch={epoch:02d} loss={row['train_loss']:.5f} val_rmse={result['rmse']:.5f} "
                f"val_pcc={result['pcc']:.3f} val_ccc={result['ccc']:.3f}", flush=True,
            )
            if bad >= config.patience:
                break
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = Config()

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    records = {fold: build_records(frames[fold], args.tag_dir) for fold in FOLDS}
    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    per_fold = []
    for fold_index, held in enumerate(FOLDS):
        other = [fold for fold in FOLDS if fold != held]
        development = [record for fold in other for record in records[fold]]
        validation = [record for record in development if ((record["musicId"] * 2654435761) % 100) < 15]
        validation_ids = {record["musicId"] for record in validation}
        training = [record for record in development if record["musicId"] not in validation_ids]
        mean, std = fit_norm(training)
        _, history, best_epoch = train_epochs(
            training, mean, std, config, device, args.seed + fold_index, config.epochs, validation,
        )

        final_mean, final_std = fit_norm(development)
        model, _, _ = train_epochs(
            development, final_mean, final_std, config, device, args.seed + 100 + fold_index, best_epoch,
        )
        held_loader = DataLoader(
            SongDataset(records[held], final_mean, final_std), batch_size=config.batch_size, collate_fn=collate,
        )
        result = evaluate(model, held_loader, device)
        fold_dir = args.out_dir / held
        fold_dir.mkdir(exist_ok=True)
        torch.save({
            "model_state": model.state_dict(), "config": asdict(config), "input_mean": final_mean,
            "input_std": final_std, "best_epoch": best_epoch,
        }, fold_dir / "model.pt")
        pd.DataFrame(history).to_csv(fold_dir / "selection_history.csv", index=False)
        prediction_frame = pd.DataFrame({
            "musicId": result["song"], "time_sec": result["time_sec"],
            "target_valence": result["target"], "pred_valence": result["pred"],
        })
        prediction_frame.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        per_fold.append({
            "fold": held, "best_epoch": best_epoch, "selection_train_songs": len(training),
            "selection_val_songs": len(validation), "refit_songs": len(development), **result["metrics"],
        })

    selected_macro = macro(per_fold)
    base_macro = macro(base_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Held fold excluded from epoch selection, refit, and normalization; final locked test excluded",
        "base_dir": str(args.base_dir), "tag_dir": str(args.tag_dir), "config": asdict(config),
        "base_macro": base_macro, "macro": selected_macro, "per_fold": per_fold,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
