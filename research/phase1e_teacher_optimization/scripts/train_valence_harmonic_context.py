from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_gated_fusion import MERT_ROOT, ROOT, set_seed
from train_multilayer_valence_fusion import load_aligned


HARMONIC_ROOT = ROOT / "data/harmonic_context"


def scalar_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    error = prediction - target
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    covariance = np.mean(pred_centered * target_centered)
    pcc = covariance / (
        np.sqrt(np.mean(pred_centered**2) * np.mean(target_centered**2)) + 1e-8
    )
    ccc = 2 * covariance / (
        prediction.var()
        + target.var()
        + (prediction.mean() - target.mean()) ** 2
        + 1e-8
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pcc": float(pcc),
        "ccc": float(ccc),
    }


def load_song(song_id: int, sequence_path: str | Path):
    layer6, final, _, target = load_aligned(song_id, sequence_path)
    with np.load(HARMONIC_ROOT / f"{song_id}.npz") as data:
        harmonic = data["harmonic_context"].astype(np.float32)
        harmonic_times = data["time_sec"].astype(np.float32)
    with np.load(sequence_path) as sequence:
        target_times = sequence["time_sec"].astype(np.float32)
    if harmonic.shape[0] != len(target_times) or not np.allclose(
        harmonic_times, target_times, atol=1e-4
    ):
        raise ValueError(f"Song {song_id}: harmonic time alignment failed.")
    if not np.isfinite(harmonic).all():
        raise ValueError(f"Song {song_id}: harmonic feature contains NaN/Inf.")
    return layer6, final, harmonic, target[:, :1]


def build_norm(index: pd.DataFrame, output: Path):
    if output.is_file():
        data = np.load(output)
        return tuple(
            data[key]
            for key in (
                "layer6_mean",
                "layer6_std",
                "final_mean",
                "final_std",
                "harmonic_mean",
                "harmonic_std",
            )
        )
    sums = squares = None
    count = 0
    for row in index[index["split"] == "train"].itertuples():
        features = load_song(int(row.song_id), row.sequence_path)[:3]
        if sums is None:
            sums = [values.astype(np.float64).sum(axis=0) for values in features]
            squares = [
                (values.astype(np.float64) ** 2).sum(axis=0) for values in features
            ]
        else:
            for i, values in enumerate(features):
                values = values.astype(np.float64)
                sums[i] += values.sum(axis=0)
                squares[i] += (values**2).sum(axis=0)
        count += len(features[0])
    means = [(total / count).astype(np.float32) for total in sums]
    stds = [
        np.sqrt(
            np.maximum(square / count - mean.astype(np.float64) ** 2, 1e-8)
        ).astype(np.float32)
        for square, mean in zip(squares, means)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        layer6_mean=means[0],
        layer6_std=stds[0],
        final_mean=means[1],
        final_std=stds[1],
        harmonic_mean=means[2],
        harmonic_std=stds[2],
        frame_count=count,
    )
    return means[0], stds[0], means[1], stds[1], means[2], stds[2]


class ValenceDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str, norms):
        self.rows = (
            index[index["split"] == split].sort_values("song_id").to_dict("records")
        )
        self.norms = norms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        song_id = int(row["song_id"])
        layer6, final, harmonic, target = load_song(song_id, row["sequence_path"])
        values = (
            (layer6 - self.norms[0]) / self.norms[1],
            (final - self.norms[2]) / self.norms[3],
            (harmonic - self.norms[4]) / self.norms[5],
            target,
        )
        return tuple(torch.from_numpy(value) for value in values) + (song_id,)


def collate(batch):
    lengths = torch.tensor([len(item[3]) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    arrays = [
        torch.zeros(len(batch), maximum, batch[0][i].shape[1]) for i in range(4)
    ]
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    song_ids = []
    for batch_index, item in enumerate(batch):
        length = len(item[3])
        for feature_index in range(4):
            arrays[feature_index][batch_index, :length] = item[feature_index]
        mask[batch_index, :length] = True
        song_ids.append(item[4])
    return (*arrays, mask, lengths, song_ids)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = values
        values = self.conv(values)
        values = self.norm(values)
        values = F.gelu(values)
        values = self.dropout(values)
        return values + residual


class HarmonicContextValenceRegressor(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], config: dict):
        super().__init__()
        projection = int(config["projection_dim"])
        dropout = float(config["dropout"])

        def projector(input_dim):
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, projection),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.projectors = nn.ModuleList(projector(dim) for dim in dimensions)
        self.gate = nn.Linear(projection * 3, projection * 3)
        self.projection = projection
        channels = int(config["tcn_channels"])
        self.tcn_input = nn.Conv1d(projection, channels, kernel_size=1)
        self.tcn = nn.Sequential(
            *[
                ResidualTemporalBlock(channels, int(dilation), dropout)
                for dilation in config["tcn_dilations"]
            ]
        )
        hidden = int(config["hidden_dim"])
        self.bilstm = nn.LSTM(
            channels, hidden, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden * 2, 1)

    def forward(self, layer6, final, harmonic):
        projected = [
            projector(values)
            for projector, values in zip(
                self.projectors, (layer6, final, harmonic)
            )
        ]
        concatenated = torch.cat(projected, dim=-1)
        logits = self.gate(concatenated).reshape(
            *concatenated.shape[:-1], 3, self.projection
        )
        weights = torch.softmax(logits, dim=-2)
        fused = (weights * torch.stack(projected, dim=-2)).sum(dim=-2)
        temporal = self.tcn(self.tcn_input(fused.transpose(1, 2))).transpose(1, 2)
        hidden, _ = self.bilstm(temporal)
        return self.output(self.dropout(hidden))


def valence_loss(prediction, target, mask, config):
    losses = []
    for index in range(len(prediction)):
        pred = prediction[index][mask[index]]
        truth = target[index][mask[index]]
        huber = F.huber_loss(
            pred, truth, delta=float(config["huber_delta"])
        )
        mse = F.mse_loss(pred, truth)
        pred_centered = pred - pred.mean()
        truth_centered = truth - truth.mean()
        covariance = (pred_centered * truth_centered).mean()
        pcc = covariance / (
            torch.sqrt(
                (pred_centered**2).mean() * (truth_centered**2).mean()
                + 1e-8
            )
        )
        ccc = 2 * covariance / (
            pred_centered.var(unbiased=False)
            + truth_centered.var(unbiased=False)
            + (pred.mean() - truth.mean()) ** 2
            + 1e-8
        )
        derivative = F.mse_loss(pred[1:] - pred[:-1], truth[1:] - truth[:-1])
        losses.append(
            huber
            + float(config["mse"]) * mse
            + float(config["pearson"]) * (1 - pcc)
            + float(config["ccc"]) * (1 - ccc)
            + float(config["derivative"]) * derivative
        )
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for layer6, final, harmonic, target, mask, _, _ in loader:
        inputs = [values.to(device) for values in (layer6, final, harmonic)]
        prediction = model(*inputs)
        predictions.append(prediction[mask.to(device)].cpu().numpy())
        targets.append(target[mask].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return scalar_metrics(prediction, target), prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/valence_harmonic_context.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    index = pd.read_csv(MERT_ROOT / "data/datasets/mert_va/sequence_index.csv")
    index = index[index["split"].isin(["train", "val"])].copy()
    output_dir = ROOT / "outputs" / config["trial_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    norms = build_norm(index, output_dir / "feature_norm.npz")
    datasets = {
        split: ValenceDataset(index, split, norms) for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            collate_fn=collate,
        )
        for split, dataset in datasets.items()
    }
    first = datasets["train"][0]
    model = HarmonicContextValenceRegressor(
        (first[0].shape[1], first[1].shape[1], first[2].shape[1]), config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = output_dir / "best.pt"
    history = []
    best_rmse = float("inf")
    best_epoch = bad_epochs = 0
    print(
        json.dumps(
            {
                "trial_id": config["trial_id"],
                "device": str(device),
                "songs": {key: len(value) for key, value in datasets.items()},
                "parameters": sum(p.numel() for p in model.parameters()),
                "selection_metric": "valence_rmse",
                "test_locked": True,
            },
            indent=2,
        )
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for layer6, final, harmonic, target, mask, _, _ in loaders["train"]:
            inputs = [values.to(device) for values in (layer6, final, harmonic)]
            target, mask = target.to(device), mask.to(device)
            optimizer.zero_grad()
            prediction = model(*inputs)
            loss = valence_loss(prediction, target, mask, config["loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _ = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"valence_rmse={validation['rmse']:.6f} "
            f"pcc={validation['pcc']:.6f} ccc={validation['ccc']:.6f}"
        )
        if validation["rmse"] < best_rmse:
            best_rmse, best_epoch, bad_epochs = validation["rmse"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_metrics, train_prediction = evaluate(model, loaders["train"], device)
    val_metrics, val_prediction = evaluate(model, loaders["val"], device)
    summary = {
        "trial_id": config["trial_id"],
        "seed": int(config["seed"]),
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint),
        "test_evaluated": False,
        "selection_metric": "valence_rmse",
        "config": config,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    np.save(output_dir / "train_pred.npy", train_prediction.astype(np.float32))
    np.save(output_dir / "val_pred.npy", val_prediction.astype(np.float32))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
