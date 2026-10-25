from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics_np(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred - target
    centered_pred = pred - pred.mean(axis=0, keepdims=True)
    centered_target = target - target.mean(axis=0, keepdims=True)
    pcc = np.sum(centered_pred * centered_target, axis=0) / np.maximum(
        np.sqrt(
            np.sum(centered_pred**2, axis=0)
            * np.sum(centered_target**2, axis=0)
        ),
        1e-8,
    )
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = 2 * covariance / np.maximum(
        pred.var(axis=0)
        + target.var(axis=0)
        + (pred_mean - target_mean) ** 2,
        1e-8,
    )
    return {
        "rmse_valence": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_valence": float(np.mean(np.abs(error[:, 0]))),
        "mae_arousal": float(np.mean(np.abs(error[:, 1]))),
        "pcc_valence": float(pcc[0]),
        "pcc_arousal": float(pcc[1]),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_ccc": float(np.mean(ccc)),
    }


class AudioFeatureDataset(Dataset):
    def __init__(self, feature_dir: Path, split: str):
        self.paths = []
        for path in sorted(feature_dir.glob("*.npz"), key=lambda item: int(item.stem)):
            with np.load(path) as data:
                if str(data["split"]) == split:
                    self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index]) as data:
            log_mel = data["log_mel"].astype(np.float32)
            handcrafted = data["handcrafted"].astype(np.float32)
            labels = data["labels"].astype(np.float32)
            song_id = int(data["song_id"])
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return (
            torch.from_numpy(log_mel),
            torch.from_numpy(handcrafted),
            torch.from_numpy(labels),
            song_id,
        )


def make_collate(spectral_frames_per_target: int):
    def collate(batch):
        resized_mels = []
        for mel, _, target, _ in batch:
            target_mel_frames = max(
                len(target), len(target) * spectral_frames_per_target
            )
            resized = F.interpolate(
                mel[None, :, :],
                size=target_mel_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            resized_mels.append(resized)
        mel_lengths = torch.tensor([mel.shape[1] for mel in resized_mels])
        target_lengths = torch.tensor([item[2].shape[0] for item in batch])
        max_mel = int(mel_lengths.max())
        max_target = int(target_lengths.max())
        mels = torch.zeros(len(batch), 1, batch[0][0].shape[0], max_mel)
        handcrafted = torch.zeros(
            len(batch), max_target, batch[0][1].shape[1]
        )
        labels = torch.zeros(len(batch), max_target, 2)
        mask = torch.zeros(len(batch), max_target, dtype=torch.bool)
        song_ids = []
        for i, ((_, hand, target, song_id), mel) in enumerate(
            zip(batch, resized_mels)
        ):
            mels[i, 0, :, : mel.shape[1]] = mel
            handcrafted[i, : len(hand)] = hand
            labels[i, : len(target)] = target
            mask[i, : len(target)] = True
            song_ids.append(song_id)
        return (
            mels,
            handcrafted,
            labels,
            mask,
            mel_lengths,
            target_lengths,
            song_ids,
        )

    return collate


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 8)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(x, 1)
        maximum = F.adaptive_max_pool2d(x, 1)
        x = x * torch.sigmoid(self.channel(average) + self.channel(maximum))
        spatial = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1
        )
        return x * torch.sigmoid(self.spatial(spatial))


class MultiScaleSpectralEncoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, channels, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                )
                for kernel in (3, 5, 7)
            ]
        )
        self.attention = ChannelSpatialAttention(channels * 3)

    def forward(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        encoded = self.attention(
            torch.cat([branch(mel) for branch in self.branches], dim=1)
        ).mean(dim=2)
        sequences = []
        for index in range(len(encoded)):
            valid = encoded[index : index + 1, :, : int(mel_lengths[index])]
            resized = F.interpolate(
                valid,
                size=int(target_lengths[index]),
                mode="linear",
                align_corners=False,
            )
            sequences.append(resized.squeeze(0).transpose(0, 1))
        return sequences


class VARegressor(nn.Module):
    def __init__(
        self, mode: str, handcrafted_dim: int, config: dict
    ):
        super().__init__()
        model_config = config["model"]
        self.mode = mode
        spectral_dim = int(model_config["cnn_channels"]) * 3
        projection_dim = int(model_config["handcrafted_projection"])
        self.spectral = MultiScaleSpectralEncoder(
            int(model_config["cnn_channels"])
        )
        self.handcrafted = nn.Sequential(
            nn.LayerNorm(handcrafted_dim),
            nn.Linear(handcrafted_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(float(model_config["dropout"])),
        )
        if mode == "logmel":
            input_dim = spectral_dim
        elif mode == "handcrafted":
            input_dim = projection_dim
        elif mode == "fusion":
            input_dim = spectral_dim + projection_dim
        else:
            raise ValueError(mode)
        self.temporal = nn.LSTM(
            input_dim,
            int(model_config["bilstm_hidden"]),
            batch_first=True,
            bidirectional=True,
        )
        temporal_dim = int(model_config["bilstm_hidden"]) * 2
        self.dropout = nn.Dropout(float(model_config["dropout"]))
        self.valence = nn.Linear(temporal_dim, 1)
        self.arousal = nn.Linear(temporal_dim, 1)

    def forward(
        self,
        mel: torch.Tensor,
        handcrafted: torch.Tensor,
        mel_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        spectral_sequences = None
        if self.mode in {"logmel", "fusion"}:
            spectral_sequences = self.spectral(mel, mel_lengths, target_lengths)
        inputs = []
        for index, length in enumerate(target_lengths.tolist()):
            hand = self.handcrafted(handcrafted[index, :length])
            if self.mode == "logmel":
                inputs.append(spectral_sequences[index])
            elif self.mode == "handcrafted":
                inputs.append(hand)
            else:
                inputs.append(torch.cat([spectral_sequences[index], hand], dim=-1))
        padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        temporal, _ = self.temporal(padded)
        temporal = self.dropout(temporal)
        return torch.cat([self.valence(temporal), self.arousal(temporal)], dim=-1)


def song_balanced_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    losses = []
    for index in range(len(prediction)):
        losses.append(((prediction[index][mask[index]] - target[index][mask[index]]) ** 2).mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    predictions = []
    targets = []
    losses = []
    for mel, handcrafted, labels, mask, mel_lengths, target_lengths, _ in loader:
        mel = mel.to(device)
        handcrafted = handcrafted.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        prediction = model(
            mel, handcrafted, mel_lengths.to(device), target_lengths.to(device)
        )
        losses.append(float(song_balanced_mse(prediction, labels, mask).cpu()))
        predictions.append(prediction[mask].cpu().numpy())
        targets.append(labels[mask].cpu().numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    return float(np.mean(losses)), metrics_np(pred, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spectral/feature VA model.")
    parser.add_argument("--mode", choices=["logmel", "handcrafted", "fusion"], required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with (ROOT / "configs" / "experiment.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    training = config["training"]
    set_seed(int(training["seed"]))
    datasets = {
        split: AudioFeatureDataset(ROOT / "data" / "features", split)
        for split in ("train", "val", "test")
    }
    if len(datasets["train"]) == 0:
        raise RuntimeError("No extracted train features.")
    batch_size = int(training["batch_size"])
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=min(2, len(dataset)) if args.smoke_test else batch_size,
            shuffle=split == "train" and not args.smoke_test,
            collate_fn=make_collate(
                int(config["model"]["spectral_frames_per_target"])
            ),
        )
        for split, dataset in datasets.items()
        if len(dataset)
    }
    loader = loaders["train"]
    batch = next(iter(loader))
    mel, handcrafted, labels, mask, mel_lengths, target_lengths, _ = batch
    model = VARegressor(args.mode, handcrafted.shape[-1], config)
    device = torch.device(args.device)
    model.to(device)
    mel = mel.to(device)
    handcrafted = handcrafted.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    prediction = model(mel, handcrafted, mel_lengths, target_lengths)
    loss = song_balanced_mse(prediction, labels, mask)
    loss.backward()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "songs": len(datasets["train"]),
                "mel_shape": list(mel.shape),
                "handcrafted_shape": list(handcrafted.shape),
                "target_shape": list(labels.shape),
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.detach()),
                "parameters": sum(p.numel() for p in model.parameters()),
                "status": "forward_backward_ok",
            },
            indent=2,
        )
    )
    if args.smoke_test:
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    result_dir = ROOT / "outputs" / args.mode
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / f"{args.mode}_seed{training['seed']}_best.pt"
    history = []
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        train_losses = []
        for mel, handcrafted, labels, mask, mel_lengths, target_lengths, _ in loaders["train"]:
            mel = mel.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            prediction = model(
                mel,
                handcrafted,
                mel_lengths.to(device),
                target_lengths.to(device),
            )
            train_loss = song_balanced_mse(prediction, labels, mask)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["grad_clip"])
            )
            optimizer.step()
            train_losses.append(float(train_loss.detach().cpu()))
        _, val_metrics = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"mode={args.mode} epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_ccc={val_metrics['mean_ccc']:.6f}"
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(training["patience"]):
                break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    summary = {
        "mode": args.mode,
        "seed": int(training["seed"]),
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint),
        "input_is_pure_audio": True,
        "uses_historical_emotion_features": False,
    }
    for split in ("train", "val", "test"):
        _, split_metrics = evaluate(model, loaders[split], device)
        summary.update(
            {f"{split}_{key}": value for key, value in split_metrics.items()}
        )
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (result_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
