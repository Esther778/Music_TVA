from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset

from train_gated_fusion import (
    FEATURE_ROOT,
    MERT_ROOT,
    ROOT,
    metrics,
    set_seed,
    song_balanced_loss,
)
from train_layer6_gated_fusion import LAYER6_ROOT, time_indices


def load_aligned(song_id: int, sequence_path: str | Path):
    with np.load(sequence_path) as sequence:
        times = sequence["time_sec"].astype(np.float32)
        final = sequence["X"].astype(np.float32)
        target = sequence["Y"].astype(np.float32)
    with np.load(LAYER6_ROOT / f"{song_id}.npz") as layer_file:
        layer_times = layer_file["time_sec"].astype(np.float32)
        layer6 = layer_file["embeddings"][:, 0].astype(np.float32)
    layer6 = layer6[time_indices(layer_times, times, song_id)]
    with np.load(FEATURE_ROOT / f"{song_id}.npz") as hand_file:
        hand_times = hand_file["target_time_sec"].astype(np.float32)
        hand = hand_file["handcrafted"].astype(np.float32)
        hand_target = hand_file["labels"].astype(np.float32)
    hand_indices = time_indices(hand_times, times, song_id)
    hand = hand[hand_indices]
    hand_target = hand_target[hand_indices]
    if not np.allclose(target, hand_target, atol=1e-6):
        raise ValueError(f"Song {song_id}: target mismatch.")
    arrays = (layer6, final, hand, target)
    if len({len(values) for values in arrays}) != 1:
        raise ValueError(f"Song {song_id}: aligned lengths differ.")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError(f"Song {song_id}: NaN or Inf after alignment.")
    return arrays


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
                "hand_mean",
                "hand_std",
            )
        )
    sums = squares = None
    count = 0
    for row in index[index["split"] == "train"].itertuples():
        features = load_aligned(int(row.song_id), row.sequence_path)[:3]
        if sums is None:
            sums = [x.astype(np.float64).sum(axis=0) for x in features]
            squares = [(x.astype(np.float64) ** 2).sum(axis=0) for x in features]
        else:
            for i, values in enumerate(features):
                values = values.astype(np.float64)
                sums[i] += values.sum(axis=0)
                squares[i] += (values**2).sum(axis=0)
        count += len(features[0])
    means = [(total / count).astype(np.float32) for total in sums]
    stds = [
        np.sqrt(
            np.maximum(
                square / count - mean.astype(np.float64) ** 2,
                1e-8,
            )
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
        hand_mean=means[2],
        hand_std=stds[2],
        frame_count=count,
    )
    return means[0], stds[0], means[1], stds[1], means[2], stds[2]


class MultilayerDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        norms,
        valence_target_sigma_frames: float = 0.0,
    ):
        self.rows = (
            index[index["split"] == split].sort_values("song_id").to_dict("records")
        )
        self.norms = norms
        self.valence_target_sigma_frames = valence_target_sigma_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        song_id = int(row["song_id"])
        layer6, final, hand, target = load_aligned(song_id, row["sequence_path"])
        if self.valence_target_sigma_frames > 0:
            target = target.copy()
            target[:, 0] = gaussian_filter1d(
                target[:, 0],
                sigma=self.valence_target_sigma_frames,
                mode="nearest",
            )
        layer6 = (layer6 - self.norms[0]) / self.norms[1]
        final = (final - self.norms[2]) / self.norms[3]
        hand = (hand - self.norms[4]) / self.norms[5]
        return tuple(
            torch.from_numpy(x) for x in (layer6, final, hand, target)
        ) + (song_id,)


def collate(batch):
    lengths = torch.tensor([len(item[3]) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    outputs = [
        torch.zeros(len(batch), maximum, batch[0][i].shape[1])
        for i in range(4)
    ]
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    song_ids = []
    for batch_index, item in enumerate(batch):
        length = len(item[3])
        for feature_index in range(4):
            outputs[feature_index][batch_index, :length] = item[feature_index]
        mask[batch_index, :length] = True
        song_ids.append(item[4])
    return (*outputs, mask, lengths, song_ids)


class ThreeWayDualBranchRegressor(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], config: dict):
        super().__init__()
        projection = int(config["projection_dim"])
        dropout = float(config["dropout"])

        def projector(input_dim: int):
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, projection),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.projectors = nn.ModuleList(projector(dimension) for dimension in dimensions)
        self.valence_gate = nn.Linear(projection * 3, projection * 3)
        self.arousal_gate = nn.Linear(projection * 3, projection * 3)
        hidden = int(config["hidden_dim"])
        self.valence_lstm = nn.LSTM(
            projection, hidden, batch_first=True, bidirectional=True
        )
        self.arousal_lstm = nn.LSTM(
            projection, hidden, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.valence_output = nn.Linear(hidden * 2, 1)
        self.arousal_output = nn.Linear(hidden * 2, 1)
        self.projection = projection

    def fuse(self, features: list[torch.Tensor], gate: nn.Module):
        concatenated = torch.cat(features, dim=-1)
        logits = gate(concatenated).reshape(
            *concatenated.shape[:-1], 3, self.projection
        )
        weights = torch.softmax(logits, dim=-2)
        stacked = torch.stack(features, dim=-2)
        return (weights * stacked).sum(dim=-2)

    def forward(self, layer6, final, hand):
        projected = [
            projector(values)
            for projector, values in zip(self.projectors, (layer6, final, hand))
        ]
        valence_hidden, _ = self.valence_lstm(
            self.fuse(projected, self.valence_gate)
        )
        arousal_hidden, _ = self.arousal_lstm(
            self.fuse(projected, self.arousal_gate)
        )
        return torch.cat(
            [
                self.valence_output(self.dropout(valence_hidden)),
                self.arousal_output(self.dropout(arousal_hidden)),
            ],
            dim=-1,
        )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for layer6, final, hand, target, mask, _, _ in loader:
        inputs = [values.to(device) for values in (layer6, final, hand)]
        prediction = model(*inputs)
        predictions.append(prediction[mask.to(device)].cpu().numpy())
        targets.append(target[mask].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return metrics(prediction, target), prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/multilayer_valence_fusion.json",
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
    train_sigma = float(config.get("train_valence_target_sigma_frames", 0.0))
    datasets = {
        split: MultilayerDataset(
            index,
            split,
            norms,
            valence_target_sigma_frames=train_sigma if split == "train" else 0.0,
        )
        for split in ("train", "val")
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
    model = ThreeWayDualBranchRegressor(
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
    selection_metric = config.get("selection_metric", "rmse")
    if selection_metric not in {"rmse", "valence_rmse"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    print(
        json.dumps(
            {
                "trial_id": config["trial_id"],
                "device": str(device),
                "songs": {key: len(value) for key, value in datasets.items()},
                "parameters": sum(p.numel() for p in model.parameters()),
                "selection_metric": selection_metric,
                "train_valence_target_sigma_frames": train_sigma,
                "test_locked": True,
            },
            indent=2,
        )
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for layer6, final, hand, target, mask, _, _ in loaders["train"]:
            inputs = [values.to(device) for values in (layer6, final, hand)]
            target, mask = target.to(device), mask.to(device)
            optimizer.zero_grad()
            prediction = model(*inputs)
            loss = song_balanced_loss(prediction, target, mask, config["loss"])
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
            f"val_rmse={validation['rmse']:.6f} "
            f"val_pcc={validation['mean_pcc']:.6f} "
            f"val_ccc={validation['mean_ccc']:.6f}"
        )
        selected_value = (
            validation["rmse"]
            if selection_metric == "rmse"
            else validation["rmse_valence"]
        )
        if selected_value < best_rmse:
            best_rmse, best_epoch, bad_epochs = selected_value, epoch, 0
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
        "selection_metric": selection_metric,
        "best_selection_value": best_rmse,
        "feature_layers": [6, "final"],
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
