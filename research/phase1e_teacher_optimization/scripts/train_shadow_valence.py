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

from train_gated_fusion import FEATURE_ROOT, MERT_ROOT, ROOT, set_seed
from train_layer6_gated_fusion import LAYER6_ROOT, time_indices
from train_valence_harmonic_context import scalar_metrics


def moving_average(values: torch.Tensor, frames: int) -> torch.Tensor:
    padding_left = (frames - 1) // 2
    padding_right = frames // 2
    padded = F.pad(
        values.transpose(0, 1).unsqueeze(0),
        (padding_left, padding_right),
        mode="replicate",
    )
    return F.avg_pool1d(padded, kernel_size=frames, stride=1).squeeze(0).transpose(0, 1)


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor, ccc: bool):
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    covariance = (pred_centered * target_centered).mean()
    if ccc:
        score = 2 * covariance / (
            pred_centered.var(unbiased=False)
            + target_centered.var(unbiased=False)
            + (prediction.mean() - target.mean()) ** 2
            + 1e-8
        )
    else:
        score = covariance / (
            torch.sqrt(
                (pred_centered**2).mean() * (target_centered**2).mean() + 1e-8
            )
        )
    return 1 - score


def load_song(song_id: int, sequence_path: str | Path):
    with np.load(sequence_path) as sequence:
        times = sequence["time_sec"].astype(np.float32)
        final = sequence["X"].astype(np.float32)
        target = sequence["Y"][:, :1].astype(np.float32)
    with np.load(LAYER6_ROOT / f"{song_id}.npz") as layer_file:
        layer_times = layer_file["time_sec"].astype(np.float32)
        layer6 = layer_file["embeddings"][:, 0].astype(np.float32)
    layer6 = layer6[time_indices(layer_times, times, song_id)]
    with np.load(FEATURE_ROOT / f"{song_id}.npz") as hand_file:
        hand_times = hand_file["target_time_sec"].astype(np.float32)
        hand = hand_file["handcrafted"].astype(np.float32)
    hand = hand[time_indices(hand_times, times, song_id)]
    arrays = (layer6, final, hand, target)
    if len({len(values) for values in arrays}) != 1:
        raise ValueError(f"Song {song_id}: feature lengths differ.")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError(f"Song {song_id}: non-finite value.")
    return arrays


def build_norm(rows: pd.DataFrame, output: Path, training_split: str = "train_sub"):
    if output.is_file():
        data = np.load(output)
        return tuple(
            data[key]
            for key in (
                "layer6_mean", "layer6_std", "final_mean", "final_std",
                "hand_mean", "hand_std",
            )
        )
    sums = squares = None
    count = 0
    for row in rows[rows["shadow_split"] == training_split].itertuples():
        features = load_song(int(row.song_id), row.sequence_path)[:3]
        if sums is None:
            sums = [x.astype(np.float64).sum(axis=0) for x in features]
            squares = [(x.astype(np.float64) ** 2).sum(axis=0) for x in features]
        else:
            for index, values in enumerate(features):
                values = values.astype(np.float64)
                sums[index] += values.sum(axis=0)
                squares[index] += (values**2).sum(axis=0)
        count += len(features[0])
    means = [(value / count).astype(np.float32) for value in sums]
    stds = [
        np.sqrt(np.maximum(square / count - mean.astype(np.float64) ** 2, 1e-8)).astype(np.float32)
        for square, mean in zip(squares, means)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        layer6_mean=means[0], layer6_std=stds[0],
        final_mean=means[1], final_std=stds[1],
        hand_mean=means[2], hand_std=stds[2], frame_count=count,
    )
    return means[0], stds[0], means[1], stds[1], means[2], stds[2]


class ShadowDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, split: str, norms):
        self.rows = rows[rows["shadow_split"] == split].sort_values("song_id").to_dict("records")
        self.norms = norms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        song_id = int(row["song_id"])
        layer6, final, hand, target = load_song(song_id, row["sequence_path"])
        values = (
            (layer6 - self.norms[0]) / self.norms[1],
            (final - self.norms[2]) / self.norms[3],
            (hand - self.norms[4]) / self.norms[5],
            target,
        )
        return tuple(torch.from_numpy(x) for x in values) + (song_id,)


def collate(batch):
    lengths = torch.tensor([len(item[3]) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    outputs = [torch.zeros(len(batch), maximum, batch[0][i].shape[1]) for i in range(4)]
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    song_ids = []
    for batch_index, item in enumerate(batch):
        length = len(item[3])
        for feature_index in range(4):
            outputs[feature_index][batch_index, :length] = item[feature_index]
        mask[batch_index, :length] = True
        song_ids.append(item[4])
    return (*outputs, mask, lengths, song_ids)


class SharedEncoder(nn.Module):
    def __init__(self, dimensions, config):
        super().__init__()
        projection = int(config["projection_dim"])
        dropout = float(config["dropout"])
        self.projectors = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, projection),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for dimension in dimensions
        )
        self.gate = nn.Linear(projection * 3, projection * 3)
        self.projection = projection
        hidden = int(config["hidden_dim"])
        self.temporal = nn.LSTM(projection, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden * 2

    def forward(self, layer6, final, hand):
        projected = [
            projector(values)
            for projector, values in zip(self.projectors, (layer6, final, hand))
        ]
        joined = torch.cat(projected, dim=-1)
        weights = torch.softmax(
            self.gate(joined).reshape(*joined.shape[:-1], 3, self.projection),
            dim=-2,
        )
        fused = (weights * torch.stack(projected, dim=-2)).sum(dim=-2)
        hidden, _ = self.temporal(fused)
        return self.dropout(hidden)


class BaselineRegressor(nn.Module):
    def __init__(self, dimensions, config):
        super().__init__()
        self.encoder = SharedEncoder(dimensions, config)
        self.output = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, layer6, final, hand):
        return {"prediction": self.output(self.encoder(layer6, final, hand))}


class TrendResidualRegressor(nn.Module):
    def __init__(self, dimensions, config):
        super().__init__()
        self.encoder = SharedEncoder(dimensions, config)
        dimension = self.encoder.output_dim
        self.trend5 = nn.Linear(dimension, 1)
        self.trend10 = nn.Linear(dimension, 1)
        self.residual = nn.Linear(dimension, 1)
        self.residual_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, layer6, final, hand):
        hidden = self.encoder(layer6, final, hand)
        trend5 = self.trend5(hidden)
        trend10 = self.trend10(hidden)
        residual = self.residual(hidden)
        scale = torch.sigmoid(self.residual_logit)
        prediction = 0.5 * (trend5 + trend10) + scale * residual
        return {
            "prediction": prediction,
            "trend5": trend5,
            "trend10": trend10,
            "residual": residual,
            "residual_scale": scale,
        }


def baseline_loss(output, target, mask, config):
    losses = []
    for index in range(len(target)):
        pred = output["prediction"][index][mask[index]]
        truth = target[index][mask[index]]
        losses.append(
            float(config["huber"]) * F.huber_loss(pred, truth, delta=0.15)
            + float(config["mse"]) * F.mse_loss(pred, truth)
            + float(config["ccc"]) * correlation_loss(pred, truth, True)
            + float(config["pearson"]) * correlation_loss(pred, truth, False)
        )
    return torch.stack(losses).mean()


def trend_loss(output, target, mask, config):
    losses = []
    for index in range(len(target)):
        truth = target[index][mask[index]]
        pred = output["prediction"][index][mask[index]]
        trend5_target = moving_average(truth, 10)
        trend10_target = moving_average(truth, 20)
        trend5 = output["trend5"][index][mask[index]]
        trend10 = output["trend10"][index][mask[index]]
        residual_target = truth - 0.5 * (trend5_target + trend10_target)
        residual = output["residual"][index][mask[index]] * output["residual_scale"]
        loss = (
            float(config["huber"]) * F.huber_loss(pred, truth, delta=0.15)
            + float(config["mse"]) * F.mse_loss(pred, truth)
            + float(config["frame_ccc"]) * correlation_loss(pred, truth, True)
            + float(config["frame_pearson"]) * correlation_loss(pred, truth, False)
            + float(config["trend_5s"]) * F.mse_loss(trend5, trend5_target)
            + float(config["trend_10s"]) * F.mse_loss(trend10, trend10_target)
            + float(config["residual"]) * F.mse_loss(residual, residual_target)
            + float(config["window_5s_ccc"])
            * correlation_loss(moving_average(pred, 10), trend5_target, True)
            + float(config["window_10s_ccc"])
            * correlation_loss(moving_average(pred, 20), trend10_target, True)
        )
        if len(truth) > 10:
            true_diff = truth[10:] - truth[:-10]
            pred_diff = pred[10:] - pred[:-10]
            valid = true_diff.abs() > 0.08
            if bool(valid.any()):
                rank = F.softplus(-torch.sign(true_diff[valid]) * pred_diff[valid]).mean()
                loss = loss + float(config["ranking"]) * rank
        variance = (pred.std(unbiased=False) - truth.std(unbiased=False)) ** 2
        loss = loss + float(config["variance"]) * variance
        losses.append(loss)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets, song_ids = [], [], []
    for layer6, final, hand, target, mask, _, ids in loader:
        output = model(layer6.to(device), final.to(device), hand.to(device))
        predictions.append(output["prediction"][mask.to(device)].cpu().numpy())
        targets.append(target[mask].numpy())
        song_ids.extend(ids)
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return scalar_metrics(prediction, target), prediction, target, song_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "trend"], required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/shadow_valence_experiment.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fit-full", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    index = pd.read_csv(MERT_ROOT / "data/datasets/mert_va/sequence_index.csv")
    shadow = pd.read_csv(ROOT / "data/shadow_split.csv")[["song_id", "shadow_split"]]
    rows = index[index["split"] == "train"].merge(shadow, on="song_id")
    if args.fit_full:
        rows["shadow_split"] = "full_train"
    trial_id = (
        f"full_valence_{args.mode}" if args.fit_full else f"shadow_valence_{args.mode}"
    )
    output_dir = ROOT / "outputs" / trial_id
    output_dir.mkdir(parents=True, exist_ok=True)
    training_split = "full_train" if args.fit_full else "train_sub"
    norm_path = (
        ROOT / "outputs/full_candidate_feature_norm.npz"
        if args.fit_full
        else ROOT / "outputs/shadow_feature_norm.npz"
    )
    norms = build_norm(rows, norm_path, training_split=training_split)
    dataset_splits = ("full_train",) if args.fit_full else ("train_sub", "shadow_val")
    datasets = {split: ShadowDataset(rows, split, norms) for split in dataset_splits}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split in {"train_sub", "full_train"},
            collate_fn=collate,
        )
        for split, dataset in datasets.items()
    }
    first = datasets[training_split][0]
    dimensions = (first[0].shape[1], first[1].shape[1], first[2].shape[1])
    model_class = BaselineRegressor if args.mode == "baseline" else TrendResidualRegressor
    model = model_class(dimensions, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = output_dir / "best.pt"
    history = []
    best_rmse, best_epoch, bad_epochs = float("inf"), 0, 0
    print(json.dumps({
        "trial_id": trial_id,
        "mode": args.mode,
        "songs": {key: len(value) for key, value in datasets.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "selection": "fixed shadow-selected epoch" if args.fit_full else "shadow_val_rmse",
        "frozen_validation_used": False,
        "test_locked": True,
    }, indent=2))
    fixed_epochs = 21 if args.mode == "baseline" else 36
    maximum_epochs = fixed_epochs if args.fit_full else int(config["epochs"])
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        losses = []
        for layer6, final, hand, target, mask, _, _ in loaders[training_split]:
            layer6, final, hand = layer6.to(device), final.to(device), hand.to(device)
            target, mask = target.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(layer6, final, hand)
            if args.mode == "baseline":
                loss = baseline_loss(output, target, mask, config["baseline_loss"])
            else:
                loss = trend_loss(output, target, mask, config["trend_loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        if args.fit_full:
            validation = None
            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        else:
            validation, _, _, _ = evaluate(model, loaders["shadow_val"], device)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(row)
        if args.fit_full:
            print(f"epoch={epoch:03d} train_loss={row['train_loss']:.6f}")
            if epoch == maximum_epochs:
                torch.save(model.state_dict(), checkpoint)
                best_epoch = epoch
        else:
            print(
                f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
                f"shadow_rmse={validation['rmse']:.6f} "
                f"pcc={validation['pcc']:.6f} ccc={validation['ccc']:.6f}"
            )
        if not args.fit_full and validation["rmse"] < best_rmse:
            best_rmse, best_epoch, bad_epochs = validation["rmse"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        elif not args.fit_full:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_metrics, train_pred, train_target, _ = evaluate(
        model, loaders[training_split], device
    )
    summary = {
        "trial_id": trial_id,
        "mode": args.mode,
        "seed": int(config["seed"]),
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint),
        "selection_split": "shadow_val" if not args.fit_full else "shadow_val_fixed_epoch",
        "fit_full_train": bool(args.fit_full),
        "frozen_validation_evaluated": False,
        "test_evaluated": False,
        **{f"train_{key}": value for key, value in train_metrics.items()},
    }
    np.save(output_dir / "train_pred.npy", train_pred.astype(np.float32))
    np.save(output_dir / "train_target.npy", train_target.astype(np.float32))
    if not args.fit_full:
        shadow_metrics, shadow_pred, shadow_target, _ = evaluate(
            model, loaders["shadow_val"], device
        )
        summary.update(
            {f"shadow_{key}": value for key, value in shadow_metrics.items()}
        )
        np.save(output_dir / "shadow_pred.npy", shadow_pred.astype(np.float32))
        np.save(output_dir / "shadow_target.npy", shadow_target.astype(np.float32))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
