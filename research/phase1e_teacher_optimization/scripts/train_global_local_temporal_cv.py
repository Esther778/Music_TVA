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

from cross_validate_mert_layer_mix import (
    INDEX,
    ROOT,
    macro_metrics,
    window_metrics,
)
from train_layer6_gated_fusion import LAYER6_ROOT, time_indices
from train_valence_harmonic_context import scalar_metrics
from train_gated_fusion import FEATURE_ROOT
from build_harmonic_context_features import MAJOR, MINOR, template_strength


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_song(song_id: int, sequence_path: str | Path):
    with np.load(sequence_path) as sequence:
        times = sequence["time_sec"].astype(np.float32)
        target = sequence["Y"][:, :1].astype(np.float32)
    with np.load(LAYER6_ROOT / f"{song_id}.npz") as layer_file:
        layer_times = layer_file["time_sec"].astype(np.float32)
        features = layer_file["embeddings"][:, 0].astype(np.float32)
    features = features[time_indices(layer_times, times, song_id)]
    if len(features) != len(target):
        raise ValueError(f"Song {song_id}: feature/target length mismatch.")
    return features, target


def build_norm(rows: pd.DataFrame, output: Path):
    if output.is_file():
        data = np.load(output)
        return data["mean"], data["std"]
    count = 0
    total = square = None
    for row in rows.itertuples():
        features, _ = load_song(int(row.song_id), row.sequence_path)
        values = features.astype(np.float64)
        if total is None:
            total = values.sum(axis=0)
            square = (values**2).sum(axis=0)
        else:
            total += values.sum(axis=0)
            square += (values**2).sum(axis=0)
        count += len(values)
    mean = (total / count).astype(np.float32)
    std = np.sqrt(
        np.maximum(square / count - mean.astype(np.float64) ** 2, 1e-8)
    ).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mean=mean, std=std, frame_count=count)
    return mean, std


class SongDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        auxiliary_music: bool = False,
    ):
        self.rows = rows.sort_values("song_id").to_dict("records")
        self.mean = mean
        self.std = std
        self.auxiliary_music = auxiliary_music

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        song_id = int(row["song_id"])
        features, target = load_song(song_id, row["sequence_path"])
        features = (features - self.mean) / self.std
        if self.auxiliary_music:
            with np.load(FEATURE_ROOT / f"{song_id}.npz") as hand_file:
                chroma = hand_file["handcrafted"][:, 31:43].astype(np.float32)
            major_scores = template_strength(
                chroma.mean(axis=0, keepdims=True), MAJOR
            )[0]
            minor_scores = template_strength(
                chroma.mean(axis=0, keepdims=True), MINOR
            )[0]
            key_mode = int(
                np.argmax(np.concatenate([major_scores, minor_scores]))
            )
            chord_change = np.zeros((len(chroma), 1), dtype=np.float32)
            chord_change[1:, 0] = np.linalg.norm(
                chroma[1:] - chroma[:-1], axis=1
            )
            if len(chord_change) != len(target):
                raise ValueError(f"Song {song_id}: auxiliary length mismatch.")
            return (
                torch.from_numpy(features),
                torch.from_numpy(target),
                song_id,
                key_mode,
                torch.from_numpy(chord_change),
            )
        return torch.from_numpy(features), torch.from_numpy(target), song_id


def collate(batch):
    lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    features = torch.zeros(len(batch), maximum, batch[0][0].shape[1])
    target = torch.zeros(len(batch), maximum, 1)
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    song_ids = []
    auxiliary_music = len(batch[0]) == 5
    key_modes = (
        torch.zeros(len(batch), dtype=torch.long) if auxiliary_music else None
    )
    chord_change = (
        torch.zeros(len(batch), maximum, 1) if auxiliary_music else None
    )
    for index, item in enumerate(batch):
        values, labels, song_id = item[:3]
        length = len(labels)
        features[index, :length] = values
        target[index, :length] = labels
        mask[index, :length] = True
        song_ids.append(song_id)
        if auxiliary_music:
            key_modes[index] = int(item[3])
            chord_change[index, :length] = item[4]
    if auxiliary_music:
        return (
            features,
            target,
            mask,
            lengths,
            song_ids,
            key_modes,
            chord_change,
        )
    return features, target, mask, lengths, song_ids


class GlobalLocalTemporalRegressor(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        projection = int(config["projection_dim"])
        dropout = float(config["dropout"])
        bottleneck = int(config.get("adapter_bottleneck", 0))
        rng_state = torch.random.get_rng_state()
        self.adapter = (
            ResidualEmbeddingAdapter(
                input_dim,
                bottleneck,
                float(config.get("adapter_dropout", 0.0)),
            )
            if bottleneck > 0
            else nn.Identity()
        )
        if bottleneck > 0:
            torch.random.set_rng_state(rng_state)
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(projection, 1)
        self.global_head = nn.Sequential(
            nn.Linear(projection, projection // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection // 2, 1),
        )
        hidden = int(config["hidden_dim"])
        self.local_temporal = nn.LSTM(
            projection, hidden, batch_first=True, bidirectional=True
        )
        self.local_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-0.5))
        self.auxiliary_music = bool(config.get("auxiliary_music", False))
        if self.auxiliary_music:
            self.key_mode_head = nn.Linear(projection, 24)
            self.chord_change_head = nn.Linear(hidden * 2, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        adapted = self.adapter(features)
        projected = self.project(adapted)
        scores = self.attention(projected).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        global_embedding = (weights * projected).sum(dim=1)
        global_prediction = self.global_head(global_embedding)
        centered_features = projected - global_embedding.unsqueeze(1)
        local_hidden, _ = self.local_temporal(centered_features)
        local_raw = self.local_head(local_hidden)
        local_raw = local_raw * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        local_mean = local_raw.sum(dim=1, keepdim=True) / lengths
        local_prediction = (local_raw - local_mean) * mask.unsqueeze(-1)
        residual_scale = torch.sigmoid(self.residual_logit)
        prediction = global_prediction.unsqueeze(1) + residual_scale * local_prediction
        output = {
            "prediction": prediction,
            "global": global_prediction,
            "global_embedding": global_embedding,
            "local": local_prediction,
            "residual_scale": residual_scale,
            "attention": weights,
        }
        if self.auxiliary_music:
            output["key_mode_logits"] = self.key_mode_head(global_embedding)
            output["chord_change"] = F.softplus(
                self.chord_change_head(local_hidden)
            )
        return output


class ResidualEmbeddingAdapter(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck)
        self.up = nn.Linear(bottleneck, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor):
        residual = self.up(
            self.dropout(F.gelu(self.down(self.norm(features))))
        )
        return features + torch.sigmoid(self.gate_logit) * residual


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
        score = covariance / torch.sqrt(
            (pred_centered**2).mean() * (target_centered**2).mean() + 1e-8
        )
    return 1 - score


def loss_function(
    output,
    target,
    mask,
    config,
    key_modes: torch.Tensor | None = None,
    chord_change: torch.Tensor | None = None,
):
    losses = []
    for index in range(len(target)):
        truth = target[index][mask[index]]
        prediction = output["prediction"][index][mask[index]]
        true_mean = truth.mean(dim=0)
        true_local = truth - true_mean
        predicted_local = (
            output["residual_scale"] * output["local"][index][mask[index]]
        )
        loss = (
            float(config["frame_huber"])
            * F.huber_loss(prediction, truth, delta=0.15)
            + float(config["frame_mse"]) * F.mse_loss(prediction, truth)
            + float(config["global_mse"])
            * F.mse_loss(output["global"][index], true_mean)
            + float(config["local_mse"]) * F.mse_loss(predicted_local, true_local)
            + float(config["local_pearson"])
            * correlation_loss(predicted_local, true_local, False)
            + float(config["local_ccc"])
            * correlation_loss(predicted_local, true_local, True)
        )
        if len(truth) > 1:
            loss = loss + float(config["derivative"]) * F.mse_loss(
                prediction[1:] - prediction[:-1],
                truth[1:] - truth[:-1],
            )
        loss = loss + float(config["variance"]) * (
            prediction.std(unbiased=False) - truth.std(unbiased=False)
        ) ** 2
        losses.append(loss)
    total = torch.stack(losses).mean()
    metric_weight = float(config.get("metric_weight", 0.0))
    if metric_weight > 0:
        song_means = torch.stack(
            [target[index][mask[index]].mean() for index in range(len(target))]
        )
        embedding = F.normalize(output["global_embedding"], dim=-1)
        pairwise_distance = torch.cdist(embedding, embedding, p=2)
        triplets = []
        positive_margin = float(config["positive_valence_margin"])
        negative_margin = float(config["negative_valence_margin"])
        margin = float(config["triplet_margin"])
        for anchor in range(len(song_means)):
            label_distance = (song_means - song_means[anchor]).abs()
            positive = (label_distance <= positive_margin)
            positive[anchor] = False
            negative = label_distance >= negative_margin
            if bool(positive.any()) and bool(negative.any()):
                hardest_positive = pairwise_distance[anchor][positive].max()
                hardest_negative = pairwise_distance[anchor][negative].min()
                triplets.append(
                    F.relu(hardest_positive - hardest_negative + margin)
                )
        if triplets:
            total = total + metric_weight * torch.stack(triplets).mean()
    if key_modes is not None and chord_change is not None:
        total = total + float(
            config.get("key_mode_weight", 0.0)
        ) * F.cross_entropy(output["key_mode_logits"], key_modes)
        total = total + float(
            config.get("chord_change_weight", 0.0)
        ) * F.smooth_l1_loss(
            output["chord_change"][mask],
            chord_change[mask],
            beta=0.05,
        )
    return total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets, song_ids = [], [], []
    for batch in loader:
        features, target, mask, _, ids = batch[:5]
        features, mask = features.to(device), mask.to(device)
        output = model(features, mask)
        predictions.append(output["prediction"][mask].cpu().numpy().reshape(-1))
        targets.append(target[mask.cpu()].numpy().reshape(-1))
        song_ids.extend(ids)
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    expanded_ids = []
    dataset_rows = loader.dataset.rows
    for row in dataset_rows:
        with np.load(row["sequence_path"]) as data:
            expanded_ids.append(
                np.full(len(data["Y"]), int(row["song_id"]), dtype=np.int32)
            )
    expanded_ids = np.concatenate(expanded_ids)
    micro = scalar_metrics(prediction, target)
    macro = macro_metrics(prediction, target, expanded_ids)
    window5 = window_metrics(prediction, target, expanded_ids, 10)
    window10 = window_metrics(prediction, target, expanded_ids, 20)
    centered_prediction, centered_target = [], []
    mean_prediction, mean_target = [], []
    first_prediction, first_target = [], []
    for song_id in np.unique(expanded_ids):
        song_mask = expanded_ids == song_id
        pred_song = prediction[song_mask]
        target_song = target[song_mask]
        centered_prediction.append(pred_song - pred_song.mean())
        centered_target.append(target_song - target_song.mean())
        mean_prediction.append(float(pred_song.mean()))
        mean_target.append(float(target_song.mean()))
        if len(pred_song) > 1:
            first_prediction.append(np.diff(pred_song))
            first_target.append(np.diff(target_song))
    within = scalar_metrics(
        np.concatenate(centered_prediction), np.concatenate(centered_target)
    )
    means = scalar_metrics(np.asarray(mean_prediction), np.asarray(mean_target))
    first = scalar_metrics(np.concatenate(first_prediction), np.concatenate(first_target))
    return {
        **{f"micro_{key}": value for key, value in micro.items()},
        **{f"macro_{key}": value for key, value in macro.items()},
        **{f"window5_{key}": value for key, value in window5.items()},
        **{f"window10_{key}": value for key, value in window10.items()},
        **{f"within_{key}": value for key, value in within.items()},
        **{f"song_mean_{key}": value for key, value in means.items()},
        **{f"first_difference_{key}": value for key, value in first.items()},
    }, prediction, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/global_local_temporal.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]) + args.fold)
    device = torch.device(args.device)
    index = pd.read_csv(INDEX)
    folds = pd.read_csv(ROOT / "data/five_fold_manifest.csv")[["song_id", "fold"]]
    rows = index[index["split"] == "train"].merge(folds, on="song_id")
    train_rows = rows[rows["fold"] != args.fold]
    held_rows = rows[rows["fold"] == args.fold]
    has_adapter = int(config.get("adapter_bottleneck", 0)) > 0
    has_metric_learning = float(config["loss"].get("metric_weight", 0.0)) > 0
    auxiliary_music = bool(config.get("auxiliary_music", False))
    if has_adapter and auxiliary_music:
        trial_name = "global_local_adapter_music_auxiliary"
    elif has_adapter:
        trial_name = "global_local_embedding_adapter"
    elif has_metric_learning:
        trial_name = "global_local_metric_learning"
    elif auxiliary_music:
        trial_name = "global_local_music_auxiliary"
    else:
        trial_name = "global_local_temporal"
    output_dir = ROOT / f"outputs/{trial_name}_fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = build_norm(train_rows, output_dir / "feature_norm.npz")
    datasets = {
        "train": SongDataset(train_rows, mean, std, auxiliary_music),
        "held": SongDataset(held_rows, mean, std, auxiliary_music),
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
    model = GlobalLocalTemporalRegressor(768, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = output_dir / "best.pt"
    history = []
    best_rmse, best_epoch, bad_epochs = float("inf"), 0, 0
    print(
        json.dumps(
            {
                "fold": args.fold,
                "train_songs": len(train_rows),
                "held_songs": len(held_rows),
                "parameters": sum(p.numel() for p in model.parameters()),
                "embedding_adapter": has_adapter,
                "adapter_bottleneck": int(config.get("adapter_bottleneck", 0)),
                "metric_learning": has_metric_learning,
                "music_auxiliary": auxiliary_music,
                "test_used": False,
                "legacy_validation_used": False,
            },
            indent=2,
        )
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            features, target, mask, _, _ = batch[:5]
            features, target, mask = (
                features.to(device),
                target.to(device),
                mask.to(device),
            )
            optimizer.zero_grad()
            output = model(features, mask)
            key_modes = batch[5].to(device) if auxiliary_music else None
            chord_change = batch[6].to(device) if auxiliary_music else None
            loss = loss_function(
                output,
                target,
                mask,
                config["loss"],
                key_modes,
                chord_change,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["grad_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        held_metrics, _, _ = evaluate(model, loaders["held"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **held_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"rmse={held_metrics['micro_rmse']:.6f} "
            f"pcc={held_metrics['micro_pcc']:.6f} "
            f"within_pcc={held_metrics['within_pcc']:.6f}"
        )
        if held_metrics["micro_rmse"] < best_rmse:
            best_rmse, best_epoch, bad_epochs = held_metrics["micro_rmse"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    held_metrics, prediction, target = evaluate(model, loaders["held"], device)
    summary = {
        "fold": args.fold,
        "seed": int(config["seed"]) + args.fold,
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "embedding_adapter": has_adapter,
        "metric_learning": has_metric_learning,
        "music_auxiliary": auxiliary_music,
        "metric_learning": has_metric_learning,
        "checkpoint": str(checkpoint),
        "test_used": False,
        "legacy_validation_used": False,
        "config": config,
        **held_metrics,
    }
    np.save(output_dir / "held_pred.npy", prediction.astype(np.float32))
    np.save(output_dir / "held_target.npy", target.astype(np.float32))
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
