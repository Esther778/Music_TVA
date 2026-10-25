from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from train_gated_fusion import FEATURE_ROOT, metrics
from train_layer6_gated_fusion import LAYER6_ROOT, time_indices


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
INDEX = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence_index.csv"
)
PROTOCOL = ROOT / "data/full_song_protocol/nested_full_song_protocol.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_song(song_id: int, sequence_path: str | Path):
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
        handcrafted = hand_file["handcrafted"].astype(np.float32)
        hand_target = hand_file["labels"].astype(np.float32)
    indices = time_indices(hand_times, times, song_id)
    handcrafted = handcrafted[indices]
    hand_target = hand_target[indices]
    arrays = (layer6, final, handcrafted, target)
    if len({len(values) for values in arrays}) != 1:
        raise ValueError(f"Song {song_id}: feature lengths differ.")
    if not np.allclose(target, hand_target, atol=1e-6):
        raise ValueError(f"Song {song_id}: target alignment mismatch.")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError(f"Song {song_id}: non-finite feature or target.")
    return arrays


def build_norm(rows: pd.DataFrame, output: Path):
    if output.is_file():
        with np.load(output) as data:
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
    totals = squares = None
    count = 0
    for row in rows.itertuples():
        features = load_song(int(row.song_id), row.sequence_path)[:3]
        if totals is None:
            totals = [
                values.astype(np.float64).sum(axis=0)
                for values in features
            ]
            squares = [
                (values.astype(np.float64) ** 2).sum(axis=0)
                for values in features
            ]
        else:
            for index, values in enumerate(features):
                values = values.astype(np.float64)
                totals[index] += values.sum(axis=0)
                squares[index] += (values**2).sum(axis=0)
        count += len(features[0])
    means = [(total / count).astype(np.float32) for total in totals]
    stds = [
        np.sqrt(
            np.maximum(
                square / count - mean.astype(np.float64) ** 2, 1e-8
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


class FullSongDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, norms):
        self.rows = rows.sort_values("song_id").to_dict("records")
        self.norms = norms
        self.lengths = [int(row["frames"]) for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        song_id = int(row["song_id"])
        layer6, final, hand, target = load_song(
            song_id, row["sequence_path"]
        )
        layer6 = (layer6 - self.norms[0]) / self.norms[1]
        final = (final - self.norms[2]) / self.norms[3]
        hand = (hand - self.norms[4]) / self.norms[5]
        return (
            torch.from_numpy(layer6),
            torch.from_numpy(final),
            torch.from_numpy(hand),
            torch.from_numpy(target),
            song_id,
            row["domain"],
        )


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_size: int = 128,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        buckets = [
            indices[start : start + self.bucket_size]
            for start in range(0, len(indices), self.bucket_size)
        ]
        batches = []
        for bucket in buckets:
            bucket.sort(key=lambda index: self.lengths[index])
            batches.extend(
                bucket[start : start + self.batch_size]
                for start in range(0, len(bucket), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def collate(batch):
    lengths = torch.tensor([len(item[3]) for item in batch], dtype=torch.long)
    maximum = int(lengths.max())
    outputs = [
        torch.zeros(len(batch), maximum, batch[0][index].shape[1])
        for index in range(4)
    ]
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    song_ids, domains = [], []
    for batch_index, item in enumerate(batch):
        length = len(item[3])
        for feature_index in range(4):
            outputs[feature_index][batch_index, :length] = item[feature_index]
        mask[batch_index, :length] = True
        song_ids.append(item[4])
        domains.append(item[5])
    return (*outputs, mask, lengths, song_ids, domains)


class WindowAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, dimension),
            nn.Dropout(dropout),
        )

    def forward(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = values + attended
        values = values + self.ffn(self.norm2(values))
        return values * valid.unsqueeze(-1)


class LocalWindowEncoder(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        ffn: int,
        dropout: float,
        window: int,
        layers: int,
    ):
        super().__init__()
        self.window = window
        self.blocks = nn.ModuleList(
            WindowAttentionBlock(dimension, heads, ffn, dropout)
            for _ in range(layers)
        )

    def _window_pass(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        block: nn.Module,
        offset: int,
    ):
        batch, frames, dimension = values.shape
        left = offset
        total = left + frames
        right = (-total) % self.window
        padded = F.pad(values, (0, 0, left, right))
        padded_mask = F.pad(mask, (left, right), value=False)
        windows = padded.reshape(-1, self.window, dimension)
        window_mask = padded_mask.reshape(-1, self.window)
        empty = ~window_mask.any(dim=1)
        if bool(empty.any()):
            window_mask = window_mask.clone()
            window_mask[empty, 0] = True
        windows = block(windows, window_mask)
        restored = windows.reshape(batch, -1, dimension)
        return restored[:, left : left + frames] * mask.unsqueeze(-1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        for index, block in enumerate(self.blocks):
            offset = 0 if index % 2 == 0 else self.window // 2
            values = self._window_pass(values, mask, block, offset)
        return values


class GlobalPooledEncoder(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        ffn: int,
        dropout: float,
        pool_frames: int,
        layers: int,
    ):
        super().__init__()
        self.pool_frames = pool_frames
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.song_token = nn.Parameter(torch.zeros(1, 1, dimension))
        nn.init.normal_(self.song_token, std=0.02)

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        batch, frames, dimension = values.shape
        right = (-frames) % self.pool_frames
        padded = F.pad(values, (0, 0, 0, right))
        padded_mask = F.pad(mask, (0, right), value=False)
        chunks = padded.reshape(
            batch, -1, self.pool_frames, dimension
        )
        chunk_mask = padded_mask.reshape(batch, -1, self.pool_frames)
        denominator = chunk_mask.sum(dim=2, keepdim=True).clamp_min(1)
        pooled = (
            chunks * chunk_mask.unsqueeze(-1)
        ).sum(dim=2) / denominator
        pooled_mask = chunk_mask.any(dim=2)
        token = self.song_token.expand(batch, -1, -1)
        sequence = torch.cat([token, pooled], dim=1)
        sequence_mask = torch.cat(
            [
                torch.ones(batch, 1, dtype=torch.bool, device=mask.device),
                pooled_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(
            sequence, src_key_padding_mask=~sequence_mask
        )
        song_context = encoded[:, :1]
        pooled_context = encoded[:, 1:]
        expanded = pooled_context.repeat_interleave(
            self.pool_frames, dim=1
        )[:, :frames]
        song_context = song_context.expand(-1, frames, -1)
        return expanded * mask.unsqueeze(-1), song_context


class DualScaleBranch(nn.Module):
    def __init__(self, dimension: int, config: dict):
        super().__init__()
        self.local = LocalWindowEncoder(
            dimension,
            int(config["attention_heads"]),
            int(config["ffn_dim"]),
            float(config["dropout"]),
            int(config["local_window_frames"]),
            int(config["local_layers"]),
        )
        self.global_encoder = GlobalPooledEncoder(
            dimension,
            int(config["attention_heads"]),
            int(config["ffn_dim"]),
            float(config["dropout"]),
            int(config["global_pool_frames"]),
            int(config["global_layers"]),
        )
        self.fusion = nn.Linear(dimension * 3, 3)
        self.output = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(dimension, 1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        local = self.local(values, mask)
        global_frames, song_context = self.global_encoder(values, mask)
        weights = torch.softmax(
            self.fusion(torch.cat([local, global_frames, song_context], -1)),
            dim=-1,
        )
        fused = (
            weights[..., :1] * local
            + weights[..., 1:2] * global_frames
            + weights[..., 2:] * song_context
        )
        return self.output(fused) * mask.unsqueeze(-1)


class DualScaleVARegressor(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], config: dict):
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
        self.valence_source_gate = nn.Linear(projection * 3, 3)
        self.arousal_source_gate = nn.Linear(projection * 3, 3)
        self.valence = DualScaleBranch(projection, config)
        self.arousal = DualScaleBranch(projection, config)

    @staticmethod
    def source_fusion(projected, gate):
        joined = torch.cat(projected, dim=-1)
        weights = torch.softmax(gate(joined), dim=-1)
        stacked = torch.stack(projected, dim=-2)
        return (weights.unsqueeze(-1) * stacked).sum(dim=-2)

    def forward(self, layer6, final, hand, mask):
        projected = [
            projector(values)
            for projector, values in zip(
                self.projectors, (layer6, final, hand)
            )
        ]
        valence_features = self.source_fusion(
            projected, self.valence_source_gate
        )
        arousal_features = self.source_fusion(
            projected, self.arousal_source_gate
        )
        valence = self.valence(valence_features, mask)
        arousal = self.arousal(arousal_features, mask)
        return torch.cat([valence, arousal], dim=-1)


def ccc_loss(prediction: torch.Tensor, target: torch.Tensor):
    pred_mean = prediction.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    covariance = (pred_centered * target_centered).mean(dim=0)
    ccc = 2 * covariance / (
        pred_centered.var(dim=0, unbiased=False)
        + target_centered.var(dim=0, unbiased=False)
        + (pred_mean - target_mean) ** 2
        + 1e-8
    )
    return 1.0 - ccc.mean()


def moving_average(values: torch.Tensor, frames: int):
    left = (frames - 1) // 2
    right = frames // 2
    padded = F.pad(
        values.transpose(0, 1).unsqueeze(0),
        (left, right),
        mode="replicate",
    )
    return (
        F.avg_pool1d(padded, kernel_size=frames, stride=1)
        .squeeze(0)
        .transpose(0, 1)
    )


def song_balanced_loss(prediction, target, mask, config):
    losses = []
    for index in range(len(target)):
        pred = prediction[index][mask[index]]
        truth = target[index][mask[index]]
        loss = (
            float(config["huber"])
            * F.huber_loss(pred, truth, delta=0.15)
            + float(config["mse"]) * F.mse_loss(pred, truth)
            + float(config["ccc"]) * ccc_loss(pred, truth)
            + float(config["trend"])
            * F.mse_loss(
                moving_average(pred, 10), moving_average(truth, 10)
            )
        )
        losses.append(loss)
    return torch.stack(losses).mean()


def expanded_song_ids(rows: list[dict]) -> np.ndarray:
    result = []
    for row in rows:
        with np.load(row["sequence_path"]) as data:
            result.append(
                np.full(len(data["Y"]), int(row["song_id"]), dtype=np.int32)
            )
    return np.concatenate(result)


def scalar_dimension_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict:
    return metrics(prediction, target)


def window_metrics(prediction, target, song_ids, window_frames):
    pred_windows, target_windows = [], []
    for song_id in np.unique(song_ids):
        selected = song_ids == song_id
        pred_song = prediction[selected]
        target_song = target[selected]
        for start in range(0, len(pred_song), window_frames):
            stop = min(start + window_frames, len(pred_song))
            pred_windows.append(pred_song[start:stop].mean(axis=0))
            target_windows.append(target_song[start:stop].mean(axis=0))
    return metrics(np.asarray(pred_windows), np.asarray(target_windows))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for layer6, final, hand, target, mask, _, _, _ in loader:
        prediction = model(
            layer6.to(device),
            final.to(device),
            hand.to(device),
            mask.to(device),
        )
        predictions.append(
            prediction[mask.to(device)].cpu().numpy()
        )
        targets.append(target[mask].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    song_ids = expanded_song_ids(loader.dataset.rows)
    frame = metrics(prediction, target)
    window5 = window_metrics(prediction, target, song_ids, 10)
    window10 = window_metrics(prediction, target, song_ids, 20)
    mean_prediction, mean_target = [], []
    centered_prediction, centered_target = [], []
    for song_id in np.unique(song_ids):
        selected = song_ids == song_id
        pred_song = prediction[selected]
        target_song = target[selected]
        mean_prediction.append(pred_song.mean(axis=0))
        mean_target.append(target_song.mean(axis=0))
        centered_prediction.append(pred_song - pred_song.mean(axis=0))
        centered_target.append(target_song - target_song.mean(axis=0))
    song_mean = metrics(
        np.asarray(mean_prediction), np.asarray(mean_target)
    )
    within = metrics(
        np.concatenate(centered_prediction), np.concatenate(centered_target)
    )
    result = {
        **{f"frame_{key}": value for key, value in frame.items()},
        **{f"window5_{key}": value for key, value in window5.items()},
        **{f"window10_{key}": value for key, value in window10.items()},
        **{f"song_mean_{key}": value for key, value in song_mean.items()},
        **{f"within_{key}": value for key, value in within.items()},
    }
    return result, prediction, target, song_ids


def make_loader(dataset, batch_size, shuffle, seed):
    sampler = BucketBatchSampler(
        dataset.lengths, batch_size, shuffle, seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )
    return loader, sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/dual_scale_full_song_va.json",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-train-songs", type=int)
    parser.add_argument("--limit-validation-songs", type=int)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"]) + args.outer_fold
    set_seed(seed)
    device = torch.device(args.device)

    protocol = pd.read_csv(PROTOCOL)
    roles = protocol[protocol["outer_fold"] == args.outer_fold].copy()
    index = pd.read_csv(INDEX)[["song_id", "frames", "sequence_path"]]
    roles = roles.drop(columns=["sequence_path"]).merge(index, on="song_id")
    train_rows = roles[roles["role"] == "train"].copy()
    validation_rows = roles[roles["role"] == "validation"].copy()
    if args.limit_train_songs:
        train_rows = train_rows.head(args.limit_train_songs)
    if args.limit_validation_songs:
        validation_rows = validation_rows.head(
            args.limit_validation_songs
        )

    output_name = (
        f"{config['trial_id']}_fold{args.outer_fold}{args.output_suffix}"
    )
    output_dir = ROOT / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    norms = build_norm(train_rows, output_dir / "feature_norm.npz")
    datasets = {
        "train": FullSongDataset(train_rows, norms),
        "validation": FullSongDataset(validation_rows, norms),
    }
    loaders, samplers = {}, {}
    for split, dataset in datasets.items():
        loaders[split], samplers[split] = make_loader(
            dataset,
            int(config["batch_size"]) if split == "train" else 1,
            split == "train",
            seed,
        )
    sample = datasets["train"][0]
    model = DualScaleVARegressor(
        (sample[0].shape[1], sample[1].shape[1], sample[2].shape[1]),
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = output_dir / "best.pt"
    epochs = args.epochs or int(config["epochs"])
    best_score, best_epoch, bad_epochs = float("inf"), 0, 0
    history = []
    started = time.time()
    print(
        json.dumps(
            {
                "outer_fold": args.outer_fold,
                "device": str(device),
                "train_songs": len(train_rows),
                "train_short": int((train_rows["domain"] == "short_clip").sum()),
                "train_full": int((train_rows["domain"] == "full_song").sum()),
                "validation_full": len(validation_rows),
                "test_used": False,
                "parameters": sum(p.numel() for p in model.parameters()),
                "selection": "0.65*valence_rmse + 0.35*arousal_rmse",
            },
            indent=2,
        ),
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        samplers["train"].set_epoch(epoch)
        losses = []
        epoch_started = time.time()
        for layer6, final, hand, target, mask, _, _, _ in loaders["train"]:
            layer6 = layer6.to(device)
            final = final.to(device)
            hand = hand.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(layer6, final, hand, mask)
            loss = song_balanced_loss(
                prediction, target, mask, config["loss"]
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "Non-finite training loss detected; stopping before "
                    "the optimizer can corrupt the last valid checkpoint."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["grad_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation, _, _, _ = evaluate(
            model, loaders["validation"], device
        )
        if not all(np.isfinite(value) for value in validation.values()):
            raise RuntimeError(
                "Non-finite validation metric detected; the last finite "
                "checkpoint remains intact."
            )
        score = (
            float(config["selection_valence_weight"])
            * validation["frame_rmse_valence"]
            + float(config["selection_arousal_weight"])
            * validation["frame_rmse_arousal"]
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            "epoch_sec": time.time() - epoch_started,
            **validation,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"score={score:.6f} "
            f"v_rmse={validation['frame_rmse_valence']:.6f} "
            f"a_rmse={validation['frame_rmse_arousal']:.6f} "
            f"mean_pcc={validation['frame_mean_pcc']:.6f} "
            f"mean_ccc={validation['frame_mean_ccc']:.6f} "
            f"sec={row['epoch_sec']:.1f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    validation, prediction, target, song_ids = evaluate(
        model, loaders["validation"], device
    )
    summary = {
        "trial_id": config["trial_id"],
        "outer_fold": args.outer_fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "elapsed_sec": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_songs": len(train_rows),
        "validation_songs": len(validation_rows),
        "test_used": False,
        "config": config,
        **validation,
    }
    np.save(output_dir / "validation_pred.npy", prediction.astype(np.float32))
    np.save(output_dir / "validation_target.npy", target.astype(np.float32))
    np.save(output_dir / "validation_song_ids.npy", song_ids.astype(np.int32))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
