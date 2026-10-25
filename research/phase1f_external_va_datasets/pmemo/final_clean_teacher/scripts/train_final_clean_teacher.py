#!/usr/bin/env python3
"""Train strict OOF and final-fit clean PMEmo VA teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


PMEMO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PMEMO_ROOT.parents[2]
DEFAULT_CONFIG = PMEMO_ROOT / "final_clean_teacher/configs/final_clean_teacher_v1.json"
DEFAULT_MERT = PMEMO_ROOT / "outputs/mert_pmemo_hidden6/features"
DEFAULT_SEQUENCE = PMEMO_ROOT / "outputs/opensmile_baseline/dataset/sequence"
DEFAULT_SPLITS = PMEMO_ROOT / "data/splits/pmemo_final_locked_20260713"
DEFAULT_VALENCE_TEACHER = PMEMO_ROOT / "outputs/valence_locked_global_shift_amp_guarded_20260715"
DEFAULT_AROUSAL_TEACHER = PMEMO_ROOT / "outputs/arousal_locked_opensmile_static_songmean_fine_20260715"
DEFAULT_OUTPUT = PMEMO_ROOT / "final_clean_teacher/outputs/final_clean_teacher_v1"
FOLDS = ("fold_01", "fold_02", "fold_03")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ccc_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = prediction.mean()
    target_mean = target.mean()
    pred_var = prediction.var(unbiased=False)
    target_var = target.var(unbiased=False)
    covariance = ((prediction - pred_mean) * (target - target_mean)).mean()
    ccc = 2.0 * covariance / (
        pred_var + target_var + (pred_mean - target_mean).square() + 1e-8
    )
    return 1.0 - ccc


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    pred_std = float(prediction.std())
    target_std = float(target.std())
    pcc = float(np.corrcoef(prediction, target)[0, 1]) if pred_std > 1e-12 and target_std > 1e-12 else 0.0
    covariance = float(np.mean((prediction - prediction.mean()) * (target - target.mean())))
    ccc = float(
        2.0 * covariance
        / (prediction.var() + target.var() + (prediction.mean() - target.mean()) ** 2 + 1e-12)
    )
    return {"rmse": rmse, "mae": mae, "pcc": pcc, "ccc": ccc}


def clip_metrics(frame: pd.DataFrame, dimension: str) -> dict[str, float]:
    grouped = frame.groupby("musicId", sort=False)[
        [f"pred_{dimension}", f"target_{dimension}"]
    ].mean()
    return regression_metrics(
        grouped[f"pred_{dimension}"].to_numpy(),
        grouped[f"target_{dimension}"].to_numpy(),
    )


def threshold_rows(result: dict[str, float]) -> list[dict[str, Any]]:
    minimum = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
    good = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
    excellent = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}
    rows = []
    for metric, value in result.items():
        lower_is_better = metric in {"rmse", "mae"}
        gap = value - minimum[metric] if lower_is_better else minimum[metric] - value
        rows.append({
            "metric": metric,
            "current": value,
            "minimum_pass": minimum[metric],
            "gap_to_pass": gap,
            "good": good[metric],
            "excellent": excellent[metric],
            "status": "pass" if gap <= 0 else "not_pass",
        })
    return rows


@dataclass
class SongRecord:
    music_id: int
    fold: str
    feature: np.ndarray
    time_sec: np.ndarray
    target: np.ndarray
    teacher: np.ndarray


class SongDataset(Dataset):
    def __init__(self, records: list[SongRecord], mean: np.ndarray, std: np.ndarray):
        self.records = records
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        return {
            "music_id": row.music_id,
            "feature": ((row.feature - self.mean) / self.std).astype(np.float32),
            "time_sec": row.time_sec,
            "target": row.target,
            "teacher": row.teacher,
        }


def collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    max_length = max(len(row["target"]) for row in batch)
    feature_dim = batch[0]["feature"].shape[1]
    size = len(batch)
    feature = torch.zeros(size, max_length, feature_dim)
    target = torch.zeros(size, max_length)
    teacher = torch.zeros(size, max_length)
    time_sec = torch.zeros(size, max_length)
    mask = torch.zeros(size, max_length, dtype=torch.bool)
    lengths = torch.zeros(size, dtype=torch.long)
    music_id = torch.zeros(size, dtype=torch.long)
    for index, row in enumerate(batch):
        length = len(row["target"])
        feature[index, :length] = torch.from_numpy(row["feature"])
        target[index, :length] = torch.from_numpy(row["target"])
        teacher[index, :length] = torch.from_numpy(row["teacher"])
        time_sec[index, :length] = torch.from_numpy(row["time_sec"])
        mask[index, :length] = True
        lengths[index] = length
        music_id[index] = int(row["music_id"])
    return feature, target, teacher, time_sec, mask, lengths, music_id


class GlobalLocalBiGRU(nn.Module):
    def __init__(self, input_dim: int, config: dict[str, Any], local_scale: float):
        super().__init__()
        architecture = config["architecture"]
        projection_dim = int(architecture["projection_dim"])
        hidden_dim = int(architecture["gru_hidden_dim"])
        dropout = float(architecture["dropout"])
        self.local_scale = float(local_scale)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            projection_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        width = hidden_dim * 2
        self.local_head = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 1)
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, int(architecture["global_hidden_dim"])),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(architecture["global_hidden_dim"]), 1),
        )

    def forward(
        self, feature: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(feature)
        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=feature.shape[1]
        )
        weights = mask.float().unsqueeze(-1)
        mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        variance = ((hidden - mean.unsqueeze(1)).square() * weights).sum(dim=1)
        variance = variance / weights.sum(dim=1).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        global_mean = torch.tanh(
            self.global_head(torch.cat([mean, variance.clamp_min(1e-8).sqrt(), maximum], dim=-1)).squeeze(-1)
        )
        local = torch.tanh(self.local_head(hidden).squeeze(-1))
        local_mean = (local * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        local = (local - local_mean.unsqueeze(1)) * mask.float()
        prediction = torch.clamp(global_mean.unsqueeze(1) + self.local_scale * local, -1.0, 1.0)
        return prediction, global_mean


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_normalizer(records: list[SongRecord]) -> tuple[np.ndarray, np.ndarray]:
    total = sum(len(row.feature) for row in records)
    feature_dim = records[0].feature.shape[1]
    sum_x = np.zeros(feature_dim, dtype=np.float64)
    sum_x2 = np.zeros(feature_dim, dtype=np.float64)
    for row in records:
        values = row.feature.astype(np.float64)
        sum_x += values.sum(axis=0)
        sum_x2 += np.square(values).sum(axis=0)
    mean = sum_x / total
    variance = np.maximum(sum_x2 / total - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def align_embedding(
    embedding: np.ndarray, source_time: np.ndarray, target_time: np.ndarray
) -> tuple[np.ndarray, bool]:
    embedding = np.asarray(embedding, dtype=np.float32)
    source_time = np.asarray(source_time, dtype=np.float32)
    target_time = np.asarray(target_time, dtype=np.float32)
    if len(embedding) == len(target_time) and np.allclose(source_time, target_time, atol=1e-4):
        return embedding, False
    if len(source_time) < 2 or np.any(np.diff(source_time) <= 0):
        raise RuntimeError("Cannot align a non-monotonic MERT time axis")
    aligned = np.stack([
        np.interp(target_time, source_time, embedding[:, column])
        for column in range(embedding.shape[1])
    ], axis=1).astype(np.float32)
    return aligned, True


def load_development_records(
    dimension: str,
    teacher_dir: Path,
    mert_dir: Path,
    sequence_dir: Path,
    final_ids: set[int],
) -> list[SongRecord]:
    target_col = f"target_{dimension}"
    pred_col = f"pred_{dimension}"
    records: list[SongRecord] = []
    seen: set[int] = set()
    resampled_ids: list[int] = []
    for fold in FOLDS:
        table = pd.read_csv(teacher_dir / f"{fold}_val_predictions.csv")
        for music_id, song in table.groupby("musicId", sort=False):
            music_id = int(music_id)
            if music_id in final_ids:
                raise RuntimeError(f"Final locked song leaked into development: {music_id}")
            if music_id in seen:
                raise RuntimeError(f"Duplicate OOF song: {music_id}")
            seen.add(music_id)
            mert = np.load(mert_dir / f"{music_id}.npz")
            sequence = np.load(sequence_dir / f"{music_id}.npz")
            time_sec = song["time_sec"].to_numpy(dtype=np.float32)
            if not (
                len(song) == len(sequence["X"])
                and np.allclose(time_sec, sequence["time_sec"], atol=1e-4)
            ):
                raise RuntimeError(f"Feature/time mismatch for development song {music_id}")
            embedding, resampled = align_embedding(
                mert["embedding"], mert["time_sec"], time_sec
            )
            if resampled:
                resampled_ids.append(music_id)
            feature = np.concatenate(
                [embedding, sequence["X"].astype(np.float32)],
                axis=1,
            )
            records.append(SongRecord(
                music_id=music_id,
                fold=fold,
                feature=feature,
                time_sec=time_sec,
                target=song[target_col].to_numpy(dtype=np.float32),
                teacher=song[pred_col].to_numpy(dtype=np.float32),
            ))
    if len(records) != 652 or len(seen) != 652:
        raise RuntimeError(f"Expected 652 development songs, got {len(records)}")
    print(json.dumps({
        "dimension": dimension,
        "development_songs": len(records),
        "resampled_mert_songs": sorted(set(resampled_ids)),
        "final_overlap": 0,
    }), flush=True)
    return records


def inner_split(records: list[SongRecord], seed: int, fraction: float) -> tuple[list[SongRecord], list[SongRecord]]:
    means = np.asarray([row.target.mean() for row in records])
    ranks = pd.Series(means).rank(method="first")
    bins = pd.qcut(ranks, q=8, labels=False, duplicates="drop").to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_index, val_index = next(splitter.split(np.zeros(len(records)), bins))
    return [records[i] for i in train_index], [records[i] for i in val_index]


def make_loader(
    records: list[SongRecord], mean: np.ndarray, std: np.ndarray,
    config: dict[str, Any], shuffle: bool,
) -> DataLoader:
    return DataLoader(
        SongDataset(records, mean, std),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        collate_fn=collate,
    )


def loss_function(
    prediction: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    global_mean: torch.Tensor,
    mask: torch.Tensor,
    settings: dict[str, float],
) -> torch.Tensor:
    valid_prediction = prediction[mask]
    valid_target = target[mask]
    valid_teacher = teacher[mask]
    if float(settings["extreme_weight"]) != 1.0:
        weight = 1.0 + (float(settings["extreme_weight"]) - 1.0) * valid_target.abs()
        frame_mse = (weight * (valid_prediction - valid_target).square()).mean()
    else:
        frame_mse = nn.functional.mse_loss(valid_prediction, valid_target)
    target_mean = (target * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    clip_loss = nn.functional.mse_loss(global_mean, target_mean)
    teacher_loss = nn.functional.mse_loss(valid_prediction, valid_teacher)
    transition = mask[:, 1:] & mask[:, :-1]
    smoothness = prediction.diff(dim=1)[transition].square().mean()
    return (
        frame_mse
        + float(settings["clip_mean_loss_weight"]) * clip_loss
        + float(settings["ccc_loss_weight"]) * ccc_loss(valid_prediction, valid_target)
        + float(settings["smoothness_loss_weight"]) * smoothness
        + float(settings["teacher_loss_weight"]) * teacher_loss
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, dimension: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    model.eval()
    frames = []
    for feature, target, _, time_sec, mask, lengths, music_id in loader:
        prediction, _ = model(feature.to(device), mask.to(device), lengths)
        prediction = prediction.cpu()
        for index in range(len(music_id)):
            valid = mask[index]
            frames.append(pd.DataFrame({
                "musicId": int(music_id[index]),
                "time_sec": time_sec[index, valid].numpy(),
                f"target_{dimension}": target[index, valid].numpy(),
                f"pred_{dimension}": prediction[index, valid].numpy(),
            }))
    table = pd.concat(frames, ignore_index=True)
    result = {
        "frame": regression_metrics(
            table[f"pred_{dimension}"].to_numpy(), table[f"target_{dimension}"].to_numpy()
        ),
        "clip": clip_metrics(table, dimension),
    }
    return table, result


def selection_score(result: dict[str, dict[str, float]]) -> float:
    clip = result["clip"]
    frame = result["frame"]
    return float(
        clip["rmse"] + 0.25 * frame["rmse"]
        + 0.05 * max(0.0, 0.80 - clip["ccc"])
        + 0.03 * max(0.0, 0.80 - clip["pcc"])
    )


def train_with_validation(
    dimension: str,
    train_records: list[SongRecord],
    val_records: list[SongRecord],
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[int, Path, np.ndarray, np.ndarray, list[dict[str, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = fit_normalizer(train_records)
    np.savez_compressed(output_dir / "normalizer.npz", mean=mean, std=std)
    train_loader = make_loader(train_records, mean, std, config, True)
    val_loader = make_loader(val_records, mean, std, config, False)
    settings = config[dimension]
    input_dim = train_records[0].feature.shape[1]
    set_seed(seed)
    model = GlobalLocalBiGRU(input_dim, config, float(settings["local_scale"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint = output_dir / "model.pt"
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        losses = []
        for feature, target, teacher, _, mask, lengths, _ in train_loader:
            feature = feature.to(device)
            target = target.to(device)
            teacher = teacher.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            prediction, global_mean = model(feature, mask, lengths)
            loss = loss_function(prediction, target, teacher, global_mean, mask, settings)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, result = evaluate(model, val_loader, device, dimension)
        score = selection_score(result)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            **{f"frame_{key}": value for key, value in result["frame"].items()},
            **{f"clip_{key}": value for key, value in result["clip"].items()},
        }
        history.append(row)
        print(
            f"{dimension} epoch={epoch:02d} loss={row['train_loss']:.5f} "
            f"frame_rmse={row['frame_rmse']:.5f} clip_rmse={row['clip_rmse']:.5f} "
            f"clip_pcc={row['clip_pcc']:.3f} clip_ccc={row['clip_ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "dimension": dimension,
                "config": config,
                "selected_epoch": epoch,
            }, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["training"]["patience"]):
                break
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    return best_epoch, checkpoint, mean, std, history


def load_model(checkpoint: Path, device: torch.device) -> GlobalLocalBiGRU:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GlobalLocalBiGRU(
        int(payload["input_dim"]), payload["config"],
        float(payload["config"][payload["dimension"]]["local_scale"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    return model


def train_fixed_epochs(
    dimension: str,
    records: list[SongRecord],
    config: dict[str, Any],
    epochs: int,
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[Path, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = fit_normalizer(records)
    np.savez_compressed(output_dir / "normalizer.npz", mean=mean, std=std)
    loader = make_loader(records, mean, std, config, True)
    input_dim = records[0].feature.shape[1]
    settings = config[dimension]
    set_seed(seed)
    model = GlobalLocalBiGRU(input_dim, config, float(settings["local_scale"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for feature, target, teacher, _, mask, lengths, _ in loader:
            feature = feature.to(device)
            target = target.to(device)
            teacher = teacher.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            prediction, global_mean = model(feature, mask, lengths)
            loss = loss_function(prediction, target, teacher, global_mean, mask, settings)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"{dimension} final_fit epoch={epoch:02d}/{epochs} loss={history[-1]['train_loss']:.5f}", flush=True)
    checkpoint = output_dir / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "dimension": dimension,
        "config": config,
        "selected_epoch": epochs,
        "training_songs": len(records),
    }, checkpoint)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    return checkpoint, mean, std


def load_final_features(
    music_id: int, mert_dir: Path, sequence_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    mert = np.load(mert_dir / f"{music_id}.npz")
    sequence = np.load(sequence_dir / f"{music_id}.npz")
    # Deliberately access only audio features and timestamps. Never access sequence["Y"].
    embedding = mert["embedding"].astype(np.float32)
    opensmile_x = sequence["X"].astype(np.float32)
    mert_time = mert["time_sec"].astype(np.float32)
    opensmile_time = sequence["time_sec"].astype(np.float32)
    embedding, _ = align_embedding(embedding, mert_time, opensmile_time)
    return np.concatenate([embedding, opensmile_x], axis=1), opensmile_time


@torch.no_grad()
def predict_final_dimension(
    dimension: str,
    checkpoint: Path,
    mean: np.ndarray,
    std: np.ndarray,
    final_index: pd.DataFrame,
    mert_dir: Path,
    sequence_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    model = load_model(checkpoint, device)
    model.eval()
    rows = []
    for item in final_index.itertuples(index=False):
        music_id = int(item.musicId)
        feature, time_sec = load_final_features(music_id, mert_dir, sequence_dir)
        normalized = ((feature - mean) / std).astype(np.float32)
        feature_tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
        mask = torch.ones(1, len(normalized), dtype=torch.bool, device=device)
        lengths = torch.tensor([len(normalized)], dtype=torch.long)
        prediction, _ = model(feature_tensor, mask, lengths)
        rows.append(pd.DataFrame({
            "musicId": music_id,
            "time_sec": time_sec,
            f"pred_{dimension}": prediction[0, :len(time_sec)].cpu().numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dimension(
    dimension: str,
    records: list[SongRecord],
    config: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], Path, np.ndarray, np.ndarray]:
    seed = int(config["seed"]) + (0 if dimension == "valence" else 1000)
    per_fold = []
    tables = []
    selected_epochs = []
    for fold_index, held_fold in enumerate(FOLDS, start=1):
        outer_train = [row for row in records if row.fold != held_fold]
        held = [row for row in records if row.fold == held_fold]
        inner_train, inner_val = inner_split(
            outer_train,
            seed + fold_index,
            float(config["training"]["inner_validation_fraction"]),
        )
        fold_dir = output_root / "oof" / dimension / held_fold
        print(
            f"{dimension} {held_fold}: inner_train={len(inner_train)} "
            f"inner_val={len(inner_val)} held={len(held)}",
            flush=True,
        )
        epoch, checkpoint, mean, std, _ = train_with_validation(
            dimension, inner_train, inner_val, config, fold_dir,
            seed + 100 * fold_index, device,
        )
        selected_epochs.append(epoch)
        refit_dir = fold_dir / "outer_refit"
        checkpoint, mean, std = train_fixed_epochs(
            dimension,
            outer_train,
            config,
            epoch,
            refit_dir,
            seed + 5000 + 100 * fold_index,
            device,
        )
        model = load_model(checkpoint, device)
        held_loader = make_loader(held, mean, std, config, False)
        table, result = evaluate(model, held_loader, device, dimension)
        table["fold"] = held_fold
        table.to_csv(fold_dir / "held_predictions.csv", index=False)
        tables.append(table)
        per_fold.append({"fold": held_fold, "selected_epoch": epoch, **result})
        print(json.dumps(per_fold[-1], indent=2), flush=True)

    oof = pd.concat(tables, ignore_index=True)
    oof.to_csv(output_root / f"oof_{dimension}_predictions.csv", index=False)
    frame_macro = {
        key: float(np.mean([row["frame"][key] for row in per_fold]))
        for key in ("rmse", "mae", "pcc", "ccc")
    }
    clip_macro = {
        key: float(np.mean([row["clip"][key] for row in per_fold]))
        for key in ("rmse", "mae", "pcc", "ccc")
    }
    final_epochs = int(np.median(selected_epochs))
    final_dir = output_root / "final_fit" / dimension
    checkpoint, mean, std = train_fixed_epochs(
        dimension, records, config, final_epochs, final_dir, seed + 9999, device
    )
    summary = {
        "dimension": dimension,
        "model_id": f"pmemo_final_clean_{dimension}_teacher_v1",
        "selection_boundary": "strict song-level inner validation and outer OOF; final locked labels excluded",
        "outer_fold_policy": "inner epoch selection followed by complete outer-train refit",
        "input": "MERT hidden-6 + OpenSMILE dynamic pure-audio features",
        "development_songs": len(records),
        "per_fold": per_fold,
        "selected_epochs": selected_epochs,
        "final_fit_epochs": final_epochs,
        "oof_frame_macro": frame_macro,
        "oof_clip_macro": clip_macro,
        "oof_clip_thresholds": threshold_rows(clip_macro),
        "final_fit_checkpoint": str(checkpoint),
        "final_locked_test": "labels excluded",
    }
    (output_root / f"summary_{dimension}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, checkpoint, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--valence-teacher", type=Path, default=DEFAULT_VALENCE_TEACHER)
    parser.add_argument("--arousal-teacher", type=Path, default=DEFAULT_AROUSAL_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_index_path = args.split_dir / "final_locked_test_index.csv"
    final_index = pd.read_csv(final_index_path)
    final_ids = set(final_index["musicId"].astype(int))
    if len(final_ids) != 115:
        raise RuntimeError(f"Expected 115 final songs, got {len(final_ids)}")

    device = torch.device(args.device)
    dimension_inputs = {
        "valence": args.valence_teacher,
        "arousal": args.arousal_teacher,
    }
    summaries = {}
    final_components = {}
    for dimension, teacher_dir in dimension_inputs.items():
        records = load_development_records(
            dimension, teacher_dir, args.mert_dir, args.sequence_dir, final_ids
        )
        summary, checkpoint, mean, std = run_dimension(
            dimension, records, config, args.output_dir, device
        )
        summaries[dimension] = summary
        final_components[dimension] = (checkpoint, mean, std)

    valence = predict_final_dimension(
        "valence", *final_components["valence"], final_index,
        args.mert_dir, args.sequence_dir, config, device,
    )
    arousal = predict_final_dimension(
        "arousal", *final_components["arousal"], final_index,
        args.mert_dir, args.sequence_dir, config, device,
    )
    predictions = valence.merge(arousal, on=["musicId", "time_sec"], validate="one_to_one")
    expected_frames = int(final_index["frames"].sum())
    if len(predictions) != expected_frames or predictions["musicId"].nunique() != 115:
        raise RuntimeError(
            f"Final prediction coverage mismatch: frames={len(predictions)}/{expected_frames}, "
            f"songs={predictions['musicId'].nunique()}/115"
        )
    if not np.isfinite(predictions[["pred_valence", "pred_arousal"]].to_numpy()).all():
        raise RuntimeError("Non-finite final predictions")
    prediction_path = args.output_dir / "final_locked_predictions_LABEL_FREE.csv"
    predictions.to_csv(prediction_path, index=False)
    manifest = {
        "created_at": "2026-07-15",
        "protocol": "prospective final prediction generated before label access",
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "final_index_path": str(final_index_path),
        "final_index_sha256": sha256(final_index_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "songs": int(predictions["musicId"].nunique()),
        "frames": len(predictions),
        "prediction_columns": list(predictions.columns),
        "labels_read_during_prediction": False,
        "model_summaries": summaries,
        "post_test_tuning_allowed": False,
    }
    manifest_path = args.output_dir / "FINAL_PREDICTION_FREEZE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "label_free_predictions_frozen",
        "prediction_path": str(prediction_path),
        "prediction_sha256": manifest["prediction_sha256"],
        "songs": manifest["songs"],
        "frames": manifest["frames"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
