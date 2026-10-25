#!/usr/bin/env python3
"""Gate a zero-initialized low-rank MERT Valence residual on locked song folds."""

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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, threshold_rows
from train_pmemo_valence_locked_joint_semantic_residual import (
    SongDataset,
    collate,
    evaluate,
    macro,
    set_seed,
)


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_global_shift_amp_guarded_20260715"
DEFAULT_MERT = ROOT / "outputs/mert_pmemo_h468_concat/features"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_effnet_moodtheme_tags_20260714/per_song"
DEFAULT_FINAL = ROOT / "data/splits/pmemo_final_locked_20260713/final_locked_test_index.csv"
DEFAULT_OUT = ROOT / "outputs/valence_locked_mert_lowrank_residual_20260715"


@dataclass(frozen=True)
class Config:
    seed: int = 20260715
    frame_proj_dim: int = 64
    hidden_dim: int = 64
    song_context_dim: int = 24
    dropout: float = 0.30
    frame_cap: float = 0.010
    song_cap: float = 0.015
    learning_rate: float = 1e-4
    weight_decay: float = 2e-3
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 10
    inner_val_fraction: float = 0.20
    song_mean_loss_weight: float = 0.80
    ccc_loss_weight: float = 0.02
    smoothness_loss_weight: float = 0.02
    multiscale_loss_weight: float = 0.50
    correction_l2_weight: float = 0.01


class LowRankTemporalResidual(nn.Module):
    def __init__(self, frame_dim: int, song_dim: int, config: Config):
        super().__init__()
        self.frame_cap = float(config.frame_cap)
        self.song_cap = float(config.song_cap)
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, config.frame_proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(config.frame_proj_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.song_projection = nn.Sequential(
            nn.LayerNorm(song_dim),
            nn.Linear(song_dim, config.song_context_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.frame_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim + config.song_context_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim + config.song_context_dim, 1),
        )
        pooled_dim = config.hidden_dim * 3 + config.song_context_dim
        self.song_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.frame_head[-1].weight)
        nn.init.zeros_(self.frame_head[-1].bias)
        nn.init.zeros_(self.song_head[-1].weight)
        nn.init.zeros_(self.song_head[-1].bias)

    def forward(
        self,
        frame: torch.Tensor,
        song: torch.Tensor,
        base: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.temporal(self.frame_projection(frame).transpose(1, 2)).transpose(1, 2)
        song_context = self.song_projection(song)
        repeated_context = song_context[:, None].expand(-1, hidden.shape[1], -1)
        frame_shift = self.frame_cap * torch.tanh(
            self.frame_head(torch.cat([hidden, repeated_context], dim=-1)).squeeze(-1)
        )
        mask_float = mask.float().unsqueeze(-1)
        count = mask_float.sum(dim=1).clamp_min(1.0)
        mean = (hidden * mask_float).sum(dim=1) / count
        variance = (((hidden - mean[:, None]).square()) * mask_float).sum(dim=1) / count
        maximum = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        pooled = torch.cat([mean, variance.clamp_min(1e-6).sqrt(), maximum, song_context], dim=-1)
        song_shift = self.song_cap * torch.tanh(self.song_head(pooled).squeeze(-1))
        correction = frame_shift + song_shift[:, None]
        prediction = torch.clamp(base + correction, -1.0, 1.0)
        return prediction, correction


def base_summary(values: np.ndarray) -> np.ndarray:
    position = np.linspace(-1.0, 1.0, len(values), dtype=np.float32)
    slope = float(np.polyfit(position, values, 1)[0]) if len(values) > 1 else 0.0
    return np.asarray([
        values.mean(), values.std(), values.min(), values.max(),
        *np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90]),
        values[0], values[-1], slope,
    ], dtype=np.float32)


def load_records(
    base_dir: Path,
    mert_dir: Path,
    tag_dir: Path,
    final_index: Path,
) -> dict[str, list[dict[str, Any]]]:
    final_ids = set(pd.read_csv(final_index)["musicId"].astype(int))
    development_ids: set[int] = set()
    resampled_ids: list[int] = []
    output: dict[str, list[dict[str, Any]]] = {}
    for fold in FOLDS:
        frame_table = pd.read_csv(base_dir / f"{fold}_val_predictions.csv")
        records = []
        for music_id, song in frame_table.groupby("musicId", sort=False):
            music_id = int(music_id)
            if music_id in final_ids:
                raise ValueError(f"Final locked song reached residual training: {music_id}")
            feature = np.load(mert_dir / f"{music_id}.npz")
            embedding = feature["embedding"].astype(np.float32)
            feature_time = feature["time_sec"].astype(np.float32)
            target_time = song["time_sec"].to_numpy(dtype=np.float32)
            if len(embedding) != len(song) or not np.allclose(feature_time, target_time, atol=1e-4):
                if len(feature_time) < 2 or np.any(np.diff(feature_time) <= 0):
                    raise ValueError(f"Cannot resample MERT features for {music_id}")
                embedding = np.stack([
                    np.interp(target_time, feature_time, embedding[:, column])
                    for column in range(embedding.shape[1])
                ], axis=1).astype(np.float32)
                resampled_ids.append(music_id)
            tags = np.load(tag_dir / f"{music_id}.npz")["tags"].astype(np.float32)
            base = song["pred_valence"].to_numpy(dtype=np.float32)
            tag_summary = np.concatenate([tags.mean(axis=0), tags.std(axis=0), tags.max(axis=0)])
            records.append({
                "song_id": music_id,
                "frame": embedding,
                "song": np.concatenate([tag_summary, base_summary(base)]).astype(np.float32),
                "base": base,
                "target": song["target_valence"].to_numpy(dtype=np.float32),
                "time": target_time,
            })
            development_ids.add(music_id)
        output[fold] = records
    if len(development_ids) != 652 or development_ids.intersection(final_ids):
        raise ValueError("Development coverage or final isolation failed")
    print(json.dumps({
        "development_songs": len(development_ids),
        "resampled_feature_songs": len(resampled_ids),
        "resampled_music_ids": sorted(resampled_ids),
        "final_locked_overlap": 0,
    }), flush=True)
    return output


def inner_split(
    records: list[dict[str, Any]], config: Config, outer_index: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    means = np.asarray([row["target"].mean() for row in records])
    ranks = pd.Series(means).rank(method="first")
    bins = pd.qcut(ranks, q=8, labels=False, duplicates="drop").to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.inner_val_fraction,
        random_state=config.seed + outer_index,
    )
    train_index, val_index = next(splitter.split(np.zeros(len(records)), bins))
    return [records[index] for index in train_index], [records[index] for index in val_index]


def loader(records: list[dict[str, Any]], config: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        SongDataset(records), batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate
    )


def multiscale_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    windows: tuple[int, ...] = (4, 10, 20),
) -> torch.Tensor:
    losses = []
    for window in windows:
        pred_windows = []
        target_windows = []
        for row in range(len(prediction)):
            length = int(mask[row].sum().item())
            for start in range(0, length, window):
                end = min(start + window, length)
                pred_windows.append(prediction[row, start:end].mean())
                target_windows.append(target[row, start:end].mean())
        losses.append(nn.functional.mse_loss(torch.stack(pred_windows), torch.stack(target_windows)))
    return torch.stack(losses).mean()


def selection_score(result: dict[str, float]) -> float:
    return (
        result["rmse"]
        + 0.05 * max(0.0, 0.80 - result["ccc"])
        + 0.03 * max(0.0, 0.80 - result["pcc"])
    )


def train_multiscale_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: Config,
    epochs: int,
    val_loader: DataLoader | None = None,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_epoch = epochs
    best_score = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for frame, song, base, target, _, mask, _ in train_loader:
            frame, song = frame.to(device), song.to(device)
            base, target, mask = base.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad()
            prediction, correction = model(frame, song, base, mask)
            squared = (prediction - target).square() * mask.float()
            frame_loss = (squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean()
            target_mean = (target * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            prediction_mean = (prediction * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            song_mean_loss = nn.functional.mse_loss(prediction_mean, target_mean)
            valid_prediction = prediction[mask]
            valid_target = target[mask]
            smooth_mask = mask[:, 1:] & mask[:, :-1]
            smoothness = correction.diff(dim=1)[smooth_mask].square().mean()
            loss = frame_loss
            loss = loss + config.song_mean_loss_weight * song_mean_loss
            loss = loss + config.ccc_loss_weight * ccc_loss(valid_prediction, valid_target)
            loss = loss + config.smoothness_loss_weight * smoothness
            loss = loss + config.multiscale_loss_weight * multiscale_mse(prediction, target, mask)
            loss = loss + config.correction_l2_weight * correction[mask].square().mean()
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
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                bad_epochs += 1
            print(
                f"epoch={epoch:02d} loss={row['train_loss']:.6f} "
                f"val_rmse={result['rmse']:.6f} val_pcc={result['pcc']:.4f} "
                f"val_ccc={result['ccc']:.4f}", flush=True,
            )
        history.append(row)
        if val_loader is not None and bad_epochs >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--frame-cap", type=float, default=Config.frame_cap)
    parser.add_argument("--song-cap", type=float, default=Config.song_cap)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--multiscale-weight", type=float, default=Config.multiscale_loss_weight)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    config = Config(
        max_epochs=args.max_epochs or Config.max_epochs,
        frame_cap=args.frame_cap,
        song_cap=args.song_cap,
        learning_rate=args.learning_rate,
        multiscale_loss_weight=args.multiscale_weight,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_records(args.base_dir, args.mert_dir, args.tag_dir, args.final_index)
    device = torch.device(args.device)
    per_fold: list[dict[str, Any]] = []
    for outer_index, held in enumerate(args.folds, start=1):
        outer_train = [row for fold in FOLDS if fold != held for row in raw[fold]]
        train_records, val_records = inner_split(outer_train, config, outer_index)
        print(f"outer={held} inner-selection", flush=True)
        set_seed(config.seed + outer_index)
        model = LowRankTemporalResidual(2304, 180, config).to(device)
        model, best_epoch, history = train_multiscale_epochs(
            model,
            loader(train_records, config, True),
            device,
            config,
            config.max_epochs,
            loader(val_records, config, False),
        )
        fold_dir = args.out_dir / held
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(fold_dir / "inner_history.csv", index=False)

        print(f"outer={held} refit_epochs={best_epoch}", flush=True)
        set_seed(config.seed + 100 + outer_index)
        model = LowRankTemporalResidual(2304, 180, config).to(device)
        model, _, refit_history = train_multiscale_epochs(
            model, loader(outer_train, config, True), device, config, best_epoch
        )
        pd.DataFrame(refit_history).to_csv(fold_dir / "refit_history.csv", index=False)
        torch.save({
            "model_state": model.state_dict(),
            "config": asdict(config),
            "held_fold": held,
            "best_epoch": best_epoch,
        }, fold_dir / "model.pt")
        held_eval = evaluate(model, loader(raw[held], config, False), device)
        held_eval["table"].to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        row = {"fold": held, "selected_epoch": best_epoch, **held_eval["metrics"]}
        per_fold.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary: dict[str, Any] = {
        "created_at": "2026-07-15",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Stratified inner epoch selection; held song fold; final locked test excluded",
        "inputs": "Current robust Valence skip + frozen MERT hidden-4/6/8 + EffNet mood/theme tags",
        "architecture": "Zero-initialized low-rank temporal residual with frame and song heads",
        "config": asdict(config),
        "per_fold": per_fold,
        "final_locked_overlap": 0,
        "final_locked_test": "excluded",
    }
    if len(args.folds) == 3:
        result = macro(per_fold)
        summary["macro"] = result
        summary["gap_rows"] = threshold_rows(result)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
