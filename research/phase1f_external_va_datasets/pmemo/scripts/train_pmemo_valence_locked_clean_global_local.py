#!/usr/bin/env python3
"""Train a clean pure-audio global/local Valence model on locked PMEmo folds."""

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_REFERENCE = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_MERT = ROOT / "outputs/mert_pmemo_hidden6/features"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_effnet_moodtheme_tags_20260714/per_song"
DEFAULT_FINAL = ROOT / "data/splits/pmemo_final_locked_20260713/final_locked_test_index.csv"
DEFAULT_OUT = ROOT / "outputs/valence_locked_clean_global_local_20260714"
REFERENCE_METRICS = {
    "rmse": 0.18154631579428374,
    "mae": 0.14231915639241396,
    "pcc": 0.8234038487655991,
    "ccc": 0.8068739659711618,
}


@dataclass(frozen=True)
class Config:
    seed: int = 20260714
    local_projection_dim: int = 128
    local_hidden_dim: int = 64
    global_hidden_dim: int = 64
    dropout: float = 0.20
    local_scale: float = 0.50
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    batch_size: int = 20
    max_epochs: int = 40
    patience: int = 7
    inner_val_fraction: float = 0.15
    song_mean_loss_weight: float = 0.50
    ccc_loss_weight: float = 0.05
    smoothness_loss_weight: float = 0.01


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SongDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    max_length = max(len(row["target"]) for row in batch)
    frame_dim = batch[0]["frame"].shape[1]
    global_dim = batch[0]["global"].shape[0]
    frame = torch.zeros(len(batch), max_length, frame_dim, dtype=torch.float32)
    global_feature = torch.zeros(len(batch), global_dim, dtype=torch.float32)
    target = torch.zeros(len(batch), max_length, dtype=torch.float32)
    time = torch.zeros(len(batch), max_length, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    song_id = torch.zeros(len(batch), dtype=torch.long)
    for index, row in enumerate(batch):
        length = len(row["target"])
        frame[index, :length] = torch.from_numpy(row["frame"])
        global_feature[index] = torch.from_numpy(row["global"])
        target[index, :length] = torch.from_numpy(row["target"])
        time[index, :length] = torch.from_numpy(row["time"])
        mask[index, :length] = True
        lengths[index] = length
        song_id[index] = int(row["song_id"])
    return frame, global_feature, target, time, mask, lengths, song_id


class GlobalLocalValence(nn.Module):
    def __init__(self, frame_dim: int, global_dim: int, config: Config):
        super().__init__()
        self.local_scale = float(config.local_scale)
        self.local_projection = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, config.local_projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.local_gru = nn.GRU(
            config.local_projection_dim,
            config.local_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        local_width = config.local_hidden_dim * 2
        self.local_head = nn.Sequential(
            nn.LayerNorm(local_width),
            nn.Dropout(config.dropout),
            nn.Linear(local_width, 1),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, config.global_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.global_hidden_dim, 1),
        )

    def forward(
        self,
        frame: torch.Tensor,
        global_feature: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.local_projection(frame)
        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_hidden, _ = self.local_gru(packed)
        hidden, _ = pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=frame.shape[1]
        )
        local = torch.tanh(self.local_head(hidden).squeeze(-1)) * self.local_scale
        local = (local - masked_mean(local, mask).unsqueeze(1)) * mask.float()
        global_mean = torch.tanh(self.global_head(global_feature).squeeze(-1))
        prediction = torch.clamp(global_mean.unsqueeze(1) + local, -1.0, 1.0)
        return prediction, global_mean, local


def load_records(reference_dir: Path, mert_dir: Path, tag_dir: Path, final_index: Path):
    final_ids = set(pd.read_csv(final_index)["musicId"].astype(int))
    output: dict[str, list[dict[str, Any]]] = {}
    development_ids: set[int] = set()
    for fold in FOLDS:
        frame = pd.read_csv(reference_dir / f"{fold}_val_predictions.csv")
        rows = []
        for music_id, indices in frame.groupby("musicId", sort=False).groups.items():
            music_id = int(music_id)
            if music_id in final_ids:
                raise ValueError(f"Final locked song reached clean model: {music_id}")
            development_ids.add(music_id)
            loc = np.asarray(indices, dtype=np.int64)
            feature_file = np.load(mert_dir / f"{music_id}.npz")
            mert = feature_file["embedding"].astype(np.float32)
            mert_time = feature_file["time_sec"].astype(np.float32)
            target_time = frame.loc[loc, "time_sec"].to_numpy(dtype=np.float32)
            if len(mert) != len(loc) or not np.allclose(mert_time, target_time, atol=1e-4):
                raise ValueError(f"MERT/label alignment mismatch for song {music_id}")
            tags = np.load(tag_dir / f"{music_id}.npz")["tags"].astype(np.float32)
            global_raw = np.concatenate([tags.mean(axis=0), tags.std(axis=0), tags.max(axis=0)])
            rows.append({
                "song_id": music_id,
                "frame_raw": mert,
                "global_raw": global_raw.astype(np.float32),
                "target": frame.loc[loc, "target_valence"].to_numpy(dtype=np.float32),
                "time": target_time,
            })
        output[fold] = rows
    if development_ids.intersection(final_ids):
        raise ValueError("Final locked leakage after record construction")
    return output, development_ids


def fit_scalers(rows: list[dict[str, Any]]) -> tuple[StandardScaler, StandardScaler]:
    frame_scaler = StandardScaler().fit(np.concatenate([row["frame_raw"] for row in rows]))
    global_scaler = StandardScaler().fit(np.stack([row["global_raw"] for row in rows]))
    return frame_scaler, global_scaler


def transform(rows, frame_scaler: StandardScaler, global_scaler: StandardScaler):
    output = []
    for row in rows:
        item = dict(row)
        item["frame"] = frame_scaler.transform(row["frame_raw"]).astype(np.float32)
        item["global"] = global_scaler.transform(row["global_raw"][None])[0].astype(np.float32)
        output.append(item)
    return output


def stratified_inner_split(rows: list[dict[str, Any]], config: Config, outer_index: int):
    means = np.asarray([row["target"].mean() for row in rows])
    ranked = pd.Series(means).rank(method="first")
    bins = pd.qcut(ranked, q=8, labels=False, duplicates="drop").to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.inner_val_fraction,
        random_state=config.seed + outer_index,
    )
    train_indices, val_indices = next(splitter.split(np.zeros(len(rows)), bins))
    return [rows[i] for i in train_indices], [rows[i] for i in val_indices]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tables = []
    for frame, global_feature, target, time, mask, lengths, song_id in loader:
        prediction, global_mean, local = model(
            frame.to(device), global_feature.to(device), mask.to(device), lengths
        )
        prediction = prediction.cpu()
        global_mean = global_mean.cpu()
        local = local.cpu()
        for index in range(len(song_id)):
            valid = mask[index]
            tables.append(pd.DataFrame({
                "musicId": int(song_id[index]),
                "time_sec": time[index, valid].numpy(),
                "target_valence": target[index, valid].numpy(),
                "pred_valence": prediction[index, valid].numpy(),
                "pred_global_mean": float(global_mean[index]),
                "pred_local_residual": local[index, valid].numpy(),
            }))
    table = pd.concat(tables, ignore_index=True)
    return table, metrics(table["pred_valence"], table["target_valence"])


def selection_score(result: dict[str, float]) -> float:
    return result["rmse"] + 0.05 * max(0.0, 0.80 - result["ccc"]) + 0.03 * max(0.0, 0.80 - result["pcc"])


def train_model(model, loader, device, config: Config, epochs: int, val_loader=None):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_epoch = epochs
    best_score = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for frame, global_feature, target, _, mask, lengths, _ in loader:
            frame = frame.to(device)
            global_feature = global_feature.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            prediction, global_mean, local = model(frame, global_feature, mask, lengths)
            frame_loss = (((prediction - target).square() * mask.float()).sum(dim=1)
                          / mask.sum(dim=1).clamp_min(1)).mean()
            target_mean = masked_mean(target, mask)
            song_mean_loss = nn.functional.mse_loss(global_mean, target_mean)
            valid_prediction = prediction[mask]
            valid_target = target[mask]
            transition_mask = mask[:, 1:] & mask[:, :-1]
            smoothness = local.diff(dim=1)[transition_mask].square().mean()
            loss = frame_loss
            loss = loss + config.song_mean_loss_weight * song_mean_loss
            loss = loss + config.ccc_loss_weight * ccc_loss(valid_prediction, valid_target)
            loss = loss + config.smoothness_loss_weight * smoothness
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            _, val_metrics = evaluate(model, val_loader, device)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = selection_score(val_metrics)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                bad_epochs += 1
            print(
                f"epoch={epoch:02d} loss={row['train_loss']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} val_pcc={val_metrics['pcc']:.4f} "
                f"val_ccc={val_metrics['ccc']:.4f}",
                flush=True,
            )
        history.append(row)
        if val_loader is not None and bad_epochs >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def loader(rows, config: Config, shuffle: bool):
    return DataLoader(
        SongDataset(rows),
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    args = parser.parse_args()
    config = Config(max_epochs=args.max_epochs or Config.max_epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    raw, development_ids = load_records(args.reference_dir, args.mert_dir, args.tag_dir, args.final_index)
    per_fold = []
    for outer_index, held in enumerate(args.folds, start=1):
        print(f"outer={held} inner-selection", flush=True)
        outer_train = [row for fold in FOLDS if fold != held for row in raw[fold]]
        inner_train, inner_val = stratified_inner_split(outer_train, config, outer_index)
        frame_scaler, global_scaler = fit_scalers(inner_train)
        train_rows = transform(inner_train, frame_scaler, global_scaler)
        val_rows = transform(inner_val, frame_scaler, global_scaler)
        set_seed(config.seed + outer_index)
        model = GlobalLocalValence(768, 168, config).to(device)
        model, best_epoch, history = train_model(
            model, loader(train_rows, config, True), device, config, config.max_epochs,
            loader(val_rows, config, False),
        )
        fold_dir = args.out_dir / held
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(fold_dir / "inner_history.csv", index=False)

        print(f"outer={held} refit_epochs={best_epoch}", flush=True)
        frame_scaler, global_scaler = fit_scalers(outer_train)
        train_rows = transform(outer_train, frame_scaler, global_scaler)
        held_rows = transform(raw[held], frame_scaler, global_scaler)
        set_seed(config.seed + 100 + outer_index)
        model = GlobalLocalValence(768, 168, config).to(device)
        model, _, refit_history = train_model(
            model, loader(train_rows, config, True), device, config, best_epoch
        )
        torch.save({
            "model_state": model.state_dict(),
            "config": asdict(config),
            "held_fold": held,
            "best_epoch": best_epoch,
            "frame_scaler_mean": frame_scaler.mean_,
            "frame_scaler_scale": frame_scaler.scale_,
            "global_scaler_mean": global_scaler.mean_,
            "global_scaler_scale": global_scaler.scale_,
        }, fold_dir / "model.pt")
        pd.DataFrame(refit_history).to_csv(fold_dir / "refit_history.csv", index=False)
        table, fold_metrics = evaluate(model, loader(held_rows, config, False), device)
        table.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        row = {"fold": held, "selected_epoch": best_epoch, **fold_metrics}
        per_fold.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result = macro(per_fold)
    guarded = len(args.folds) == 3 and (
        result["rmse"] < REFERENCE_METRICS["rmse"]
        and result["mae"] <= REFERENCE_METRICS["mae"]
        and result["pcc"] >= REFERENCE_METRICS["pcc"]
        and result["ccc"] >= REFERENCE_METRICS["ccc"]
    )
    summary = {
        "created_at": "2026-07-14",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Song-level outer folds; stratified inner validation for epoch selection; final locked test excluded",
        "inputs": "MERT hidden-6 frame sequence + pooled MTG mood/theme EffNet tags; pure audio only",
        "architecture": "EffNet global song-mean MLP + MERT BiGRU zero-mean local residual",
        "config": asdict(config),
        "development_songs": len(development_ids),
        "final_locked_overlap": 0,
        "reference_macro": REFERENCE_METRICS,
        "macro": result,
        "per_fold": per_fold,
        "delta_vs_reference": {key: result[key] - REFERENCE_METRICS[key] for key in REFERENCE_METRICS},
        "guarded_promotion": guarded,
        "gap_rows": threshold_rows(result),
        "final_locked_test": "excluded",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
