#!/usr/bin/env python3
"""Train a leakage-safe MERT attention song-mean head on locked PMEmo folds."""

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
from torch.utils.data import DataLoader, Dataset

from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_global_shift_amp_guarded_20260715"
DEFAULT_MERT = ROOT / "outputs/mert_pmemo_h468_concat/features"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_effnet_moodtheme_tags_20260714/per_song"
DEFAULT_FINAL = ROOT / "data/splits/pmemo_final_locked_20260713/final_locked_test_index.csv"
DEFAULT_OUT = ROOT / "outputs/valence_locked_mert_attention_songmean_20260715"


@dataclass(frozen=True)
class Config:
    seed: int = 20260715
    projection_dim: int = 96
    attention_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 20
    max_epochs: int = 80
    patience: int = 12
    inner_val_fraction: float = 0.20
    ccc_weight: float = 0.03
    rank_weight: float = 0.03
    target_mode: str = "direct"
    residual_scale: float = 0.20


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


class SongDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    max_length = max(len(row["frame"]) for row in batch)
    frame_dim = batch[0]["frame"].shape[1]
    tag_dim = batch[0]["tags"].shape[0]
    frame = torch.zeros(len(batch), max_length, frame_dim, dtype=torch.float32)
    tags = torch.zeros(len(batch), tag_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    target = torch.zeros(len(batch), dtype=torch.float32)
    base_mean = torch.zeros(len(batch), dtype=torch.float32)
    song_id = torch.zeros(len(batch), dtype=torch.long)
    for index, row in enumerate(batch):
        length = len(row["frame"])
        frame[index, :length] = torch.from_numpy(row["frame"])
        tags[index] = torch.from_numpy(row["tags"])
        mask[index, :length] = True
        target[index] = float(row["target_mean"])
        base_mean[index] = float(row["base_mean"])
        song_id[index] = int(row["musicId"])
    return frame, tags, mask, target, base_mean, song_id


class AttentionSongMean(nn.Module):
    def __init__(self, frame_dim: int, tag_dim: int, config: Config):
        super().__init__()
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(config.projection_dim, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1),
        )
        pooled_dim = config.projection_dim * 4
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + tag_dim),
            nn.Linear(pooled_dim + tag_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.target_mode = config.target_mode
        self.residual_scale = float(config.residual_scale)
        if self.target_mode == "residual":
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, frame: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.frame_projection(frame)
        score = self.attention(hidden).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        attention_mean = (hidden * weight).sum(dim=1)
        attention_var = ((hidden - attention_mean.unsqueeze(1)).square() * weight).sum(dim=1)
        mask_float = mask.float().unsqueeze(-1)
        plain_mean = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        masked_hidden = hidden.masked_fill(~mask.unsqueeze(-1), -1e4)
        maximum = masked_hidden.max(dim=1).values
        pooled = torch.cat([attention_mean, attention_var.clamp_min(1e-6).sqrt(), plain_mean, maximum, tags], dim=-1)
        output = torch.tanh(self.head(pooled).squeeze(-1))
        return output * self.residual_scale if self.target_mode == "residual" else output


def ranking_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_diff = prediction[:, None] - prediction[None, :]
    target_diff = target[:, None] - target[None, :]
    mask = target_diff.abs() >= 0.05
    if not mask.any():
        return prediction.new_zeros(())
    return nn.functional.smooth_l1_loss(pred_diff[mask], target_diff[mask])


def load_rows(base_dir: Path, mert_dir: Path, tag_dir: Path, final_index: Path) -> dict[str, list[dict[str, Any]]]:
    final_ids = set(pd.read_csv(final_index)["musicId"].astype(int))
    result: dict[str, list[dict[str, Any]]] = {}
    development_ids: set[int] = set()
    resampled_ids: list[int] = []
    for fold in FOLDS:
        frame_table = pd.read_csv(base_dir / f"{fold}_val_predictions.csv")
        rows = []
        for music_id, song in frame_table.groupby("musicId", sort=False):
            music_id = int(music_id)
            if music_id in final_ids:
                raise ValueError(f"Final locked song reached training data: {music_id}")
            mert = np.load(mert_dir / f"{music_id}.npz")
            embedding = mert["embedding"].astype(np.float32)
            time_sec = mert["time_sec"].astype(np.float32)
            target_time = song["time_sec"].to_numpy(dtype=np.float32)
            if len(embedding) != len(song) or not np.allclose(time_sec, target_time, atol=1e-4):
                if len(time_sec) < 2 or np.any(np.diff(time_sec) <= 0):
                    raise ValueError(f"Cannot resample MERT features for {music_id}")
                embedding = np.stack([
                    np.interp(target_time, time_sec, embedding[:, column])
                    for column in range(embedding.shape[1])
                ], axis=1).astype(np.float32)
                resampled_ids.append(music_id)
            tags = np.load(tag_dir / f"{music_id}.npz")["tags"].astype(np.float32)
            tag_summary = np.concatenate([tags.mean(axis=0), tags.std(axis=0), tags.max(axis=0)]).astype(np.float32)
            base = song["pred_valence"].to_numpy(dtype=np.float32)
            position = np.linspace(-1.0, 1.0, len(base), dtype=np.float32)
            slope = float(np.polyfit(position, base, 1)[0]) if len(base) > 1 else 0.0
            base_summary = np.asarray([
                base.mean(), base.std(), base.min(), base.max(),
                *np.quantile(base, [0.10, 0.25, 0.50, 0.75, 0.90]),
                base[0], base[-1], slope,
            ], dtype=np.float32)
            rows.append({
                "fold": fold,
                "musicId": music_id,
                "frame": embedding,
                "tags": np.concatenate([tag_summary, base_summary]).astype(np.float32),
                "target_mean": float(song["target_valence"].mean()),
                "base_mean": float(base.mean()),
            })
            development_ids.add(music_id)
        result[fold] = rows
    if len(development_ids) != 652 or development_ids.intersection(final_ids):
        raise ValueError("Development coverage or final-lock isolation failed")
    print(json.dumps({
        "development_songs": len(development_ids),
        "resampled_feature_songs": len(resampled_ids),
        "resampled_music_ids": sorted(resampled_ids),
        "final_locked_overlap": 0,
    }), flush=True)
    return result


def inner_split(rows: list[dict[str, Any]], config: Config, outer_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = np.asarray([row["target_mean"] for row in rows])
    ranks = pd.Series(targets).rank(method="first")
    bins = pd.qcut(ranks, q=8, labels=False, duplicates="drop").to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.inner_val_fraction,
        random_state=config.seed + outer_index,
    )
    train_index, val_index = next(splitter.split(np.zeros(len(rows)), bins))
    return [rows[index] for index in train_index], [rows[index] for index in val_index]


def make_loader(rows: list[dict[str, Any]], config: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        SongDataset(rows), batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate
    )


@torch.no_grad()
def evaluate_song(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows = []
    for frame, tags, mask, target, base_mean, song_id in loader:
        output = model(frame.to(device), tags.to(device), mask.to(device)).cpu()
        prediction = base_mean + output if model.target_mode == "residual" else output
        for sid, truth, pred in zip(song_id.numpy(), target.numpy(), prediction.numpy()):
            rows.append({"musicId": int(sid), "target_song_mean": float(truth), "pred_song_mean": float(pred)})
    table = pd.DataFrame(rows)
    return table, metrics(table["pred_song_mean"], table["target_song_mean"])


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    epochs: int,
    val_loader: DataLoader | None = None,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_epoch = epochs
    best_score = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for frame, tags, mask, target, base_mean, _ in loader:
            frame, tags, mask, target = frame.to(device), tags.to(device), mask.to(device), target.to(device)
            base_mean = base_mean.to(device)
            optimizer.zero_grad()
            output = model(frame, tags, mask)
            prediction = base_mean + output if model.target_mode == "residual" else output
            loss = nn.functional.mse_loss(prediction, target)
            loss = loss + config.ccc_weight * ccc_loss(prediction, target)
            loss = loss + config.rank_weight * ranking_loss(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            _, val_metrics = evaluate_song(model, val_loader, device)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            score = val_metrics["rmse"] + 0.03 * max(0.0, 0.80 - val_metrics["pcc"]) + 0.03 * max(0.0, 0.80 - val_metrics["ccc"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                bad_epochs += 1
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.6f} "
                f"song_rmse={val_metrics['rmse']:.6f} song_pcc={val_metrics['pcc']:.4f} "
                f"song_ccc={val_metrics['ccc']:.4f}", flush=True,
            )
        history.append(row)
        if val_loader is not None and bad_epochs >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def apply_correction(
    base: pd.DataFrame,
    song_prediction: pd.DataFrame,
    strength: float,
    cap: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    output = base.copy()
    mapping = song_prediction.set_index("musicId")["pred_song_mean"].to_dict()
    predicted_mean = output["musicId"].astype(int).map(mapping)
    if predicted_mean.isna().any():
        raise ValueError("Missing song-mean prediction")
    base_mean = output.groupby("musicId")["pred_valence"].transform("mean")
    shift = np.clip(strength * (predicted_mean - base_mean), -cap, cap)
    output["base_pred_valence"] = output["pred_valence"]
    output["attention_pred_song_mean"] = predicted_mean
    output["pred_valence"] = np.clip(output["pred_valence"] + shift, -1.0, 1.0)
    return output, metrics(output["pred_valence"], output["target_valence"])


def select_correction(base: pd.DataFrame, song_prediction: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    base_metrics = metrics(base["pred_valence"], base["target_valence"])
    rows = [{"strength": 0.0, "cap": 0.0, **base_metrics, "guarded": True}]
    for strength in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        for cap in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]:
            _, result = apply_correction(base, song_prediction, strength, cap)
            guarded = (
                result["mae"] <= base_metrics["mae"]
                and result["pcc"] >= base_metrics["pcc"]
                and result["ccc"] >= base_metrics["ccc"]
            )
            rows.append({"strength": strength, "cap": cap, **result, "guarded": guarded})
    table = pd.DataFrame(rows)
    eligible = table[table["guarded"]]
    selected = eligible.sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False]).iloc[0]
    return float(selected["strength"]), float(selected["cap"]), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    config = Config(
        max_epochs=args.max_epochs or Config.max_epochs,
        target_mode=args.target_mode,
        residual_scale=args.residual_scale,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_rows(args.base_dir, args.mert_dir, args.tag_dir, args.final_index)
    device = torch.device(args.device)
    per_fold = []
    for outer_index, held in enumerate(args.folds, start=1):
        print(f"outer={held} inner-selection", flush=True)
        outer_train = [row for fold in FOLDS if fold != held for row in raw[fold]]
        inner_train, inner_val = inner_split(outer_train, config, outer_index)
        set_seed(config.seed + outer_index)
        model = AttentionSongMean(2304, 180, config).to(device)
        model, best_epoch, history = train_model(
            model,
            make_loader(inner_train, config, True),
            device,
            config,
            config.max_epochs,
            make_loader(inner_val, config, False),
        )
        fold_dir = args.out_dir / held
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(fold_dir / "inner_history.csv", index=False)
        inner_prediction, inner_song_metrics = evaluate_song(model, make_loader(inner_val, config, False), device)
        base_all = pd.concat(
            [pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS],
            ignore_index=True,
        )
        inner_ids = set(inner_prediction["musicId"].astype(int))
        inner_base = base_all[base_all["musicId"].astype(int).isin(inner_ids)].copy()
        strength, cap, correction_table = select_correction(inner_base, inner_prediction)
        correction_table.to_csv(fold_dir / "inner_correction_search.csv", index=False)

        print(f"outer={held} refit_epochs={best_epoch} strength={strength} cap={cap}", flush=True)
        set_seed(config.seed + 100 + outer_index)
        model = AttentionSongMean(2304, 180, config).to(device)
        model, _, refit_history = train_model(
            model, make_loader(outer_train, config, True), device, config, best_epoch
        )
        pd.DataFrame(refit_history).to_csv(fold_dir / "refit_history.csv", index=False)
        torch.save({
            "model_state": model.state_dict(),
            "config": asdict(config),
            "held_fold": held,
            "best_epoch": best_epoch,
            "correction_strength": strength,
            "correction_cap": cap,
        }, fold_dir / "model.pt")
        held_prediction, held_song_metrics = evaluate_song(model, make_loader(raw[held], config, False), device)
        held_prediction.to_csv(fold_dir / "held_song_predictions.csv", index=False)
        held_base = pd.read_csv(args.base_dir / f"{held}_val_predictions.csv")
        output, held_frame_metrics = apply_correction(held_base, held_prediction, strength, cap)
        output.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        row = {
            "fold": held,
            "selected_epoch": best_epoch,
            "selected_strength": strength,
            "selected_cap": cap,
            "inner_song_rmse": inner_song_metrics["rmse"],
            "held_song_rmse": held_song_metrics["rmse"],
            **held_frame_metrics,
        }
        per_fold.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary: dict[str, Any] = {
        "created_at": "2026-07-15",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Outer song folds; stratified inner epoch/correction selection; final locked test excluded",
        "inputs": "Frozen MERT hidden-4/6/8 frame embeddings + pooled EffNet mood/theme audio tags",
        "architecture": "Attention statistics pooling song-mean head with ranking regularization",
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
