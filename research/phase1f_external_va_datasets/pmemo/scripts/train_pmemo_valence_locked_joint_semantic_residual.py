#!/usr/bin/env python3
"""Train one preregistered semantic residual BiGRU on locked PMEmo dev folds.

The model uses only audio-derived inputs: the current OOF Valence prediction,
past-context MTG mood/theme tags, and a full-song MAEST representation.  Every
outer fold fits preprocessing and model weights on the other songs only.  The
final locked test is neither loaded nor evaluated.
"""

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from search_pmemo_valence_locked_effnet_moodtheme_tags import (
    FOLDS,
    feature_matrix,
    interpolate_tags,
    macro,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows


DEFAULT_BASE = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_effnet_moodtheme_tags_20260714"
DEFAULT_MAEST = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_maest_30s_layer7_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_joint_semantic_residual_20260714"

BASE_METRICS = {
    "rmse": 0.18154631579428374,
    "mae": 0.14231915639241396,
    "pcc": 0.8234038487655991,
    "ccc": 0.8068739659711618,
}


@dataclass(frozen=True)
class Config:
    seed: int = 20260714
    maest_pca_dim: int = 8
    frame_proj_dim: int = 64
    hidden_dim: int = 32
    song_context_dim: int = 16
    dropout: float = 0.30
    correction_cap: float = 0.02
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    batch_size: int = 20
    max_epochs: int = 40
    patience: int = 8
    inner_val_fraction: float = 0.15
    song_mean_loss_weight: float = 0.50
    ccc_loss_weight: float = 0.03
    smoothness_loss_weight: float = 0.01


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class SongDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    length = max(len(row["target"]) for row in batch)
    frame_dim = batch[0]["frame"].shape[1]
    song_dim = batch[0]["song"].shape[0]
    frame = torch.zeros(len(batch), length, frame_dim, dtype=torch.float32)
    song = torch.zeros(len(batch), song_dim, dtype=torch.float32)
    base = torch.zeros(len(batch), length, dtype=torch.float32)
    target = torch.zeros(len(batch), length, dtype=torch.float32)
    time = torch.zeros(len(batch), length, dtype=torch.float32)
    mask = torch.zeros(len(batch), length, dtype=torch.bool)
    song_id = torch.zeros(len(batch), dtype=torch.long)
    for i, row in enumerate(batch):
        n = len(row["target"])
        frame[i, :n] = torch.from_numpy(row["frame"])
        song[i] = torch.from_numpy(row["song"])
        base[i, :n] = torch.from_numpy(row["base"])
        target[i, :n] = torch.from_numpy(row["target"])
        time[i, :n] = torch.from_numpy(row["time"])
        mask[i, :n] = True
        song_id[i] = int(row["song_id"])
    return frame, song, base, target, time, mask, song_id


class JointSemanticResidual(nn.Module):
    def __init__(self, frame_dim: int, song_dim: int, config: Config):
        super().__init__()
        self.cap = float(config.correction_cap)
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, config.frame_proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.gru = nn.GRU(
            config.frame_proj_dim,
            config.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        width = config.hidden_dim * 2
        self.song_proj = nn.Sequential(
            nn.LayerNorm(song_dim),
            nn.Linear(song_dim, config.song_context_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.frame_head = nn.Sequential(
            nn.LayerNorm(width + config.song_context_dim),
            nn.Dropout(config.dropout),
            nn.Linear(width + config.song_context_dim, 1),
        )
        self.song_head = nn.Sequential(
            nn.LayerNorm(width * 3 + config.song_context_dim),
            nn.Linear(width * 3 + config.song_context_dim, width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width, 1),
        )
        # Start exactly at the registered base prediction. This makes the
        # residual branch conservative and lets validation retain identity.
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
        hidden, _ = self.gru(self.frame_proj(frame))
        song_context = self.song_proj(song)
        repeated_context = song_context.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        frame_shift = self.cap * torch.tanh(
            self.frame_head(torch.cat([hidden, repeated_context], dim=-1)).squeeze(-1)
        )
        mask_f = mask.float().unsqueeze(-1)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (hidden * mask_f).sum(dim=1) / count
        variance = (((hidden - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / count
        maximum = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        song_shift = self.cap * torch.tanh(
            self.song_head(
                torch.cat([mean, torch.sqrt(variance + 1e-6), maximum, song_context], dim=-1)
            ).squeeze(-1)
        )
        prediction = torch.clamp(base + frame_shift + song_shift.unsqueeze(1), -1.0, 1.0)
        return prediction, frame_shift + song_shift.unsqueeze(1)


def load_raw_records(base_dir: Path, tag_dir: Path, maest_dir: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for fold in FOLDS:
        frame = pd.read_csv(base_dir / f"{fold}_val_predictions.csv")
        tags, _ = interpolate_tags(frame, tag_dir)
        past_tags = feature_matrix(frame, tags, "all_past_dense")
        fold_records = []
        for music_id, indices in frame.groupby("musicId", sort=False).groups.items():
            loc = np.asarray(indices, dtype=np.int64)
            base = frame.loc[loc, "pred_valence"].to_numpy(dtype=np.float32)
            delta = np.diff(base, prepend=base[0]).astype(np.float32)
            position = np.linspace(0.0, 1.0, len(loc), dtype=np.float32)
            frame_feature = np.column_stack([past_tags[loc], base, delta, position]).astype(np.float32)
            maest = np.load(maest_dir / "per_song" / f"{int(music_id)}.npz")["embedding"]
            fold_records.append({
                "song_id": int(music_id),
                "frame_raw": frame_feature,
                "song_raw": maest.mean(axis=0).astype(np.float32),
                "base": base,
                "target": frame.loc[loc, "target_valence"].to_numpy(dtype=np.float32),
                "time": frame.loc[loc, "time_sec"].to_numpy(dtype=np.float32),
            })
        output[fold] = fold_records
    return output


def fit_preprocess(records: list[dict[str, Any]], config: Config) -> tuple[StandardScaler, StandardScaler, PCA]:
    frame_scaler = StandardScaler().fit(np.concatenate([row["frame_raw"] for row in records]))
    song_raw = np.stack([row["song_raw"] for row in records])
    song_scaler = StandardScaler().fit(song_raw)
    pca = PCA(n_components=config.maest_pca_dim, whiten=True, random_state=config.seed)
    pca.fit(song_scaler.transform(song_raw))
    return frame_scaler, song_scaler, pca


def transform_records(
    records: list[dict[str, Any]],
    frame_scaler: StandardScaler,
    song_scaler: StandardScaler,
    pca: PCA,
) -> list[dict[str, Any]]:
    output = []
    for row in records:
        item = dict(row)
        item["frame"] = frame_scaler.transform(row["frame_raw"]).astype(np.float32)
        item["song"] = pca.transform(song_scaler.transform(row["song_raw"][None]))[0].astype(np.float32)
        output.append(item)
    return output


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    rows = []
    for frame, song, base, target, time, mask, song_id in loader:
        prediction, _ = model(frame.to(device), song.to(device), base.to(device), mask.to(device))
        prediction = prediction.cpu()
        for i in range(len(song_id)):
            valid = mask[i]
            rows.append(pd.DataFrame({
                "musicId": int(song_id[i]),
                "time_sec": time[i, valid].numpy(),
                "target_valence": target[i, valid].numpy(),
                "base_pred_valence": base[i, valid].numpy(),
                "pred_valence": prediction[i, valid].numpy(),
            }))
    table = pd.concat(rows, ignore_index=True)
    result = metrics(table["pred_valence"].to_numpy(), table["target_valence"].to_numpy())
    return {"table": table, "metrics": result}


def selection_score(result: dict[str, float]) -> float:
    return (
        result["rmse"]
        + 0.05 * max(0.0, 0.80 - result["ccc"])
        + 0.03 * max(0.0, 0.80 - result["pcc"])
    )


def train_epochs(
    model: JointSemanticResidual,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    epochs: int,
    val_loader: DataLoader | None = None,
) -> tuple[JointSemanticResidual, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_epoch = epochs
    best_score = float("inf")
    bad = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for frame, song, base, target, _, mask, _ in loader:
            frame = frame.to(device)
            song = song.to(device)
            base = base.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            prediction, correction = model(frame, song, base, mask)
            squared = (prediction - target).square() * mask.float()
            frame_loss = (squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean()
            mean_loss = nn.functional.mse_loss(masked_mean(prediction, mask), masked_mean(target, mask))
            valid_prediction = prediction[mask]
            valid_target = target[mask]
            smooth_mask = mask[:, 1:] & mask[:, :-1]
            smoothness = correction.diff(dim=1)[smooth_mask].square().mean()
            loss = frame_loss
            loss = loss + config.song_mean_loss_weight * mean_loss
            loss = loss + config.ccc_loss_weight * ccc_loss(valid_prediction, valid_target)
            loss = loss + config.smoothness_loss_weight * smoothness
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row: dict[str, float] = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            val_result = evaluate(model, val_loader, device)["metrics"]
            row.update({f"val_{key}": float(value) for key, value in val_result.items()})
            score = selection_score(val_result)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                bad = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                bad += 1
        history.append(row)
        if val_loader is not None:
            print(
                f"epoch={epoch:02d} loss={row['train_loss']:.6f} "
                f"val_rmse={row['val_rmse']:.6f} val_pcc={row['val_pcc']:.4f} "
                f"val_ccc={row['val_ccc']:.4f}",
                flush=True,
            )
            if bad >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def split_inner(records: list[dict[str, Any]], config: Config, outer_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed + outer_index)
    order = rng.permutation(len(records))
    count = max(1, int(round(len(records) * config.inner_val_fraction)))
    val_indices = set(order[:count].tolist())
    train = [row for i, row in enumerate(records) if i not in val_indices]
    val = [row for i, row in enumerate(records) if i in val_indices]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--maest-dir", type=Path, default=DEFAULT_MAEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    config = Config(max_epochs=args.max_epochs or Config.max_epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    raw = load_raw_records(args.base_dir, args.tag_dir, args.maest_dir)
    per_fold = []
    for outer_index, held in enumerate(FOLDS, start=1):
        print(f"outer={held} selection", flush=True)
        outer_train = [row for fold in FOLDS if fold != held for row in raw[fold]]
        inner_train, inner_val = split_inner(outer_train, config, outer_index)
        frame_scaler, song_scaler, pca = fit_preprocess(inner_train, config)
        train_records = transform_records(inner_train, frame_scaler, song_scaler, pca)
        val_records = transform_records(inner_val, frame_scaler, song_scaler, pca)
        train_loader = DataLoader(
            SongDataset(train_records), batch_size=config.batch_size, shuffle=True, collate_fn=collate
        )
        val_loader = DataLoader(
            SongDataset(val_records), batch_size=config.batch_size, shuffle=False, collate_fn=collate
        )
        set_seed(config.seed + outer_index)
        model = JointSemanticResidual(
            train_records[0]["frame"].shape[1], train_records[0]["song"].shape[0], config
        ).to(device)
        _, best_epoch, history = train_epochs(
            model, train_loader, device, config, config.max_epochs, val_loader
        )
        fold_dir = args.out_dir / held
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(fold_dir / "inner_history.csv", index=False)

        print(f"outer={held} refit_epochs={best_epoch}", flush=True)
        frame_scaler, song_scaler, pca = fit_preprocess(outer_train, config)
        refit_records = transform_records(outer_train, frame_scaler, song_scaler, pca)
        held_records = transform_records(raw[held], frame_scaler, song_scaler, pca)
        refit_loader = DataLoader(
            SongDataset(refit_records), batch_size=config.batch_size, shuffle=True, collate_fn=collate
        )
        held_loader = DataLoader(
            SongDataset(held_records), batch_size=config.batch_size, shuffle=False, collate_fn=collate
        )
        set_seed(config.seed + 100 + outer_index)
        model = JointSemanticResidual(
            refit_records[0]["frame"].shape[1], refit_records[0]["song"].shape[0], config
        ).to(device)
        model, _, _ = train_epochs(model, refit_loader, device, config, best_epoch)
        checkpoint = {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "best_epoch": best_epoch,
            "held_fold": held,
            "frame_scaler_mean": frame_scaler.mean_,
            "frame_scaler_scale": frame_scaler.scale_,
            "song_scaler_mean": song_scaler.mean_,
            "song_scaler_scale": song_scaler.scale_,
            "pca_components": pca.components_,
            "pca_mean": pca.mean_,
            "pca_explained_variance": pca.explained_variance_,
        }
        torch.save(checkpoint, fold_dir / "model.pt")
        held_eval = evaluate(model, held_loader, device)
        held_eval["table"].to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        row = {"fold": held, "selected_epoch": best_epoch, **held_eval["metrics"]}
        per_fold.append(row)
        print(json.dumps(row, indent=2), flush=True)

    result = macro(per_fold)
    guarded = (
        result["rmse"] < BASE_METRICS["rmse"]
        and result["mae"] <= BASE_METRICS["mae"]
        and result["pcc"] >= BASE_METRICS["pcc"]
        and result["ccc"] >= BASE_METRICS["ccc"]
    )
    summary = {
        "created_at": "2026-07-14",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Per outer fold: preprocessing and epoch selection on other songs only; final locked test excluded",
        "inputs": "OOF base Valence + past-8s MTG mood/theme tags + full-song MAEST layer-7 representation",
        "config": asdict(config),
        "base_macro": BASE_METRICS,
        "macro": result,
        "per_fold": per_fold,
        "delta_vs_base": {key: result[key] - BASE_METRICS[key] for key in BASE_METRICS},
        "guarded_promotion": guarded,
        "gap_rows": threshold_rows(result),
        "final_locked_test": "excluded",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
