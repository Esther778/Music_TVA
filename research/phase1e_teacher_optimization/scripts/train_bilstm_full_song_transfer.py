from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    squared = error**2
    result = {
        "rmse": float(np.sqrt(squared.mean())),
        "mae": float(absolute.mean()),
    }
    for index, name in enumerate(("valence", "arousal")):
        pred = prediction[:, index]
        true = target[:, index]
        result[f"rmse_{name}"] = float(np.sqrt(squared[:, index].mean()))
        result[f"mae_{name}"] = float(absolute[:, index].mean())
        result[f"pcc_{name}"] = (
            float(np.corrcoef(pred, true)[0, 1])
            if np.std(pred) > 1e-8 and np.std(true) > 1e-8
            else 0.0
        )
        pred_mean = float(pred.mean())
        true_mean = float(true.mean())
        covariance = float(np.mean((pred - pred_mean) * (true - true_mean)))
        denominator = (
            float(pred.var())
            + float(true.var())
            + (pred_mean - true_mean) ** 2
        )
        result[f"ccc_{name}"] = (
            2.0 * covariance / denominator if denominator > 1e-12 else 0.0
        )
    result["mean_pcc"] = (
        result["pcc_valence"] + result["pcc_arousal"]
    ) / 2.0
    result["mean_ccc"] = (
        result["ccc_valence"] + result["ccc_arousal"]
    ) / 2.0
    return result


def load_sequence(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        features = data["X"].astype(np.float32)
        target = data["Y"].astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError(f"Non-finite sequence data: {path}")
    return features, target


def build_normalization(rows: pd.DataFrame, path: Path):
    if path.is_file():
        with np.load(path) as data:
            return data["mean"], data["std"]
    total = None
    square = None
    frames = 0
    for row in rows.itertuples():
        features, _ = load_sequence(row.sequence_path)
        values = features.astype(np.float64)
        if total is None:
            total = values.sum(axis=0)
            square = (values**2).sum(axis=0)
        else:
            total += values.sum(axis=0)
            square += (values**2).sum(axis=0)
        frames += len(values)
    mean = (total / frames).astype(np.float32)
    variance = square / frames - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std, frames=frames, songs=len(rows))
    return mean, std


class SongDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        self.rows = rows.sort_values("song_id").to_dict("records")
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        features, target = load_sequence(row["sequence_path"])
        features = (features - self.mean) / self.std
        return (
            torch.from_numpy(features),
            torch.from_numpy(target),
            int(row["song_id"]),
        )


def collate(batch):
    lengths = [len(item[0]) for item in batch]
    maximum = max(lengths)
    features = torch.zeros(len(batch), maximum, batch[0][0].shape[1])
    targets = torch.zeros(len(batch), maximum, 2)
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    song_ids = []
    for index, (song_features, song_target, song_id) in enumerate(batch):
        length = len(song_features)
        features[index, :length] = song_features
        targets[index, :length] = song_target
        mask[index, :length] = True
        song_ids.append(song_id)
    return features, targets, mask, song_ids


class TransferBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        hidden = int(config["hidden_dim"])
        layers = int(config["num_layers"])
        dropout = float(config["dropout"])
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.valence_head = nn.Linear(hidden * 2, 1)
        self.arousal_head = nn.Linear(hidden * 2, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(features)
        encoded = self.dropout(encoded)
        return torch.cat(
            [self.valence_head(encoded), self.arousal_head(encoded)], dim=-1
        )


def song_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    squared = (prediction - target) ** 2 * weights
    per_frame = squared.mean(dim=-1)
    denominator = mask.sum(dim=1).clamp_min(1)
    per_song = (per_frame * mask).sum(dim=1) / denominator
    return per_song.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    song_ids = []
    for features, target, mask, batch_song_ids in loader:
        features = features.to(device)
        prediction = model(features).cpu()
        predictions.append(prediction[mask].numpy())
        targets.append(target[mask].numpy())
        for batch_index, song_id in enumerate(batch_song_ids):
            song_ids.extend([song_id] * int(mask[batch_index].sum()))
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return metrics(prediction, target), prediction, target, np.asarray(song_ids)


def make_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
    )


def selection_score(result: dict[str, float], config: dict) -> float:
    return (
        float(config["selection_valence_weight"]) * result["rmse_valence"]
        + float(config["selection_arousal_weight"]) * result["rmse_arousal"]
    )


def train_stage(
    stage: str,
    model: nn.Module,
    train_loader,
    validation_loader,
    device,
    config,
    output_dir: Path,
    epoch_override: int | None,
):
    stage_config = config[stage]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_config["learning_rate"]),
        weight_decay=float(stage_config["weight_decay"]),
    )
    weights = torch.tensor(
        [
            float(config["valence_loss_weight"]),
            float(config["arousal_loss_weight"]),
        ],
        device=device,
    )
    epochs = epoch_override or int(stage_config["epochs"])
    patience = int(stage_config["patience"])
    checkpoint = output_dir / f"{stage}_best.pt"
    history = []
    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        losses = []
        for features, target, mask, _ in train_loader:
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = song_balanced_loss(prediction, target, mask, weights)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{stage}: non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["grad_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation, _, _, _ = evaluate(model, validation_loader, device)
        score = selection_score(validation, config)
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            "epoch_sec": time.time() - epoch_started,
            **validation,
        }
        history.append(row)
        print(
            f"stage={stage} epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"score={score:.6f} v_rmse={validation['rmse_valence']:.6f} "
            f"a_rmse={validation['rmse_arousal']:.6f} "
            f"mean_pcc={validation['mean_pcc']:.6f} "
            f"mean_ccc={validation['mean_ccc']:.6f} "
            f"sec={row['epoch_sec']:.1f}",
            flush=True,
        )
        if not np.isfinite(score):
            raise RuntimeError(f"{stage}: non-finite validation score")
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return history, best_epoch, best_score, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/bilstm_full_song_transfer.json",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--limit-short", type=int)
    parser.add_argument("--limit-full", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"]) + args.outer_fold
    set_seed(seed)
    device = torch.device(args.device)

    protocol = pd.read_csv(PROTOCOL)
    fold = protocol[protocol["outer_fold"] == args.outer_fold].copy()
    index = pd.read_csv(INDEX)[["song_id", "sequence_path"]]
    fold = fold.drop(columns=["sequence_path"]).merge(index, on="song_id")
    short_rows = fold[
        (fold["role"] == "train") & (fold["domain"] == "short_clip")
    ].copy()
    full_rows = fold[
        (fold["role"] == "train") & (fold["domain"] == "full_song")
    ].copy()
    validation_rows = fold[fold["role"] == "validation"].copy()
    test_rows = fold[fold["role"] == "test"].copy()
    if args.limit_short:
        short_rows = short_rows.head(args.limit_short)
    if args.limit_full:
        full_rows = full_rows.head(args.limit_full)
    if args.limit_validation:
        validation_rows = validation_rows.head(args.limit_validation)

    overlap = (
        set(short_rows.song_id)
        | set(full_rows.song_id)
    ) & (set(validation_rows.song_id) | set(test_rows.song_id))
    if overlap:
        raise RuntimeError(f"Train/held-out song overlap: {sorted(overlap)}")

    output_dir = ROOT / "outputs" / (
        f"{config['trial_id']}_fold{args.outer_fold}{args.output_suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_training = pd.concat([short_rows, full_rows], ignore_index=True)
    mean, std = build_normalization(
        allowed_training, output_dir / "feature_norm.npz"
    )
    datasets = {
        "short": SongDataset(short_rows, mean, std),
        "full": SongDataset(full_rows, mean, std),
        "validation": SongDataset(validation_rows, mean, std),
    }
    validation_loader = make_loader(datasets["validation"], 1, False)
    loaders = {
        "pretrain": make_loader(
            datasets["short"], int(config["pretrain"]["batch_size"]), True
        ),
        "finetune": make_loader(
            datasets["full"], int(config["finetune"]["batch_size"]), True
        ),
    }
    model = TransferBiLSTM(768, config).to(device)
    initial, _, _, _ = evaluate(model, validation_loader, device)
    started = time.time()
    print(
        json.dumps(
            {
                "outer_fold": args.outer_fold,
                "device": str(device),
                "short_pretrain_songs": len(short_rows),
                "full_finetune_songs": len(full_rows),
                "validation_full_songs": len(validation_rows),
                "locked_test_full_songs": len(test_rows),
                "test_used": False,
                "normalization_songs": len(allowed_training),
                "parameters": sum(p.numel() for p in model.parameters()),
                "initial_validation": initial,
            },
            indent=2,
        ),
        flush=True,
    )

    pretrain_history, pretrain_epoch, pretrain_score, pretrain_checkpoint = (
        train_stage(
            "pretrain",
            model,
            loaders["pretrain"],
            validation_loader,
            device,
            config,
            output_dir,
            args.pretrain_epochs,
        )
    )
    pretrained, _, _, _ = evaluate(model, validation_loader, device)
    finetune_history, finetune_epoch, finetune_score, finetune_checkpoint = (
        train_stage(
            "finetune",
            model,
            loaders["finetune"],
            validation_loader,
            device,
            config,
            output_dir,
            args.finetune_epochs,
        )
    )
    final, prediction, target, song_ids = evaluate(
        model, validation_loader, device
    )
    summary = {
        "trial_id": config["trial_id"],
        "outer_fold": args.outer_fold,
        "seed": seed,
        "elapsed_sec": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "short_pretrain_songs": len(short_rows),
        "full_finetune_songs": len(full_rows),
        "validation_full_songs": len(validation_rows),
        "locked_test_full_songs": len(test_rows),
        "test_used": False,
        "pretrain_best_epoch": pretrain_epoch,
        "pretrain_best_score": pretrain_score,
        "finetune_best_epoch": finetune_epoch,
        "finetune_best_score": finetune_score,
        "pretrain_checkpoint": str(pretrain_checkpoint),
        "finetune_checkpoint": str(finetune_checkpoint),
        "config": config,
        "initial_validation": initial,
        "pretrained_validation": pretrained,
        "finetuned_validation": final,
    }
    np.save(output_dir / "validation_pred.npy", prediction.astype(np.float32))
    np.save(output_dir / "validation_target.npy", target.astype(np.float32))
    np.save(output_dir / "validation_song_ids.npy", song_ids.astype(np.int32))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history = pretrain_history + finetune_history
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
