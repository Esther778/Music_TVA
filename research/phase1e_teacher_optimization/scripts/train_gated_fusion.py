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
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"
FEATURE_ROOT = (
    PROJECT_ROOT / "research" / "phase1c_spectral_feature_fusion_va" / "data" / "features"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    pcc = np.sum(pred_centered * target_centered, axis=0) / np.maximum(
        np.sqrt(
            np.sum(pred_centered**2, axis=0)
            * np.sum(target_centered**2, axis=0)
        ),
        1e-8,
    )
    pred_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = ((prediction - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = 2 * covariance / np.maximum(
        prediction.var(axis=0)
        + target.var(axis=0)
        + (pred_mean - target_mean) ** 2,
        1e-8,
    )
    rmse_dim = np.sqrt(np.mean(error**2, axis=0))
    mae_dim = np.mean(np.abs(error), axis=0)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_valence": float(rmse_dim[0]),
        "rmse_arousal": float(rmse_dim[1]),
        "mae_valence": float(mae_dim[0]),
        "mae_arousal": float(mae_dim[1]),
        "pcc_valence": float(pcc[0]),
        "pcc_arousal": float(pcc[1]),
        "mean_pcc": float(np.mean(pcc)),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "mean_ccc": float(np.mean(ccc)),
    }


def build_handcrafted_norm(index: pd.DataFrame, output: Path) -> tuple[np.ndarray, np.ndarray]:
    if output.is_file():
        norm = np.load(output)
        return norm["mean"], norm["std"]
    count = 0
    total = None
    total_square = None
    for song_id in index[index["split"] == "train"]["song_id"]:
        with np.load(FEATURE_ROOT / f"{int(song_id)}.npz") as data:
            values = data["handcrafted"].astype(np.float64)
        if total is None:
            total = values.sum(axis=0)
            total_square = (values**2).sum(axis=0)
        else:
            total += values.sum(axis=0)
            total_square += (values**2).sum(axis=0)
        count += len(values)
    mean = (total / count).astype(np.float32)
    variance = np.maximum(total_square / count - mean.astype(np.float64) ** 2, 1e-8)
    std = np.sqrt(variance).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mean=mean, std=std, frame_count=count)
    return mean, std


class FusionDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        mert_mean: np.ndarray,
        mert_std: np.ndarray,
        handcrafted_mean: np.ndarray,
        handcrafted_std: np.ndarray,
    ):
        self.rows = (
            index[index["split"] == split]
            .sort_values("song_id")
            .to_dict("records")
        )
        self.mert_mean = mert_mean
        self.mert_std = mert_std
        self.handcrafted_mean = handcrafted_mean
        self.handcrafted_std = handcrafted_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        song_id = int(row["song_id"])
        with np.load(row["sequence_path"]) as data:
            mert = data["X"].astype(np.float32)
            target = data["Y"].astype(np.float32)
        with np.load(FEATURE_ROOT / f"{song_id}.npz") as data:
            handcrafted = data["handcrafted"].astype(np.float32)
            feature_target = data["labels"].astype(np.float32)
        if target.shape != feature_target.shape or not np.allclose(
            target, feature_target, atol=1e-6, equal_nan=False
        ):
            raise ValueError(f"Target mismatch for song {song_id}.")
        if len(mert) != len(handcrafted):
            raise ValueError(
                f"Feature length mismatch song {song_id}: MERT={len(mert)} hand={len(handcrafted)}"
            )
        mert = (mert - self.mert_mean) / self.mert_std
        handcrafted = (
            handcrafted - self.handcrafted_mean
        ) / self.handcrafted_std
        return (
            torch.from_numpy(mert),
            torch.from_numpy(handcrafted),
            torch.from_numpy(target),
            song_id,
        )


def collate(batch):
    lengths = torch.tensor([len(item[2]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    mert = torch.zeros(len(batch), max_length, batch[0][0].shape[1])
    handcrafted = torch.zeros(len(batch), max_length, batch[0][1].shape[1])
    target = torch.zeros(len(batch), max_length, 2)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    song_ids = []
    for index, (mert_values, hand_values, labels, song_id) in enumerate(batch):
        length = len(labels)
        mert[index, :length] = mert_values
        handcrafted[index, :length] = hand_values
        target[index, :length] = labels
        mask[index, :length] = True
        song_ids.append(song_id)
    return mert, handcrafted, target, mask, lengths, song_ids


class GatedDualBranchRegressor(nn.Module):
    def __init__(self, mert_dim: int, handcrafted_dim: int, config: dict):
        super().__init__()
        projection = int(config["mert_projection"])
        hand_projection = int(config["handcrafted_projection"])
        if projection != hand_projection:
            raise ValueError("The first gated-fusion trial requires equal projection sizes.")
        dropout = float(config["dropout"])
        self.mert_projection = nn.Sequential(
            nn.LayerNorm(mert_dim),
            nn.Linear(mert_dim, projection),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hand_projection = nn.Sequential(
            nn.LayerNorm(handcrafted_dim),
            nn.Linear(handcrafted_dim, projection),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.valence_gate = nn.Linear(projection * 2, projection)
        self.arousal_gate = nn.Linear(projection * 2, projection)
        hidden = int(config["hidden_dim"])
        self.valence_temporal = nn.LSTM(
            projection, hidden, batch_first=True, bidirectional=True
        )
        self.arousal_temporal = nn.LSTM(
            projection, hidden, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.valence_output = nn.Linear(hidden * 2, 1)
        self.arousal_output = nn.Linear(hidden * 2, 1)

    def gated_features(
        self, mert: torch.Tensor, handcrafted: torch.Tensor, gate_layer: nn.Module
    ) -> torch.Tensor:
        gate = torch.sigmoid(gate_layer(torch.cat([mert, handcrafted], dim=-1)))
        return gate * mert + (1.0 - gate) * handcrafted

    def forward(
        self, mert: torch.Tensor, handcrafted: torch.Tensor
    ) -> torch.Tensor:
        mert = self.mert_projection(mert)
        handcrafted = self.hand_projection(handcrafted)
        valence_features = self.gated_features(
            mert, handcrafted, self.valence_gate
        )
        arousal_features = self.gated_features(
            mert, handcrafted, self.arousal_gate
        )
        valence_hidden, _ = self.valence_temporal(valence_features)
        arousal_hidden, _ = self.arousal_temporal(arousal_features)
        valence = self.valence_output(self.dropout(valence_hidden))
        arousal = self.arousal_output(self.dropout(arousal_hidden))
        return torch.cat([valence, arousal], dim=-1)


def correlation_loss(
    prediction: torch.Tensor, target: torch.Tensor, kind: str
) -> torch.Tensor:
    pred_mean = prediction.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    covariance = (pred_centered * target_centered).mean(dim=0)
    if kind == "pearson":
        denominator = torch.sqrt(
            (pred_centered**2).mean(dim=0)
            * (target_centered**2).mean(dim=0)
            + 1e-8
        )
        value = covariance / denominator
    elif kind == "ccc":
        value = 2 * covariance / (
            pred_centered.var(dim=0, unbiased=False)
            + target_centered.var(dim=0, unbiased=False)
            + (pred_mean - target_mean) ** 2
            + 1e-8
        )
    else:
        raise ValueError(kind)
    return 1.0 - value.mean()


def song_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_config: dict,
) -> torch.Tensor:
    losses = []
    dimension_weights = prediction.new_tensor(
        loss_config.get("dimension_weights", [1.0, 1.0])
    )
    for index in range(len(prediction)):
        pred_song = prediction[index][mask[index]]
        target_song = target[index][mask[index]]
        mse = (
            ((pred_song - target_song) ** 2) * dimension_weights
        ).mean()
        loss = float(loss_config.get("mse", 1.0)) * mse
        if float(loss_config.get("ccc", 0.0)) > 0:
            loss = loss + float(loss_config["ccc"]) * correlation_loss(
                pred_song, target_song, "ccc"
            )
        if float(loss_config.get("pearson", 0.0)) > 0:
            loss = loss + float(loss_config["pearson"]) * correlation_loss(
                pred_song, target_song, "pearson"
            )
        if float(loss_config.get("derivative", 0.0)) > 0 and len(pred_song) > 1:
            pred_delta = pred_song[1:] - pred_song[:-1]
            target_delta = target_song[1:] - target_song[:-1]
            derivative = ((pred_delta - target_delta) ** 2).mean()
            loss = loss + float(loss_config["derivative"]) * derivative
        losses.append(loss)
    return torch.stack(losses).mean()


def song_balanced_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return song_balanced_loss(
        prediction,
        target,
        mask,
        {
            "mse": 1.0,
            "ccc": 0.0,
            "pearson": 0.0,
            "derivative": 0.0,
            "dimension_weights": [1.0, 1.0],
        },
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    predictions = []
    targets = []
    for mert, handcrafted, target, mask, _, _ in loader:
        mert = mert.to(device)
        handcrafted = handcrafted.to(device)
        target = target.to(device)
        mask = mask.to(device)
        prediction = model(mert, handcrafted)
        predictions.append(prediction[mask].cpu().numpy())
        targets.append(target[mask].cpu().numpy())
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    return metrics(prediction_array, target_array), prediction_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MERT-handcrafted gated dual-branch VA model.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "gated_fusion.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)

    dataset_root = MERT_ROOT / "data" / "datasets" / "mert_va"
    index = pd.read_csv(dataset_root / "sequence_index.csv")
    mert_norm = np.load(dataset_root / "frame" / "feature_norm.npz")
    output_dir = ROOT / "outputs" / config["trial_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_mean, hand_std = build_handcrafted_norm(
        index, output_dir / "handcrafted_norm.npz"
    )
    datasets = {
        split: FusionDataset(
            index,
            split,
            mert_norm["mean"].squeeze(0),
            mert_norm["std"].squeeze(0),
            hand_mean,
            hand_std,
        )
        for split in ("train", "val", "test")
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
    model = GatedDualBranchRegressor(
        first[0].shape[1], first[1].shape[1], config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = output_dir / "best.pt"
    history = []
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    print(
        json.dumps(
            {
                "trial_id": config["trial_id"],
                "device": str(device),
                "songs": {split: len(dataset) for split, dataset in datasets.items()},
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "test_locked": not args.evaluate_test,
            },
            indent=2,
        )
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_losses = []
        for mert, handcrafted, target, mask, _, _ in loaders["train"]:
            mert = mert.to(device)
            handcrafted = handcrafted.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            prediction = model(mert, handcrafted)
            loss = song_balanced_loss(
                prediction, target, mask, config.get("loss", {})
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["grad_clip"])
            )
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_metrics, _ = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_pcc={val_metrics['mean_pcc']:.6f} "
            f"val_ccc={val_metrics['mean_ccc']:.6f}"
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
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
    train_metrics, train_prediction = evaluate(model, loaders["train"], device)
    val_metrics, val_prediction = evaluate(model, loaders["val"], device)
    summary = {
        "trial_id": config["trial_id"],
        "seed": int(config["seed"]),
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint),
        "test_evaluated": bool(args.evaluate_test),
        "config": config,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    np.save(output_dir / "train_pred.npy", train_prediction.astype(np.float32))
    np.save(output_dir / "val_pred.npy", val_prediction.astype(np.float32))
    if args.evaluate_test:
        test_metrics, test_prediction = evaluate(model, loaders["test"], device)
        summary.update(
            {f"test_{key}": value for key, value in test_metrics.items()}
        )
        np.save(output_dir / "test_pred.npy", test_prediction.astype(np.float32))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
