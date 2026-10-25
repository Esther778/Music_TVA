from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


HIDDEN = [512, 256]
DROPOUT = 0.10
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-5
MAX_EPOCHS = 120
BATCH_SIZE = 512
PATIENCE = 18


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN[1], 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred - target
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    pearson = np.sum(pred_centered * target_centered, axis=0) / np.maximum(
        np.sqrt(
            np.sum(pred_centered**2, axis=0)
            * np.sum(target_centered**2, axis=0)
        ),
        1e-8,
    )
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = (2.0 * covariance) / np.maximum(
        pred.var(axis=0)
        + target.var(axis=0)
        + (pred_mean - target_mean) ** 2,
        1e-8,
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_valence": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_valence": float(np.mean(np.abs(error[:, 0]))),
        "mae_arousal": float(np.mean(np.abs(error[:, 1]))),
        "pearson_valence": float(pearson[0]),
        "pearson_arousal": float(pearson[1]),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "mean_ccc": float(np.mean(ccc)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, dict[str, float]]:
    model.eval()
    predictions = []
    targets = []
    squared_error_sum = 0.0
    value_count = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        output = model(features)
        squared_error_sum += float(torch.sum((output - labels) ** 2).item())
        value_count += int(labels.numel())
        predictions.append(output.cpu())
        targets.append(labels.cpu())
    pred = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    return squared_error_sum / value_count, metrics(pred, target)


def load_data(dataset_dir: Path):
    norm = np.load(dataset_dir / "feature_norm.npz")
    arrays = {}
    for split in ("train", "val", "test"):
        features = np.load(dataset_dir / f"X_{split}.npy").astype(np.float32)
        labels = np.load(dataset_dir / f"Y_{split}.npy").astype(np.float32)
        features = (features - norm["mean"]) / norm["std"]
        if not np.isfinite(features).all() or not np.isfinite(labels).all():
            raise ValueError(f"Non-finite values in {split} split.")
        arrays[split] = (
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).float(),
        )
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train OpenL3+MLP on the fixed DEAM song-level split."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    set_seed(args.seed)

    baseline_root = Path(__file__).resolve().parents[1]
    dataset_dir = (
        baseline_root / "artifacts" / "datasets" / "mlp_song_level_aligned15"
    )
    model_dir = baseline_root / "outputs" / "models" / "mlp_song_level_aligned15"
    result_dir = (
        baseline_root / "outputs" / "results" / "mlp_song_level_aligned15"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    arrays = load_data(dataset_dir)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            TensorDataset(*arrays["train"]),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        ),
        "train_eval": DataLoader(
            TensorDataset(*arrays["train"]), batch_size=BATCH_SIZE, shuffle=False
        ),
        "val": DataLoader(
            TensorDataset(*arrays["val"]), batch_size=BATCH_SIZE, shuffle=False
        ),
        "test": DataLoader(
            TensorDataset(*arrays["test"]), batch_size=BATCH_SIZE, shuffle=False
        ),
    }

    model = MLPRegressor(arrays["train"][0].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    checkpoint = (
        model_dir / f"openl3_mlp_song_level_aligned15_seed{args.seed}_best.pt"
    )
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(
        f"seed={args.seed} device={device} train_frames={len(arrays['train'][0])} "
        f"val_frames={len(arrays['val'][0])} test_frames={len(arrays['test'][0])}"
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features, labels in loaders["train"]:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(features)
            sample_count += len(features)

        val_loss, val_metrics = evaluate(model, loaders["val"], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / sample_count,
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={running_loss / sample_count:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_ccc={val_metrics['mean_ccc']:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
                break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    summary = {
        "trial_id": f"openl3_mlp_song_level_aligned15_seed{args.seed}",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split_protocol": "MERT split_manifest.csv song-level split",
        "alignment": "OpenL3 15.0 s frame onward with labels from sample_15000ms",
        "config": {
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
        },
        "checkpoint": str(checkpoint),
    }
    for split, loader_name in (
        ("train", "train_eval"),
        ("val", "val"),
        ("test", "test"),
    ):
        loss, split_metrics = evaluate(model, loaders[loader_name], device)
        summary[f"{split}_loss"] = loss
        summary.update(
            {f"{split}_{key}": value for key, value in split_metrics.items()}
        )

    with (result_dir / f"{summary['trial_id']}_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (result_dir / f"{summary['trial_id']}_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
