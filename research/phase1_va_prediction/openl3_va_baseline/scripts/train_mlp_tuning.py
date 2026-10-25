from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


TRIALS = {
    "mlp_tune_w384_d015_lr3e4": {
        "hidden": [384, 192],
        "dropout": 0.15,
        "lr": 3e-4,
        "weight_decay": 5e-5,
        "epochs": 70,
        "batch_size": 512,
        "patience": 12,
        "seed": 42,
    },
    "mlp_tune_w384_d010_lr3e4_e100": {
        "hidden": [384, 192],
        "dropout": 0.10,
        "lr": 3e-4,
        "weight_decay": 5e-5,
        "epochs": 100,
        "batch_size": 512,
        "patience": 15,
        "seed": 42,
    },
    "mlp_tune_w384_d008_lr2e4_e120": {
        "hidden": [384, 192],
        "dropout": 0.08,
        "lr": 2e-4,
        "weight_decay": 5e-5,
        "epochs": 120,
        "batch_size": 512,
        "patience": 18,
        "seed": 42,
    },
    "mlp_tune_w512_d010_lr2e4_e120": {
        "hidden": [512, 256],
        "dropout": 0.10,
        "lr": 2e-4,
        "weight_decay": 5e-5,
        "epochs": 120,
        "batch_size": 512,
        "patience": 18,
        "seed": 42,
    },
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset(data_dir: Path):
    X_train = np.load(data_dir / "X_train.npy")
    Y_train = np.load(data_dir / "Y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    Y_val = np.load(data_dir / "Y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    Y_test = np.load(data_dir / "Y_test.npy")

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(Y_test).float(),
    )


def pearson_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    numerator = np.sum(pred_centered * target_centered, axis=0)
    denominator = np.sqrt(np.sum(pred_centered**2, axis=0) * np.sum(target_centered**2, axis=0))
    return numerator / np.maximum(denominator, 1e-8)


def ccc_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_var = pred.var(axis=0)
    target_var = target.var(axis=0)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(axis=0)
    return (2.0 * cov) / np.maximum(pred_var + target_var + (pred_mean - target_mean) ** 2, 1e-8)


def metrics_np(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    pearson = pearson_np(pred, target)
    ccc = ccc_np(pred, target)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "rmse_valence": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "mae_valence": float(np.mean(np.abs(err[:, 0]))),
        "mae_arousal": float(np.mean(np.abs(err[:, 1]))),
        "pearson_valence": float(pearson[0]),
        "pearson_arousal": float(pearson[1]),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "mean_ccc": float(np.mean(ccc)),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_values = 0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        total_loss += float(criterion(pred, y_batch).item())
        total_values += int(y_batch.numel())
        preds.append(pred.cpu())
        targets.append(y_batch.cpu())
    pred_np = torch.cat(preds, dim=0).numpy()
    target_np = torch.cat(targets, dim=0).numpy()
    return total_loss / total_values, metrics_np(pred_np, target_np)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune OpenL3+MLP VA regressor without overwriting the published model.")
    parser.add_argument("--trial", required=True, choices=sorted(TRIALS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override the trial seed and write seed-specific outputs.")
    args = parser.parse_args()

    config = dict(TRIALS[args.trial])
    if args.seed is not None:
        config["seed"] = args.seed
    output_id = args.trial if args.seed is None else f"{args.trial}_seed{args.seed}"
    set_seed(int(config["seed"]))

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "artifacts" / "datasets" / "mlp"
    model_dir = root / "outputs" / "models" / "mlp_tuning"
    result_dir = root / "outputs" / "results" / "mlp_tuning"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    requested_device = args.device
    if requested_device == "mps" and not torch.backends.mps.is_available():
        requested_device = "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(data_dir)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=int(config["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=int(config["batch_size"]), shuffle=False)

    model = MLPRegressor(X_train.shape[1], list(config["hidden"]), float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = model_dir / f"{output_id}_best.pt"
    history: list[dict[str, float]] = []

    print(f"trial={output_id} base_trial={args.trial} device={device} config={json.dumps(config, ensure_ascii=False)}")
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(x_batch.shape[0])
            running_samples += int(x_batch.shape[0])

        train_loss = running_loss / running_samples
        val_loss, val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        print(
            f"{output_id} epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f} "
            f"val_rv={val_metrics['pearson_valence']:.3f} val_ra={val_metrics['pearson_arousal']:.3f} "
            f"val_ccc={val_metrics['mean_ccc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    train_loss, train_metrics = evaluate(model, DataLoader(TensorDataset(X_train, Y_train), batch_size=int(config["batch_size"]), shuffle=False), device)
    val_loss, val_metrics = evaluate(model, val_loader, device)
    test_loss, test_metrics = evaluate(model, test_loader, device)

    summary = {
        "trial_id": output_id,
        "base_trial_id": args.trial,
        "seed": int(config["seed"]),
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "config_json": json.dumps(config, ensure_ascii=False),
        "checkpoint": str(best_path),
        **prefixed("train", train_metrics),
        **prefixed("val", val_metrics),
        **prefixed("test", test_metrics),
    }
    np.savez(result_dir / f"{output_id}_history.npz", rows=np.asarray(history, dtype=object))
    with (result_dir / f"{output_id}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_summary(result_dir / "mlp_tuning_summary.csv", summary)
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
