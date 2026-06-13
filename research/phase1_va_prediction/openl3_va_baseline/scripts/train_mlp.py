from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
HIDDEN1 = 256
HIDDEN2 = 128
DROPOUT = 0.2
SEED = 42


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse_torch(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def pearsonr_torch(pred, target):
    pred_mean = torch.mean(pred, dim=0)
    target_mean = torch.mean(target, dim=0)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    cov = torch.sum(pred_centered * target_centered, dim=0)
    pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=0) + 1e-8)
    target_std = torch.sqrt(torch.sum(target_centered ** 2, dim=0) + 1e-8)
    return cov / (pred_std * target_std + 1e-8)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=512, hidden1=256, hidden2=128, output_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
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

    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    X_val = torch.from_numpy(X_val).float()
    Y_val = torch.from_numpy(Y_val).float()
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean.astype(np.float32), std.astype(np.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()

        bs = X_batch.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        preds = model(X_batch)
        loss = criterion(preds, Y_batch)

        bs = X_batch.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_preds.append(preds.cpu())
        all_targets.append(Y_batch.cpu())

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    rmse = rmse_torch(all_preds, all_targets).item()
    pearson = pearsonr_torch(all_preds, all_targets)
    return avg_loss, rmse, pearson


def main():
    set_seed(SEED)

    ROOT = Path(__file__).resolve().parents[1]
    data_dir = ROOT / "artifacts" / "datasets" / "mlp"
    model_dir = ROOT / "outputs" / "models" / "mlp"
    result_dir = ROOT / "outputs" / "results" / "mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std = load_dataset(data_dir)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        output_dim=2,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_pearson_v": [],
        "val_pearson_a": [],
    }

    best_val = float("inf")
    best_path = model_dir / "mlp_best.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_rmse, va_r = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_rmse"].append(va_rmse)
        history["val_pearson_v"].append(float(va_r[0]))
        history["val_pearson_a"].append(float(va_r[1]))

        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"val_rmse={va_rmse:.4f} | val_r(V)={va_r[0]:.3f} | val_r(A)={va_r[1]:.3f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

    np.savez(result_dir / "mlp_history.npz", **history)
    np.savez(model_dir / "mlp_input_norm.npz", mean=mean, std=std)

    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_rmse, te_r = evaluate(model, test_loader, criterion, device)

    print("[TEST]")
    print(f"  loss={te_loss:.4f} | rmse={te_rmse:.4f} | r(V)={te_r[0]:.3f} | r(A)={te_r[1]:.3f}")


if __name__ == "__main__":
    main()
