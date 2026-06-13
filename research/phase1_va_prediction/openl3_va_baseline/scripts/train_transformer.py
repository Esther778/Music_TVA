from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim=512,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        output_dim=2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.fc_out(x)


def load_dataset(data_dir: Path):
    X_train = np.load(data_dir / "X_train.npy")
    Y_train = np.load(data_dir / "Y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    Y_val = np.load(data_dir / "Y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    Y_test = np.load(data_dir / "Y_test.npy")

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

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
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def make_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=BATCH_SIZE):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
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

    N, T, D = all_preds.shape
    all_preds_flat = all_preds.view(N * T, D)
    all_targets_flat = all_targets.view(N * T, D)

    rmse = rmse_torch(all_preds_flat, all_targets_flat).item()
    pearson = pearsonr_torch(all_preds_flat, all_targets_flat)

    return avg_loss, rmse, pearson


def main():
    set_seed(SEED)

    ROOT = Path(__file__).resolve().parents[1]
    data_dir = ROOT / "artifacts" / "datasets" / "seq"
    model_dir = ROOT / "outputs" / "models" / "transformer"
    result_dir = ROOT / "outputs" / "results" / "transformer"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, std = load_dataset(data_dir)
    train_loader, val_loader = make_dataloaders(X_train, Y_train, X_val, Y_val)

    model = TransformerRegressor(
        input_dim=X_train.shape[-1],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_pearson_v": [],
        "val_pearson_a": [],
    }

    best_val_loss = float("inf")
    best_model_path = model_dir / "transformer_best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_pearson = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["val_pearson_v"].append(val_pearson[0].item())
        history["val_pearson_a"].append(val_pearson[1].item())

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_rmse={val_rmse:.4f} | "
            f"val_Pearson(V)={val_pearson[0]:.3f} | "
            f"val_Pearson(A)={val_pearson[1]:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    np.savez(result_dir / "transformer_history.npz", **history)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    test_loss, test_rmse, test_pearson = evaluate(model, test_loader, criterion, device)

    print("\n[TEST RESULT - Transformer]")
    print(f"  test_loss       = {test_loss:.4f}")
    print(f"  test_RMSE       = {test_rmse:.4f}")
    print(f"  test_Pearson(V) = {test_pearson[0]:.3f}")
    print(f"  test_Pearson(A) = {test_pearson[1]:.3f}")


if __name__ == "__main__":
    main()
