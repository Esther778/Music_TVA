#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common import PHASE2_ROOT


SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TARGET_COLUMNS = ["mean_valence", "mean_arousal"]
NUMERIC_FEATURES = [
    "section_index",
    "section_duration_sec",
    "section_position",
    "n_sections",
    "frame_count",
    "std_valence",
    "entry_valence",
    "exit_valence",
    "delta_valence",
    "slope_valence",
    "peak_valence_time_rel",
    "std_arousal",
    "entry_arousal",
    "exit_arousal",
    "delta_arousal",
    "slope_arousal",
    "peak_arousal_time_rel",
    "prev_delta_mean_valence",
    "next_delta_mean_valence",
    "prev_delta_mean_arousal",
    "next_delta_mean_arousal",
]
CATEGORICAL_FEATURES = ["section_type"]


class SectionEmotionNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def pearsonr(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    values = []
    for idx in range(2):
        x = pred[:, idx]
        y = target[:, idx]
        if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return values[0], values[1]


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def to_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(x_train: np.ndarray, y_train: np.ndarray, device: torch.device) -> SectionEmotionNet:
    model = SectionEmotionNet(x_train.shape[1]).to(device)
    loader = to_loader(x_train, y_train, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: SectionEmotionNet, x: np.ndarray, device: torch.device) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 2), dtype=np.float32)
    model.eval()
    return model(torch.from_numpy(x).float().to(device)).cpu().numpy()


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(pred) == 0:
        return {"mse": np.nan, "rmse": np.nan, "pearson_valence": np.nan, "pearson_arousal": np.nan}
    mse = float(mean_squared_error(target, pred))
    rmse = float(np.sqrt(mse))
    rv, ra = pearsonr(pred, target)
    return {"mse": mse, "rmse": rmse, "pearson_valence": rv, "pearson_arousal": ra}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 section-aware emotion structure baseline.")
    parser.add_argument("--features", default=str(PHASE2_ROOT / "data" / "section_va" / "phase2_initial_70_section_features.csv"))
    parser.add_argument("--max-songs", type=int, default=70)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = pd.read_csv(args.features)
    features = features[features["selected_order"] <= args.max_songs].copy()
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].fillna(0)
    features[CATEGORICAL_FEATURES] = features[CATEGORICAL_FEATURES].fillna("unknown")

    metrics_rows: list[dict[str, float | int | str]] = []
    prediction_rows = []
    model_dir = PHASE2_ROOT / "outputs" / "models" / "section_model"
    metric_dir = PHASE2_ROOT / "outputs" / "metrics"
    prediction_dir = PHASE2_ROOT / "outputs" / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for cutoff in range(args.step, args.max_songs + 1, args.step):
        train_df = features[(features["split"] == "train") & (features["selected_order"] <= cutoff)].copy()
        val_df = features[features["split"] == "val"].copy()
        test_df = features[features["split"] == "test"].copy()
        if len(train_df) < 2:
            print(f"skip cutoff={cutoff} insufficient_train_sections={len(train_df)}")
            continue

        preprocessor = Pipeline([("columns", make_preprocessor())])
        x_train = preprocessor.fit_transform(train_df)
        y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        model = train_model(np.asarray(x_train, dtype=np.float32), y_train, device)

        torch.save(model.state_dict(), model_dir / f"section_model_cumulative_{cutoff:02d}.pt")
        with (model_dir / f"section_model_cumulative_{cutoff:02d}_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "cutoff_songs": cutoff,
                    "train_sections": int(len(train_df)),
                    "target_columns": TARGET_COLUMNS,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "note": "Targets are Phase 1 generated pseudo VA section means, not human VA annotations.",
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if split_df.empty:
                pred = np.empty((0, 2), dtype=np.float32)
                target = np.empty((0, 2), dtype=np.float32)
            else:
                x_split = preprocessor.transform(split_df)
                target = split_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
                pred = predict(model, np.asarray(x_split, dtype=np.float32), device)
                for row, values in zip(split_df.to_dict("records"), pred):
                    prediction_rows.append(
                        {
                            "cutoff_songs": cutoff,
                            "split": split_name,
                            "song_id": row["song_id"],
                            "section_index": row["section_index"],
                            "section_type": row["section_type"],
                            "target_mean_valence": row["mean_valence"],
                            "target_mean_arousal": row["mean_arousal"],
                            "pred_mean_valence": float(values[0]),
                            "pred_mean_arousal": float(values[1]),
                        }
                    )
            row_metrics = metrics(pred, target)
            metrics_rows.append(
                {
                    "cutoff_songs": cutoff,
                    "split": split_name,
                    "sections": int(len(split_df)),
                    **row_metrics,
                }
            )
        print(f"trained cutoff={cutoff} train_sections={len(train_df)}")

    pd.DataFrame(metrics_rows).to_csv(metric_dir / "phase2_section_model_batch_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(prediction_dir / "phase2_section_model_predictions.csv", index=False)
    print(f"metrics_rows={len(metrics_rows)} predictions={len(prediction_rows)}")


if __name__ == "__main__":
    main()
