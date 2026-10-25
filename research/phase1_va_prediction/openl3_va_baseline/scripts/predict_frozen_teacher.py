from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_feature_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        features = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        for key in ("features", "embeddings", "arr_0"):
            if key in archive:
                features = archive[key]
                break
        else:
            raise ValueError(f"No supported feature key found in {path}.")
    else:
        raise ValueError("Feature input must be .npy or .npz.")
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got {features.shape}.")
    return features.astype(np.float32, copy=False)


class FrozenVATeacher:
    def __init__(self, config_path: Path, device: str = "cpu"):
        self.config_path = config_path.resolve()
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        self.project_root = self.config_path.parents[4]
        self.device = torch.device(device)
        norm_path = self.project_root / self.config["normalization"]["path"]
        norm = np.load(norm_path)
        self.mean = norm["mean"].astype(np.float32)
        self.std = norm["std"].astype(np.float32)

        head = self.config["head"]
        members = self.config["ensemble"]["members"]
        if self.config["selected_mode"] == "single_checkpoint":
            members = [self.config["best_single_by_validation"]]

        self.models = []
        for member in members:
            model = MLPRegressor(
                input_dim=int(self.config["feature_backbone"]["embedding_size"]),
                hidden=list(head["hidden"]),
                dropout=float(head["dropout"]),
            )
            checkpoint = self.project_root / member["checkpoint"]
            model.load_state_dict(
                torch.load(checkpoint, map_location=self.device, weights_only=True)
            )
            model.to(self.device)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        if features.shape[1] != self.mean.shape[1]:
            raise ValueError(
                f"Expected {self.mean.shape[1]} feature columns, got {features.shape[1]}."
            )
        normalized = (features - self.mean) / self.std
        member_outputs = []
        for model in self.models:
            batches = []
            for start in range(0, len(normalized), batch_size):
                batch = torch.from_numpy(
                    normalized[start : start + batch_size]
                ).float().to(self.device)
                batches.append(model(batch).cpu().numpy())
            member_outputs.append(np.concatenate(batches, axis=0))
        return np.mean(member_outputs, axis=0)


def main() -> None:
    baseline_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the frozen OpenL3+MLP VA Teacher on precomputed features."
    )
    parser.add_argument("features", type=Path, help="Raw 512-d OpenL3 .npy/.npz file.")
    parser.add_argument("output", type=Path, help="Output frame-level VA CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=baseline_root / "configs" / "frozen_teacher_v1.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    features = load_feature_array(args.features)
    teacher = FrozenVATeacher(args.config, device=args.device)
    predictions = teacher.predict(features)
    hop_size = float(teacher.config["feature_backbone"]["hop_size_sec"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "time_sec",
                "valence",
                "arousal",
                "teacher_model_id",
            ]
        )
        for index, (valence, arousal) in enumerate(predictions):
            writer.writerow(
                [
                    index,
                    index * hop_size,
                    float(valence),
                    float(arousal),
                    teacher.config["teacher_model_id"],
                ]
            )

    print(
        f"teacher={teacher.config['teacher_model_id']} "
        f"mode={teacher.config['selected_mode']} frames={len(predictions)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
