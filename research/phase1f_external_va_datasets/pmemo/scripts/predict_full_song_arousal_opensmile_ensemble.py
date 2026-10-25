#!/usr/bin/env python3
"""Predict complete-song Arousal with the PMEmo OpenSMILE ensemble.

The PMEmo released dynamic features are 65 ComParE_2016 LLD mean/std features
plus 65 delta LLD mean/std features. We reproduce that structure by extracting
ComParE_2016 low-level descriptors and aggregating them with a trailing 1.0 s
window every 0.5 s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import opensmile
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENSEMBLE = ROOT / "outputs" / "opensmile_arousal_trials" / "arousal_ensemble_v1.json"
DEFAULT_NORM = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "feature_norm.npz"


class ArousalBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float, head_hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(y).squeeze(-1)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def index_seconds(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    idx = df.index
    starts = np.array([x.total_seconds() for x in idx.get_level_values("start")], dtype=np.float32)
    ends = np.array([x.total_seconds() for x in idx.get_level_values("end")], dtype=np.float32)
    centers = (starts + ends) / 2.0
    return df.reset_index(drop=True), centers


def extract_compare_lld(audio_path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    smile_lld = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    smile_delta = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
    )
    lld, lld_times = index_seconds(smile_lld.process_file(str(audio_path)))
    delta, delta_times = index_seconds(smile_delta.process_file(str(audio_path)))
    return lld, lld_times, delta, delta_times


def aggregate_pmemo_style(
    lld: pd.DataFrame,
    lld_times: np.ndarray,
    delta: pd.DataFrame,
    delta_times: np.ndarray,
    output_times: np.ndarray,
    feature_cols: list[str],
    window_sec: float,
) -> np.ndarray:
    rows: list[list[float]] = []
    for t in output_times:
        lld_mask = (lld_times > t - window_sec) & (lld_times <= t)
        delta_mask = (delta_times > t - window_sec) & (delta_times <= t)
        if not np.any(lld_mask):
            lld_mask[int(np.argmin(np.abs(lld_times - t)))] = True
        if not np.any(delta_mask):
            delta_mask[int(np.argmin(np.abs(delta_times - t)))] = True
        values: dict[str, float] = {}
        lld_part = lld.loc[lld_mask]
        delta_part = delta.loc[delta_mask]
        for col in lld.columns:
            values[f"{col}_mean"] = float(lld_part[col].mean())
            values[f"{col}_std"] = float(lld_part[col].std(ddof=0))
        for col in delta.columns:
            values[f"{col}_mean"] = float(delta_part[col].mean())
            values[f"{col}_std"] = float(delta_part[col].std(ddof=0))
        rows.append([values.get(col, 0.0) for col in feature_cols])
    return np.asarray(rows, dtype=np.float32)


def load_member(member: dict[str, Any], device: torch.device) -> nn.Module:
    ckpt_path = Path(member["checkpoint"])
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT.parents[2] / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["trial_config"]
    model = ArousalBiLSTM(
        int(ckpt["input_dim"]),
        int(cfg["hidden_dim"]),
        int(cfg["layers"]),
        float(cfg["dropout"]),
        int(cfg["head_hidden"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_ensemble(x_norm: np.ndarray, ensemble: dict[str, Any], device: torch.device) -> np.ndarray:
    x = torch.from_numpy(x_norm.astype(np.float32))[None, :, :].to(device)
    total = np.zeros(x_norm.shape[0], dtype=np.float32)
    for member in ensemble["members"]:
        model = load_member(member, device)
        pred = model(x).detach().cpu().numpy().squeeze(0)
        total += float(member["weight"]) * pred.astype(np.float32)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--norm", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--start-sec", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ensemble = json.loads(args.ensemble.read_text(encoding="utf-8"))
    norm = np.load(args.norm, allow_pickle=True)
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)
    feature_cols = [str(x) for x in norm["feature_cols"].tolist()]

    duration = float(librosa.get_duration(path=str(args.audio)))
    output_times = np.arange(float(args.start_sec), duration + 1e-6, float(args.hop_sec), dtype=np.float32)
    lld, lld_times, delta, delta_times = extract_compare_lld(args.audio)
    x = aggregate_pmemo_style(lld, lld_times, delta, delta_times, output_times, feature_cols, float(args.window_sec))
    x_norm = ((x - mean) / std).astype(np.float32)
    pred = predict_ensemble(x_norm, ensemble, resolve_device(args.device))
    pred = np.clip(pred, -1.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time_sec": output_times,
            "arousal_minus1_1": pred,
            "arousal_0_1": (pred + 1.0) / 2.0,
            "model_id": ensemble["model_id"],
            "feature_type": "opensmile_compare2016_lld_trailing_1s_mean_std",
        }
    ).to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "audio": str(args.audio),
                "output": str(args.output),
                "frames": int(len(output_times)),
                "duration_sec": duration,
                "model_id": ensemble["model_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
