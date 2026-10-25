#!/usr/bin/env python3
"""Predict complete-song Arousal curves with the PMEmo Librosa Arousal model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_pmemo_librosa_arousal import ArousalBiLSTM, DEFAULT_CONFIG, extract_librosa_features, resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "arousal_fullsong_librosa"


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--start-sec", type=float, default=0.5)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--smooth-frames", type=int, default=5)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or (args.model_dir / "arousal_bilstm_best.pt")
    norm = np.load(args.model_dir / "dataset" / "feature_norm.npz")
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)
    config = dict(DEFAULT_CONFIG)
    config_path = args.model_dir / "config.json"
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))

    import librosa

    duration = float(librosa.get_duration(path=str(args.audio)))
    end_sec = duration if args.end_sec is None else min(float(args.end_sec), duration)
    times = np.arange(float(args.start_sec), end_sec, float(args.hop_sec), dtype=np.float32)
    if len(times) == 0:
        raise SystemExit(f"No prediction frames for {args.audio}; duration={duration:.3f}")

    X = extract_librosa_features(args.audio, times, config)
    X = ((X - mean) / std).astype(np.float32)

    device = resolve_device(args.device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ArousalBiLSTM(int(ckpt["input_dim"]), int(config["hidden_dim"]), int(config["num_layers"]), float(config["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)[None].to(device), torch.ones(1, len(X), dtype=torch.bool, device=device)).squeeze(0).detach().cpu().numpy()
    pred_smooth = moving_average(pred, int(args.smooth_frames))
    pred_01 = np.clip((pred_smooth + 1.0) / 2.0, 0.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_sec", "arousal_minus1_1", "arousal_0_1", "model_id", "feature_type"])
        writer.writeheader()
        for t, a, a01 in zip(times, pred_smooth, pred_01):
            writer.writerow(
                {
                    "time_sec": round(float(t), 3),
                    "arousal_minus1_1": round(float(a), 6),
                    "arousal_0_1": round(float(a01), 6),
                    "model_id": "pmemo_librosa_arousal_bilstm_v1",
                    "feature_type": "librosa_handcrafted",
                }
            )
    print(json.dumps({"audio": str(args.audio), "output": str(args.output), "frames": int(len(times)), "duration_sec": duration, "checkpoint": str(checkpoint_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
