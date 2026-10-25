#!/usr/bin/env python3
"""Predict complete-song Valence with the current PMEmo MERT+harmonic ensemble.

This is the deployable counterpart of the current Valence candidate. It uses
only information available at inference time:

1. extract MERT-v1-95M hidden_6 embeddings from the MP3,
2. extract Librosa harmonic features from the same 0.5 s frame grid,
3. run the three multi-split BiLSTM checkpoints,
4. average their frame-level predictions,
5. apply the global prediction-only rescale used for deployment.

The split-specific validation rescale remains the CV metric best, but a single
global rescale is cleaner for arbitrary songs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_musicva")
os.environ.setdefault(
    "HF_MODULES_CACHE",
    str(Path(__file__).resolve().parents[1] / "outputs" / "hf_cache" / "modules"),
)

import librosa
import numpy as np
import pandas as pd
import torch

from extract_pmemo_mert_features import extract_song, load_mert
from train_pmemo_librosa_arousal import DEFAULT_CONFIG as LIBROSA_CONFIG
from train_pmemo_librosa_arousal import extract_librosa_features
from train_pmemo_mert_harmonic_valence_fusion import FusionBiLSTM, selected_indices
from train_pmemo_mert_valence_baseline import resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT_DIR = ROOT / "outputs" / "valence_fullsong_predictions"

MODEL_ID = "pmemo_valence_mert_hidden6_harmonic_bilstm_ensemble_global_rescale_v1"
FEATURE_TYPE = "mert_v1_95m_hidden6_plus_librosa_harmonic"

GLOBAL_RESCALE = {
    "mean_gain": 1.0,
    "std_gain": 0.45,
    "blend": 1.0,
    "clip": None,
}


def prediction_times(duration_sec: float, start_sec: float, end_sec: float | None, hop_sec: float) -> np.ndarray:
    end = duration_sec if end_sec is None else min(float(end_sec), duration_sec)
    times = np.arange(float(start_sec), end + 1e-6, float(hop_sec), dtype=np.float32)
    if len(times) == 0:
        raise SystemExit(f"No prediction frames: duration={duration_sec:.3f}, start={start_sec}, end={end}")
    return times


def transform_prediction(pred: np.ndarray, mean_gain: float, std_gain: float, blend: float, clip: float | None) -> np.ndarray:
    song_mean = np.full_like(pred, float(np.mean(pred)), dtype=np.float32)
    transformed = float(mean_gain) * song_mean + float(std_gain) * (pred - song_mean)
    out = (1.0 - float(blend)) * pred + float(blend) * transformed
    if clip is not None:
        out = np.clip(out, -float(clip), float(clip))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def load_checkpoint(split_dir: Path, device: torch.device) -> tuple[FusionBiLSTM, dict[str, Any]]:
    ckpt = torch.load(split_dir / "best.pt", map_location=device)
    cfg = ckpt["config"]
    model = FusionBiLSTM(
        input_dim=int(ckpt["input_dim"]),
        proj_dim=int(cfg["proj_dim"]),
        hidden=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def build_split_features(mert_embedding: np.ndarray, librosa_all: np.ndarray, split_dir: Path) -> np.ndarray:
    mert_norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    feature_idx = harm_norm["feature_idx"].astype(np.int64)
    x_mert = ((mert_embedding.astype(np.float32) - mert_norm["mean"].astype(np.float32)) / mert_norm["std"].astype(np.float32)).astype(np.float32)
    x_harm = librosa_all.astype(np.float32)[:, feature_idx]
    x_harm = ((x_harm - harm_norm["mean"].astype(np.float32)) / harm_norm["std"].astype(np.float32)).astype(np.float32)
    return np.concatenate([x_mert, x_harm], axis=1).astype(np.float32)


@torch.no_grad()
def predict_split(x: np.ndarray, split_dir: Path, device: torch.device) -> np.ndarray:
    model, _ = load_checkpoint(split_dir, device)
    xt = torch.from_numpy(x)[None, :, :].to(device)
    mask = torch.ones(1, x.shape[0], dtype=torch.bool, device=device)
    return model(xt, mask).detach().cpu().numpy().squeeze(0).astype(np.float32)


def extract_features(
    audio: Path,
    times: np.ndarray,
    device: torch.device,
    model_name: str,
    chunk_sec: float,
    overlap_sec: float,
    pool_window_sec: float,
    allow_download: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mert_model, processor, transformers_version = load_mert(model_name, device, local_files_only=not allow_download)
    mert_times, embedding, mert_meta = extract_song(
        audio,
        times,
        mert_model,
        processor,
        device,
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
        pool_window_sec=pool_window_sec,
        layer="hidden_6",
    )
    if len(mert_times) != len(times):
        times = mert_times
    librosa_config = dict(LIBROSA_CONFIG)
    librosa_all = extract_librosa_features(audio, times, librosa_config)
    meta = {
        **mert_meta,
        "transformers_version": transformers_version,
        "mert_model": model_name,
        "mert_layer": "hidden_6",
        "librosa_feature_dim": int(librosa_all.shape[1]),
    }
    return times.astype(np.float32), embedding.astype(np.float32), librosa_all.astype(np.float32), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--start-sec", type=float, default=0.5)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    parser.add_argument("--overlap-sec", type=float, default=1.0)
    parser.add_argument("--pool-window-sec", type=float, default=0.5)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-member-preds", action="store_true")
    args = parser.parse_args()

    duration = float(librosa.get_duration(path=str(args.audio)))
    times = prediction_times(duration, args.start_sec, args.end_sec, args.hop_sec)
    device = resolve_device(args.device)

    times, mert_embedding, librosa_all, feature_meta = extract_features(
        args.audio,
        times,
        device,
        args.model_name,
        float(args.chunk_sec),
        float(args.overlap_sec),
        float(args.pool_window_sec),
        bool(args.allow_download),
    )

    split_dirs = sorted(p for p in args.model_dir.glob("split_*") if (p / "best.pt").exists())
    if not split_dirs:
        raise SystemExit(f"No split checkpoints found under {args.model_dir}")
    member_preds: dict[str, np.ndarray] = {}
    for split_dir in split_dirs:
        x = build_split_features(mert_embedding, librosa_all, split_dir)
        member_preds[split_dir.name] = predict_split(x, split_dir, device)
    raw_ensemble = np.mean(np.stack(list(member_preds.values()), axis=0), axis=0).astype(np.float32)
    valence = transform_prediction(raw_ensemble, **GLOBAL_RESCALE)

    out = pd.DataFrame(
        {
            "time_sec": times,
            "valence_minus1_1": valence,
            "valence_0_1": np.clip((valence + 1.0) / 2.0, 0.0, 1.0),
            "raw_ensemble_valence_minus1_1": raw_ensemble,
            "model_id": MODEL_ID,
            "feature_type": FEATURE_TYPE,
        }
    )
    if args.save_member_preds:
        for name, pred in member_preds.items():
            out[f"{name}_raw_valence_minus1_1"] = pred

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    meta = {
        "audio": str(args.audio),
        "output": str(args.output),
        "frames": int(len(out)),
        "duration_sec": duration,
        "model_id": MODEL_ID,
        "feature_type": FEATURE_TYPE,
        "model_dir": str(args.model_dir),
        "members": list(member_preds.keys()),
        "global_rescale": GLOBAL_RESCALE,
        "label_scale": "[-1, 1]; valence_0_1 is exported for convenience",
        "training_caveat": "PMEmo dynamic annotations exclude the unstable first 15 seconds; predictions before 15s should be interpreted cautiously.",
        **feature_meta,
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
