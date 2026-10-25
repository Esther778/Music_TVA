#!/usr/bin/env python3
"""Predict complete-song Valence with the current-best deployable approximation.

The current metric-best Valence result is a validation-selected prediction
artifact:

    pmemo_valence_parsimony_output_guarded_20260707

It is not a single checkpoint. This script provides a deployable raw-MP3
approximation that keeps the same audio-only feature direction:

1. extract MERT hidden_4, hidden_6, hidden_8 embeddings;
2. extract Librosa harmonic features on the same 0.5 s grid;
3. run the older hidden_6 + harmonic split ensemble as a base prediction;
4. run the direct multi-layer hidden_4/6/8 + harmonic split_02 model;
5. blend split_02 as `0.65 * base + 0.35 * direct_multilayer`;
6. apply the split_02 smoothing window 15 selected by the guarded output search.

This is the correct operational direction for arbitrary songs, but its metrics
should be reported as an approximation to the CV metric-best artifact, not as a
separate validated model.
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
from screen_pmemo_valence_multilayer_mert import MultiLayerBiLSTM, MultiLayerConfig
from train_pmemo_librosa_arousal import DEFAULT_CONFIG as LIBROSA_CONFIG
from train_pmemo_librosa_arousal import extract_librosa_features
from train_pmemo_mert_harmonic_valence_fusion import FusionBiLSTM
from train_pmemo_mert_valence_baseline import resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_DIRECT_DIR = ROOT / "outputs" / "valence_multilayer_mert_screen_20260629" / "h468_harm_concat_proj384_h80_do035" / "split_02_seed_606"
DEFAULT_OUT_DIR = ROOT / "outputs" / "valence_fullsong_current_best_approx"

MODEL_ID = "pmemo_valence_current_best_deployable_approx_20260707"
METRIC_BEST_REFERENCE = {
    "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
    "rmse": 0.19127265472577384,
    "mae": 0.14892050958155043,
    "pcc": 0.8060728304873086,
    "ccc": 0.7915010047864408,
}
DEPLOYABLE_BASE_REFERENCE = {
    "model_id": "pmemo_valence_mert_hidden6_harmonic_bilstm_ensemble_global_rescale_v1",
    "macro_rmse": 0.19538791661216673,
    "macro_mae": 0.152277096301242,
    "macro_pcc": 0.7979096406370229,
    "macro_ccc": 0.7834203725305792,
}

DIRECT_WEIGHT = 0.35
SMOOTH_WINDOW = 15
FEATURE_TYPE = "mert_v1_95m_hidden4_hidden6_hidden8_plus_librosa_harmonic"


def prediction_times(duration_sec: float, start_sec: float, end_sec: float | None, hop_sec: float) -> np.ndarray:
    end = duration_sec if end_sec is None else min(float(end_sec), duration_sec)
    times = np.arange(float(start_sec), end + 1e-6, float(hop_sec), dtype=np.float32)
    if len(times) == 0:
        raise SystemExit(f"No prediction frames: duration={duration_sec:.3f}, start={start_sec}, end={end}")
    return times


def moving_average(pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(pred) == 0:
        return pred.astype(np.float32)
    kernel = np.ones(int(window), dtype=np.float64)
    start = (int(window) - 1) // 2
    vals = pred.astype(np.float64)
    summed = np.convolve(vals, kernel, mode="full")[start : start + len(vals)]
    counts = np.convolve(np.ones_like(vals), kernel, mode="full")[start : start + len(vals)]
    return (summed / np.maximum(counts, 1.0)).astype(np.float32)


def load_base_checkpoint(split_dir: Path, device: torch.device) -> FusionBiLSTM:
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
    return model


def load_direct_checkpoint(split_dir: Path, device: torch.device) -> MultiLayerBiLSTM:
    ckpt = torch.load(split_dir / "best.pt", map_location=device)
    cfg = ckpt["config"]
    config = MultiLayerConfig(
        trial_id=str(cfg.get("trial_id", "h468_harm_concat_proj384_h80_do035")),
        proj_dim=int(cfg["proj_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 3e-4)),
        batch_size=int(cfg.get("batch_size", 18)),
        frame_ccc_weight=float(cfg.get("frame_ccc_weight", 0.08)),
        clip_mse_weight=float(cfg.get("clip_mse_weight", 0.10)),
        clip_std_weight=float(cfg.get("clip_std_weight", 0.0)),
        extreme_weight=float(cfg.get("extreme_weight", 1.25)),
        patience=int(cfg.get("patience", 8)),
        epochs=int(cfg.get("epochs", 60)),
    )
    model = MultiLayerBiLSTM(int(ckpt["input_dim"]), config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_base_features(hidden6_embedding: np.ndarray, librosa_all: np.ndarray, split_dir: Path) -> np.ndarray:
    mert_norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    feature_idx = harm_norm["feature_idx"].astype(np.int64)
    x_mert = ((hidden6_embedding.astype(np.float32) - mert_norm["mean"].astype(np.float32)) / mert_norm["std"].astype(np.float32)).astype(np.float32)
    x_harm = librosa_all.astype(np.float32)[:, feature_idx]
    x_harm = ((x_harm - harm_norm["mean"].astype(np.float32)) / harm_norm["std"].astype(np.float32)).astype(np.float32)
    return np.concatenate([x_mert, x_harm], axis=1).astype(np.float32)


def build_direct_features(layer_embeddings: dict[str, np.ndarray], librosa_all: np.ndarray, split_dir: Path) -> np.ndarray:
    parts = []
    for layer_name in ["h4", "h6", "h8"]:
        norm = np.load(split_dir / "dataset" / f"{layer_name}_norm.npz")
        x = layer_embeddings[layer_name].astype(np.float32)
        parts.append(((x - norm["mean"].astype(np.float32)) / norm["std"].astype(np.float32)).astype(np.float32))
    harm_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    feature_idx = harm_norm["feature_idx"].astype(np.int64)
    x_harm = librosa_all.astype(np.float32)[:, feature_idx]
    x_harm = ((x_harm - harm_norm["mean"].astype(np.float32)) / harm_norm["std"].astype(np.float32)).astype(np.float32)
    return np.concatenate([*parts, x_harm], axis=1).astype(np.float32)


@torch.no_grad()
def predict_model(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(x)[None, :, :].to(device)
    mask = torch.ones(1, x.shape[0], dtype=torch.bool, device=device)
    return model(xt, mask).detach().cpu().numpy().squeeze(0).astype(np.float32)


def extract_mert_layers(
    audio: Path,
    times: np.ndarray,
    device: torch.device,
    model_name: str,
    chunk_sec: float,
    overlap_sec: float,
    pool_window_sec: float,
    allow_download: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    mert_model, processor, transformers_version = load_mert(model_name, device, local_files_only=not allow_download)
    embeddings: dict[str, np.ndarray] = {}
    common_times = times
    meta: dict[str, Any] = {}
    for layer_key, layer_spec in [("h4", "hidden_4"), ("h6", "hidden_6"), ("h8", "hidden_8")]:
        layer_times, embedding, layer_meta = extract_song(
            audio,
            common_times,
            mert_model,
            processor,
            device,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            pool_window_sec=pool_window_sec,
            layer=layer_spec,
        )
        common_times = layer_times
        embeddings[layer_key] = embedding.astype(np.float32)
        meta[f"{layer_key}_meta"] = layer_meta
    min_len = min(len(common_times), *(len(v) for v in embeddings.values()))
    common_times = common_times[:min_len]
    embeddings = {k: v[:min_len] for k, v in embeddings.items()}
    meta.update(
        {
            "transformers_version": transformers_version,
            "mert_model": model_name,
            "mert_layers": ["hidden_4", "hidden_6", "hidden_8"],
        }
    )
    return common_times.astype(np.float32), embeddings, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--direct-split-dir", type=Path, default=DEFAULT_DIRECT_DIR)
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
    times, layer_embeddings, feature_meta = extract_mert_layers(
        args.audio,
        times,
        device,
        args.model_name,
        float(args.chunk_sec),
        float(args.overlap_sec),
        float(args.pool_window_sec),
        bool(args.allow_download),
    )
    librosa_config = dict(LIBROSA_CONFIG)
    librosa_all = extract_librosa_features(args.audio, times, librosa_config)
    min_len = min(len(times), len(librosa_all), *(len(v) for v in layer_embeddings.values()))
    times = times[:min_len]
    librosa_all = librosa_all[:min_len]
    layer_embeddings = {k: v[:min_len] for k, v in layer_embeddings.items()}

    base_split_dirs = sorted(p for p in args.base_model_dir.glob("split_*") if (p / "best.pt").exists())
    if not base_split_dirs:
        raise SystemExit(f"No base checkpoints found under {args.base_model_dir}")

    base_member_preds: dict[str, np.ndarray] = {}
    for split_dir in base_split_dirs:
        model = load_base_checkpoint(split_dir, device)
        x = build_base_features(layer_embeddings["h6"], librosa_all, split_dir)
        base_member_preds[split_dir.name] = predict_model(model, x, device)
    base_raw = np.mean(np.stack(list(base_member_preds.values()), axis=0), axis=0).astype(np.float32)

    direct_model = load_direct_checkpoint(args.direct_split_dir, device)
    direct_x = build_direct_features(layer_embeddings, librosa_all, args.direct_split_dir)
    direct_pred = predict_model(direct_model, direct_x, device)

    blended = ((1.0 - DIRECT_WEIGHT) * base_raw + DIRECT_WEIGHT * direct_pred).astype(np.float32)
    valence = moving_average(blended, SMOOTH_WINDOW)
    valence = np.clip(valence, -1.0, 1.0).astype(np.float32)

    out = pd.DataFrame(
        {
            "time_sec": times,
            "valence_minus1_1": valence,
            "valence_0_1": np.clip((valence + 1.0) / 2.0, 0.0, 1.0),
            "base_raw_valence_minus1_1": base_raw,
            "direct_multilayer_valence_minus1_1": direct_pred,
            "blended_valence_minus1_1": blended,
            "model_id": MODEL_ID,
            "feature_type": FEATURE_TYPE,
        }
    )
    if args.save_member_preds:
        for name, pred in base_member_preds.items():
            out[f"{name}_base_valence_minus1_1"] = pred

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    meta = {
        "audio": str(args.audio),
        "output": str(args.output),
        "frames": int(len(out)),
        "duration_sec": duration,
        "model_id": MODEL_ID,
        "feature_type": FEATURE_TYPE,
        "metric_best_reference": METRIC_BEST_REFERENCE,
        "deployable_base_reference": DEPLOYABLE_BASE_REFERENCE,
        "approximation_notice": (
            "This script is a deployable approximation to the current metric-best CV artifact. "
            "It does not reproduce every upstream prediction-level artifact exactly."
        ),
        "base_model_dir": str(args.base_model_dir),
        "base_members": list(base_member_preds.keys()),
        "direct_split_dir": str(args.direct_split_dir),
        "direct_weight": DIRECT_WEIGHT,
        "smooth_window": SMOOTH_WINDOW,
        "label_scale": "[-1, 1]; valence_0_1 is exported for convenience",
        "training_caveat": "PMEmo dynamic annotations exclude the unstable first 15 seconds; predictions before 15s should be interpreted cautiously.",
        "librosa_feature_dim": int(librosa_all.shape[1]),
        **feature_meta,
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
