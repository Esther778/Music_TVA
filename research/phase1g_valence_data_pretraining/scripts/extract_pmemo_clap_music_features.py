#!/usr/bin/env python3
"""Extract deterministic music-CLAP window embeddings for the locked PMEmo dev pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT / "research/phase1g_valence_data_pretraining/models/laion_larger_clap_music"
DEFAULT_MANIFEST = PROJECT / "research/phase1f_external_va_datasets/pmemo/data/processed/pmemo2019_song_manifest.csv"
DEFAULT_DEV = PROJECT / "research/phase1f_external_va_datasets/pmemo/data/splits/pmemo_final_locked_20260713/development_pool_index.csv"
DEFAULT_FINAL = PROJECT / "research/phase1f_external_va_datasets/pmemo/data/splits/pmemo_final_locked_20260713/final_locked_test_index.csv"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_clap_music_10s_20260714"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def window_audio(audio: np.ndarray, sample_rate: int, window_sec: float, hop_sec: float):
    window_samples = int(round(window_sec * sample_rate))
    hop_samples = int(round(hop_sec * sample_rate))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window-sec and hop-sec must be positive")
    if len(audio) <= window_samples:
        yield 0, audio
        return
    starts = list(range(0, max(1, len(audio) - window_samples + 1), hop_samples))
    final_start = max(0, len(audio) - window_samples)
    if starts[-1] != final_start:
        starts.append(final_start)
    for start in starts:
        yield start, audio[start : start + window_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--development-index", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--final-index", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--hop-sec", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_song = args.out_dir / "per_song"
    per_song.mkdir(exist_ok=True)

    development = pd.read_csv(args.development_index)
    final = pd.read_csv(args.final_index)
    development_ids = set(development["musicId"].astype(int))
    final_ids = set(final["musicId"].astype(int))
    overlap = development_ids.intersection(final_ids)
    if overlap:
        raise ValueError(f"Final locked leakage in development index: {sorted(overlap)[:20]}")

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["musicId"].astype(int).isin(development_ids)].copy()
    manifest = manifest.sort_values("musicId")
    if len(manifest) != len(development_ids):
        raise ValueError(f"Development manifest mismatch: {len(manifest)} rows for {len(development_ids)} IDs")
    if set(manifest["musicId"].astype(int)).intersection(final_ids):
        raise ValueError("Final locked IDs reached the CLAP extraction manifest")
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    processor = ClapProcessor.from_pretrained(args.model_dir, local_files_only=True)
    sample_rate = int(processor.feature_extractor.sampling_rate)
    model = ClapModel.from_pretrained(args.model_dir, local_files_only=True)
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            model = model.to(device)
        except RuntimeError as exc:
            print(f"MPS unavailable at runtime; falling back to CPU: {exc}", flush=True)
            device = "cpu"
            model = model.to(device)
    else:
        device = args.device
        model = model.to(device)
    model.eval()

    inventory: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.time()
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        music_id = int(row.musicId)
        destination = per_song / f"{music_id}.npz"
        song_started = time.time()
        try:
            if destination.exists():
                saved = np.load(destination)
                embeddings = saved["embedding"]
                starts_sec = saved["start_sec"]
                ends_sec = saved["end_sec"]
                audio_duration = float(saved["duration_sec"])
            else:
                audio, _ = librosa.load(row.audio_path, sr=sample_rate, mono=True)
                audio = np.asarray(audio, dtype=np.float32)
                audio_duration = len(audio) / sample_rate
                vectors, starts_sec, ends_sec = [], [], []
                pending_audio = []
                for start, window in window_audio(audio, sample_rate, args.window_sec, args.hop_sec):
                    pending_audio.append(window)
                    starts_sec.append(start / sample_rate)
                    ends_sec.append(min(audio_duration, (start + len(window)) / sample_rate))
                for batch_start in range(0, len(pending_audio), args.batch_size):
                    batch = pending_audio[batch_start : batch_start + args.batch_size]
                    inputs = processor(
                        audios=batch,
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        batch_vectors = model.get_audio_features(**inputs)
                    vectors.append(batch_vectors.float().cpu().numpy().astype(np.float32))
                embeddings = np.concatenate(vectors, axis=0)
                starts_sec = np.asarray(starts_sec, dtype=np.float32)
                ends_sec = np.asarray(ends_sec, dtype=np.float32)
                if embeddings.ndim != 2 or embeddings.shape[1] != 512:
                    raise ValueError(f"Unexpected CLAP embedding shape: {embeddings.shape}")
                if not np.isfinite(embeddings).all():
                    raise ValueError("Non-finite CLAP embedding")
                np.savez_compressed(
                    destination,
                    embedding=embeddings,
                    start_sec=starts_sec,
                    end_sec=ends_sec,
                    duration_sec=np.float32(audio_duration),
                )
            inventory.append({
                "musicId": music_id,
                "windows": int(len(embeddings)),
                "feature_dim": int(embeddings.shape[1]),
                "duration_sec": audio_duration,
                "first_start_sec": float(starts_sec[0]),
                "last_end_sec": float(ends_sec[-1]),
                "elapsed_sec": time.time() - song_started,
            })
        except Exception as exc:
            failures.append({"musicId": music_id, "error": repr(exc)})
        if number % 10 == 0 or number == len(manifest):
            print(
                f"processed={number}/{len(manifest)} ok={len(inventory)} failures={len(failures)} "
                f"elapsed_min={(time.time() - started) / 60:.2f}",
                flush=True,
            )

    if inventory:
        with (args.out_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
    with (args.out_dir / "failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["musicId", "error"])
        writer.writeheader()
        writer.writerows(failures)

    model_path = args.model_dir / "pytorch_model.bin"
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "PMEmo locked development pool only; final locked test explicitly excluded",
        "model": "laion/larger_clap_music",
        "model_sha256": sha256(model_path),
        "license": "Apache-2.0",
        "sample_rate": sample_rate,
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "batch_size": args.batch_size,
        "representation": "512-D projected audio embedding from each full-coverage window",
        "device": device,
        "requested_songs": int(len(manifest)),
        "songs": len(inventory),
        "windows": int(sum(int(row["windows"]) for row in inventory)),
        "failures": failures,
        "final_locked_overlap": 0,
        "elapsed_sec": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
