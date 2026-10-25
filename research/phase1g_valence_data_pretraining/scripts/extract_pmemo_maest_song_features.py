#!/usr/bin/env python3
"""Extract full-clip MAEST window representations for the PMEmo dev pool."""

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
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT / "research/phase1g_valence_data_pretraining/models/maest_discogs_30s_pw_129e"
DEFAULT_MANIFEST = PROJECT / "research/phase1f_external_va_datasets/pmemo/data/processed/pmemo2019_song_manifest.csv"
DEFAULT_DEV = PROJECT / "research/phase1f_external_va_datasets/pmemo/outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_maest_30s_layer7_20260714"
FOLDS = ["fold_01", "fold_02", "fold_03"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def development_ids(directory: Path) -> set[int]:
    output: set[int] = set()
    for fold in FOLDS:
        frame = pd.read_csv(directory / f"{fold}_val_predictions.csv", usecols=["musicId"])
        output.update(frame["musicId"].astype(int).unique())
    return output


def windows(audio: np.ndarray, sample_rate: int, duration_sec: float) -> list[np.ndarray]:
    length = int(round(sample_rate * duration_sec))
    if len(audio) <= length:
        return [audio]
    return [audio[start : start + length] for start in range(0, len(audio), length)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dev-predictions", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_song = args.out_dir / "per_song"
    per_song.mkdir(exist_ok=True)

    allowed = development_ids(args.dev_predictions)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["musicId"].astype(int).isin(allowed)].copy()
    manifest = manifest.sort_values("musicId")
    if len(manifest) != len(allowed):
        raise ValueError(f"Development manifest mismatch: {len(manifest)} rows for {len(allowed)} IDs")
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    extractor = AutoFeatureExtractor.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_dir, local_files_only=True, output_hidden_states=True
    ).to(device).eval()

    inventory: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.time()
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        music_id = int(row.musicId)
        destination = per_song / f"{music_id}.npz"
        song_started = time.time()
        try:
            if destination.exists():
                representation = np.load(destination)["embedding"]
                audio_duration = float(row.metadata_duration)
            else:
                audio, _ = librosa.load(row.audio_path, sr=16000, mono=True)
                audio_duration = len(audio) / 16000.0
                vectors = []
                for window in windows(audio, 16000, args.window_sec):
                    inputs = extractor(window, sampling_rate=16000, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        output = model(**inputs, output_hidden_states=True)
                    hidden = output.hidden_states[args.layer][0].float().cpu()
                    vector = torch.cat([hidden[0], hidden[1], hidden[2:].mean(dim=0)])
                    vectors.append(vector.numpy().astype(np.float32))
                representation = np.stack(vectors)
                if representation.shape[1] != 2304 or not np.isfinite(representation).all():
                    raise ValueError(f"Invalid MAEST representation: {representation.shape}")
                np.savez_compressed(destination, embedding=representation)
            inventory.append({
                "musicId": music_id,
                "windows": int(len(representation)),
                "feature_dim": int(representation.shape[1]),
                "duration_sec": audio_duration,
                "elapsed_sec": time.time() - song_started,
            })
        except Exception as exc:
            failures.append({"musicId": music_id, "error": repr(exc)})
        if number % 20 == 0 or number == len(manifest):
            elapsed = time.time() - started
            print(
                f"processed={number}/{len(manifest)} ok={len(inventory)} failures={len(failures)} "
                f"elapsed_min={elapsed / 60:.2f}",
                flush=True,
            )

    if inventory:
        with (args.out_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "PMEmo three-fold development IDs only; final locked test excluded",
        "model": "mtg-upf/discogs-maest-30s-pw-129e",
        "model_sha256": sha256(args.model_dir / "model.safetensors"),
        "license": "CC BY-NC-SA 4.0",
        "layer": args.layer,
        "representation": "CLS + DIST + mean patch tokens",
        "window_sec": args.window_sec,
        "device": device,
        "songs": len(inventory),
        "windows": int(sum(int(row["windows"]) for row in inventory)),
        "failures": failures,
        "elapsed_sec": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
