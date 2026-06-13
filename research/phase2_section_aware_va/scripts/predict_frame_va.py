#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import PHASE1_ROOT, PHASE2_ROOT, ensure_dir, read_csv


TARGET_SR = 44100
INPUT_REPR = "mel256"
CONTENT_TYPE = "music"
EMBEDDING_SIZE = 512
HOP_SIZE = 0.5


class MLP(nn.Module):
    def __init__(self, input_dim: int = 512, hidden1: int = 256, hidden2: int = 128, output_dim: int = 2, dropout: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_openl3():
    try:
        import librosa  # type: ignore
        import openl3  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "openl3 is not installed in the active Python environment. "
            "Provide precomputed 512-d OpenL3 features with --features-dir, "
            "or install the extractor explicitly before using --extract-audio."
        ) from exc
    return librosa, openl3


def load_model(device: torch.device) -> tuple[MLP, np.ndarray, np.ndarray]:
    model_path = PHASE1_ROOT / "outputs" / "models" / "mlp" / "mlp_best.pt"
    norm_path = PHASE1_ROOT / "outputs" / "models" / "mlp" / "mlp_input_norm.npz"
    model = MLP()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    norm = np.load(norm_path)
    return model, norm["mean"], norm["std"]


def extract_embedding(libs: tuple[object, object], audio_path: Path, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    librosa, openl3_module = libs
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    embedding, timestamps = openl3_module.get_audio_embedding(
        audio,
        TARGET_SR,
        input_repr=INPUT_REPR,
        content_type=CONTENT_TYPE,
        embedding_size=EMBEDDING_SIZE,
        hop_size=HOP_SIZE,
    )
    embedding_norm = (embedding - mean) / (std + 1e-8)
    return embedding.astype(np.float32), embedding_norm.astype(np.float32), timestamps.astype(np.float32)


def load_precomputed_embedding(path: Path, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path)
        if "embedding_norm" in data:
            embedding_norm = data["embedding_norm"].astype(np.float32)
            embedding = data["embedding"].astype(np.float32) if "embedding" in data else (embedding_norm * (std + 1e-8) + mean).astype(np.float32)
        elif "embedding" in data:
            embedding = data["embedding"].astype(np.float32)
            embedding_norm = ((embedding - mean) / (std + 1e-8)).astype(np.float32)
        else:
            raise ValueError(f"No embedding or embedding_norm array in {path}")
        timestamps = data["time_sec"].astype(np.float32) if "time_sec" in data else np.arange(len(embedding_norm), dtype=np.float32) * HOP_SIZE
        return embedding, embedding_norm, timestamps
    embedding = np.load(path).astype(np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != EMBEDDING_SIZE:
        raise ValueError(f"Expected shape (frames, {EMBEDDING_SIZE}) in {path}, got {embedding.shape}")
    embedding_norm = ((embedding - mean) / (std + 1e-8)).astype(np.float32)
    timestamps = np.arange(len(embedding), dtype=np.float32) * HOP_SIZE
    return embedding, embedding_norm, timestamps


def find_feature(features_dir: Path, song_id: str) -> Path | None:
    for suffix in [".npz", ".npy"]:
        candidate = features_dir / f"{song_id}{suffix}"
        if candidate.is_file():
            return candidate
        candidate = features_dir / f"{song_id}_openl3{suffix}"
        if candidate.is_file():
            return candidate
    return None


@torch.no_grad()
def predict(model: MLP, embedding_norm: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(embedding_norm).float().to(device)
    return model(x).cpu().numpy().astype(np.float32)


def save_frame_csv(path: Path, song: dict[str, str], timestamps: np.ndarray, predictions: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["song_id", "title", "artist", "time_sec", "valence", "arousal"])
        for time_sec, (valence, arousal) in zip(timestamps, predictions):
            writer.writerow([song["id"], song["title"], song["artist"], float(time_sec), float(valence), float(arousal)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 0.5s frame-level VA using the Phase 1 OpenL3 + MLP model.")
    parser.add_argument("--splits", default=str(PHASE2_ROOT / "data" / "manifests" / "phase2_initial_70_splits.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--features-dir", default="", help="Directory containing precomputed per-song OpenL3 .npy/.npz features.")
    parser.add_argument("--extract-audio", action="store_true", help="Extract OpenL3 features from audio. Requires openl3 installed.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-song frame VA and feature files.")
    args = parser.parse_args()

    if not args.features_dir and not args.extract_audio:
        raise SystemExit("Provide --features-dir for precomputed OpenL3 features, or pass --extract-audio explicitly.")
    libs = load_openl3() if args.extract_audio else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_model(device)

    frame_dir = ensure_dir(PHASE2_ROOT / "data" / "frame_va")
    feature_dir = ensure_dir(PHASE2_ROOT / "data" / "openl3_features")
    rows = read_csv(Path(args.splits))
    if args.limit is not None:
        rows = rows[: args.limit]

    completed = 0
    skipped = 0
    for song in rows:
        song_id = song["id"]
        audio_path = Path(song["local_audio_path"])
        frame_path = frame_dir / f"{song_id}_frame_va.csv"
        feature_path = feature_dir / f"{song_id}_openl3_va.npz"
        if not audio_path.is_file():
            print(f"missing_audio {song_id} {audio_path}")
            skipped += 1
            continue
        if frame_path.is_file() and feature_path.is_file() and not args.force:
            print(f"preserved_existing {song_id}")
            skipped += 1
            continue
        feature_path_in = find_feature(Path(args.features_dir), song_id) if args.features_dir else None
        if feature_path_in is not None:
            embedding, embedding_norm, timestamps = load_precomputed_embedding(feature_path_in, mean, std)
        elif libs is not None:
            embedding, embedding_norm, timestamps = extract_embedding(libs, audio_path, mean, std)
        else:
            print(f"missing_openl3_features {song_id}")
            skipped += 1
            continue
        predictions = predict(model, embedding_norm, device)
        save_frame_csv(frame_path, song, timestamps, predictions)
        np.savez_compressed(
            feature_path,
            song_id=song_id,
            title=song["title"],
            artist=song["artist"],
            time_sec=timestamps,
            embedding=embedding,
            embedding_norm=embedding_norm,
            valence=predictions[:, 0],
            arousal=predictions[:, 1],
        )
        completed += 1
        print(f"generated {song_id} frames={len(timestamps)}")

    print(f"completed={completed} skipped={skipped}")


if __name__ == "__main__":
    main()
