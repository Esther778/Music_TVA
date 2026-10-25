#!/usr/bin/env python3
"""Materialize 56 MTG-Jamendo mood/theme tags from saved EffNet embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import essentia
import essentia.standard as es
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_discogs_effnet_features_20260714"
DEFAULT_MODEL = PROJECT / "research/phase1g_valence_data_pretraining/models/essentia_discogs_effnet/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_effnet_moodtheme_tags_20260714"
CLASSES = [
    "action", "adventure", "advertising", "background", "ballad", "calm", "children",
    "christmas", "commercial", "cool", "corporate", "dark", "deep", "documentary",
    "drama", "dramatic", "dream", "emotional", "energetic", "epic", "fast", "film",
    "fun", "funny", "game", "groovy", "happy", "heavy", "holiday", "hopeful",
    "inspiring", "love", "meditative", "melancholic", "melodic", "motivational",
    "movie", "nature", "party", "positive", "powerful", "relaxing", "retro", "romantic",
    "sad", "sexy", "slow", "soft", "soundscape", "space", "sport", "summer", "trailer",
    "travel", "upbeat", "uplifting",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_song = args.out_dir / "per_song"
    per_song.mkdir(exist_ok=True)

    essentia.log.warningActive = False
    predictor = es.TensorflowPredict2D(
        graphFilename=str(args.model), input="model/Placeholder", output="model/Sigmoid"
    )
    inventory = []
    failures = []
    source_files = sorted((args.input_dir / "per_song").glob("*.npz"), key=lambda path: int(path.stem))
    for number, source in enumerate(source_files, start=1):
        music_id = int(source.stem)
        destination = per_song / source.name
        try:
            if destination.exists():
                tags = np.load(destination)["tags"]
            else:
                embedding = np.load(source)["embedding"].astype(np.float32)
                tags = np.asarray(predictor(embedding), dtype=np.float32)
                if tags.ndim != 2 or tags.shape[1] != len(CLASSES):
                    raise ValueError(f"Unexpected tag shape: {tags.shape}")
                if not np.isfinite(tags).all():
                    raise ValueError("Tag sequence contains NaN/Inf")
                np.savez_compressed(destination, tags=tags)
            inventory.append({"musicId": music_id, "patches": len(tags), "tag_dim": tags.shape[1]})
        except Exception as exc:
            failures.append({"musicId": music_id, "error": repr(exc)})
        if number % 50 == 0 or number == len(source_files):
            print(f"processed={number}/{len(source_files)} ok={len(inventory)} failures={len(failures)}", flush=True)

    with (args.out_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Derived from locked-development EffNet features; final locked test excluded",
        "head": "mtg_jamendo_moodtheme-discogs-effnet-1",
        "head_sha256": sha256(args.model),
        "classes": CLASSES,
        "songs": len(inventory),
        "failures": failures,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
