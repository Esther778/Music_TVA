#!/usr/bin/env python3
"""Convert saved MusiCNN patch embeddings into 50-tag patch sequences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import essentia
import essentia.standard as es
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDINGS = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_musicnn_sequence_embeddings_20260714"
DEFAULT_MODEL = PROJECT / "research/phase1g_valence_data_pretraining/models/essentia_emomusic_musicnn/msd-musicnn-1.pb"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_musicnn_tag_sequences_20260714"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_song = args.out_dir / "per_song"
    per_song.mkdir(exist_ok=True)

    essentia.log.warningActive = False
    head = es.TensorflowPredict2D(
        graphFilename=str(args.model),
        input="model/dense/BiasAdd",
        output="model/Sigmoid",
    )
    inventory: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    sources = sorted((args.embedding_dir / "per_song").glob("*.npz"), key=lambda path: int(path.stem))
    for number, source in enumerate(sources, start=1):
        music_id = int(source.stem)
        destination = per_song / source.name
        try:
            if destination.exists():
                saved = np.load(destination)
                tags = saved["tags"]
            else:
                saved = np.load(source)
                embedding = saved["embedding"]
                tags = np.asarray(head(embedding), dtype=np.float32)
                if tags.ndim != 2 or tags.shape[1] != 50 or tags.shape[0] != embedding.shape[0]:
                    raise ValueError(f"Unexpected tag shape {tags.shape} for embedding {embedding.shape}")
                np.savez_compressed(destination, tags=tags)
            inventory.append({
                "musicId": music_id,
                "patches": int(tags.shape[0]),
                "tag_dim": int(tags.shape[1]),
                "path": str(destination),
            })
        except Exception as exc:
            failures.append({"musicId": music_id, "error": repr(exc)})
        if number % 50 == 0 or number == len(sources):
            print(f"processed={number}/{len(sources)} ok={len(inventory)} failures={len(failures)}", flush=True)

    if inventory:
        with (args.out_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Derived from locked-development MusiCNN embeddings; final locked test excluded",
        "source": str(args.embedding_dir),
        "input_node": "model/dense/BiasAdd",
        "output_node": "model/Sigmoid",
        "songs": len(inventory),
        "failures": failures,
        "tag_dim": 50,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
