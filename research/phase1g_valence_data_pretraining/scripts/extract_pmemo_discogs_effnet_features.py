#!/usr/bin/env python3
"""Extract Discogs-EffNet sequences for the locked PMEmo development pool."""

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
PMEMO = PROJECT / "research/phase1f_external_va_datasets/pmemo"
DEFAULT_INDEX = PMEMO / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv"
DEFAULT_AUDIO = PMEMO / "data/raw/PMEmo_2019_updated/extracted/PMEmo2019/chorus"
DEFAULT_MODEL = PROJECT / "research/phase1g_valence_data_pretraining/models/essentia_discogs_effnet/discogs-effnet-bs64-1.pb"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_discogs_effnet_features_20260714"
EXPECTED_SHA256 = "3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    wanted = set(args.splits)
    if not wanted <= {"train", "val"}:
        raise ValueError("Only locked-development train/val splits are allowed")
    model_sha256 = sha256(args.model)
    if model_sha256 != EXPECTED_SHA256:
        raise ValueError(f"Discogs-EffNet checksum mismatch: {model_sha256}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.out_dir / "per_song"
    feature_dir.mkdir(exist_ok=True)
    rows = [row for row in read_csv(args.index) if row["split"] in wanted]

    essentia.log.warningActive = False
    embedder = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(args.model), output="PartitionedCall:1"
    )
    failures: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        music_id = int(row["musicId"])
        path = feature_dir / f"{music_id}.npz"
        try:
            if path.exists():
                saved = np.load(path)
                embedding = saved["embedding"]
                pooled = saved["pooled"]
            else:
                audio = es.MonoLoader(
                    filename=str(args.audio_dir / f"{music_id}.mp3"), sampleRate=16000
                )()
                embedding = np.asarray(embedder(audio), dtype=np.float32)
                if embedding.ndim != 2 or embedding.shape[0] < 2 or embedding.shape[1] != 1280:
                    raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
                if not np.isfinite(embedding).all():
                    raise ValueError("Embedding contains NaN/Inf")
                pooled = np.concatenate([
                    embedding.mean(0),
                    embedding.std(0),
                    np.quantile(embedding, 0.25, axis=0),
                    np.quantile(embedding, 0.75, axis=0),
                ]).astype(np.float32)
                np.savez_compressed(path, embedding=embedding, pooled=pooled)
            inventory.append({
                "musicId": music_id,
                "split": row["split"],
                "patches": int(embedding.shape[0]),
                "sequence_dim": int(embedding.shape[1]),
                "pooled_dim": int(pooled.size),
                "path": str(path),
            })
        except Exception as exc:
            failures.append({"musicId": music_id, "error": repr(exc)})
        if number % 25 == 0 or number == len(rows):
            print(f"processed={number}/{len(rows)} ok={len(inventory)} failures={len(failures)}", flush=True)

    if inventory:
        with (args.out_dir / "inventory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Locked PMEmo development pool only; final locked test excluded",
        "backbone": "Essentia Discogs-EffNet bs64",
        "model_sha256": model_sha256,
        "output_node": "PartitionedCall:1",
        "pooling": ["mean", "std", "q25", "q75"],
        "songs": len(inventory),
        "failures": failures,
        "sequence_dim": int(inventory[0]["sequence_dim"]) if inventory else 0,
        "pooled_dim": int(inventory[0]["pooled_dim"]) if inventory else 0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
