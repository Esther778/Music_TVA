#!/usr/bin/env python3
"""Extract MusiCNN patch sequences for the locked PMEmo development pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import essentia
import essentia.standard as es
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
PMEMO = PROJECT / "research/phase1f_external_va_datasets/pmemo"
DEFAULT_INDEX = PMEMO / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv"
DEFAULT_AUDIO = PMEMO / "data/raw/PMEmo_2019_updated/extracted/PMEmo2019/chorus"
DEFAULT_MODELS = PROJECT / "research/phase1g_valence_data_pretraining/models/essentia_emomusic_musicnn"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_musicnn_sequence_embeddings_20260714"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.out_dir / "per_song"
    feature_dir.mkdir(exist_ok=True)
    rows = [row for row in read_csv(args.index) if row["split"] in {"train", "val"}]

    essentia.log.warningActive = False
    embedder = es.TensorflowPredictMusiCNN(
        graphFilename=str(args.model_dir / "msd-musicnn-1.pb"),
        output="model/dense/BiasAdd", batchSize=16,
    )
    failures = []
    completed = 0
    for number, row in enumerate(rows, start=1):
        music_id = int(row["musicId"])
        path = feature_dir / f"{music_id}.npz"
        try:
            if not path.exists():
                audio = es.MonoLoader(filename=str(args.audio_dir / f"{music_id}.mp3"), sampleRate=16000)()
                embedding = np.asarray(embedder(audio), dtype=np.float32)
                if embedding.ndim != 2 or embedding.shape[0] < 2:
                    raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
                np.savez_compressed(path, embedding=embedding)
            completed += 1
        except Exception as exc:
            failures.append({"musicId": music_id, "error": repr(exc)})
        if number % 25 == 0 or number == len(rows):
            print(f"processed={number}/{len(rows)} ok={completed} failures={len(failures)}", flush=True)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Audio-only features from locked PMEmo development pool; final locked test excluded",
        "backbone": "Essentia msd-musicnn-1 dense patch embedding",
        "songs": completed, "failures": failures,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
