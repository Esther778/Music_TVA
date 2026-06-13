#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_sample_ms(column: str) -> int | None:
    match = re.fullmatch(r"sample_(\d+)ms", column)
    return int(match.group(1)) if match else None


def split_song_ids(song_ids: list[int], seed: int, ratios: dict[str, float]) -> dict[int, str]:
    ids = list(song_ids)
    random.Random(seed).shuffle(ids)
    n_total = len(ids)
    n_train = int(n_total * ratios["train"])
    n_val = int(n_total * ratios["val"])
    split_by_id: dict[int, str] = {}
    for sid in ids[:n_train]:
        split_by_id[sid] = "train"
    for sid in ids[n_train : n_train + n_val]:
        split_by_id[sid] = "val"
    for sid in ids[n_train + n_val :]:
        split_by_id[sid] = "test"
    return split_by_id


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DEAM audio and dynamic VA labels for Phase 1B.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage1_data_audit.json"))
    args = parser.parse_args()

    config = load_config(Path(args.config))
    deam_root = PROJECT_ROOT / config["source_deam_root"]
    audio_dir = deam_root / config["audio_subdir"]
    annotation_dir = deam_root / config["dynamic_annotation_subdir"]
    valence_path = annotation_dir / config["valence_csv"]
    arousal_path = annotation_dir / config["arousal_csv"]

    valence = pd.read_csv(valence_path)
    arousal = pd.read_csv(arousal_path)
    valence_ids = set(valence["song_id"].astype(int))
    arousal_ids = set(arousal["song_id"].astype(int))
    audio_paths = {int(path.stem): path for path in audio_dir.glob("*.mp3") if path.stem.isdigit()}
    common_ids = sorted(set(audio_paths) & valence_ids & arousal_ids)

    sample_columns = [column for column in valence.columns if column != "song_id"]
    sample_ms = [parse_sample_ms(column) for column in sample_columns]
    valid_sample_ms = [value for value in sample_ms if value is not None]
    label_start_ms = min(valid_sample_ms) if valid_sample_ms else None
    label_end_ms = max(valid_sample_ms) if valid_sample_ms else None
    label_hop_ms = valid_sample_ms[1] - valid_sample_ms[0] if len(valid_sample_ms) > 1 else None
    label_frames = len(valid_sample_ms)

    split_by_id = split_song_ids(common_ids, int(config["split_seed"]), config["split_ratio"])

    manifest_rows = []
    split_rows = []
    for sid in common_ids:
        row = {
            "song_id": sid,
            "audio_path": str(audio_paths[sid]),
            "has_audio": True,
            "has_valence": sid in valence_ids,
            "has_arousal": sid in arousal_ids,
            "label_start_ms": label_start_ms,
            "label_end_ms": label_end_ms,
            "label_hop_ms": label_hop_ms,
            "label_frames": label_frames,
        }
        manifest_rows.append(row)
        split_rows.append({**row, "split": split_by_id[sid]})

    manifest_fields = [
        "song_id",
        "audio_path",
        "has_audio",
        "has_valence",
        "has_arousal",
        "label_start_ms",
        "label_end_ms",
        "label_hop_ms",
        "label_frames",
    ]
    split_fields = manifest_fields + ["split"]
    manifest_dir = ROOT / "data" / "manifests"
    write_csv(manifest_dir / "data_manifest.csv", manifest_rows, manifest_fields)
    write_csv(manifest_dir / "split_manifest.csv", split_rows, split_fields)

    summary = {
        "source_deam_root": str(deam_root),
        "audio_count": len(audio_paths),
        "valence_song_count": len(valence_ids),
        "arousal_song_count": len(arousal_ids),
        "common_song_count": len(common_ids),
        "missing_audio_for_labels": len((valence_ids & arousal_ids) - set(audio_paths)),
        "audio_without_labels": len(set(audio_paths) - (valence_ids & arousal_ids)),
        "label_start_ms": label_start_ms,
        "label_end_ms": label_end_ms,
        "label_hop_ms": label_hop_ms,
        "label_frames": label_frames,
        "split_seed": config["split_seed"],
        "split_counts": {
            "train": sum(1 for value in split_by_id.values() if value == "train"),
            "val": sum(1 for value in split_by_id.values() if value == "val"),
            "test": sum(1 for value in split_by_id.values() if value == "test"),
        },
    }
    with (manifest_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

