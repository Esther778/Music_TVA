#!/usr/bin/env python3
"""Build a model-blind human emotion annotation sheet for the frozen 15 songs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2 = PHASE2_ROOT / "data/blind_validation/stage2"
DEFAULT_STAGE5 = PHASE2_ROOT / "data/blind_validation/stage5"
OUTPUT_FIELDS = [
    "sample_index",
    "song_id",
    "title",
    "artist",
    "human_annotation_id",
    "section_order",
    "section_type",
    "start_time_sec",
    "end_time_sec",
    "human_valence_level",
    "human_arousal_level",
    "human_change_from_prev",
    "is_climax",
    "rater_id",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage5-dir", type=Path, default=DEFAULT_STAGE5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.stage2_dir / "blind_sample_15.csv").open(newline="", encoding="utf-8") as handle:
        songs = list(csv.DictReader(handle))
    labels = json.loads(
        (args.stage5_dir / "blind_sample_15_heldout_section_types.json").read_text(encoding="utf-8")
    )
    song_by_id = {row["song_id"]: row for row in songs}
    rows = []
    for label in sorted(labels, key=lambda row: (int(song_by_id[row["song_id"]]["sample_index"]), int(row["section_order"]))):
        song = song_by_id[label["song_id"]]
        rows.append(
            {
                "sample_index": int(song["sample_index"]),
                "song_id": label["song_id"],
                "title": song["title"],
                "artist": song["artist"],
                "human_annotation_id": label["human_annotation_id"],
                "section_order": int(label["section_order"]),
                "section_type": label["section_type"],
                "start_time_sec": float(label["start_time_sec"]),
                "end_time_sec": float(label["end_time_sec"]),
                "human_valence_level": "",
                "human_arousal_level": "",
                "human_change_from_prev": "" if int(label["section_order"]) > 1 else "not_applicable",
                "is_climax": "",
                "rater_id": "",
                "notes": "",
            }
        )
    output = args.stage5_dir / "blind_sample_15_human_emotion_annotation_template.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)}")
    print(f"model_prediction_columns=0")
    print(output)


if __name__ == "__main__":
    main()
