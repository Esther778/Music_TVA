#!/usr/bin/env python3
"""Build the standalone browser console for blind section emotion annotation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/outputs/human_validation/"
    "blind_sample_15_audio_clip_audit.csv"
)
TEMPLATE_PATH = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/web/human_validation_console_template.html"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/outputs/human_validation/"
    "blind_sample_15_human_validation_console.html"
)


def main() -> None:
    with AUDIT_PATH.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))

    if len(source_rows) != 179:
        raise RuntimeError(f"Expected 179 section clips, found {len(source_rows)}")

    rows = []
    for source in source_rows:
        if source["status"] != "pass":
            raise RuntimeError(
                f"Audio audit failed for {source['human_annotation_id']}: {source['status']}"
            )
        clip_path = AUDIT_PATH.parent / source["clip_relative_path"]
        if not clip_path.is_file():
            raise FileNotFoundError(clip_path)
        rows.append(
            {
                "sample_index": int(source["sample_index"]),
                "song_id": source["song_id"],
                "title": source["title"],
                "artist": source["artist"],
                "human_annotation_id": source["human_annotation_id"],
                "section_order": int(source["section_order"]),
                "start_sec": float(source["requested_start_sec"]),
                "end_sec": float(source["requested_end_sec"]),
                "duration_sec": float(source["measured_duration_sec"]),
                "clip_relative_path": source["clip_relative_path"],
            }
        )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count("__DATA_JSON__") != 1:
        raise RuntimeError("Template must contain exactly one __DATA_JSON__ placeholder")
    html = template.replace(
        "__DATA_JSON__",
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "songs": len({row["song_id"] for row in rows}),
                "sections": len(rows),
                "audio_files_verified": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
