#!/usr/bin/env python3
"""Batch complete-song Valence prediction wrapper."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "predict_full_song_valence_mert_ensemble.py"
DEFAULT_OUT = ROOT / "outputs" / "valence_fullsong_predictions_batch"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}


def safe_stem(path: Path) -> str:
    keep = []
    for ch in path.stem:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep).strip("_") or "audio"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-sec", type=float, default=0.5)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-member-preds", action="store_true")
    args = parser.parse_args()

    audio_files = sorted(p for p in args.audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if args.limit is not None:
        audio_files = audio_files[: int(args.limit)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for idx, audio in enumerate(audio_files, start=1):
        output = args.output_dir / f"{idx:04d}_{safe_stem(audio)}_valence_mert_ensemble.csv"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--audio",
            str(audio),
            "--output",
            str(output),
            "--device",
            str(args.device),
            "--start-sec",
            str(args.start_sec),
        ]
        if args.end_sec is not None:
            cmd.extend(["--end-sec", str(args.end_sec)])
        if args.allow_download:
            cmd.append("--allow-download")
        if args.save_member_preds:
            cmd.append("--save-member-preds")
        print(f"[{idx}/{len(audio_files)}] {audio} -> {output}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT.parents[2], text=True, capture_output=True, check=False)
        report_rows.append(
            {
                "audio": str(audio),
                "output": str(output),
                "status": "ok" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            }
        )
        if result.returncode != 0:
            print(result.stderr[-2000:], flush=True)

    report_path = args.output_dir / "batch_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio", "output", "status", "returncode", "stdout_tail", "stderr_tail"])
        writer.writeheader()
        writer.writerows(report_rows)
    ok = sum(1 for row in report_rows if row["status"] == "ok")
    print(f"batch complete: ok={ok}, error={len(report_rows) - ok}, report={report_path}")


if __name__ == "__main__":
    main()
