#!/usr/bin/env python3
"""Download the audio files listed in songs_manifest.csv."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import requests
import urllib3


def download_file_with_requests(
    url: str,
    output_path: Path,
    verify_ssl: bool = True,
    referer: str | None = None,
) -> None:
    headers = {"User-Agent": "music-section-segmenter/1.0"}
    if referer:
        headers["Referer"] = referer
    with requests.get(url, headers=headers, stream=True, timeout=60, verify=verify_ssl) as response:
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".part")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(output_path)


def download_file_with_curl(
    url: str,
    output_path: Path,
    insecure: bool = False,
    referer: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "2",
        "--continue-at",
        "-",
        "--max-time",
        "900",
        "-A",
        "Mozilla/5.0",
    ]
    if insecure:
        command.append("-k")
    if referer:
        command.extend(["-e", referer])
    command.extend(["-o", str(tmp_path), url])
    subprocess.run(command, check=True)
    tmp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download songs from a CSV manifest.")
    parser.add_argument("--manifest", default="songs_manifest.csv", help="CSV manifest path.")
    parser.add_argument("--out-dir", default="songs", help="Folder to store downloaded audio.")
    parser.add_argument("--force", action="store_true", help="Download again even if file exists.")
    parser.add_argument("--insecure", action="store_true", help="Skip SSL certificate verification.")
    parser.add_argument("--method", choices=["requests", "curl"], default="curl", help="Download backend.")
    parser.add_argument("--limit", type=int, help="Download only the first N songs from the manifest.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]

    for index, row in enumerate(rows, start=1):
        output_path = out_dir / row["filename"]
        if output_path.exists() and not args.force:
            print(f"[{index:02d}/{len(rows):02d}] exists: {output_path}")
            continue

        print(f"[{index:02d}/{len(rows):02d}] downloading: {row['title']}")
        referer = row.get("page_url") or None
        if args.method == "curl":
            download_file_with_curl(row["source_url"], output_path, insecure=args.insecure, referer=referer)
        else:
            download_file_with_requests(row["source_url"], output_path, verify_ssl=not args.insecure, referer=referer)
        print(f"           saved: {output_path}")

    print(f"Done. Audio folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
