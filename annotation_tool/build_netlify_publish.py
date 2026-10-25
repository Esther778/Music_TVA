#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "annotation_tool"
PUBLISH_DIR = ROOT / "netlify_publish"

FILES = [
    "index.html",
    "styles.css",
    "app.js",
    "README.md",
    "WORKFLOW_STATUS.md",
]

SONGS_DIR = ROOT / "songs"


def main() -> None:
    if PUBLISH_DIR.exists():
        shutil.rmtree(PUBLISH_DIR)
    PUBLISH_DIR.mkdir()

    for name in FILES:
        shutil.copy2(TOOL_DIR / name, PUBLISH_DIR / name)

    songs_dir = PUBLISH_DIR / "songs"
    songs_dir.mkdir()
    for audio in sorted(SONGS_DIR.glob("*.mp3")):
        shutil.copy2(audio, songs_dir / audio.name)

    (PUBLISH_DIR / "_headers").write_text(
        "/songs/*\n"
        "  Cache-Control: public, max-age=31536000, immutable\n"
        "\n"
        "/*\n"
        "  Cache-Control: public, max-age=0, must-revalidate\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
