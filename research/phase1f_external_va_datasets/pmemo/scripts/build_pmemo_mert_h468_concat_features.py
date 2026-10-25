#!/usr/bin/env python3
"""Build concatenated PMEmo MERT hidden4/6/8 feature files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train_pmemo_mert_valence_baseline import ROOT


HIDDEN_DIRS = [
    ROOT / "outputs" / "mert_pmemo_hidden4" / "features",
    ROOT / "outputs" / "mert_pmemo_hidden6" / "features",
    ROOT / "outputs" / "mert_pmemo_hidden8" / "features",
]
OUT_DIR = ROOT / "outputs" / "mert_pmemo_h468_concat" / "features"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted({p.stem for p in HIDDEN_DIRS[0].glob("*.npz")})
    written = 0
    skipped = []
    for sid in ids:
        paths = [d / f"{sid}.npz" for d in HIDDEN_DIRS]
        if not all(p.exists() for p in paths):
            skipped.append(sid)
            continue
        arrays = [np.load(p) for p in paths]
        embeddings = [a["embedding"].astype(np.float32) for a in arrays]
        n = min(x.shape[0] for x in embeddings)
        time_sec = arrays[0]["time_sec"].astype(np.float32)[:n]
        concat = np.concatenate([x[:n] for x in embeddings], axis=1).astype(np.float32)
        np.savez_compressed(OUT_DIR / f"{sid}.npz", embedding=concat, time_sec=time_sec)
        written += 1
    summary = {
        "created_at": "2026-07-09",
        "feature_type": "MERT hidden4 + hidden6 + hidden8 concatenated",
        "input_dirs": [str(d.relative_to(ROOT)) for d in HIDDEN_DIRS],
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "written": written,
        "skipped": skipped,
        "feature_dim": 2304,
    }
    (OUT_DIR.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
