#!/usr/bin/env python3
"""Guard against accidental PMEmo final-locked-test leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "pmemo_final_locked_song_level_split.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="CSV index files to check.")
    parser.add_argument("--locked-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--allow-final", action="store_true", help="Allow final_locked_test ids; use only for final evaluation scripts.")
    args = parser.parse_args()

    split = pd.read_csv(args.locked_split)
    final_ids = set(split.loc[split["locked_split"].eq("final_locked_test"), "musicId"].astype(int))
    report = {
        "locked_split": str(args.locked_split),
        "final_locked_test_count": len(final_ids),
        "allow_final": bool(args.allow_final),
        "checked": [],
        "violations": [],
    }
    for path in args.paths:
        df = pd.read_csv(path)
        if "musicId" not in df.columns:
            raise SystemExit(f"{path} has no musicId column; cannot guard final-test leakage.")
        ids = set(df["musicId"].astype(int))
        overlap = sorted(ids.intersection(final_ids))
        item = {"path": str(path), "rows": int(len(df)), "unique_music_ids": int(len(ids)), "final_overlap_count": len(overlap)}
        report["checked"].append(item)
        if overlap and not args.allow_final:
            report["violations"].append({"path": str(path), "overlap_count": len(overlap), "overlap_ids": overlap[:50]})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["violations"] and not args.allow_final:
        raise SystemExit("FINAL_LOCKED_TEST_LEAKAGE_DETECTED")


if __name__ == "__main__":
    main()
