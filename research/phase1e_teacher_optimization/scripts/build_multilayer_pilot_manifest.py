from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible MERT multilayer pilot manifest.")
    parser.add_argument("--train-songs", type=int, default=100)
    parser.add_argument("--val-songs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--source",
        type=Path,
        default=MERT_ROOT / "data" / "manifests" / "split_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "multilayer_pilot_manifest.csv",
    )
    args = parser.parse_args()
    manifest = pd.read_csv(args.source)
    train = manifest[manifest["split"] == "train"].sample(
        n=args.train_songs, random_state=args.seed
    )
    validation = manifest[manifest["split"] == "val"].sample(
        n=args.val_songs, random_state=args.seed
    )
    pilot = pd.concat([train, validation]).sort_values(["split", "song_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(args.output, index=False)
    print(
        {
            "output": str(args.output),
            "seed": args.seed,
            "train": len(train),
            "val": len(validation),
            "total": len(pilot),
        }
    )


if __name__ == "__main__":
    main()
