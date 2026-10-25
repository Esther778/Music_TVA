#!/usr/bin/env python3
"""Train the frozen Valence base model on train/val and evaluate blind holdout once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, resolve_device
from train_pmemo_valence_multisplit_best import BestConfig, summarize, train_one_split


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_blind_20260709" / "pmemo_blind_trainvaltest_index.csv"
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "valence_blind_holdout_base_20260709"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    counts = index["split"].value_counts().to_dict()
    if set(counts) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test split labels, got {counts}")

    device = resolve_device(args.device)
    config = BestConfig()
    row = train_one_split(1, args.seed, index, args, device, config)
    summary = summarize([row], args.out_dir)
    summary["blind_holdout_protocol"] = {
        "split_file": str(args.index),
        "blind_label_in_original_split": "blind_test",
        "train_songs": int(counts["train"]),
        "val_songs": int(counts["val"]),
        "blind_test_songs": int(counts["test"]),
        "rule": "test split is the sealed blind_test subset; it is evaluated after checkpoint selection by validation only",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
