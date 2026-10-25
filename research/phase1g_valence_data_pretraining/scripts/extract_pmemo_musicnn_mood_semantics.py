#!/usr/bin/env python3
"""Extract fixed MusiCNN happy/sad song features for the locked PMEmo dev pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import essentia.standard as es


PROJECT = Path(__file__).resolve().parents[3]
PMEMO = PROJECT / "research/phase1f_external_va_datasets/pmemo"
DEFAULT_INDEX = PMEMO / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv"
DEFAULT_AUDIO = PMEMO / "data/raw/PMEmo_2019_updated/extracted/PMEmo2019/chorus"
DEFAULT_LABEL = PMEMO / "outputs/opensmile_baseline/dataset/sequence"
DEFAULT_MODELS = PROJECT / "research/phase1g_valence_data_pretraining/models/essentia_emomusic_musicnn"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/pmemo_musicnn_mood_semantics_20260714"


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(np.float64); target = target.astype(np.float64)
    err = pred - target
    pcc = float(np.corrcoef(pred, target)[0, 1])
    pm, tm = pred.mean(), target.mean()
    cov = float(np.mean((pred - pm) * (target - tm)))
    ccc = float(2 * cov / (pred.var() + target.var() + (pm - tm) ** 2 + 1e-12))
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))), "pcc": pcc, "ccc": ccc}


def pair_accuracy(pred: np.ndarray, target: np.ndarray, margin: float = 0.05) -> dict[str, float | int]:
    left, right = np.triu_indices(len(target), k=1)
    delta = target[left] - target[right]
    keep = np.abs(delta) >= margin
    correct = np.sign(pred[left][keep] - pred[right][keep]) == np.sign(delta[keep])
    return {"pair_margin": margin, "pair_count": int(keep.sum()), "pair_accuracy": float(correct.mean())}


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()), f"{prefix}_max": float(values.max()),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, default=0, help="Maximum new songs to process; zero means all.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.out_dir / "song_features.csv"
    wanted_splits = set(args.splits)
    if not wanted_splits <= {"train", "val"}:
        raise ValueError("Only locked-development train/val splits are allowed")
    index = [row for row in read_csv(args.index) if row["split"] in wanted_splits]
    existing = read_csv(output_csv)
    done = {int(row["musicId"]) for row in existing}

    embedder = es.TensorflowPredictMusiCNN(
        graphFilename=str(args.model_dir / "msd-musicnn-1.pb"),
        output="model/dense/BiasAdd", batchSize=16,
    )
    happy_head = es.TensorflowPredict2D(
        graphFilename=str(args.model_dir / "mood_happy-msd-musicnn-1.pb"), output="model/Softmax",
    )
    sad_head = es.TensorflowPredict2D(
        graphFilename=str(args.model_dir / "mood_sad-msd-musicnn-1.pb"), output="model/Softmax",
    )
    rows: list[dict[str, object]] = list(existing)
    failures = []
    pending = [row for row in index if int(row["musicId"]) not in done]
    if args.limit > 0:
        pending = pending[: args.limit]
    for number, row in enumerate(pending, start=1):
        music_id = int(row["musicId"])
        try:
            audio = es.MonoLoader(filename=str(args.audio_dir / f"{music_id}.mp3"), sampleRate=16000)()
            embedding = embedder(audio)
            # Essentia mood heads use [mood, not_mood]; index 0 is positive.
            happy = happy_head(embedding)[:, 0].astype(np.float64)
            sad = sad_head(embedding)[:, 0].astype(np.float64)
            target = float(np.load(args.label_dir / f"{music_id}.npz")["Y"][:, 0].mean())
            item = {"musicId": music_id, "split": row["split"], "target_song_valence": target, "patches": len(happy)}
            item.update(describe(happy, "happy")); item.update(describe(sad, "sad"))
            item["happy_minus_sad"] = item["happy_mean"] - item["sad_mean"]
            item["happy_ratio"] = item["happy_mean"] / (item["happy_mean"] + item["sad_mean"] + 1e-8)
            rows.append(item)
        except Exception as exc:
            failures.append({"musicId": music_id, "error": repr(exc)})
        if number % 20 == 0 or number == len(pending):
            rows.sort(key=lambda item: int(item["musicId"]))
            write_csv(output_csv, rows)
            print(f"processed={number}/{len(pending)} total_ok={len(rows)} failures={len(failures)}", flush=True)

    val = [row for row in rows if row["split"] == "val"]
    if len(val) < 3:
        print(json.dumps({"status": "insufficient_val_rows", "rows": len(rows), "val_rows": len(val)}, indent=2))
        return
    candidates = {
        "happy_mean": np.asarray([float(row["happy_mean"]) for row in val]),
        "negative_sad_mean": -np.asarray([float(row["sad_mean"]) for row in val]),
        "happy_minus_sad": np.asarray([float(row["happy_minus_sad"]) for row in val]),
        "happy_ratio": np.asarray([float(row["happy_ratio"]) for row in val]),
    }
    target = np.asarray([float(row["target_song_valence"]) for row in val])
    results = []
    for name, pred in candidates.items():
        results.append({"candidate": name, **metrics(pred, target), **pair_accuracy(pred, target)})
    results.sort(key=lambda item: (float(item["pcc"]), float(item["pair_accuracy"])), reverse=True)
    write_csv(args.out_dir / "raw_semantic_gate.csv", results)
    selected = results[0]
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "PMEmo fold_03 locked development train/val only; final locked test excluded",
        "songs": {
            "all": len(rows),
            "train": sum(row["split"] == "train" for row in rows),
            "val": sum(row["split"] == "val" for row in rows),
        },
        "failures": failures, "selected_raw_semantic": selected,
        "decision_rule": "validation PCC >= 0.30 and pair accuracy >= 0.60",
        "decision": "promising_for_crossfold" if selected["pcc"] >= 0.30 and selected["pair_accuracy"] >= 0.60 else "reject_musicnn_mood_semantics",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
