#!/usr/bin/env python3
"""Read-only rigor audit for the frozen PMEmo final VA teacher result."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "final_clean_teacher/outputs/final_clean_teacher_v2"
SPLIT = ROOT / "data/splits/pmemo_final_locked_20260713/pmemo_final_locked_song_level_split.csv"
MANIFEST = ROOT / "data/processed/pmemo2019_song_manifest_normalized.csv"
PREDICTION = RUN / "final_locked_predictions_LABEL_FREE.csv"
SAVED_RESULT = RUN / "FINAL_LOCKED_TEST_RESULTS.json"
OUTPUT = RUN / "FINAL_TEST_RIGOR_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    covariance = float(np.mean((prediction - prediction.mean()) * (target - target.mean())))
    pcc = float(np.corrcoef(prediction, target)[0, 1]) if prediction.std() > 1e-12 and target.std() > 1e-12 else 0.0
    ccc = float(
        2.0 * covariance
        / (prediction.var() + target.var() + (prediction.mean() - target.mean()) ** 2 + 1e-12)
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "pcc": pcc,
        "ccc": ccc,
    }


def evaluate(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    clip = frame.groupby("musicId", sort=False)[
        ["pred_valence", "target_valence", "pred_arousal", "target_arousal"]
    ].mean()
    return {
        "frame": {
            dimension: metrics(frame[f"pred_{dimension}"], frame[f"target_{dimension}"])
            for dimension in ("valence", "arousal")
        },
        "complete_clip": {
            dimension: metrics(clip[f"pred_{dimension}"], clip[f"target_{dimension}"])
            for dimension in ("valence", "arousal")
        },
    }


def metric_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in left}


def main() -> None:
    split = pd.read_csv(SPLIT)
    metadata = pd.read_csv(MANIFEST)[
        ["musicId", "title", "artist", "audio_path", "frame_count", "label_time_min", "label_time_max"]
    ]
    data = split.merge(metadata, on="musicId", validate="one_to_one")
    development_ids = set(data.loc[data["locked_split"].eq("development_pool"), "musicId"].astype(int))
    final_ids = set(data.loc[data["locked_split"].eq("final_locked_test"), "musicId"].astype(int))

    data["audio_sha256"] = [sha256(Path(path)) for path in data["audio_path"]]
    data["metadata_key"] = [
        f"{normalize_text(title)}|{normalize_text(artist)}"
        for title, artist in zip(data["title"], data["artist"])
    ]
    exact_duplicate_groups = []
    cross_split_test_ids = []
    for digest, group in data.groupby("audio_sha256", sort=False):
        if len(group) <= 1:
            continue
        rows = group[["musicId", "locked_split", "title", "artist"]].to_dict("records")
        cross_split = group["locked_split"].nunique() > 1
        exact_duplicate_groups.append({"sha256": digest, "cross_split": cross_split, "rows": rows})
        if cross_split:
            cross_split_test_ids.extend(
                group.loc[group["locked_split"].eq("final_locked_test"), "musicId"].astype(int).tolist()
            )
    metadata_duplicate_groups = []
    for key, group in data.groupby("metadata_key", sort=False):
        if len(group) > 1 and group["locked_split"].nunique() > 1:
            metadata_duplicate_groups.append({
                "metadata_key": key,
                "rows": group[["musicId", "locked_split", "title", "artist"]].to_dict("records"),
            })

    label_rows = []
    label_ranges = {}
    time_audit = {}
    for split_name, subset in data.groupby("locked_split", sort=False):
        values = []
        starts, ends, hops, frames = [], [], [], []
        for item in subset.itertuples(index=False):
            sequence = np.load(Path(item.path))
            target = sequence["Y"].astype(np.float64)
            time_sec = sequence["time_sec"].astype(np.float64)
            values.append(target)
            starts.append(float(time_sec[0]))
            ends.append(float(time_sec[-1]))
            frames.append(len(time_sec))
            if len(time_sec) > 1:
                hops.extend(np.diff(time_sec).tolist())
            if split_name == "final_locked_test":
                label_rows.append(pd.DataFrame({
                    "musicId": int(item.musicId),
                    "time_sec": time_sec,
                    "target_valence": target[:, 0],
                    "target_arousal": target[:, 1],
                }))
        merged = np.concatenate(values)
        label_ranges[split_name] = {
            "songs": len(subset),
            "frames": int(sum(frames)),
            "valence_min": float(merged[:, 0].min()),
            "valence_max": float(merged[:, 0].max()),
            "arousal_min": float(merged[:, 1].min()),
            "arousal_max": float(merged[:, 1].max()),
        }
        time_audit[split_name] = {
            "start_min": float(min(starts)),
            "start_max": float(max(starts)),
            "end_min": float(min(ends)),
            "end_max": float(max(ends)),
            "hop_unique": sorted(set(round(value, 8) for value in hops)),
            "frames_min": int(min(frames)),
            "frames_median": float(np.median(frames)),
            "frames_max": int(max(frames)),
            "labelled_span_sec_median": float(np.median([(count - 1) * 0.5 for count in frames])),
        }

    prediction = pd.read_csv(PREDICTION)
    target = pd.concat(label_rows, ignore_index=True)
    frame = prediction.merge(target, on=["musicId", "time_sec"], validate="one_to_one")
    independent = evaluate(frame)
    dedup = frame.loc[~frame["musicId"].isin(cross_split_test_ids)].copy()
    dedup_result = evaluate(dedup)

    saved = json.loads(SAVED_RESULT.read_text(encoding="utf-8"))
    saved_reference = {
        "frame": saved["levels"]["frame_0.5s"],
        "complete_clip": saved["levels"]["complete_clip"],
    }
    recompute_deltas = {
        level: {
            dimension: metric_delta(independent[level][dimension], saved_reference[level][dimension])
            for dimension in ("valence", "arousal")
        }
        for level in ("frame", "complete_clip")
    }

    dev = data.loc[data["locked_split"].eq("development_pool")]
    dev_targets = []
    for item in dev.itertuples(index=False):
        dev_targets.append(np.load(Path(item.path))["Y"].astype(np.float64))
    dev_frame = np.concatenate(dev_targets)
    dev_clip_mean = dev[["mean_valence", "mean_arousal"]].to_numpy(dtype=np.float64).mean(axis=0)
    dev_frame_mean = dev_frame.mean(axis=0)
    test_clip = frame.groupby("musicId", sort=False)[["target_valence", "target_arousal"]].mean()
    mean_baseline = {
        "frame": {
            "valence": metrics(np.full(len(frame), dev_frame_mean[0]), frame["target_valence"]),
            "arousal": metrics(np.full(len(frame), dev_frame_mean[1]), frame["target_arousal"]),
        },
        "complete_clip": {
            "valence": metrics(np.full(len(test_clip), dev_clip_mean[0]), test_clip["target_valence"]),
            "arousal": metrics(np.full(len(test_clip), dev_clip_mean[1]), test_clip["target_arousal"]),
        },
    }

    result = {
        "audit_date": "2026-07-15",
        "scope": "read-only audit; no training, calibration, model selection, or prediction change",
        "id_disjoint": {
            "development_songs": len(development_ids),
            "final_songs": len(final_ids),
            "overlap_count": len(development_ids.intersection(final_ids)),
        },
        "split_construction": {
            "used_mean_va_for_stratification": True,
            "used_frame_count_for_stratification": True,
            "interpretation": "pre-lock stratification, not model fitting; disclose that labels were algorithmically accessed",
        },
        "content_duplicates": {
            "exact_audio_duplicate_group_count": len(exact_duplicate_groups),
            "cross_split_exact_audio_group_count": sum(row["cross_split"] for row in exact_duplicate_groups),
            "cross_split_final_music_ids": sorted(set(cross_split_test_ids)),
            "exact_audio_groups": exact_duplicate_groups,
            "cross_split_normalized_title_artist_groups": metadata_duplicate_groups,
        },
        "label_ranges": label_ranges,
        "time_axis": time_audit,
        "frozen_artifacts": {
            "prediction_sha256": sha256(PREDICTION),
            "valence_checkpoint_sha256": sha256(RUN / "final_fit/valence/model.pt"),
            "arousal_checkpoint_sha256": sha256(RUN / "final_fit/arousal/model.pt"),
            "config_sha256": sha256(ROOT / "final_clean_teacher/configs/final_clean_teacher_v1.json"),
        },
        "independent_recompute_115": independent,
        "saved_result_minus_recompute": recompute_deltas,
        "deduplicated_sensitivity_112": {
            "excluded_final_music_ids": sorted(set(cross_split_test_ids)),
            "songs": dedup["musicId"].nunique(),
            "frames": len(dedup),
            "metrics": dedup_result,
            "interpretation": "post-hoc sensitivity analysis, not a new blind test",
        },
        "training_mean_baseline_115": mean_baseline,
        "conclusions": {
            "final_id_split_disjoint": True,
            "final_audio_content_disjoint": False,
            "saved_metrics_reproduced": all(
                abs(value) < 1e-12
                for levels in recompute_deltas.values()
                for dimensions in levels.values()
                for value in dimensions.values()
            ),
            "post_test_tuning_performed": False,
            "full_length_song_generalization_proven": False,
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
