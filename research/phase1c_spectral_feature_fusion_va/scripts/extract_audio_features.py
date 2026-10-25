from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pool_at_targets(
    feature: np.ndarray,
    feature_times: np.ndarray,
    target_times: np.ndarray,
    context_sec: float,
) -> np.ndarray:
    half = context_sec / 2.0
    pooled = []
    for target in target_times:
        mask = (feature_times >= target - half) & (feature_times < target + half)
        if not np.any(mask):
            index = int(np.argmin(np.abs(feature_times - target)))
            values = feature[:, index : index + 1]
        else:
            values = feature[:, mask]
        pooled.append(np.concatenate([values.mean(axis=1), values.std(axis=1)]))
    return np.asarray(pooled, dtype=np.float32)


def extract_song(audio_path: Path, sequence_path: Path, config: dict) -> dict:
    sequence = np.load(sequence_path)
    target_times = sequence["time_sec"].astype(np.float32)
    labels = sequence["Y"].astype(np.float32)
    sample_rate = int(config["sample_rate"])
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    hop_length = int(config["spectrogram_hop_length"])
    n_fft = int(config["n_fft"])

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=int(config["n_mels"]),
        fmin=float(config["fmin"]),
        fmax=float(config["fmax"]),
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32)
    frame_times = librosa.frames_to_time(
        np.arange(log_mel.shape[1]), sr=sample_rate, hop_length=hop_length
    ).astype(np.float32)
    mel_end_sec = float(target_times[-1]) + float(config["label_hop_sec"])
    mel_mask = (frame_times >= float(config["label_start_sec"])) & (
        frame_times < mel_end_sec
    )
    log_mel = log_mel[:, mel_mask]
    mel_times = frame_times[mel_mask]

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    harmonic = librosa.effects.harmonic(audio)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=int(config["n_mfcc"]),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=sample_rate
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=sample_rate
    )
    flatness = librosa.feature.spectral_flatness(S=magnitude)
    contrast = librosa.feature.spectral_contrast(
        S=magnitude, sr=sample_rate
    )
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=n_fft, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_stft(
        S=magnitude, sr=sample_rate, hop_length=hop_length
    )
    tonnetz = librosa.feature.tonnetz(
        y=harmonic, sr=sample_rate
    )
    if tonnetz.shape[1] != magnitude.shape[1]:
        tonnetz = librosa.util.fix_length(tonnetz, size=magnitude.shape[1], axis=1)
    rms = librosa.feature.rms(S=magnitude, frame_length=n_fft)
    onset = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop_length
    )[None, :]
    base = np.concatenate(
        [
            mfcc,
            centroid,
            bandwidth,
            flatness,
            contrast,
            zcr,
            chroma,
            tonnetz,
            rms,
            onset,
        ],
        axis=0,
    ).astype(np.float32)
    feature_times = librosa.frames_to_time(
        np.arange(base.shape[1]), sr=sample_rate, hop_length=hop_length
    )
    handcrafted = pool_at_targets(
        base,
        feature_times,
        target_times,
        float(config["handcrafted_context_sec"]),
    )

    onset_envelope = onset.squeeze(0)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
    )
    bpm_values = librosa.tempo_frequencies(
        tempogram.shape[0], sr=sample_rate, hop_length=hop_length
    )
    tempo_rows = []
    for bpm in config["tempogram_bpms"]:
        index = int(np.argmin(np.abs(bpm_values - float(bpm))))
        tempo_rows.append(tempogram[index])
    tempo_features = pool_at_targets(
        np.asarray(tempo_rows, dtype=np.float32),
        feature_times,
        target_times,
        float(config["handcrafted_context_sec"]),
    )
    handcrafted = np.concatenate([handcrafted, tempo_features], axis=1)

    return {
        "target_time_sec": target_times,
        "labels": labels,
        "log_mel": log_mel.astype(np.float16),
        "mel_time_sec": mel_times,
        "handcrafted": handcrafted.astype(np.float32),
    }


def extract_row(task: tuple[dict, dict, str, bool]) -> str:
    row, config, output_dir_text, force = task
    song_id = int(row["song_id"])
    output = Path(output_dir_text) / f"{song_id}.npz"
    if output.exists() and not force:
        return f"skipped song_id={song_id}"
    features = extract_song(
        Path(row["audio_path"]), Path(row["sequence_path"]), config
    )
    np.savez_compressed(
        output,
        song_id=song_id,
        split=row["split"],
        **features,
    )
    return (
        f"song_id={song_id} split={row['split']} "
        f"targets={len(features['labels'])} mel={features['log_mel'].shape} "
        f"handcrafted={features['handcrafted'].shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract log-Mel and Librosa music features for DEAM."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MERT_ROOT / "data" / "manifests" / "split_manifest.csv",
    )
    parser.add_argument(
        "--sequence-index",
        type=Path,
        default=MERT_ROOT / "data" / "datasets" / "mert_va" / "sequence_index.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "features"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    index = pd.read_csv(args.sequence_index)[["song_id", "sequence_path"]]
    rows = manifest.merge(index, on="song_id").sort_values("song_id")
    if args.limit is not None:
        rows = rows.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (row, config, str(args.output_dir), args.force)
        for row in rows.to_dict("records")
    ]
    if args.workers <= 1:
        results = [extract_row(task) for task in tasks]
    else:
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [executor.submit(extract_row, task) for task in tasks]
            for completed_count, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                result = future.result()
                results.append(result)
                print(f"[{completed_count}/{len(futures)}] {result}", flush=True)
    if args.workers <= 1:
        for result in results:
            print(result)
    completed = sum(not result.startswith("skipped") for result in results)
    print(
        f"completed={completed} requested={len(rows)} workers={args.workers}"
    )


if __name__ == "__main__":
    main()
