from pathlib import Path
import numpy as np
import librosa
import openl3


def extract_openl3_2hz(audio_path: Path, target_sr: int = 44100) -> np.ndarray:
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    emb, _ = openl3.get_audio_embedding(
        audio,
        sr,
        input_repr="mel256",
        content_type="music",
        embedding_size=512,
        hop_size=0.5,
    )
    return emb.astype(np.float32)


def main():
    ROOT = Path(__file__).resolve().parents[1]
    in_dir = ROOT / "data" / "DEAM" / "MEMD_audio"
    out_dir = ROOT / "artifacts" / "openl3_2hz_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(in_dir.glob("*.mp3"), key=lambda p: int(p.stem))

    saved = 0
    for audio_path in audio_files:
        sid = audio_path.stem
        out_path = out_dir / f"{sid}.npy"
        if out_path.exists():
            continue

        emb = extract_openl3_2hz(audio_path)
        np.save(out_path, emb)
        saved += 1

    print(f"[DONE] Saved {saved} OpenL3 feature files → {out_dir}")


if __name__ == "__main__":
    main()
