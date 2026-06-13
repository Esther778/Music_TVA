import os
import glob
import numpy as np
import librosa
import openl3
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MYSONGS_DIR = os.path.join(ROOT_DIR, "data", "my_songs")
MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "models", "mlp", "mlp_best.pt")
NORM_PATH = os.path.join(ROOT_DIR, "outputs", "models", "mlp", "mlp_input_norm.npz")
PRED_DIR = os.path.join(ROOT_DIR, "outputs", "predictions")

TARGET_SR = 44100
INPUT_REPR = "mel256"
CONTENT_TYPE = "music"
EMBEDDING_SIZE = 512
HOP_SIZE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_dim=512, hidden1=256, hidden2=128, output_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_trained_mlp():
    model = MLP()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_input_norm():
    d = np.load(NORM_PATH)
    return d["mean"], d["std"]


def extract_openl3(audio_path, mean, std):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    emb, ts = openl3.get_audio_embedding(
        audio,
        TARGET_SR,
        input_repr=INPUT_REPR,
        content_type=CONTENT_TYPE,
        embedding_size=EMBEDDING_SIZE,
        hop_size=HOP_SIZE
    )

    emb_norm = (emb - mean) / (std + 1e-8)
    return emb_norm, ts


def predict_va(model, emb_norm):
    with torch.no_grad():
        x = torch.from_numpy(emb_norm).float().to(DEVICE)
        preds = model(x).cpu().numpy()
    return preds[:, 0], preds[:, 1]


def save_csv(time, valence, arousal, audio_path):
    os.makedirs(PRED_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(audio_path))[0]
    path = os.path.join(PRED_DIR, f"{name}_va.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "valence", "arousal"])
        for t, v, a in zip(time, valence, arousal):
            writer.writerow([float(t), float(v), float(a)])


def save_plot(time, valence, arousal, audio_path):
    os.makedirs(PRED_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(audio_path))[0]
    path = os.path.join(PRED_DIR, f"{name}_va_curve.png")

    plt.figure(figsize=(10, 4))
    plt.plot(time, valence, label="Valence")
    plt.plot(time, arousal, label="Arousal")
    plt.xlabel("Time (s)")
    plt.ylabel("Predicted VA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    model = load_trained_mlp()
    mean, std = load_input_norm()

    files = sorted(
        glob.glob(os.path.join(MYSONGS_DIR, "*.mp3")) +
        glob.glob(os.path.join(MYSONGS_DIR, "*.wav"))
    )

    for audio_path in files:
        emb_norm, ts = extract_openl3(audio_path, mean, std)
        valence, arousal = predict_va(model, emb_norm)
        save_csv(ts, valence, arousal, audio_path)
        save_plot(ts, valence, arousal, audio_path)


if __name__ == "__main__":
    main()
