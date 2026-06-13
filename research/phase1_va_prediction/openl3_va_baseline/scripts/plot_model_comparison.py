from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_history(path: Path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    ROOT = Path(__file__).resolve().parents[1]

    model_files = [
        ("MLP", ROOT / "outputs" / "results" / "mlp" / "mlp_history.npz"),
        ("BiLSTM", ROOT / "outputs" / "results" / "bilstm" / "bilstm_history.npz"),
        ("Transformer", ROOT / "outputs" / "results" / "transformer" / "transformer_history.npz"),
    ]

    out_dir = ROOT / "outputs" / "plots" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    min_len = None

    for name, path in model_files:
        if not path.exists():
            print(f"[WARN] {name} history file not found: {path}")
            continue

        hist = load_history(path)
        histories[name] = hist

        L = len(hist["val_loss"])
        min_len = L if min_len is None else min(min_len, L)
        print(f"[INFO] {name} epochs: {L}")

    if not histories:
        print("[ERROR] No history files loaded.")
        return

    epochs = np.arange(1, min_len + 1)

    # 1) val_loss
    plt.figure()
    for name, hist in histories.items():
        plt.plot(epochs, hist["val_loss"][:min_len], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "compare_val_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) val_rmse
    plt.figure()
    for name, hist in histories.items():
        plt.plot(epochs, hist["val_rmse"][:min_len], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "compare_val_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3) val_pearson_v
    plt.figure()
    for name, hist in histories.items():
        plt.plot(epochs, hist["val_pearson_v"][:min_len], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Pearson r (Valence)")
    plt.title("Validation Pearson(V) Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "compare_pearson_v.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4) val_pearson_a
    plt.figure()
    for name, hist in histories.items():
        plt.plot(epochs, hist["val_pearson_a"][:min_len], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Pearson r (Arousal)")
    plt.title("Validation Pearson(A) Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(out_dir / "compare_pearson_a.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Saved model comparison plots → {out_dir}")


if __name__ == "__main__":
    main()
