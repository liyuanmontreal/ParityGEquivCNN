# ================================================================
# experiment_compare_symmetry.py
# ---------------------------------------------------------------
# Compare Baseline CNN vs Parity-Invariant CNN
# under different mirror probabilities (0.0, 0.5, 1.0)
# Author: Yuan Li · Mila / Université de Montréal
# ================================================================

import os
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']        # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False          # 解决负号乱码
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader

from utils.data_tools import SynthShapes
from utils.train_utils import split_dataset, train_one, eval_acc
from models.cnn_baseline import CNNBaseline
from models.parity_invariant import ParityInvariantCNN

# ================================================================
# Helper functions
# ================================================================

@torch.no_grad()
def eval_mirrored(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    for x, y in loader:
        xm = torch.flip(x, dims=[-1]).to(device)
        y = y.to(device)
        logits = model(xm)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def make_loaders(n=4000, mirror_prob=0.5, batch_size=64, seed=0):
    ds = SynthShapes(n=n, mirror_prob=mirror_prob, seed=seed)
    tr, va = split_dataset(ds, 0.8, seed=seed)
    return DataLoader(tr, batch_size=batch_size, shuffle=True), DataLoader(va, batch_size=batch_size)


# ================================================================
# Main experiment
# ================================================================

def run_experiment(mirror_prob, device="cpu"):
    print(f"\n=== Running experiment: mirror_prob={mirror_prob} ===")

    tr, va = make_loaders(n=4000, mirror_prob=mirror_prob, batch_size=64, seed=42)

    results = []

    # ---------- Baseline CNN ----------
    base = CNNBaseline()
    base = train_one(base, tr, va, epochs=6, lr=1e-3, device=device)
    acc_base = eval_acc(base, va, device)
    acc_base_m = eval_mirrored(base, va, device)
    results.append({
        "mirror_prob": mirror_prob,
        "model": "baseline",
        "acc_original": acc_base,
        "acc_mirror": acc_base_m
    })

    # ---------- Parity-Invariant CNN ----------
    pinv = ParityInvariantCNN()
    pinv = train_one(pinv, tr, va, epochs=6, lr=1e-3, device=device)
    acc_inv = eval_acc(pinv, va, device)
    acc_inv_m = eval_mirrored(pinv, va, device)
    results.append({
        "mirror_prob": mirror_prob,
        "model": "parity_invariant",
        "acc_original": acc_inv,
        "acc_mirror": acc_inv_m
    })

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)

    mirror_probs = [0.0, 0.5, 1.0]
    all_results = []

    for p in mirror_probs:
        results = run_experiment(p, device)
        all_results.extend(results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(out_dir, f"symmetry_comparison_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    # ============================================================
    # Plot Results
    # ============================================================
    plt.figure(figsize=(8,5))
    for model, style, color in [
        ("baseline", "-", "tab:blue"),
        ("parity_invariant", "-", "tab:red")
    ]:
        subset = df[df["model"] == model]
        plt.plot(subset["mirror_prob"], subset["acc_original"],
                 linestyle=style, color=color, label=f"{model} (original)")
        plt.plot(subset["mirror_prob"], subset["acc_mirror"],
                 linestyle="--", color=color, alpha=0.7, label=f"{model} (mirror)")

    plt.xlabel("Mirror Probability / 镜像概率", fontsize=11)
    plt.ylabel("Accuracy / 准确率", fontsize=11)
    plt.title("Baseline CNN vs Parity-Invariant CNN — Mirror Robustness Comparison\n基线CNN与镜像不变CNN镜像鲁棒性比较",
              fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"symmetry_comparison_{timestamp}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"📊 Figure saved to {png_path}")


if __name__ == "__main__":
    main()
