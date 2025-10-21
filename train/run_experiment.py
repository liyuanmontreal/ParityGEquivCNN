import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os
from utils.data_tools import SynthShapes
from models.cnn_baseline import CNNBaseline
from models.parity_invariant import ParityInvariantCNN
from utils.train_utils import split_dataset, train_one, eval_acc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def make_loaders(n=4000, mirror_prob=0.5, batch_size=64, seed=0):
    ds = SynthShapes(n=n, mirror_prob=mirror_prob, seed=seed)
    tr, va = split_dataset(ds, 0.8, seed=seed)
    return DataLoader(tr, batch_size=batch_size, shuffle=True), DataLoader(va, batch_size=batch_size)

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
    return correct/total if total>0 else 0.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = make_loaders(n=4000, mirror_prob=0.5, batch_size=64, seed=0)

    # Baseline
    base = CNNBaseline()
    base = train_one(base, tr, va, epochs=6, lr=1e-3, device=device)
    acc_base = eval_acc(base, va, device)
    acc_base_m = eval_mirrored(base, va, device)
    torch.save(base.state_dict(), OUT / "baseline.pth")

    # Parity-invariant
    pinv = ParityInvariantCNN()
    pinv = train_one(pinv, tr, va, epochs=6, lr=1e-3, device=device)
    acc_inv = eval_acc(pinv, va, device)
    acc_inv_m = eval_mirrored(pinv, va, device)
    torch.save(pinv.state_dict(), OUT / "parity_invariant.pth")

    with open(OUT / "results.txt", "w") as f:
        f.write(f"Baseline: acc={acc_base:.3f}, acc_mirror={acc_base_m:.3f}\n")
        f.write(f"ParityInvariant: acc={acc_inv:.3f}, acc_mirror={acc_inv_m:.3f}\n")
    print(f"[DONE] Results -> {OUT/'results.txt'}")

if __name__ == "__main__":
    main()
