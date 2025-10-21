import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def split_dataset(ds, train_ratio=0.8, seed=0):
    n = len(ds)
    n_train = int(n*train_ratio)
    n_val = n - n_train
    return random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

def train_one(model, train_loader, val_loader, epochs=6, lr=1e-3, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = {"val_acc": 0.0, "state": None}
    for ep in range(1, epochs+1):
        model.train()
        correct = total = 0
        for x, y in tqdm(train_loader, desc=f"Train ep{ep}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_acc = correct/total
        va = eval_acc(model, val_loader, device)
        if va > best["val_acc"]:
            best = {"val_acc": va, "state": {k: v.cpu() for k, v in model.state_dict().items()}}
        print(f"Epoch {ep}: train_acc={train_acc:.3f} val_acc={va:.3f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model

@torch.no_grad()
def eval_acc(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct/total if total>0 else 0.0
