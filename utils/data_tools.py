import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

# Synthetic shapes dataset: triangle / L / arrow, with random mirroring.
# Goal: test mirror (parity) robustness.

def draw_triangle(img, rng):
    h, w = img.shape
    pts = np.array([
        (rng.integers(6, w-6), rng.integers(6, h-6)),
        (rng.integers(6, w-6), rng.integers(6, h-6)),
        (rng.integers(6, w-6), rng.integers(6, h-6))
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, 255, thickness=2)

def draw_L(img, rng):
    h, w = img.shape
    x = rng.integers(6, w-10)
    y = rng.integers(6, h-10)
    cv2.line(img, (x, y), (x, y+12), 255, 2)
    cv2.line(img, (x, y+12), (x+10, y+12), 255, 2)

def draw_arrow(img, rng):
    h, w = img.shape
    x1 = rng.integers(6, w-6)
    x2 = rng.integers(6, w-6)
    y = rng.integers(8, h-8)
    cv2.line(img, (x1, y), (x2, y), 255, 2)
    xh = (x1 + x2)//2
    cv2.line(img, (xh, y), (xh-4, y-4), 255, 2)
    cv2.line(img, (xh, y), (xh-4, y+4), 255, 2)

DRAW_FUNCS = [draw_triangle, draw_L, draw_arrow]

class SynthShapes(Dataset):
    def __init__(self, n=4000, mirror_prob=0.5, seed=0):
        self.n = n
        self.mirror_prob = mirror_prob
        self.rng = np.random.default_rng(seed)
        self.data = []
        self.labels = []
        for i in range(n):
            cls = int(self.rng.integers(0, len(DRAW_FUNCS)))
            img = np.zeros((32, 32), np.uint8)
            DRAW_FUNCS[cls](img, self.rng)
            if self.rng.random() < 0.5:
                k = int(self.rng.integers(1, 3)) * 2 + 1
                img = cv2.GaussianBlur(img, (k, k), 0)
            if self.rng.random() < mirror_prob:
                img = cv2.flip(img, 1)
            img = img.astype(np.float32) / 255.0
            self.data.append(img[None, ...])
            self.labels.append(cls)
        self.data = np.stack(self.data, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = int(self.labels[idx])
        return x, y
