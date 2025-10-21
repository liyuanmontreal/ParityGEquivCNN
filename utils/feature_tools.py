import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

def register_conv_hooks(model: nn.Module):
    outs = {}
    def hook_fn(name):
        def hook(m, inp, outp):
            outs[name] = outp.detach().cpu()
        return hook
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook_fn(name))
    return outs

def tensor_to_mean_map(t: torch.Tensor) -> np.ndarray:
    a = t[0].numpy()
    if a.ndim == 3:
        return a.mean(0)
    return a

def corr2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten(); b = b.flatten()
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0,1])

def mirror_np(arr: np.ndarray) -> np.ndarray:
    import numpy as np
    return np.flip(arr, axis=1)

def ensure_sample_image(sample_path: Path):
    import cv2, numpy as np
    if not sample_path.exists():
        img = np.zeros((32,32), np.uint8)
        cv2.line(img, (8,8), (8,24), 255, 2)
        cv2.line(img, (8,24), (20,24), 255, 2)
        cv2.imwrite(str(sample_path), img)
    return str(sample_path)
