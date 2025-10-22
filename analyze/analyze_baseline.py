import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch, cv2, csv
from pathlib import Path
from utils.feature_tools import register_conv_hooks, tensor_to_mean_map, corr2, ensure_sample_image
from utils.plot_utils import save_triptych, save_bar
from models.cnn_baseline import CNNBaseline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
(OUT / "feature_maps/baseline").mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNBaseline().to(device).eval()
outs = register_conv_hooks(model)

img_path = ensure_sample_image(ROOT / "analyze" / "sample_input.png")
import cv2
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
x  = torch.tensor(img[None,None,:,:]/255.0, dtype=torch.float32, device=device)
xm = torch.flip(x, dims=[-1])

_ = model(x);  orig = {k:v.clone() for k,v in outs.items()}
_ = model(xm); mirr = {k:v.clone() for k,v in outs.items()}

rows, layers, vals = [("layer","corr(original,mirror)")], [], []
for name in orig.keys():
    A = tensor_to_mean_map(orig[name])
    B = tensor_to_mean_map(mirr[name])
    c = corr2(A,B)
    rows.append((name, c))
    layers.append(name); vals.append(c)
    save_triptych(A, B, str(OUT / f"feature_maps/baseline/{name}_compare.png"), titleA=f"{name} orig", titleB=f"{name} mirror")

with open(OUT / "results_baseline.csv","w",newline="") as f:
    w = csv.writer(f); w.writerows(rows)

save_bar(layers, vals, str(OUT / "baseline_corr.png"), ylabel="corr(orig, mirror)", title="Baseline CNN: layer-wise correlation", ylim=(-1,1))
print(f"[DONE] Saved baseline analysis to {OUT}")
