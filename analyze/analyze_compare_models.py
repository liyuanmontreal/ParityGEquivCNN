import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch, cv2, csv
from pathlib import Path
from utils.feature_tools import register_conv_hooks, tensor_to_mean_map, corr2, mirror_np, ensure_sample_image
from utils.plot_utils import save_triptych, save_bar
from models.cnn_baseline import CNNBaseline
from models.parity_invariant import ParityInvariantCNN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
(OUT / "feature_maps/parity_invariant").mkdir(parents=True, exist_ok=True)
(OUT / "feature_maps/baseline").mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
base = CNNBaseline().to(device).eval()
pinv = ParityInvariantCNN().to(device).eval()

outs_b = register_conv_hooks(base)
outs_p = register_conv_hooks(pinv)

img_path = ensure_sample_image(ROOT / "analyze" / "sample_input.png")
import cv2
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
x  = torch.tensor(img[None,None,:,:]/255.0, dtype=torch.float32, device=device)
xm = torch.flip(x, dims=[-1])

_ = base(x);  b_o = {k:v.clone() for k,v in outs_b.items()}
_ = base(xm); b_m = {k:v.clone() for k,v in outs_b.items()}
_ = pinv(x);  p_o = {k:v.clone() for k,v in outs_p.items()}
_ = pinv(xm); p_m = {k:v.clone() for k,v in outs_p.items()}

rows = [("model","layer","corr_raw(orig,mirr)","corr_equiv(orig,mirror_back)")]

def proc(model_tag, o_dict, m_dict, subdir):
    layers, raw_vals, eq_vals = [], [], []
    for name in o_dict.keys():
        A = tensor_to_mean_map(o_dict[name])
        B = tensor_to_mean_map(m_dict[name])
        c_raw = corr2(A, B)
        c_eq  = corr2(A, mirror_np(B))
        rows.append((model_tag, name, c_raw, c_eq))
        layers.append(name); raw_vals.append(c_raw); eq_vals.append(c_eq)
        save_triptych(A, B, str(OUT / f"feature_maps/{subdir}/{name}_compare.png"), titleA=f"{name} orig", titleB=f"{name} mirror")
    return layers, raw_vals, eq_vals

layers_b, raw_b, eq_b = proc("baseline", b_o, b_m, "baseline")
layers_p, raw_p, eq_p = proc("parity_invariant", p_o, p_m, "parity_invariant")

with open(OUT / "results_compare.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(rows[0]); w.writerows(rows[1:])

save_bar(layers_b, raw_b, str(OUT / "baseline_corr_raw.png"), ylabel="corr_raw", title="Baseline: corr(F(x), F(mirror(x)))", ylim=(-1,1))
save_bar(layers_b, eq_b,  str(OUT / "baseline_corr_equiv.png"), ylabel="corr_equiv", title="Baseline: corr(F(x), mirror(F(mirror(x))))", ylim=(-1,1))
save_bar(layers_p, raw_p, str(OUT / "parityinv_corr_raw.png"), ylabel="corr_raw", title="Parity-invariant: raw corr", ylim=(-1,1))
save_bar(layers_p, eq_p,  str(OUT / "parityinv_corr_equiv.png"), ylabel="corr_equiv", title="Parity-invariant: equivariant corr", ylim=(-1,1))

print(f"[DONE] Saved comparison analysis to {OUT}")
