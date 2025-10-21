# ParityGEquivCNN 
Learning from Parity: Mirror-Equivariant CNNs with Layer-wise Analysis  

---

##  Overview / 实验简介
- 普通 CNN 只**平移等变**，对**镜像（反演）**不等变 → 在镜像输入下表现不稳。  
- 本仓库包含：**模型训练**（Baseline vs Parity-Invariant）与**层级可视化分析**（镜像敏感性 & 对称性恢复）。

---

##  Structure
```
ParityGEquivCNN/
 ├── models/                    # CNNBaseline / ParityInvariantCNN
 ├── utils/                     # data, training, hooks, plotting
 ├── train/
 │    └── run_experiment.py     # train & export weights to outputs/
 ├── analyze/
 │    ├── analyze_baseline.py   # single-model mirror sensitivity
 │    └── analyze_compare_models.py # baseline vs parity-invariant
 ├── outputs/                   # weights + figures + csv
 ├── docs/figures/              # put cover images / schematics
 ├── requirements.txt
 └── README.md
```

---

##  Quick Start
```bash
pip install -r requirements.txt

# 1) Train models and export weights
python train/run_experiment.py

# 2) Analyze layer-wise mirror response
python analyze/analyze_baseline.py

# 3) Compare baseline vs parity-invariant
python analyze/analyze_compare_models.py
```

Outputs will appear in `outputs/`:
```
outputs/
 ├── baseline.pth, parity_invariant.pth
 ├── results.txt, results_baseline.csv, results_compare.csv
 ├── feature_maps/
 ├── baseline_corr.png, *_equiv.png
```

---

##  Method — Parity-Invariant Block
```
phi(x) = 0.5 * [ Conv(x) + mirror_back(Conv(mirror(x))) ]
```
Enforces **Z₂** mirror equivariance **layer-wise** (structure, not just augmentation).

---

##  Expected Results (examples)
```
Baseline:         acc≈0.92, acc_mirror≈0.60
Parity-Invariant: acc≈0.91, acc_mirror≈0.90
```
And in analysis:
- **Baseline**: low `corr_raw` & low `corr_equiv` across layers.  
- **Parity-Invariant**: `corr_equiv` notably higher than `corr_raw`.

---

##  Figures (White Minimal)
Put images under `docs/figures/`, e.g.:
- `mirror_pipeline.png` — pipeline schematic
- `feature_diff_heatmap.png` — layer heatmap example
- `symmetry_recovery_chart.png` — correlation comparison

---

##  References papers
- Cohen & Welling (2016) — Group Equivariant CNNs  
- Weiler & Cesa (2019) — General E(2)-Steerable CNNs

## License
MIT
