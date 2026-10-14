# PALM — Pathology-Aware Lightweight Mustard-disease framework

Imbalance-robust, field-deployable, **explainability-validated** deep learning
for mustard (*Brassica juncea*) leaf disease classification.

Three classes: **Healthy**, **Alternaria Blight** (*Alternaria brassicae*),
**White Rust** (*Albugo candida*) — 5,783 DSLR field images curated at
CCS HAU Bawal (Dec 2025 – Feb 2026).

> This repo accompanies the paper *"Beyond Accuracy: An Imbalance-Robust,
> Lesion-Attentive Framework with Quantitative Explainability for Field-Condition
> Mustard Leaf Disease Diagnosis."* It contains everything needed to reproduce
> the results; the images are supplied separately (see [`data/README.md`](data/README.md)).

---

## Why this repo exists (the research gap)

Prior mustard disease work reports very high accuracy with standard CNNs, but
three problems are left unaddressed together:

1. **Severe natural class imbalance** — Alternaria Blight is only 783 of 5,783
   images (13.5%) while White Rust is 3,000 (51.9%). Headline accuracy hides
   poor minority-class recall.
2. **Field-condition clutter** — images contain soil, weeds and neighbouring
   leaves, so a model can "cheat" using background cues that will not transfer.
3. **Unverified explanations** — Grad-CAM is usually shown as a few cherry-picked
   pictures, never *measured*.

PALM tackles all three: a lightweight backbone + **CBAM** lesion attention,
trained with **class-balanced focal loss** and an inverse-frequency sampler, and
audited with a new **Leaf-Attention-Fidelity (LAF)** metric that quantifies how
much of each saliency map lands on plant tissue rather than background.

---

## Method at a glance

```
image ─► timm backbone (EfficientNet-B0) ─► CBAM (channel+spatial attention)
        ─► GAP ─► dropout ─► linear ─► {Healthy, Alternaria, White Rust}

loss = class-balanced focal loss   (effective-number reweighting, gamma=2)
sampler = inverse-frequency weighted random sampler
explainability = Grad-CAM  +  Leaf-Attention-Fidelity (LAF)
```

`model.use_cbam` and `loss.name` are single switches, so every component is a
clean ablation (see `run_all.sh`).

---

## Installation

**Linux / macOS**

```bash
git clone <your-repo-url> palm-mustard-disease
cd palm-mustard-disease
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (CMD)**

```cmd
git clone <your-repo-url> palm-mustard-disease
cd palm-mustard-disease
setup.cmd
.venv\Scripts\activate.bat
```

`setup.cmd` / `setup.ps1` install PyTorch with CUDA 12.1 when available, then the
rest of the stack. Exact versions used on a reference Windows machine are listed
in `requirements-lock-win.txt`. The virtual environment (`.venv/`) is **not**
committed — see [`GITHUB.md`](GITHUB.md) for publishing to GitHub.

Tested with Python 3.10+ and PyTorch 2.x (CUDA optional — the code runs on CPU
and Apple MPS too, just slower).

---

## Quick start

```bash
# 1. Put images under data/raw/{Healthy,Alternaria_Blight,White_Rust}/  (see data/README.md)

# 2. Make reproducible stratified splits
python scripts/prepare_splits.py --config configs/default.yaml

# 3. Train the proposed model
python scripts/train.py --config configs/default.yaml

# 4. Explainability: Grad-CAM grid + Leaf-Attention-Fidelity
python scripts/explain.py --config configs/default.yaml --checkpoint runs/palm/best.pt

# 5. Predict on a new leaf photo
python scripts/predict.py --checkpoint runs/palm/best.pt --image my_leaf.jpg --cam
```

Reproduce **all** paper experiments (baselines + ablations + PALM):

```bash
bash run_all.sh
python scripts/collect_results.py     # prints a Markdown comparison table
```

**Windows:** `run_all.cmd` (CMD) or `.\run_all.ps1` (PowerShell) — same pipeline.

---

## Repository structure

```
palm-mustard-disease/
├── configs/default.yaml      # all hyper-parameters in one place
├── data/README.md            # dataset layout + how to get the images
├── src/
│   ├── config.py             # YAML + CLI override handling
│   ├── dataset.py            # split-CSV dataset, weighted sampler
│   ├── transforms.py         # field-aware augmentation
│   ├── losses.py             # ce / weighted_ce / focal / cb_focal
│   ├── models/
│   │   ├── cbam.py           # channel + spatial attention
│   │   └── build.py          # timm backbone + CBAM + classifier
│   ├── engine.py             # train / eval loops, latency benchmark
│   ├── metrics.py            # macro-F1, balanced acc, per-class, AUC
│   ├── segmentation.py       # label-free leaf foreground mask
│   ├── gradcam.py            # Grad-CAM + Leaf-Attention-Fidelity
│   └── utils.py              # seeding, checkpoints, plots
├── scripts/
│   ├── prepare_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   └── collect_results.py
├── run_all.sh / run_all.cmd / run_all.ps1
├── setup.cmd / setup.ps1
├── requirements.txt
├── requirements-lock-win.txt
├── GITHUB.md
├── LICENSE  (MIT — code only)
└── CITATION.cff
```

---

## Configuration

Everything lives in `configs/default.yaml`; override per run on the CLI, e.g.

```bash
python scripts/train.py --model convnext_tiny --loss cb_focal --epochs 60 \
    --batch-size 24 --output-dir runs/convnext_palm
```

Swap `--model` for any [timm](https://github.com/huggingface/pytorch-image-models)
identifier (`resnet50`, `mobilenetv3_large_100`, `efficientnet_b0`,
`convnext_tiny`, `vit_small_patch16_224`, …).

---

## Reproducibility

* Fixed seeds across `random`, `numpy`, `torch`; cuDNN set deterministic.
* Splits are written to CSV once and reused by every run.
* Each run dumps `test_metrics.json`, `confusion_matrix.png`, `training_curves.png`
  and the best/last checkpoints into its `--output-dir`.

---

## License & data

Code is MIT-licensed. The image dataset is **not** included and is governed by
its own terms — keep it private if required (the `.gitignore` already excludes
`data/`).
"# palm-mustard-disease" 
