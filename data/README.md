# Dataset

This repository **does not** ship the image data. Place the mustard leaf images
yourself, then generate the splits.

## 1. Expected layout

```
data/
└── raw/
    ├── Healthy/            # 2,000 images
    ├── Alternaria_Blight/  #   783 images
    └── White_Rust/         # 3,000 images
```

Folder names must match exactly (they map to the fixed class order in
`src/dataset.py`). Sub-folders inside each class are fine — the loader recurses.

## 2. Obtaining the data

The dataset was curated at the Regional Research Station, Bawal (Rewari),
CCS Haryana Agricultural University, over Dec 2025 – Feb 2026 (5,783 DSLR
field images across the three classes above).

> Set your own download source here. Keep it private if your data should not be
> shared publicly:
>
> ```
> DATASET_SOURCE = <add-your-link-or-leave-blank>
> ```

If your raw export uses different folder names (e.g. `whiterust`, `alternaria`),
rename them to the three canonical names above, or adjust `CLASS_NAMES` in
`src/dataset.py`.

## 3. Build splits

```bash
python scripts/prepare_splits.py --config configs/default.yaml
```

This writes deterministic `train.csv`, `val.csv`, `test.csv` into `data/splits/`
(70 / 15 / 15, stratified by class). Re-running with the same `seed` reproduces
the identical partition.

## Notes

* `.gitignore` excludes `data/raw/` and `data/splits/` so images and generated
  splits are never committed by accident.
* The preprocessing (aspect-ratio-preserving resize to 224×224, ImageNet
  normalisation) is applied on the fly in `src/transforms.py`; you do **not**
  need to pre-resize the native 6000×4000 captures.
