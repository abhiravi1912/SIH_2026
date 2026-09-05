# V2 Dataset Provenance

> **Person 2 document** — read-only reference for all V2 benchmark work.

---

## 1. Dataset Identity

| Field | Value |
|---|---|
| Dataset name | V2 Sentinel-2 Land-Cover Classification Benchmark |
| Version | v2.0 |
| Created by | Person 2 (V2 dataset preparation and classifier evaluation) |
| Creation date | 2026-09-05 |
| Random seed | 42 (used everywhere: splitting, classifiers) |

---

## 2. Source Imagery

All scenes are real **Sentinel-2 MSI Level-2A BOA (Bottom-of-Atmosphere) Surface Reflectance** imagery downloaded from AWS Open Data COGs / Copernicus Open Access Hub.

**License:** Creative Commons Attribution 4.0 International (CC-BY 4.0) — open access.

| Scene ID | Location | Acquisition Date | Sentinel-2 Product ID | Split |
|---|---|---|---|---|
| `vit_ap_ref` | VIT-AP University, Amaravati, AP | 2021-03-06 | S2B_44QMD_20210306_2_L2A | train |
| `forest_ref` | Garbhanga Forest Reserve, Assam | 2024-02-10 | (Sentinel-2 L2A) | train |
| `river_ref_2023` | Brahmaputra River, Assam | 2023-01-21 | (Sentinel-2 L2A) | train |
| `wetland_ref` | Deepor Beel Ramsar Wetland, Assam | 2024-02-10 | (Sentinel-2 L2A) | train |
| `urban_ref` | Dispur / Guwahati urban corridor | 2024-02-10 | (Sentinel-2 L2A) | validation |
| `river_ref_2024` | Brahmaputra River, Assam | 2024-02-10 | (Sentinel-2 L2A) | validation |
| `vit_ap_tgt` | VIT-AP University, Amaravati, AP | 2026-03-05 | S2C_44QMD_20260305_0_L2A | test |

All scenes are **4-band GeoTIFFs** stored as `uint8` (0–255), bands ordered: Band 1 = Red (B04), Band 2 = Green (B03), Band 3 = Blue (B02), Band 4 = NIR (B08).

---

## 3. Preprocessing

1. **Band loading:** Bands 1–4 are read from each GeoTIFF using `rasterio`.
2. **Normalisation:** All band values divided by `255.0` → float32 in `[0, 1]`.
3. **Feature computation (per pixel):**

| Feature | Formula |
|---|---|
| R | Raw Red band (normalised) |
| G | Raw Green band (normalised) |
| B | Raw Blue band (normalised) |
| NIR | Raw NIR band (normalised) |
| NDVI | `(NIR − R) / (NIR + R + 1e-5)` |
| NDWI | `(G − NIR) / (G + NIR + 1e-5)` |

All four classifiers receive **identical** 6-feature vectors. No classifier gets extra features.

---

## 4. Label Mapping

Because no human-annotated pixel-level labels exist for these scenes, labels are derived via **reproducible spectral thresholds** applied to real Sentinel-2 reflectance.

| Class ID | Name | Spectral Rule | Priority |
|---|---|---|---|
| 2 | Water | NDWI > 0.15 | 1 (highest) |
| 1 | Vegetation | NDVI > 0.25 | 2 |
| 4 | Building | R > 150 AND NIR > 130 AND NDVI < 0.10 | 3 |
| 3 | Road | \|R−G\| < 15 AND \|G−B\| < 15 AND 60 < R < 150 | 4 |
| 0 | Bare land | All remaining pixels | 5 (lowest / default) |

Rules are applied in strict priority order (Water first, Bare land last). Thresholds operate on raw uint8-equivalent scale (0–255) for R, G, NIR comparisons, and on the derived NDVI/NDWI for index-based rules.

> **Important caveat:** These labels are algorithmically derived, not human-verified.
> The benchmark is a valid **relative** comparison between RF, SVM, XGBoost, and KNN —
> all are trained and tested on the same labels. Absolute accuracy numbers should be
> interpreted as agreement with the spectral rules, not as a claim about real-world accuracy.

---

## 5. Train / Validation / Test Split

**Strategy:** Scene-level splitting — pixels from the same scene always stay in the same split. This prevents spatial leakage (adjacent pixels from the same image cannot appear in both train and test).

| Split | Scenes | Expected pixels |
|---|---|---|
| Train | vit_ap_ref, forest_ref, river_ref_2023, wetland_ref | ~1.0 M |
| Validation | urban_ref, river_ref_2024 | ~220 K |
| Test | vit_ap_tgt | ~262 K |

Exact pixel counts are written to `data/v2/splits/{split}/manifest.json` after `prepare_dataset.py` runs.

---

## 6. Feature Definitions (Canonical)

```python
epsilon = 1e-5
NDVI = (NIR - R) / (NIR + R + epsilon)   # Normalised Difference Vegetation Index
NDWI = (G - NIR) / (G + NIR + epsilon)   # Normalised Difference Water Index
```

`R, G, B, NIR` are band values normalised to `[0, 1]` (uint8 ÷ 255).

---

## 7. Output Files

After running `python data/v2/prepare_dataset.py`:

```
data/v2/
├── dataset_summary.json          ← overall dataset metadata
├── label_config.json             ← threshold parameters (reproducibility)
├── processed/
│   ├── <scene_id>_features.npy   ← (n_pixels, 6) float32
│   └── ...
├── labels/
│   ├── <scene_id>_labels.npy     ← (n_pixels,) uint8
│   └── ...
└── splits/
    ├── train/
    │   ├── X.npy                 ← concatenated features
    │   ├── y.npy                 ← concatenated labels
    │   └── manifest.json
    ├── validation/
    │   └── ...
    └── test/
        └── ...
```

---

## 8. Reproducibility Checklist

- [x] All random operations seeded with `42`
- [x] Scene-level splitting (no spatial leakage)
- [x] Identical features for all classifiers
- [x] Thresholds documented in `label_config.json`
- [x] No hard-coded results or fake numbers
- [x] Person 1's files untouched

---

## 9. Related Files

| File | Purpose |
|---|---|
| [`data/v2/prepare_dataset.py`](prepare_dataset.py) | Run once to generate all processed data |
| [`eval/v2/benchmark.py`](../../eval/v2/benchmark.py) | Run the 4-classifier benchmark |
| [`eval/v2/metrics.py`](../../eval/v2/metrics.py) | Metric computation functions |
| [`eval/v2/confusion_matrix.py`](../../eval/v2/confusion_matrix.py) | Confusion matrix generation |
| [`data/v2/label_config.json`](label_config.json) | Auto-generated threshold config |
| [`data/v2/dataset_summary.json`](dataset_summary.json) | Auto-generated dataset metadata |
