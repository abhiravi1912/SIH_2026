# V2 Land-Cover Semantic Segmentation Classifier Framework

This directory contains the isolated **V2 Classifier Foundation** designed to benchmark four supervised machine learning algorithms under identical feature distributions and experimental conditions:

1. **Random Forest** (`RandomForestV2` in `random_forest.py`)
2. **Support Vector Machine** (`SVMV2` in `svm.py`)
3. **Extreme Gradient Boosting** (`XGBoostV2` in `xgboost_model.py`)
4. **K-Nearest Neighbors** (`KNNV2` in `knn.py`)

---

## 1. Input Feature Specification

All classifiers receive a 2D numpy array of shape `(N, 6)` or `(H * W, 6)` where each pixel has 6 spectral features:

| Feature Index | Name | Description | Range |
|---|---|---|---|
| `0` | **R** | Red band reflectance (normalized) | `[0.0, 1.0]` |
| `1` | **G** | Green band reflectance (normalized) | `[0.0, 1.0]` |
| `2` | **B** | Blue band reflectance (normalized) | `[0.0, 1.0]` |
| `3` | **NIR** | Near-Infrared reflectance (normalized) | `[0.0, 1.0]` |
| `4` | **NDVI** | Normalized Difference Vegetation Index: `(NIR - R) / (NIR + R + 1e-5)` | `[-1.0, 1.0]` |
| `5` | **NDWI** | Normalized Difference Water Index: `(G - NIR) / (G + NIR + 1e-5)` | `[-1.0, 1.0]` |

The utility `compute_spectral_features(img_normalized)` transforms a 4-band normalized image `(4, H, W)` into `(H * W, 6)` ready for classification.

---

## 2. Land-Cover Class Taxonomy

All models predict categorical labels mapping to the standard 5-class remote sensing taxonomy:

| Class ID | Name | Spectral Characteristics |
|---|---|---|
| `0` | **Bare land** | Higher Red than NIR, negative or low NDVI |
| `1` | **Vegetation** | High NIR reflectance, high NDVI (> 0.35) |
| `2` | **Water** | High Blue reflectance, high NIR absorption, high NDWI (> 0.30) |
| `3` | **Road** | Neutral/balanced spectral signature across R/G/B/NIR |
| `4` | **Building** | High albedo in Red and NIR, low NDVI |

---

## 3. Uniform Model Interface

Each classifier inherits from `BaseV2Classifier` and provides the following core methods:

```python
# 1. Training
model.train(X_train, y_train, **kwargs)

# 2. Pixel-level Prediction
preds = model.predict(X)              # Returns (N,) uint8 array (values 0-4)
proba = model.predict_proba(X)        # Returns (N, 5) float32 array (where supported)

# 3. Full-Image Spatial Inference
seg_map = model.predict_image(img_norm)                 # Returns (H, W) uint8 array
seg_map, proba_cube = model.predict_image_with_proba(img_norm) # (H, W) & (H, W, 5)

# 4. Serialization
model.save_model("experiments/v2/rf/model.joblib")
model.load_model("experiments/v2/rf/model.joblib")

# Or direct instantiation from saved checkpoint:
from backend.models.v2 import RandomForestV2
model = RandomForestV2.from_saved("experiments/v2/rf/model.joblib")
```

---

## 4. Preprocessing & Scaler Mechanism

- **Distance & Margin Models (SVM, KNN)**: Highly sensitive to feature variance. `SVMV2` and `KNNV2` default to `use_scaler=True`.
  - The `StandardScaler` is fitted **strictly during `train()`** on `X_train`.
  - During `predict()`, the fitted scaler transforms `X_test` **without refitting**.
  - When saved via `save_model()`, the fitted scaler is packaged alongside the estimator.
- **Tree-Based Models (Random Forest, XGBoost)**: Invariant to monotonic scaling, so `use_scaler=False` by default.

---

## 5. Usage Example

```python
import numpy as np
from backend.models.v2 import get_classifier

# Instantiate via factory or direct class import
clf = get_classifier("svm", C=1.0, kernel="rbf")

# X_train: (N, 6), y_train: (N,)
clf.train(X_train, y_train)

# Inference on new pixels
predictions = clf.predict(X_test)      # shape: (N,)

# Inference directly on a 4-band tile (4, 512, 512)
seg_map = clf.predict_image(tile_norm) # shape: (512, 512)

# Save checkpoint with integrated scaler
clf.save_model("experiments/v2/svm/best_svm.joblib")
```
