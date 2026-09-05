# DRISHTI V2 Classifier Evaluation Framework

## Overview
The V2 Evaluation Framework provides a scientifically defensible benchmark comparing four land-cover classifiers (**Random Forest**, **Support Vector Machine**, **XGBoost**, and **K-Nearest Neighbors**) on standardized multi-spectral Sentinel-2 features across multiple independent geographic scenes.

```
       Real Sentinel-2 Dataset (data/v2/splits/)
                          ↓
      Identical 6 Features [R, G, B, NIR, NDVI, NDWI]
                          ↓
      ┌───────────┬───────────┬───────────┬───────────┐
      │    RF     │    SVM    │  XGBoost  │    KNN    │
      └───────────┴───────────┴───────────┴───────────┘
                          ↓
            MULTIPLE HELD-OUT TEST SCENES
             (Andhra Pradesh & River West)
                          ↓
      Accuracy / Precision / Recall / F1 / IoU / CM
                          ↓
              Empirically Ranked Winner
```

## Scientific Principles
1. **Model-Agnostic Adapters**: Evaluates Person 1's classifiers (`backend/models/v2/**`) through adapter wrappers without modifying backend code.
2. **Zero Leakage**: All train, validation, and test splits are strictly disjoint in space and time. Verified by `eval/v2/leakage_check.py`.
3. **Two Evaluation Tracks**:
   - **Track A (Weak-Label Benchmark)**: Measures agreement with physical spectral threshold rules.
   - **Track B (Annotated Benchmark)**: Measures agreement with expert human ground-truth annotations.
   *Statement*: "The weak-label benchmark measures agreement with spectral labeling rules, whereas the annotated benchmark provides a stronger estimate of real-world classification performance."
4. **Multi-Scene Held-Out Evaluation**: Models are tested on multiple independent scenes (not just one scene) to quantify cross-scene generalization error (mean ± standard deviation).
5. **Standardized Features**: All models receive exactly $[R, G, B, \text{NIR}, \text{NDVI}, \text{NDWI}]$. Scalers are fit strictly on training data.

---

## Evaluation Scripts

### 1. Data Leakage Verification
Runs automated tests verifying scene uniqueness, geographic spatial disjointness (zero bounding box overlap), centroid independence, and array integrity:
```bash
python eval/v2/leakage_check.py
```

### 2. Full Multi-Classifier Benchmark
Trains all 4 models on the training split (default: 300,000 stratified samples) and evaluates on the 786,432 held-out test pixels across all 3 test scenes:
```bash
python eval/v2/benchmark.py --max-train-samples 300000
```
Options:
- `--max-train-samples N`: Cap training samples (default: 300000, 0 = no cap).

Outputs:
- `eval/v2/results/benchmark_results.json`: Full metrics, per-class breakdown, per-scene breakdown, confusion matrices, and execution timing.
- `eval/v2/results/benchmark_results.csv`: Tabular export of all metrics across models.
- `eval/v2/results/confusion_matrix_<model>.png`: Visual confusion matrix plots for each classifier.

### 3. Multi-Scene & Cross-Geographic Generalization
Evaluates models against individual unseen scenes to report mean and standard deviation across scenes, testing geographic transfer between Assam and Andhra Pradesh:
```bash
python eval/v2/scene_evaluation.py
```
Outputs:
- `eval/v2/results/scene_evaluation_results.json`: Per-scene Macro F1, OA, mIoU, and aggregate mean ± std.

---

## Evaluation Metrics Implemented
- **Overall Accuracy (OA)**: Fraction of correctly predicted pixels.
- **Macro Precision & Recall**: Unweighted average across all 5 classes.
- **Macro F1**: Unweighted harmonic mean of precision and recall across classes. Primary ranking metric.
- **Weighted Precision, Recall, and F1**: Support-weighted class averages.
- **Mean Intersection-over-Union (mIoU)**: Average Jaccard index across all classes with positive support.
- **Per-Class Metrics**: Precision, Recall, F1, and IoU for Bare land, Vegetation, Water, Road, and Building.
- **Confusion Matrix**: Full $5 \times 5$ multi-class confusion matrix.
