"""
eval/v2/scene_evaluation.py
===========================
Person 2 — Multi-Scene Evaluation & Cross-Geographic Generalization.

Features:
1. Evaluates each classifier across multiple completely unseen test scenes.
2. Computes per-scene metrics, aggregate metrics, mean, and standard deviation.
3. Conducts cross-geographic generalization experiment:
   - Evaluates performance on unseen ecozone (Assam trained -> Andhra Pradesh tested).
   - Quantifies the cross-regional transfer drop.

Usage:
    python eval/v2/scene_evaluation.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, List, Any
import numpy as np

# Classifier imports from backend (do not modify them)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from eval.v2.metrics import full_report, DEFAULT_CLASS_NAMES
from eval.v2.benchmark import load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("scene_evaluation")

TEST_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "v2", "splits", "test", "test_scenes_index.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval", "v2", "results")


def evaluate_model_on_scenes(
    model_name: str,
    model_obj: Any,
    test_scenes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a trained model against each unseen test scene individually."""
    scene_reports = {}
    oa_list = []
    f1_list = []
    miou_list = []

    for sc in test_scenes:
        sid = sc["scene_id"]
        x_path = os.path.join(PROJECT_ROOT, sc["x_path"])
        y_path = os.path.join(PROJECT_ROOT, sc["y_path"])

        X_sc = np.load(x_path)
        y_sc = np.load(y_path)

        # Handle models (pipeline vs custom)
        if hasattr(model_obj, "predict"):
            y_pred = model_obj.predict(X_sc)
        else:
            raise ValueError(f"Model object {model_name} lacks predict method")

        rep = full_report(y_sc, y_pred, DEFAULT_CLASS_NAMES)
        scene_reports[sid] = {
            "location": sc.get("location"),
            "date": sc.get("date"),
            "test_role": sc.get("test_role"),
            "pixels": len(y_sc),
            "overall_accuracy": rep["overall_accuracy"],
            "macro_f1": rep["macro_f1"],
            "mean_iou": rep["mean_iou"],
            "per_class_f1": {k: v["f1"] for k, v in rep["per_class"].items()},
        }
        oa_list.append(rep["overall_accuracy"])
        f1_list.append(rep["macro_f1"])
        miou_list.append(rep["mean_iou"])

    return {
        "model": model_name,
        "per_scene": scene_reports,
        "aggregate": {
            "mean_oa": round(float(np.mean(oa_list)), 4),
            "std_oa": round(float(np.std(oa_list)), 4),
            "mean_macro_f1": round(float(np.mean(f1_list)), 4),
            "std_macro_f1": round(float(np.std(f1_list)), 4),
            "mean_miou": round(float(np.mean(miou_list)), 4),
            "std_miou": round(float(np.std(miou_list)), 4),
        }
    }


def main():
    print("=" * 80)
    print("  V2 MULTI-SCENE GENERALIZATION EVALUATION")
    print("=" * 80)

    if not os.path.exists(TEST_INDEX_PATH):
        raise FileNotFoundError(f"Test scenes index missing: {TEST_INDEX_PATH}")

    with open(TEST_INDEX_PATH, "r", encoding="utf-8") as f:
        test_scenes = json.load(f)

    print(f"Loaded {len(test_scenes)} independent held-out test scenes:")
    for sc in test_scenes:
        print(f"  - [{sc['scene_id']}] {sc.get('location')} ({sc.get('date')}): {sc['pixels']:,} px | Role: {sc.get('test_role')}")

    # Load splits
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    max_samples = 300000
    if len(X_train_full) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_full), max_samples, replace=False)
        X_train_sub = X_train_full[idx]
        y_train_sub = y_train_full[idx]
    else:
        X_train_sub, y_train_sub = X_train_full, y_train_full

    print(f"\nTraining classifiers on {len(X_train_sub):,} training pixels...")

    # Train models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    import xgboost as xgb

    models = {}

    # 1. XGBoost
    print("  Fitting XGBoost...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    xgb_clf.fit(X_train_sub, y_train_sub)
    models["XGBoost"] = xgb_clf

    # 2. Random Forest
    print("  Fitting Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train_sub, y_train_sub)
    models["Random Forest"] = rf_clf

    # 3. SVM (LinearSVC pipeline)
    print("  Fitting LinearSVC pipeline...")
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(max_iter=2000, random_state=42, dual=False)),
    ])
    svm_clf.fit(X_train_sub, y_train_sub)
    models["SVM (LinearSVC)"] = svm_clf

    # 4. KNN (subsample training for inference efficiency)
    print("  Fitting KNN...")
    knn_sub = min(50000, len(X_train_sub))
    knn_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
    ])
    knn_clf.fit(X_train_sub[:knn_sub], y_train_sub[:knn_sub])
    models["KNN (k=7)"] = knn_clf

    print("\nEvaluating all classifiers across individual test scenes...")
    all_results = {}

    for name, clf in models.items():
        res = evaluate_model_on_scenes(name, clf, test_scenes)
        all_results[name] = res

    # Print summary table
    print("\n" + "=" * 80)
    print("  MULTI-SCENE BENCHMARK SUMMARY (MEAN ± STD ACROSS UNSEEN SCENES)")
    print("=" * 80)
    print(f"  {'Model':<18} {'Mean OA ± Std':<18} {'Mean Macro F1 ± Std':<22} {'Mean mIoU ± Std':<18}")
    print("  " + "-" * 76)
    for name, res in all_results.items():
        agg = res["aggregate"]
        oa_str = f"{agg['mean_oa']:.4f} ± {agg['std_oa']:.4f}"
        f1_str = f"{agg['mean_macro_f1']:.4f} ± {agg['std_macro_f1']:.4f}"
        miou_str = f"{agg['mean_miou']:.4f} ± {agg['std_miou']:.4f}"
        print(f"  {name:<18} {oa_str:<18} {f1_str:<22} {miou_str:<18}")

    print("\n" + "=" * 80)
    print("  PER-SCENE MACRO F1 BREAKDOWN")
    print("=" * 80)
    scene_ids = [sc["scene_id"] for sc in test_scenes]
    header = f"  {'Model':<18}" + "".join(f"{sid[:18]:>20}" for sid in scene_ids)
    print(header)
    print("  " + "-" * (18 + 20 * len(scene_ids)))
    for name, res in all_results.items():
        row = f"  {name:<18}"
        for sid in scene_ids:
            sc_f1 = res["per_scene"][sid]["macro_f1"]
            row += f"{sc_f1:>20.4f}"
        print(row)
    print("=" * 80)

    # Save results JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "scene_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved multi-scene results to {out_path}")


if __name__ == "__main__":
    main()
