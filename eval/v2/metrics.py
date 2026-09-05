"""
eval/v2/metrics.py
==================
Person 2 — V2 evaluation metrics.

Pure functions: no model imports, no I/O side-effects.
All functions operate on numpy arrays (y_true, y_pred).

Metrics implemented
-------------------
- overall_accuracy
- per_class_precision / recall / f1 / iou
- macro_f1 / weighted_f1
- mean_iou
- full_report  ← single entry-point returning all metrics as a dict

Usage
-----
    from eval.v2.metrics import full_report
    results = full_report(y_true, y_pred, class_names)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ─── Class constants (must match backend/models/segmentation.py CLASS_MAP) ───
DEFAULT_CLASS_NAMES = {
    0: "Bare land",
    1: "Vegetation",
    2: "Water",
    3: "Road",
    4: "Building",
}


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError(f"Expected 1-D arrays, got shape {y_true.shape}")


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of pixels correctly classified."""
    _validate(y_true, y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))


def confusion_matrix_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Build a confusion matrix entirely in numpy (no sklearn dependency).

    Returns:
        cm: (n_classes, n_classes) int64 array
            cm[i, j] = number of pixels with true label i predicted as j
    """
    _validate(y_true, y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(
    cm: np.ndarray,
) -> dict[int, dict[str, float]]:
    """
    Compute per-class precision, recall, F1, IoU from a confusion matrix.

    Formulae
    --------
    TP_c  = cm[c, c]
    FP_c  = sum(cm[:, c]) - TP_c
    FN_c  = sum(cm[c, :]) - TP_c
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = TP / (TP + FP + FN + eps)
    """
    n = cm.shape[0]
    results = {}
    eps = 1e-10

    for c in range(n):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        support = float(cm[c, :].sum())

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        results[c] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "iou": round(iou, 6),
            "support": int(support),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
        }
    return results


def macro_f1(per_class: dict[int, dict[str, float]]) -> float:
    """Unweighted mean F1 across all classes."""
    f1s = [v["f1"] for v in per_class.values()]
    return round(float(np.mean(f1s)), 6)


def weighted_f1(per_class: dict[int, dict[str, float]]) -> float:
    """Support-weighted mean F1 across all classes."""
    total_support = sum(v["support"] for v in per_class.values())
    if total_support == 0:
        return 0.0
    wf1 = sum(
        v["f1"] * v["support"] for v in per_class.values()
    ) / total_support
    return round(float(wf1), 6)


def mean_iou(per_class: dict[int, dict[str, float]]) -> float:
    """Mean IoU (mIoU) across all classes that have any support."""
    ious = [v["iou"] for v in per_class.values() if v["support"] > 0]
    return round(float(np.mean(ious)) if ious else 0.0, 6)


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict[int, str] | None = None,
    n_classes: int = 5,
) -> dict:
    """
    Compute all evaluation metrics for a single classifier run.

    Parameters
    ----------
    y_true : (N,) uint8/int array of ground-truth labels
    y_pred : (N,) uint8/int array of predicted labels
    class_names : optional dict {class_id: name}
    n_classes : number of classes (default 5)

    Returns
    -------
    dict with keys:
        overall_accuracy, macro_f1, weighted_f1, mean_iou,
        per_class: {class_name: {precision, recall, f1, iou, support}},
        confusion_matrix: [[…]]
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    oa = overall_accuracy(y_true, y_pred)
    cm = confusion_matrix_array(y_true, y_pred, n_classes)
    pc = per_class_metrics(cm)

    # Build per_class dict keyed by class name
    per_class_named = {}
    for cid, stats in pc.items():
        name = class_names.get(cid, f"class_{cid}")
        per_class_named[name] = stats

    return {
        "overall_accuracy": round(oa, 6),
        "macro_f1": macro_f1(pc),
        "weighted_f1": weighted_f1(pc),
        "mean_iou": mean_iou(pc),
        "per_class": per_class_named,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
        "n_classes": n_classes,
    }


def print_report(model_name: str, report: dict, class_names: dict[int, str] | None = None) -> None:
    """Pretty-print a full_report dict to stdout."""
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    width = 64
    print("=" * width)
    print(f"  Model: {model_name}")
    print("=" * width)
    print(f"  Overall Accuracy : {report['overall_accuracy']:.4f}")
    print(f"  Macro F1         : {report['macro_f1']:.4f}")
    print(f"  Weighted F1      : {report['weighted_f1']:.4f}")
    print(f"  Mean IoU (mIoU)  : {report['mean_iou']:.4f}")
    print()
    print(f"  {'Class':<14} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7} {'Support':>9}")
    print("  " + "-" * 54)
    for cls_name, stats in report["per_class"].items():
        print(
            f"  {cls_name:<14} "
            f"{stats['precision']:>7.4f} "
            f"{stats['recall']:>7.4f} "
            f"{stats['f1']:>7.4f} "
            f"{stats['iou']:>7.4f} "
            f"{stats['support']:>9,}"
        )
    print("=" * width)
