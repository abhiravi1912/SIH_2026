"""
eval/v2/benchmark.py
====================
Person 2 — V2 four-classifier benchmark.

Pipeline
--------
1. Load train/validation/test splits from data/v2/splits/
2. Train each classifier fresh on the SAME train split
3. Evaluate each classifier on the SAME test split
4. Compute metrics via eval/v2/metrics.py (identical for all classifiers)
5. Save results to eval/v2/results/benchmark_results.json  &  .csv
6. Save confusion matrices to eval/v2/results/

Design principles
-----------------
- ADAPTER LAYER: Each classifier is wrapped in a uniform ClassifierAdapter
  so the benchmark loop is model-agnostic.  Person 1's segmentation.py is
  NOT imported or modified.
- NO hard-coded results.  All numbers come from actual model.predict() calls.
- NO winner is selected in advance.  The final comparison table is the output.
- Random seed 42 is passed to every classifier that accepts it.

Classifiers benchmarked
-----------------------
1. Random Forest  (sklearn)
2. SVM            (sklearn)
3. XGBoost        (xgboost)
4. KNN            (sklearn)

Dependencies
------------
    pip install scikit-learn xgboost matplotlib pandas

Run
---
    # Step 1 — prepare data (run once)
    python data/v2/prepare_dataset.py

    # Step 2 — run benchmark
    python eval/v2/benchmark.py

Outputs
-------
    eval/v2/results/benchmark_results.json
    eval/v2/results/benchmark_results.csv
    eval/v2/results/confusion_matrix_<model>.png   (one per classifier)
    eval/v2/results/confusion_matrix_<model>.json  (raw counts)
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import numpy as np

# ─── Path bootstrap (run from project root or eval/v2/) ──────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_V2_ROOT = os.path.join(PROJECT_ROOT, "data", "v2")
SPLITS_DIR = os.path.join(DATA_V2_ROOT, "splits")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

N_CLASSES = 5
CLASS_NAMES = {
    0: "Bare land",
    1: "Vegetation",
    2: "Water",
    3: "Road",
    4: "Building",
}
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2.benchmark")


# ─── Adapter layer ────────────────────────────────────────────────────────────

class ClassifierAdapter:
    """
    Uniform wrapper around a classifier so the benchmark loop is model-agnostic.

    Each subclass must implement:
        fit(X_train, y_train)
        predict(X_test) -> np.ndarray
        get_name() -> str
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassifierAdapter":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


class RandomForestAdapter(ClassifierAdapter):
    """Random Forest via sklearn.  Matches feature space of segmentation.py."""

    def __init__(self, n_estimators: int = 200, seed: int = RANDOM_SEED):
        from sklearn.ensemble import RandomForestClassifier

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        self._name = "Random Forest"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.int64)

    def get_name(self):
        return self._name


class SVMAdapter(ClassifierAdapter):
    """
    Support Vector Machine via sklearn.
    Uses LinearSVC for scalability on large pixel arrays.
    """

    def __init__(self, C: float = 1.0, seed: int = RANDOM_SEED, max_iter: int = 2000):
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(C=C, max_iter=max_iter, random_state=seed)),
        ])
        self._name = "SVM (LinearSVC)"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.int64)

    def get_name(self):
        return self._name


class XGBoostAdapter(ClassifierAdapter):
    """XGBoost gradient boosting classifier."""

    def __init__(self, seed: int = RANDOM_SEED):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the XGBoost adapter.\n"
                "Install with: pip install xgboost"
            ) from exc

        self._model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        self._name = "XGBoost"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.int64)

    def get_name(self):
        return self._name


class KNNAdapter(ClassifierAdapter):
    """k-Nearest Neighbours via sklearn."""

    def __init__(self, k: int = 7):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm="auto")),
        ])
        self._name = f"KNN (k={k})"

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.int64)

    def get_name(self):
        return self._name


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X (features) and y (labels) from a pre-built split directory.

    Returns
    -------
    X : (N, 6) float32
    y : (N,) int64
    """
    split_dir = os.path.join(SPLITS_DIR, split_name)
    x_path = os.path.join(split_dir, "X.npy")
    y_path = os.path.join(split_dir, "y.npy")

    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"Split '{split_name}' not found at {split_dir}.\n"
            "Run 'python data/v2/prepare_dataset.py' first."
        )

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    return X, y


def subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optionally subsample to max_samples while preserving class balance
    (stratified sampling).  Returns the full array if max_samples >= len(X).
    """
    if max_samples is None or len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max_samples // len(classes)

    idx_list = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        n = min(per_class, len(idx_c))
        idx_list.append(rng.choice(idx_c, size=n, replace=False))

    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    return X[idx], y[idx]


def evaluate_one(
    adapter: ClassifierAdapter,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_train_samples: int | None = 300_000,
) -> dict:
    """
    Train adapter on (X_train, y_train) and evaluate on (X_test, y_test).

    Returns a dict with timing + full metrics report.
    """
    from eval.v2.metrics import full_report

    name = adapter.get_name()
    log.info(f"  Training {name} ...")

    X_tr, y_tr = subsample(X_train, y_train, max_train_samples)

    t0 = time.perf_counter()
    adapter.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0
    log.info(f"    Train time : {train_time:.2f}s  ({len(X_tr):,} samples)")

    t1 = time.perf_counter()
    y_pred = adapter.predict(X_test)
    infer_time = time.perf_counter() - t1
    log.info(f"    Infer time : {infer_time:.3f}s  ({len(X_test):,} samples)")

    report = full_report(y_test, y_pred, class_names=CLASS_NAMES, n_classes=N_CLASSES)
    log.info(
        f"    OA={report['overall_accuracy']:.4f} | "
        f"macroF1={report['macro_f1']:.4f} | "
        f"mIoU={report['mean_iou']:.4f}"
    )

    return {
        "model_name": name,
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_test)),
        "train_time_s": round(train_time, 4),
        "infer_time_s": round(infer_time, 4),
        **report,
    }


def save_results(
    all_results: dict[str, dict],
    run_meta: dict,
) -> tuple[str, str]:
    """
    Save benchmark_results.json and benchmark_results.csv.

    Returns (json_path, csv_path).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    output = {
        "benchmark_version": "v2.0",
        "run_metadata": run_meta,
        "models": all_results,
    }
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── CSV ───────────────────────────────────────────────────────────────────
    try:
        import csv

        csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
        rows = []
        for model_name, res in all_results.items():
            row = {
                "model": model_name,
                "overall_accuracy": res.get("overall_accuracy"),
                "macro_f1": res.get("macro_f1"),
                "weighted_f1": res.get("weighted_f1"),
                "mean_iou": res.get("mean_iou"),
                "train_time_s": res.get("train_time_s"),
                "infer_time_s": res.get("infer_time_s"),
                "train_samples": res.get("train_samples"),
                "test_samples": res.get("test_samples"),
            }
            # Per-class columns
            for cls_name, stats in res.get("per_class", {}).items():
                safe = cls_name.lower().replace(" ", "_")
                row[f"{safe}_precision"] = stats.get("precision")
                row[f"{safe}_recall"] = stats.get("recall")
                row[f"{safe}_f1"] = stats.get("f1")
                row[f"{safe}_iou"] = stats.get("iou")
            rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    except Exception as exc:
        log.warning(f"Could not write CSV: {exc}")
        csv_path = None

    return json_path, csv_path


def print_comparison_table(all_results: dict[str, dict]) -> None:
    """Print a human-readable comparison table to stdout."""
    cols = ["Model", "OA", "Macro F1", "Wt. F1", "mIoU", "Train(s)", "Infer(s)"]
    widths = [22, 8, 10, 8, 8, 10, 9]
    sep = "  "

    header = sep.join(c.ljust(w) for c, w in zip(cols, widths))
    divider = sep.join("-" * w for w in widths)

    print()
    print("=" * 85)
    print("  V2 BENCHMARK — COMPARISON TABLE  (test split)")
    print("=" * 85)
    print(header)
    print(divider)

    for name, res in all_results.items():
        row = [
            name,
            f"{res.get('overall_accuracy', 0):.4f}",
            f"{res.get('macro_f1', 0):.4f}",
            f"{res.get('weighted_f1', 0):.4f}",
            f"{res.get('mean_iou', 0):.4f}",
            f"{res.get('train_time_s', 0):.2f}",
            f"{res.get('infer_time_s', 0):.3f}",
        ]
        print(sep.join(str(v).ljust(w) for v, w in zip(row, widths)))

    print("=" * 85)
    print()
    print("  NOTE: No winner is pre-selected. The table above reflects")
    print("  measured performance on the held-out test split.")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(max_train_samples: int | None = 300_000) -> None:
    """
    Main benchmark entry point.

    Parameters
    ----------
    max_train_samples : cap on training set size (stratified).
        Set to None to use all training pixels.
        Default 300_000 to keep SVM and KNN feasible on a laptop.
    """
    from eval.v2.metrics import print_report
    from eval.v2.confusion_matrix import plot_and_save

    log.info("=" * 60)
    log.info("V2 Classifier Benchmark")
    log.info(f"Random seed : {RANDOM_SEED}")
    log.info("=" * 60)

    # ── Load splits ──────────────────────────────────────────────────────────
    log.info("Loading splits ...")
    try:
        X_train, y_train = load_split("train")
        X_val, y_val = load_split("validation")
        X_test, y_test = load_split("test")
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(f"  Train      : {len(X_train):>9,} pixels")
    log.info(f"  Validation : {len(X_val):>9,} pixels")
    log.info(f"  Test       : {len(X_test):>9,} pixels")
    log.info(f"  Features   : {X_train.shape[1]} (R, G, B, NIR, NDVI, NDWI)")

    # Combine train + validation for final model training
    # (validation was used for threshold tuning during development;
    #  the test split is held out until this function runs)
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    log.info(f"  Train+Val  : {len(X_train_full):>9,} pixels (used for training)")

    # ── Classifier registry ──────────────────────────────────────────────────
    adapters: list[ClassifierAdapter] = [
        RandomForestAdapter(n_estimators=200, seed=RANDOM_SEED),
        SVMAdapter(C=1.0, seed=RANDOM_SEED),
        XGBoostAdapter(seed=RANDOM_SEED),
        KNNAdapter(k=7),
    ]

    # ── Run benchmark ────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for adapter in adapters:
        log.info("")
        try:
            result = evaluate_one(
                adapter,
                X_train_full,
                y_train_full,
                X_test,
                y_test,
                max_train_samples=max_train_samples,
            )
            all_results[adapter.get_name()] = result

            # Pretty print per-model report
            print_report(adapter.get_name(), result, class_names=CLASS_NAMES)

            # Confusion matrix
            cm = result.get("confusion_matrix")
            if cm is not None:
                png_path = plot_and_save(
                    cm,
                    adapter.get_name(),
                    results_dir=RESULTS_DIR,
                )
                log.info(f"    CM saved : {png_path}")

        except Exception as exc:
            log.error(f"  {adapter.get_name()} failed: {exc}", exc_info=True)
            all_results[adapter.get_name()] = {"error": str(exc)}

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    import platform, datetime

    run_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "n_classes": N_CLASSES,
        "class_names": CLASS_NAMES,
        "features": ["R", "G", "B", "NIR", "NDVI", "NDWI"],
        "max_train_samples_cap": max_train_samples,
        "test_samples": int(len(X_test)),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset_dir": DATA_V2_ROOT,
        "note": (
            "Labels are spectral-threshold derived from real Sentinel-2 imagery. "
            "All four classifiers trained on the same train+val split and evaluated "
            "on the same held-out test split with identical 6-feature vectors."
        ),
    }

    json_path, csv_path = save_results(all_results, run_meta)
    log.info(f"Results saved:")
    log.info(f"  JSON : {json_path}")
    if csv_path:
        log.info(f"  CSV  : {csv_path}")
    log.info("Benchmark complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V2 Classifier Benchmark")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=300_000,
        help=(
            "Cap on training samples (stratified). "
            "Use 0 for no cap (very slow for SVM/KNN). "
            "Default: 300000."
        ),
    )
    args = parser.parse_args()

    max_samples = args.max_train_samples if args.max_train_samples > 0 else None
    run_benchmark(max_train_samples=max_samples)
