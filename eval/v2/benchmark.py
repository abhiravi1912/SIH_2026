"""
eval/v2/benchmark.py
====================
Person 2 — V2 four-classifier benchmark.

Pipeline
--------
    Real Dataset  (data/v2/splits/)
         ↓
    Same features [R, G, B, NIR, NDVI, NDWI]
         ↓
    ┌──────────┬──────────┬──────────┬──────────┐
    │    RF    │   SVM    │ XGBoost  │   KNN    │
    └──────────┴──────────┴──────────┴──────────┘
                       ↓
                 SAME TEST SET
                       ↓
       Accuracy / Precision / Recall / F1 / IoU
                 Confusion Matrix
                       ↓
                 🏆 BEST MODEL

Design
------
- ADAPTER LAYER: Each classifier is wrapped so the benchmark loop is
  model-agnostic.  Adapters call Person 1's BaseV2Classifier subclasses
  (.train() / .predict()) WITHOUT modifying any of Person 1's files.
- NO hard-coded results. All numbers come from actual model.predict().
- NO winner is selected in advance. The comparison table IS the output.
- Random seed 42 everywhere.

Run
---
    # Step 1 — prepare data (run once)
    python data/v2/prepare_dataset.py

    # Step 2 — run benchmark (all 4 classifiers)
    python eval/v2/benchmark.py

    # Optional: no cap on training samples (slower)
    python eval/v2/benchmark.py --max-train-samples 0

Outputs
-------
    eval/v2/results/benchmark_results.json
    eval/v2/results/benchmark_results.csv
    eval/v2/results/confusion_matrix_<model>.png  (one per classifier)
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import numpy as np

# ─── Path bootstrap ───────────────────────────────────────────────────────────
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
# Wraps Person 1's BaseV2Classifier subclasses with a uniform interface.
# Person 1's files are NEVER modified — only instantiated and called here.

class ClassifierAdapter:
    """Uniform thin wrapper over Person 1's BaseV2Classifier subclasses."""

    def __init__(self, v2_instance, display_name: str):
        """
        Parameters
        ----------
        v2_instance : BaseV2Classifier subclass instance
            One of: RandomForestV2, SVMV2, XGBoostV2, KNNV2
        display_name : str
            Human-readable label for tables and filenames.
        """
        self._clf = v2_instance
        self._name = display_name

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassifierAdapter":
        # Person 1's API: .train(X_train, y_train)
        self._clf.train(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Person 1's API: .predict(X) → np.ndarray uint8
        return self._clf.predict(X).astype(np.int64)

    def get_name(self) -> str:
        return self._name


def build_adapters() -> list[ClassifierAdapter]:
    """
    Instantiate all four classifiers using Person 1's V2 classes and wrap
    them in ClassifierAdapter.  Import errors are caught per-classifier so
    one missing dependency does not abort the full benchmark.
    """
    adapters = []

    # ── 1. Random Forest ──────────────────────────────────────────────────────
    try:
        from backend.models.v2.random_forest import RandomForestV2
        rf = RandomForestV2(
            n_estimators=200,
            max_depth=12,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            use_scaler=False,
        )
        adapters.append(ClassifierAdapter(rf, "Random Forest"))
        log.info("  ✓ Random Forest loaded")
    except Exception as exc:
        log.error(f"  ✗ Random Forest unavailable: {exc}")

    # ── 2. SVM ────────────────────────────────────────────────────────────────
    # NOTE: Person 1's SVMV2 wraps sklearn.svm.SVC which is O(n²) in training
    # samples — not feasible on 265K pixels (would take hours).
    # The adapter uses sklearn LinearSVC directly (liblinear solver, O(n)),
    # which is the standard scalable SVM for large pixel datasets.
    # Person 1's svm.py is NOT modified.
    try:
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        class _LinearSVCAdapter(ClassifierAdapter):
            """Thin local wrapper — uses sklearn LinearSVC, not SVMV2(SVC)."""
            def __init__(self):
                self._pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm", LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_SEED)),
                ])
                self._name = "SVM (LinearSVC)"

            def fit(self, X, y):
                self._pipeline.fit(X, y.astype(np.int32))
                return self

            def predict(self, X):
                return self._pipeline.predict(X).astype(np.int64)

            def get_name(self):
                return self._name

        adapters.append(_LinearSVCAdapter())
        log.info("  ✓ SVM (LinearSVC) loaded")
    except Exception as exc:
        log.error(f"  ✗ SVM unavailable: {exc}")

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    try:
        from backend.models.v2.xgboost_model import XGBoostV2
        xgb_clf = XGBoostV2(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            use_scaler=False,
        )
        adapters.append(ClassifierAdapter(xgb_clf, "XGBoost"))
        log.info("  ✓ XGBoost loaded")
    except Exception as exc:
        log.error(f"  ✗ XGBoost unavailable: {exc}")

    # ── 4. KNN ────────────────────────────────────────────────────────────────
    try:
        from backend.models.v2.knn import KNNV2
        knn = KNNV2(
            n_neighbors=7,
            use_scaler=True,   # KNN needs feature scaling
        )
        adapters.append(ClassifierAdapter(knn, "KNN (k=7)"))
        log.info("  ✓ KNN loaded")
    except Exception as exc:
        log.error(f"  ✗ KNN unavailable: {exc}")

    return adapters


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load X (N,6) float32 and y (N,) int64 from a pre-built split."""
    split_dir = os.path.join(SPLITS_DIR, split_name)
    x_path = os.path.join(split_dir, "X.npy")
    y_path = os.path.join(split_dir, "y.npy")

    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"Split '{split_name}' not found at {split_dir}.\n"
            "Run: python data/v2/prepare_dataset.py"
        )
    return (
        np.load(x_path).astype(np.float32),
        np.load(y_path).astype(np.int64),
    )


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally cap training data with stratified sampling."""
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


# ─── Single model evaluation ──────────────────────────────────────────────────

def evaluate_one(
    adapter: ClassifierAdapter,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_train_samples: int | None = 300_000,
) -> dict:
    """Train adapter → predict on test → compute full metrics."""
    from eval.v2.metrics import full_report

    name = adapter.get_name()
    log.info(f"  Training {name} ...")

    X_tr, y_tr = stratified_subsample(X_train, y_train, max_train_samples)

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
        f"    OA={report['overall_accuracy']:.4f}  "
        f"macroF1={report['macro_f1']:.4f}  "
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


# ─── Comparison table ─────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict[str, dict]) -> None:
    cols   = ["Model", "OA", "Macro F1", "Wt. F1", "mIoU", "Train(s)", "Infer(s)"]
    widths = [22, 8, 10, 8, 8, 10, 9]
    sep = "  "
    header  = sep.join(c.ljust(w) for c, w in zip(cols, widths))
    divider = sep.join("-" * w for w in widths)

    print()
    print("=" * 85)
    print("  V2 BENCHMARK — COMPARISON TABLE  (test split, held-out)")
    print("=" * 85)
    print(header)
    print(divider)

    ranked = sorted(
        [(n, r) for n, r in all_results.items() if "error" not in r],
        key=lambda x: x[1].get("macro_f1", 0),
        reverse=True,
    )
    for i, (name, res) in enumerate(ranked):
        crown = " [BEST]" if i == 0 else "       "
        row = [
            name + crown,
            f"{res.get('overall_accuracy', 0):.4f}",
            f"{res.get('macro_f1', 0):.4f}",
            f"{res.get('weighted_f1', 0):.4f}",
            f"{res.get('mean_iou', 0):.4f}",
            f"{res.get('train_time_s', 0):.2f}",
            f"{res.get('infer_time_s', 0):.3f}",
        ]
        print(sep.join(str(v).ljust(w) for v, w in zip(row, widths)))

    # Failed models
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<22}  [FAILED] {res['error']}")

    print("=" * 85)
    print()
    print("  [BEST] = Best model by Macro F1 on the held-out test split.")
    print("  Winner determined by measured performance — not pre-selected.")
    print()


# ─── Result persistence ───────────────────────────────────────────────────────

def save_results(all_results: dict, run_meta: dict) -> tuple[str, str | None]:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {"benchmark_version": "v2.1", "run_metadata": run_meta, "models": all_results}
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = None
    try:
        import csv
        csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
        rows = []
        for model_name, res in all_results.items():
            if "error" in res:
                rows.append({"model": model_name, "error": res["error"]})
                continue
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
            for cls_name, stats in res.get("per_class", {}).items():
                safe = cls_name.lower().replace(" ", "_")
                for m in ("precision", "recall", "f1", "iou"):
                    row[f"{safe}_{m}"] = stats.get(m)
            rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    except Exception as exc:
        log.warning(f"CSV export failed: {exc}")

    return json_path, csv_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(max_train_samples: int | None = 300_000) -> None:
    from eval.v2.metrics import print_report
    from eval.v2.confusion_matrix import plot_and_save

    log.info("=" * 60)
    log.info("V2 Classifier Benchmark")
    log.info(f"Random seed   : {RANDOM_SEED}")
    log.info(f"Train cap     : {max_train_samples or 'None (all pixels)'}")
    log.info("=" * 60)

    # ── Load data splits ──────────────────────────────────────────────────────
    log.info("Loading splits ...")
    try:
        X_train, y_train = load_split("train")
        X_val,   y_val   = load_split("validation")
        X_test,  y_test  = load_split("test")
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(f"  Train      : {len(X_train):>9,} pixels")
    log.info(f"  Validation : {len(X_val):>9,} pixels")
    log.info(f"  Test       : {len(X_test):>9,} pixels  ← held-out")
    log.info(f"  Features   : {X_train.shape[1]}  [R, G, B, NIR, NDVI, NDWI]")

    # Train on train+val combined; test split untouched until evaluation
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    log.info(f"  Train+Val  : {len(X_train_full):>9,} pixels (used for fitting)")

    # ── Instantiate classifiers via Person 1's V2 classes ────────────────────
    log.info("")
    log.info("Instantiating classifiers ...")
    adapters = build_adapters()

    if not adapters:
        log.error("No classifiers available. Check backend/models/v2/ imports.")
        sys.exit(1)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for adapter in adapters:
        log.info("")
        try:
            result = evaluate_one(
                adapter,
                X_train_full, y_train_full,
                X_test, y_test,
                max_train_samples=max_train_samples,
            )
            all_results[adapter.get_name()] = result
            print_report(adapter.get_name(), result, class_names=CLASS_NAMES)

            cm = result.get("confusion_matrix")
            if cm is not None:
                png = plot_and_save(cm, adapter.get_name(), results_dir=RESULTS_DIR)
                log.info(f"    CM saved : {png}")

        except Exception as exc:
            log.error(f"  {adapter.get_name()} FAILED: {exc}", exc_info=True)
            all_results[adapter.get_name()] = {"error": str(exc)}

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Persist results ───────────────────────────────────────────────────────
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
        "classifier_source": "backend/models/v2/ (Person 1's V2 classifier classes)",
        "note": (
            "All classifiers trained on train+val split, evaluated on the same "
            "held-out test split with identical 6-feature vectors. "
            "No winner pre-selected — ranking is by measured Macro F1."
        ),
    }

    json_path, csv_path = save_results(all_results, run_meta)
    log.info(f"Results → JSON : {json_path}")
    if csv_path:
        log.info(f"Results → CSV  : {csv_path}")
    log.info("Benchmark complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V2 Classifier Benchmark")
    parser.add_argument(
        "--max-train-samples", type=int, default=300_000,
        help="Stratified training sample cap. 0 = no cap. Default: 300000.",
    )
    args = parser.parse_args()
    max_s = args.max_train_samples if args.max_train_samples > 0 else None
    run_benchmark(max_train_samples=max_s)
