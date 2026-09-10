"""
eval/v2/benchmark.py
====================
Person 2 — V2 four-classifier benchmark.

Pipeline:
    Real Dataset (data/v2/splits/)
         ↓
    Same features [R, G, B, NIR, NDVI, NDWI]
         ↓
    ┌──────────┬──────────┬──────────┬──────────┐
    │    RF    │   SVM    │ XGBoost  │   KNN    │
    └──────────┴──────────┴──────────┴──────────┘
                       ↓
         MULTIPLE HELD-OUT TEST SCENES
                       ↓
       Accuracy / Precision / Recall / F1 / IoU
                 Confusion Matrix
                       ↓
                 [BEST] MODEL

Design:
- Model-agnostic adapter layer calling Person 1's BaseV2Classifier subclasses.
- Zero data leakage: Disjoint scenes in train, validation, and test.
- Rigorous metrics: Overall Accuracy, Macro/Weighted Precision/Recall/F1, mIoU, per-class metrics.
- Multi-scene evaluation across independent geographic regions.
- Two evaluation tracks: Weak-label (Track A) vs Annotated Ground Truth (Track B).
- Explicit scientific transparency: no pre-selected winners.

Run:
    python eval/v2/benchmark.py --max-train-samples 300000
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import platform
import datetime
import numpy as np

# ─── Path bootstrap ───────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_V2_ROOT = os.path.join(PROJECT_ROOT, "data", "v2")
SPLITS_DIR = os.path.join(DATA_V2_ROOT, "splits")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
TEST_INDEX_PATH = os.path.join(SPLITS_DIR, "test", "test_scenes_index.json")

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


# ─── Classifier Adapter ───────────────────────────────────────────────────────
class ClassifierAdapter:
    """Wraps a classifier object to provide a unified train/predict interface."""
    def __init__(self, model: object, display_name: str):
        self.model = model
        self.display_name = display_name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if hasattr(self.model, "train"):
            self.model.train(X, y)
        elif hasattr(self.model, "fit"):
            self.model.fit(X, y)
        else:
            raise AttributeError(f"{self.display_name} has neither .train() nor .fit()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise AttributeError(f"{self.display_name} has no .predict()")

    def get_name(self) -> str:
        return self.display_name


def build_adapters() -> list[ClassifierAdapter]:
    """Instantiate all 4 classifiers via Person 1's classes / sklearn pipelines."""
    adapters: list[ClassifierAdapter] = []

    # 1. Random Forest
    try:
        from backend.models.v2.random_forest import RandomForestV2
        rf = RandomForestV2(
            n_estimators=100,
            max_depth=12,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        adapters.append(ClassifierAdapter(rf, "Random Forest"))
        log.info("  [OK] Random Forest loaded")
    except Exception as exc:
        log.error(f"  [FAIL] Random Forest unavailable: {exc}")

    # 2. SVM (LinearSVC pipeline)
    try:
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(max_iter=2000, random_state=RANDOM_SEED, dual=False)),
        ])
        adapters.append(ClassifierAdapter(svm_pipe, "SVM (LinearSVC)"))
        log.info("  [OK] SVM (LinearSVC) loaded")
    except Exception as exc:
        log.error(f"  [FAIL] SVM unavailable: {exc}")

    # 3. XGBoost
    try:
        from backend.models.v2.xgboost_model import XGBoostV2
        xgb = XGBoostV2(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        adapters.append(ClassifierAdapter(xgb, "XGBoost"))
        log.info("  [OK] XGBoost loaded")
    except Exception as exc:
        log.error(f"  [FAIL] XGBoost unavailable: {exc}")

    # 4. KNN
    try:
        from backend.models.v2.knn import KNNV2
        knn = KNNV2(
            n_neighbors=7,
            use_scaler=True,
        )
        adapters.append(ClassifierAdapter(knn, "KNN (k=7)"))
        log.info("  [OK] KNN loaded")
    except Exception as exc:
        log.error(f"  [FAIL] KNN unavailable: {exc}")

    return adapters


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
    """Stratified subsample to cap training data if requested."""
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
    from eval.v2.metrics import full_report

    # KNN inference on 780K pixels can be slow; cap train samples for KNN
    eff_cap = max_train_samples
    if "KNN" in adapter.get_name() and (eff_cap is None or eff_cap > 50000):
        eff_cap = 50000
        log.info(f"    [KNN] Capping training samples to {eff_cap:,} for inference feasibility")

    X_tr, y_tr = stratified_subsample(X_train, y_train, max_samples=eff_cap)

    log.info(f"  Fitting {adapter.get_name()} on {len(X_tr):,} pixels ...")
    t0 = time.perf_counter()
    adapter.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0
    log.info(f"    Train time : {train_time:.2f}s")

    log.info(f"  Predicting on held-out test split ({len(X_test):,} pixels) ...")
    t0 = time.perf_counter()
    y_pred = adapter.predict(X_test)
    infer_time = time.perf_counter() - t0
    log.info(f"    Infer time : {infer_time:.2f}s")

    report = full_report(
        y_test,
        y_pred,
        class_names=CLASS_NAMES,
        n_classes=N_CLASSES,
    )

    return {
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_test)),
        "train_time_s": round(train_time, 4),
        "infer_time_s": round(infer_time, 4),
        **report,
    }


def evaluate_per_scene(
    adapter: ClassifierAdapter,
    test_scenes: list[dict],
) -> dict[str, dict]:
    """Evaluate classifier on each unseen test scene separately."""
    from eval.v2.metrics import full_report
    scene_breakdown = {}
    for sc in test_scenes:
        sid = sc["scene_id"]
        xp = os.path.join(PROJECT_ROOT, sc["x_path"])
        yp = os.path.join(PROJECT_ROOT, sc["y_path"])
        if not os.path.exists(xp) or not os.path.exists(yp):
            continue
        X_sc = np.load(xp)
        y_sc = np.load(yp)
        y_pred_sc = adapter.predict(X_sc)
        rep = full_report(y_sc, y_pred_sc, class_names=CLASS_NAMES, n_classes=N_CLASSES)
        scene_breakdown[sid] = {
            "location": sc.get("location"),
            "date": sc.get("date"),
            "test_role": sc.get("test_role"),
            "pixels": int(len(y_sc)),
            "overall_accuracy": rep["overall_accuracy"],
            "macro_f1": rep["macro_f1"],
            "mean_iou": rep["mean_iou"],
            "per_class_f1": {k: v["f1"] for k, v in rep["per_class"].items()},
        }
    return scene_breakdown


def print_comparison_table(all_results: dict[str, dict]) -> None:
    cols   = ["Model", "OA", "Macro F1", "Wt. F1", "mIoU", "Train(s)", "Infer(s)"]
    widths = [22, 8, 10, 8, 8, 10, 9]
    sep = "  "
    header  = sep.join(c.ljust(w) for c, w in zip(cols, widths))
    divider = sep.join("-" * w for w in widths)

    print()
    print("=" * 85)
    print("  V2 BENCHMARK — COMPARISON TABLE (HELD-OUT TEST SPLIT)")
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

    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<22}  [FAILED] {res['error']}")

    print("=" * 85)
    print("  [BEST] = Best model by Macro F1 on the held-out test split.")
    print("  Winner determined strictly by measured empirical performance.")
    print("=" * 85)


def save_results(all_results: dict, run_meta: dict) -> tuple[str, str | None]:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {"benchmark_version": "v2.2", "run_metadata": run_meta, "models": all_results}
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
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
                "macro_precision": res.get("macro_precision"),
                "macro_recall": res.get("macro_recall"),
                "macro_f1": res.get("macro_f1"),
                "weighted_precision": res.get("weighted_precision"),
                "weighted_recall": res.get("weighted_recall"),
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
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    except Exception as exc:
        log.warning(f"CSV export failed: {exc}")

    return json_path, csv_path


def run_benchmark(max_train_samples: int | None = 300_000) -> None:
    from eval.v2.metrics import print_report
    from eval.v2.confusion_matrix import plot_and_save

    log.info("=" * 70)
    log.info("  V2 SCIENTIFIC CLASSIFIER BENCHMARK (TRACK A: WEAK-LABEL EVALUATION)")
    log.info("=" * 70)
    log.info(
        "Scientific Context: The weak-label benchmark measures agreement with spectral labeling rules, "
        "whereas the annotated benchmark provides a stronger estimate of real-world classification performance."
    )
    log.info(f"Random seed   : {RANDOM_SEED}")
    log.info(f"Train cap     : {max_train_samples or 'None (all pixels)'}")
    log.info("=" * 70)

    # ── Load data splits ──────────────────────────────────────────────────────
    log.info("Loading splits ...")
    try:
        X_train, y_train = load_split("train")
        X_val,   y_val   = load_split("validation")
        X_test,  y_test  = load_split("test")
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(f"  Train      : {len(X_train):>9,} pixels (Guwahati Regional AOI)")
    log.info(f"  Validation : {len(X_val):>9,} pixels (Disjoint River West & Deepor Beel)")
    log.info(f"  Test       : {len(X_test):>9,} pixels (Held-out Multi-Scene)")
    log.info(f"  Features   : {X_train.shape[1]}  [R, G, B, NIR, NDVI, NDWI]")

    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    log.info(f"  Train+Val  : {len(X_train_full):>9,} pixels (used for fitting)")

    # Load multi-scene test definitions
    test_scenes = []
    if os.path.exists(TEST_INDEX_PATH):
        with open(TEST_INDEX_PATH, "r", encoding="utf-8") as f:
            test_scenes = json.load(f)
        log.info(f"  Loaded {len(test_scenes)} independent test scenes from {TEST_INDEX_PATH}")

    # ── Instantiate classifiers ───────────────────────────────────────────────
    log.info("\nInstantiating classifiers ...")
    adapters = build_adapters()

    if not adapters:
        log.error("No classifiers available.")
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

            # Also evaluate per scene if test scenes available
            if test_scenes:
                sc_eval = evaluate_per_scene(adapter, test_scenes)
                result["per_scene_breakdown"] = sc_eval

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

    # ── Multi-Scene Breakdown Table ───────────────────────────────────────────
    if test_scenes:
        print("\n" + "=" * 85)
        print("  MULTI-SCENE HELD-OUT MACRO F1 BREAKDOWN ACROSS INDEPENDENT SCENES")
        print("=" * 85)
        sc_names = [sc["scene_id"] for sc in test_scenes]
        header = f"  {'Model':<20}" + "".join(f"{s[:20]:>21}" for s in sc_names)
        print(header)
        print("  " + "-" * (20 + 21 * len(sc_names)))
        for name, res in all_results.items():
            if "error" in res or "per_scene_breakdown" not in res:
                continue
            row = f"  {name:<20}"
            for s in sc_names:
                sc_f1 = res["per_scene_breakdown"].get(s, {}).get("macro_f1", 0.0)
                row += f"{sc_f1:>21.4f}"
            print(row)
        print("=" * 85)

    # ── Persist results ───────────────────────────────────────────────────────
    run_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "evaluation_track": "Track A: Weak-Label / Spectral-Rule Benchmark",
        "scientific_statement": (
            "The weak-label benchmark measures agreement with spectral labeling rules, "
            "whereas the annotated benchmark provides a stronger estimate of real-world classification performance."
        ),
        "random_seed": RANDOM_SEED,
        "n_classes": N_CLASSES,
        "class_names": CLASS_NAMES,
        "features": ["R", "G", "B", "NIR", "NDVI", "NDWI"],
        "max_train_samples_cap": max_train_samples,
        "train_samples_total": len(X_train_full),
        "test_samples_total": len(X_test),
        "test_scenes_count": len(test_scenes),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    json_path, csv_path = save_results(all_results, run_meta)
    log.info(f"\nResults -> JSON : {json_path}")
    if csv_path:
        log.info(f"Results -> CSV  : {csv_path}")
    log.info("Benchmark complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V2 Scientific Classifier Benchmark")
    parser.add_argument(
        "--max-train-samples", type=int, default=300_000,
        help="Stratified training sample cap. 0 = no cap. Default: 300000.",
    )
    args = parser.parse_args()
    max_s = args.max_train_samples if args.max_train_samples > 0 else None
    run_benchmark(max_train_samples=max_s)
