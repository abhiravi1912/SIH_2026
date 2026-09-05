"""
eval/v2/confusion_matrix.py
============================
Person 2 — generates and saves confusion matrix figures
from actual model predictions.

Usage
-----
Standalone:
    python eval/v2/confusion_matrix.py

Or called from benchmark.py:
    from eval.v2.confusion_matrix import plot_and_save

Outputs
-------
    eval/v2/results/confusion_matrix_<model_name>.png
    eval/v2/results/confusion_matrix_<model_name>.json  (raw counts)
"""

from __future__ import annotations

import os
import json
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

CLASS_NAMES = ["Bare land", "Vegetation", "Water", "Road", "Building"]
N_CLASSES = len(CLASS_NAMES)


def _safe_model_name(name: str) -> str:
    """Sanitise a model name for use as a filename component."""
    return name.lower().replace(" ", "_").replace("-", "_")


def plot_and_save(
    cm: list[list[int]] | np.ndarray,
    model_name: str,
    results_dir: str | None = None,
    normalise: bool = True,
) -> str:
    """
    Plot a confusion matrix and save it as a PNG.

    Parameters
    ----------
    cm : (N, N) confusion matrix (counts) — as list-of-lists or ndarray
    model_name : display name of the model (e.g. "Random Forest")
    results_dir : directory to save outputs (default: eval/v2/results/)
    normalise : if True, also plot the row-normalised version

    Returns
    -------
    Path to the saved PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe for CI
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for confusion matrix plots.\n"
            "Install with: pip install matplotlib"
        ) from exc

    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    cm_arr = np.array(cm, dtype=np.int64)
    safe_name = _safe_model_name(model_name)

    # ── Save raw JSON counts ─────────────────────────────────────────────────
    json_path = os.path.join(results_dir, f"confusion_matrix_{safe_name}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "class_names": CLASS_NAMES,
                "confusion_matrix": cm_arr.tolist(),
            },
            f,
            indent=2,
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2 if normalise else 1,
        figsize=(14 if normalise else 7, 6),
    )
    if not normalise:
        axes = [axes]

    def _draw(ax, data, title, fmt, cmap):
        im = ax.imshow(data, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        tick_marks = np.arange(N_CLASSES)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)

        thresh = data.max() / 2.0
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                val = data[i, j]
                label = fmt.format(val) if not np.isnan(val) else "—"
                ax.text(
                    j, i, label,
                    ha="center", va="center", fontsize=8,
                    color="white" if val > thresh else "black",
                )

    # Raw count plot
    _draw(
        axes[0],
        cm_arr.astype(float),
        f"{model_name} — Confusion Matrix (counts)",
        "{:.0f}",
        "Blues",
    )

    if normalise:
        # Row-normalised (recall per class on diagonal)
        row_sums = cm_arr.sum(axis=1, keepdims=True).astype(float)
        cm_norm = np.where(row_sums > 0, cm_arr / row_sums, np.nan)
        _draw(
            axes[1],
            cm_norm,
            f"{model_name} — Normalised (row = recall)",
            "{:.2f}",
            "Greens",
        )

    fig.suptitle(
        f"V2 Benchmark — {model_name}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    png_path = os.path.join(results_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return png_path


def plot_all_from_results(benchmark_json_path: str, results_dir: str | None = None) -> list[str]:
    """
    Read benchmark_results.json and regenerate all confusion matrix PNGs.

    Useful for re-plotting without rerunning the full benchmark.
    """
    with open(benchmark_json_path) as f:
        data = json.load(f)

    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(benchmark_json_path))

    paths = []
    for model_name, result in data.get("models", {}).items():
        cm = result.get("confusion_matrix")
        if cm is not None:
            path = plot_and_save(cm, model_name, results_dir=results_dir)
            paths.append(path)
            print(f"  Saved: {path}")
    return paths


if __name__ == "__main__":
    import sys

    results_json = os.path.join(RESULTS_DIR, "benchmark_results.json")
    if not os.path.isfile(results_json):
        print(
            f"[ERROR] {results_json} not found.\n"
            "Run 'python eval/v2/benchmark.py' first to generate results."
        )
        sys.exit(1)

    print("Regenerating confusion matrix plots from existing results ...")
    saved = plot_all_from_results(results_json)
    print(f"Done. {len(saved)} plot(s) saved to {RESULTS_DIR}")
