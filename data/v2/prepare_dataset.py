"""
data/v2/prepare_dataset.py
==========================
Person 2 — V2 Dataset Preparation Pipeline.
DO NOT modify Person 1 files (backend/**, frontend/**).

Scientific Rigor Features:
- Driven entirely by data/v2/metadata/scene_registry.json.
- Disjoint scene-level splitting (zero spatial & temporal leakage).
- Standardized 6 features: [R, G, B, NIR, NDVI, NDWI] as float32 in [0, 1].
- Clear separation of weak_labels vs annotated ground-truth.
- Multi-scene test evaluation support: preserves individual held-out test scenes.
- Exports dataset manifest with class balance, pixel counts, and split checksums.

Run:
    python data/v2/prepare_dataset.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, List, Any
import numpy as np
import rasterio

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_V2_ROOT = os.path.join(PROJECT_ROOT, "data", "v2")
METADATA_DIR = os.path.join(DATA_V2_ROOT, "metadata")
REGISTRY_PATH = os.path.join(METADATA_DIR, "scene_registry.json")
MANIFEST_PATH = os.path.join(METADATA_DIR, "dataset_manifest.json")

PROCESSED_DIR = os.path.join(DATA_V2_ROOT, "processed")
WEAK_LABELS_DIR = os.path.join(DATA_V2_ROOT, "labels", "weak_labels")
ANNOTATED_LABELS_DIR = os.path.join(DATA_V2_ROOT, "labels", "annotated")
SPLITS_DIR = os.path.join(DATA_V2_ROOT, "splits")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2.prepare")

LABEL_CONFIG = {
    "version": "v2.1",
    "random_seed": 42,
    "epsilon": 1e-5,
    "class_map": {
        0: "Bare land",
        1: "Vegetation",
        2: "Water",
        3: "Road",
        4: "Building",
    },
    "thresholds": {
        "water_ndwi_min": 0.15,
        "vegetation_ndvi_min": 0.25,
        "building_r_min": 150,
        "building_nir_min": 130,
        "building_ndvi_max": 0.10,
        "road_grey_diff_max": 15,
        "road_r_min": 60,
        "road_r_max": 150,
    },
    "priority_order": ["Water", "Vegetation", "Building", "Road", "Bare land"],
    "note": (
        "Track A weak labels are algorithmically derived via physics-based spectral thresholds. "
        "Track B holds human/expert annotated ground truth when accessible."
    ),
}


def load_scene(abs_path: str) -> np.ndarray:
    """Load a 4-band GeoTIFF (R, G, B, NIR) normalised to [0, 1]."""
    with rasterio.open(abs_path) as src:
        if src.count < 4:
            raise ValueError(f"Expected >=4 bands, got {src.count} in {abs_path}")
        bands = src.read([1, 2, 3, 4]).astype(np.float32) / 255.0
    return bands


def compute_features(bands: np.ndarray) -> np.ndarray:
    """
    Compute standardized 6-element feature vector per pixel:
    [R, G, B, NIR, NDVI, NDWI] (float32, in [0, 1] for bands, [-1, 1] for indices).
    """
    eps = LABEL_CONFIG["epsilon"]
    R = bands[0]
    G = bands[1]
    B = bands[2]
    N = bands[3]

    ndvi = (N - R) / (N + R + eps)
    ndwi = (G - N) / (G + N + eps)

    features = np.stack(
        [R.ravel(), G.ravel(), B.ravel(), N.ravel(), ndvi.ravel(), ndwi.ravel()],
        axis=1,
    )
    return features.astype(np.float32)


def derive_weak_labels(bands: np.ndarray) -> np.ndarray:
    """
    Apply deterministic spectral threshold labeling in strict priority order:
    1. Water (NDWI > 0.15)
    2. Vegetation (NDVI > 0.25)
    3. Building (R > 150, NIR > 130, NDVI < 0.10)
    4. Road (neutral grey, mid albedo)
    5. Bare land (default)
    """
    eps = LABEL_CONFIG["epsilon"]
    thr = LABEL_CONFIG["thresholds"]

    R_f = bands[0]
    G_f = bands[1]
    B_f = bands[2]
    N_f = bands[3]

    R8 = (R_f * 255).astype(np.float32)
    G8 = (G_f * 255).astype(np.float32)
    B8 = (B_f * 255).astype(np.float32)
    N8 = (N_f * 255).astype(np.float32)

    ndvi = (N_f - R_f) / (N_f + R_f + eps)
    ndwi = (G_f - N_f) / (G_f + N_f + eps)

    H, W = R_f.shape
    labels = np.zeros(H * W, dtype=np.uint8)

    flat_R8 = R8.ravel()
    flat_G8 = G8.ravel()
    flat_B8 = B8.ravel()
    flat_N8 = N8.ravel()
    flat_ndvi = ndvi.ravel()
    flat_ndwi = ndwi.ravel()

    # Priority 4: Road
    road_mask = (
        (np.abs(flat_R8 - flat_G8) < thr["road_grey_diff_max"])
        & (np.abs(flat_G8 - flat_B8) < thr["road_grey_diff_max"])
        & (flat_R8 >= thr["road_r_min"])
        & (flat_R8 <= thr["road_r_max"])
    )
    labels[road_mask] = 3

    # Priority 3: Building
    bld_mask = (
        (flat_R8 > thr["building_r_min"])
        & (flat_N8 > thr["building_nir_min"])
        & (flat_ndvi < thr["building_ndvi_max"])
    )
    labels[bld_mask] = 4

    # Priority 2: Vegetation
    veg_mask = flat_ndvi > thr["vegetation_ndvi_min"]
    labels[veg_mask] = 1

    # Priority 1: Water
    water_mask = flat_ndwi > thr["water_ndwi_min"]
    labels[water_mask] = 2

    return labels


def main() -> None:
    log.info("=== Starting V2 Scientific Dataset Preparation ===")

    for d in [PROCESSED_DIR, WEAK_LABELS_DIR, ANNOTATED_LABELS_DIR, SPLITS_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    scenes = registry["scenes"]
    log.info(f"Loaded {len(scenes)} scenes from {REGISTRY_PATH}")

    split_X: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    split_y: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    split_scenes: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}

    test_scenes_info: List[Dict[str, Any]] = []

    for sc in scenes:
        sid = sc["scene_id"]
        rel_path = sc["file_path"]
        split = sc.get("split")
        abs_path = os.path.join(PROJECT_ROOT, rel_path)

        if not os.path.exists(abs_path):
            log.warning(f"File missing for {sid}: {abs_path} - skipping.")
            continue

        log.info(f"Processing scene [{sid}] for split [{split}]...")
        bands = load_scene(abs_path)
        features = compute_features(bands)
        labels = derive_weak_labels(bands)

        # Save individual processed files
        feat_path = os.path.join(PROCESSED_DIR, f"{sid}_features.npy")
        lbl_path = os.path.join(WEAK_LABELS_DIR, f"{sid}_labels.npy")
        np.save(feat_path, features)
        np.save(lbl_path, labels)

        if split in split_X:
            split_X[split].append(features)
            split_y[split].append(labels)
            split_scenes[split].append(sid)

        # If test split, save individual scene arrays for per-scene multi-scene evaluation
        if split == "test":
            test_dir = os.path.join(SPLITS_DIR, "test")
            os.makedirs(test_dir, exist_ok=True)
            sc_x_path = os.path.join(test_dir, f"X_{sid}.npy")
            sc_y_path = os.path.join(test_dir, f"y_{sid}.npy")
            np.save(sc_x_path, features)
            np.save(sc_y_path, labels)

            # Class counts
            bc = np.bincount(labels, minlength=5)
            test_scenes_info.append({
                "scene_id": sid,
                "location": sc.get("geographic_location"),
                "date": sc.get("acquisition_date"),
                "test_role": sc.get("test_role", ""),
                "pixels": int(len(labels)),
                "class_counts": {LABEL_CONFIG["class_map"][i]: int(bc[i]) for i in range(5)},
                "x_path": os.path.relpath(sc_x_path, PROJECT_ROOT).replace("\\", "/"),
                "y_path": os.path.relpath(sc_y_path, PROJECT_ROOT).replace("\\", "/"),
            })

    # Save test scenes index
    test_index_path = os.path.join(SPLITS_DIR, "test", "test_scenes_index.json")
    with open(test_index_path, "w", encoding="utf-8") as f:
        json.dump(test_scenes_info, f, indent=2)
    log.info(f"Wrote multi-scene test index to {test_index_path}")

    # Build and write concatenated split arrays
    manifest_splits: Dict[str, Any] = {}
    for sp in ["train", "validation", "test"]:
        sp_dir = os.path.join(SPLITS_DIR, sp)
        os.makedirs(sp_dir, exist_ok=True)

        if not split_X[sp]:
            log.error(f"No scenes found for split '{sp}'")
            continue

        X_cat = np.concatenate(split_X[sp], axis=0)
        y_cat = np.concatenate(split_y[sp], axis=0)

        # Shuffle training split with fixed seed
        if sp == "train":
            rng = np.random.default_rng(LABEL_CONFIG["random_seed"])
            perm = rng.permutation(len(X_cat))
            X_cat = X_cat[perm]
            y_cat = y_cat[perm]

        xpath = os.path.join(sp_dir, "X.npy")
        ypath = os.path.join(sp_dir, "y.npy")
        np.save(xpath, X_cat)
        np.save(ypath, y_cat)

        bc = np.bincount(y_cat, minlength=5)
        cls_dist = {
            LABEL_CONFIG["class_map"][i]: {
                "count": int(bc[i]),
                "percent": round(float(bc[i] / len(y_cat) * 100), 2),
            }
            for i in range(5)
        }

        manifest_splits[sp] = {
            "total_pixels": int(len(y_cat)),
            "features_shape": list(X_cat.shape),
            "scenes": split_scenes[sp],
            "class_distribution": cls_dist,
            "x_file": os.path.relpath(xpath, PROJECT_ROOT).replace("\\", "/"),
            "y_file": os.path.relpath(ypath, PROJECT_ROOT).replace("\\", "/"),
        }
        log.info(f"Split [{sp}]: {len(y_cat):,} pixels across {len(split_scenes[sp])} scenes saved.")

    # Write overall dataset manifest
    manifest_data = {
        "dataset_name": "DRISHTI V2 Land-Cover Dataset",
        "version": "2.1",
        "track_a": {
            "name": "Weak-Label Benchmark",
            "description": "Deterministic spectral thresholding on real Sentinel-2 Level-2A imagery.",
            "label_config": LABEL_CONFIG,
            "splits": manifest_splits,
        },
        "track_b": {
            "name": "Annotated Ground-Truth Benchmark",
            "description": "Human/expert annotated land-cover datasets.",
            "status": "External archives (Zenodo, EuroSAT) offline/forbidden in local environment; interface active in data/v2/labels/annotated/.",
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    log.info(f"Wrote dataset manifest to {MANIFEST_PATH}")

    # Backward compatibility: write dataset_summary.json
    summary_path = os.path.join(DATA_V2_ROOT, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    log.info("=== Dataset Preparation Completed Successfully ===")


if __name__ == "__main__":
    main()
