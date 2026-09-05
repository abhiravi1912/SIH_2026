"""
V2 Dataset Preparation
======================
Person 2 script — do NOT modify Person 1 files.

Reads real Sentinel-2 Level-2A GeoTIFF scenes that are already present locally
(under data/locations/), applies reproducible spectral-threshold labeling to
derive per-pixel land-cover labels, saves feature arrays and label arrays as
.npy files under data/v2/processed/ and data/v2/labels/, then writes
scene-level train / validation / test split manifests.

Label classes (matches backend/models/segmentation.py CLASS_MAP):
    0 = Bare land
    1 = Vegetation
    2 = Water
    3 = Road
    4 = Building

Spectral thresholds are applied in strict priority order:
    Water      (NDWI > 0.15)            — highest priority
    Vegetation (NDVI > 0.25)
    Building   (R > 150 AND NIR > 130 AND NDVI < 0.1)
    Road       (neutral grey: |R-G|<15, |G-B|<15, 60<R<150)
    Bare land  (all remaining pixels)   — lowest priority

Features: [R, G, B, NIR, NDVI, NDWI]  (normalised 0–1 float32)
    NDVI = (NIR - R) / (NIR + R + 1e-5)
    NDWI = (G - NIR) / (G + NIR + 1e-5)

Spatial leakage prevention:
    Pixels from the same scene always stay in the same split.
    No random pixel-level mixing across scenes.

Random seed: 42 (fixed)

Run from the project root:
    python data/v2/prepare_dataset.py
"""

import os
import sys
import json
import logging
import numpy as np
import rasterio

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_V2_ROOT = os.path.join(PROJECT_ROOT, "data", "v2")
PROCESSED_DIR = os.path.join(DATA_V2_ROOT, "processed")
LABELS_DIR = os.path.join(DATA_V2_ROOT, "labels")
SPLITS_DIR = os.path.join(DATA_V2_ROOT, "splits")
LOCATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "locations")

# ─── Scene registry ───────────────────────────────────────────────────────────
# Scene-level split assignment (spatial leakage prevention).
# Each entry: (scene_id, relative_path_from_project_root, split)
SCENE_REGISTRY = [
    # ── TRAIN ──────────────────────────────────────────────────────────────
    {
        "scene_id": "vit_ap_ref",
        "file_path": "data/locations/vit_ap/reference/scene_20210306.tif",
        "split": "train",
        "location": "VIT-AP University, Amaravati, Andhra Pradesh",
        "date": "2021-03-06",
        "sensor": "Sentinel-2 MSI L2A (S2B_44QMD_20210306_2_L2A)",
        "dominant_cover": "mixed: campus built-up, bare land, vegetation",
    },
    {
        "scene_id": "forest_ref",
        "file_path": "data/locations/forest/reference/scene_20240210.tif",
        "split": "train",
        "location": "Garbhanga Forest Reserve, Assam",
        "date": "2024-02-10",
        "sensor": "Sentinel-2 MSI L2A",
        "dominant_cover": "dense vegetation / forest canopy",
    },
    {
        "scene_id": "river_ref_2023",
        "file_path": "data/locations/river/reference/scene_20230121.tif",
        "split": "train",
        "location": "Brahmaputra River, Assam (2023 acquisition)",
        "date": "2023-01-21",
        "sensor": "Sentinel-2 MSI L2A",
        "dominant_cover": "water, sandbar (bare land)",
    },
    {
        "scene_id": "wetland_ref",
        "file_path": "data/locations/wetland/reference/scene_20240210.tif",
        "split": "train",
        "location": "Deepor Beel Ramsar Wetland, Assam",
        "date": "2024-02-10",
        "sensor": "Sentinel-2 MSI L2A",
        "dominant_cover": "open water, aquatic macrophytes",
    },
    # ── VALIDATION ──────────────────────────────────────────────────────────
    {
        "scene_id": "urban_ref",
        "file_path": "data/locations/urban/reference/scene_20240210.tif",
        "split": "validation",
        "location": "Dispur, Guwahati (urban expansion corridor)",
        "date": "2024-02-10",
        "sensor": "Sentinel-2 MSI L2A",
        "dominant_cover": "urban: road, building, bare land",
    },
    {
        "scene_id": "river_ref_2024",
        "file_path": "data/locations/river/reference/scene_20240210.tif",
        "split": "validation",
        "location": "Brahmaputra River, Assam (2024 acquisition)",
        "date": "2024-02-10",
        "sensor": "Sentinel-2 MSI L2A",
        "dominant_cover": "water, sandbar, riverside vegetation",
    },
    # ── TEST ────────────────────────────────────────────────────────────────
    {
        "scene_id": "vit_ap_tgt",
        "file_path": "data/locations/vit_ap/target/scene_20260305.tif",
        "split": "test",
        "location": "VIT-AP University, Amaravati (5-year later scene)",
        "date": "2026-03-05",
        "sensor": "Sentinel-2 MSI L2A (S2C_44QMD_20260305_0_L2A)",
        "dominant_cover": "expanded built-up, less bare land",
    },
]

# ─── Labeling configuration ────────────────────────────────────────────────
LABEL_CONFIG = {
    "version": "v2.0",
    "random_seed": 42,
    "epsilon": 1e-5,
    "class_map": {
        "0": "Bare land",
        "1": "Vegetation",
        "2": "Water",
        "3": "Road",
        "4": "Building",
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
        "Labels are algorithmically derived via spectral thresholds applied to "
        "real Sentinel-2 BOA reflectance. They are NOT human-annotated. "
        "Classifier comparisons are valid (same labels, same test set), but "
        "absolute accuracy numbers reflect agreement with spectral rules, not "
        "manually verified ground truth."
    ),
}

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2.prepare")


# ─── Core helpers ─────────────────────────────────────────────────────────────

def load_scene(abs_path: str) -> np.ndarray:
    """
    Load a 4-band GeoTIFF (R, G, B, NIR) and return a float32 array
    normalised to [0, 1].

    Returns:
        bands: (4, H, W) float32 in [0, 1]
    """
    with rasterio.open(abs_path) as src:
        if src.count < 4:
            raise ValueError(
                f"Expected 4 bands (R,G,B,NIR), got {src.count} in {abs_path}"
            )
        bands = src.read([1, 2, 3, 4]).astype(np.float32) / 255.0  # uint8 → [0,1]
    return bands  # (4, H, W)


def compute_features(bands: np.ndarray) -> np.ndarray:
    """
    Compute 6-element feature vector per pixel.

    Args:
        bands: (4, H, W) float32 — R, G, B, NIR in [0,1]
    Returns:
        features: (H*W, 6) float32  [R, G, B, NIR, NDVI, NDWI]
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


def derive_labels(bands: np.ndarray) -> np.ndarray:
    """
    Apply spectral threshold labeling in priority order.

    Args:
        bands: (4, H, W) float32 in [0,1]
    Returns:
        labels: (H*W,) uint8
    """
    eps = LABEL_CONFIG["epsilon"]
    thr = LABEL_CONFIG["thresholds"]

    R_f = bands[0]
    G_f = bands[1]
    N_f = bands[3]

    # Work in raw uint8-equivalent scale for threshold comparisons
    R8 = (R_f * 255).astype(np.float32)
    G8 = (G_f * 255).astype(np.float32)
    N8 = (N_f * 255).astype(np.float32)
    B8 = (bands[2] * 255).astype(np.float32)

    ndvi = (N_f - R_f) / (N_f + R_f + eps)
    ndwi = (G_f - N_f) / (G_f + N_f + eps)

    H, W = R_f.shape
    labels = np.zeros(H * W, dtype=np.uint8)  # default: Bare land (0)

    flat_R8 = R8.ravel()
    flat_G8 = G8.ravel()
    flat_B8 = B8.ravel()
    flat_N8 = N8.ravel()
    flat_ndvi = ndvi.ravel()
    flat_ndwi = ndwi.ravel()

    # Priority 4 (lowest): Road — neutral grey, mid reflectance
    road_mask = (
        (np.abs(flat_R8 - flat_G8) < thr["road_grey_diff_max"])
        & (np.abs(flat_G8 - flat_B8) < thr["road_grey_diff_max"])
        & (flat_R8 > thr["road_r_min"])
        & (flat_R8 < thr["road_r_max"])
    )
    labels[road_mask] = 3

    # Priority 3: Building — high albedo Red+NIR, low NDVI
    building_mask = (
        (flat_R8 > thr["building_r_min"])
        & (flat_N8 > thr["building_nir_min"])
        & (flat_ndvi < thr["building_ndvi_max"])
    )
    labels[building_mask] = 4

    # Priority 2: Vegetation — high NDVI
    veg_mask = flat_ndvi > thr["vegetation_ndvi_min"]
    labels[veg_mask] = 1

    # Priority 1 (highest): Water — high NDWI
    water_mask = flat_ndwi > thr["water_ndwi_min"]
    labels[water_mask] = 2

    return labels


def class_distribution(labels: np.ndarray) -> dict:
    """Return pixel counts and percentages per class."""
    total = len(labels)
    dist = {}
    class_map = LABEL_CONFIG["class_map"]
    for cid_str, name in class_map.items():
        cid = int(cid_str)
        count = int(np.sum(labels == cid))
        dist[name] = {"class_id": cid, "count": count, "pct": round(count / total * 100, 2)}
    return dist


# ─── Main pipeline ─────────────────────────────────────────────────────────

def prepare():
    np.random.seed(LABEL_CONFIG["random_seed"])
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    for split in ("train", "validation", "test"):
        os.makedirs(os.path.join(SPLITS_DIR, split), exist_ok=True)

    split_manifests = {"train": [], "validation": [], "test": []}
    dataset_summary = {
        "version": "v2.0",
        "random_seed": LABEL_CONFIG["random_seed"],
        "feature_names": ["R", "G", "B", "NIR", "NDVI", "NDWI"],
        "class_map": LABEL_CONFIG["class_map"],
        "label_config": LABEL_CONFIG,
        "scenes": [],
        "split_pixel_counts": {"train": 0, "validation": 0, "test": 0},
        "total_pixels": 0,
    }

    log.info("=" * 60)
    log.info("V2 Dataset Preparation")
    log.info("=" * 60)

    processed_count = 0
    skipped_count = 0

    for entry in SCENE_REGISTRY:
        scene_id = entry["scene_id"]
        rel_path = entry["file_path"]
        split = entry["split"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path.replace("/", os.sep))

        if not os.path.isfile(abs_path):
            log.warning(f"  [SKIP] {scene_id}: file not found at {abs_path}")
            skipped_count += 1
            continue

        log.info(f"  Processing [{split}] {scene_id} ...")

        try:
            bands = load_scene(abs_path)
        except Exception as exc:
            log.error(f"  [ERROR] Failed to load {scene_id}: {exc}")
            skipped_count += 1
            continue

        _, H, W = bands.shape
        n_pixels = H * W

        features = compute_features(bands)   # (n_pixels, 6)
        labels = derive_labels(bands)         # (n_pixels,)

        dist = class_distribution(labels)
        log.info(
            f"    {H}x{W} = {n_pixels:,} px | "
            + " | ".join(f"{k}: {v['pct']:.1f}%" for k, v in dist.items())
        )

        # Save features and labels
        feat_path = os.path.join(PROCESSED_DIR, f"{scene_id}_features.npy")
        lbl_path = os.path.join(LABELS_DIR, f"{scene_id}_labels.npy")
        np.save(feat_path, features)
        np.save(lbl_path, labels)

        # Record in split manifest
        manifest_entry = {
            "scene_id": scene_id,
            "split": split,
            "location": entry["location"],
            "date": entry["date"],
            "sensor": entry["sensor"],
            "dominant_cover": entry["dominant_cover"],
            "source_file": rel_path,
            "features_file": f"processed/{scene_id}_features.npy",
            "labels_file": f"labels/{scene_id}_labels.npy",
            "height": H,
            "width": W,
            "n_pixels": n_pixels,
            "class_distribution": dist,
        }
        split_manifests[split].append(manifest_entry)
        dataset_summary["scenes"].append(manifest_entry)
        dataset_summary["split_pixel_counts"][split] += n_pixels
        dataset_summary["total_pixels"] += n_pixels

        processed_count += 1

    if processed_count == 0:
        log.error("No scenes could be processed. Check file paths.")
        sys.exit(1)

    # ── Save per-split concatenated arrays ──────────────────────────────────
    log.info("")
    log.info("Assembling split arrays ...")

    for split in ("train", "validation", "test"):
        entries = split_manifests[split]
        if not entries:
            log.warning(f"  Split '{split}' has no scenes — skipping.")
            continue

        feat_arrays = []
        lbl_arrays = []
        for e in entries:
            feat_arrays.append(
                np.load(os.path.join(DATA_V2_ROOT, e["features_file"]))
            )
            lbl_arrays.append(
                np.load(os.path.join(DATA_V2_ROOT, e["labels_file"]))
            )

        X = np.concatenate(feat_arrays, axis=0)
        y = np.concatenate(lbl_arrays, axis=0)

        split_dir = os.path.join(SPLITS_DIR, split)
        np.save(os.path.join(split_dir, "X.npy"), X)
        np.save(os.path.join(split_dir, "y.npy"), y)

        dist_combined = class_distribution(y)
        log.info(
            f"  [{split:10}] {len(X):>10,} pixels from {len(entries)} scene(s)"
        )
        for cls_name, info in dist_combined.items():
            log.info(f"              {cls_name:12}: {info['count']:>8,} ({info['pct']:.1f}%)")

        # Save per-split manifest JSON
        with open(os.path.join(split_dir, "manifest.json"), "w") as f:
            json.dump(
                {
                    "split": split,
                    "n_pixels": int(len(X)),
                    "n_scenes": len(entries),
                    "class_distribution": dist_combined,
                    "scenes": entries,
                },
                f,
                indent=2,
            )

    # ── Save label config ────────────────────────────────────────────────────
    with open(os.path.join(DATA_V2_ROOT, "label_config.json"), "w") as f:
        json.dump(LABEL_CONFIG, f, indent=2)

    # ── Save dataset summary ─────────────────────────────────────────────────
    with open(os.path.join(DATA_V2_ROOT, "dataset_summary.json"), "w") as f:
        json.dump(dataset_summary, f, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info(f"Done. Processed {processed_count} scene(s), skipped {skipped_count}.")
    log.info(f"Total pixels: {dataset_summary['total_pixels']:,}")
    log.info(f"Outputs written to: {DATA_V2_ROOT}")
    log.info("=" * 60)


if __name__ == "__main__":
    prepare()
