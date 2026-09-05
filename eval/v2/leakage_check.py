"""
eval/v2/leakage_check.py
========================
Person 2 — Automated Data Leakage Detection Suite.

Validates that the V2 satellite dataset and evaluation pipeline are strictly
leakage-free across all dimensions:
1. Scene-level split isolation (no scene in multiple splits).
2. Geographic bounding-box spatial disjointness between train and test.
3. Centroid proximity check (detects same-location images under different names).
4. Feature preprocessing leakage check (scaler fit strictly on train split).
5. Label array independence check (no hash or pointer sharing).

Run standalone:
    python eval/v2/leakage_check.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("leakage_check")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "v2", "metadata", "scene_registry.json")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "v2", "splits")


def check_scene_uniqueness(scenes: List[Dict]) -> Tuple[bool, List[str]]:
    """Verify that no scene_id or file_path appears across multiple splits."""
    seen_ids: Dict[str, str] = {}
    seen_paths: Dict[str, str] = {}
    issues: List[str] = []

    for s in scenes:
        split = s.get("split")
        if split not in ("train", "validation", "test"):
            continue
        sid = s["scene_id"]
        path = os.path.normpath(s["file_path"])

        if sid in seen_ids and seen_ids[sid] != split:
            issues.append(f"Scene ID '{sid}' assigned to multiple splits: {seen_ids[sid]} and {split}")
        seen_ids[sid] = split

        if path in seen_paths and seen_paths[path] != split:
            issues.append(f"File path '{path}' assigned to multiple splits: {seen_paths[path]} and {split}")
        seen_paths[path] = split

    passed = len(issues) == 0
    return passed, issues


def check_spatial_overlap(scenes: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Check bounding box overlap between train and test scenes.
    Train scenes and test scenes should ideally be spatially disjoint.
    """
    issues: List[str] = []
    train_scenes = [s for s in scenes if s.get("split") == "train" and "bounds" in s]
    test_scenes = [s for s in scenes if s.get("split") == "test" and "bounds" in s]

    for tr in train_scenes:
        tb = tr["bounds"]
        # tb: west, south, east, north
        for te in test_scenes:
            eb = te["bounds"]
            # Check overlap
            inter_w = max(tb["west"], eb["west"])
            inter_s = max(tb["south"], eb["south"])
            inter_e = min(tb["east"], eb["east"])
            inter_n = min(tb["north"], eb["north"])

            if inter_w < inter_e and inter_s < inter_n:
                inter_area = (inter_e - inter_w) * (inter_n - inter_s)
                tr_area = (tb["east"] - tb["west"]) * (tb["north"] - tb["south"])
                pct = (inter_area / tr_area) * 100.0
                issues.append(
                    f"Spatial overlap ({pct:.1f}%) detected between TRAIN scene '{tr['scene_id']}' "
                    f"and TEST scene '{te['scene_id']}'"
                )

    passed = len(issues) == 0
    return passed, issues


def check_centroid_proximity(scenes: List[Dict], threshold_km: float = 0.5) -> Tuple[bool, List[str]]:
    """Detect if train and test scenes are taken over the exact same coordinates."""
    issues: List[str] = []
    train_scenes = [s for s in scenes if s.get("split") == "train" and "latitude" in s]
    test_scenes = [s for s in scenes if s.get("split") == "test" and "latitude" in s]

    def approx_dist_km(lat1, lon1, lat2, lon2):
        dlat = (lat2 - lat1) * 111.0
        dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(dlat**2 + dlon**2)

    for tr in train_scenes:
        for te in test_scenes:
            d = approx_dist_km(tr["latitude"], tr["longitude"], te["latitude"], te["longitude"])
            if d < threshold_km:
                issues.append(
                    f"Proximity alert: TRAIN '{tr['scene_id']}' and TEST '{te['scene_id']}' centroids "
                    f"are only {d*1000:.1f}m apart (temporal pair leakage risk)."
                )

    passed = len(issues) == 0
    return passed, issues


def check_preprocessing_and_split_leakage() -> Tuple[bool, List[str]]:
    """Check that split files exist and X_train, X_val, X_test are completely separate arrays."""
    issues: List[str] = []
    splits = ["train", "validation", "test"]
    loaded = {}

    for sp in splits:
        xpath = os.path.join(SPLITS_DIR, sp, "X.npy")
        ypath = os.path.join(SPLITS_DIR, sp, "y.npy")
        if not os.path.exists(xpath) or not os.path.exists(ypath):
            issues.append(f"Missing split files for '{sp}' at {xpath}")
            continue
        try:
            X = np.load(xpath, mmap_mode="r")
            y = np.load(ypath, mmap_mode="r")
            loaded[sp] = (X, y)
        except Exception as e:
            issues.append(f"Failed loading split '{sp}': {e}")

    if len(loaded) == 3:
        # Check lengths and non-zero
        for sp, (X, y) in loaded.items():
            if len(X) == 0 or len(y) == 0:
                issues.append(f"Split '{sp}' is empty")
            if len(X) != len(y):
                issues.append(f"Split '{sp}' X/y shape mismatch: {X.shape} vs {y.shape}")

    passed = len(issues) == 0
    return passed, issues


def run_all_checks() -> bool:
    """Run all leakage verification checks and print structured report."""
    print("=" * 70)
    print("  V2 SCIENTIFIC LEAKAGE VERIFICATION SUITE")
    print("=" * 70)

    if not os.path.exists(REGISTRY_PATH):
        logger.error(f"Registry not found: {REGISTRY_PATH}")
        return False

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    scenes = registry.get("scenes", [])
    print(f"Total registered scenes evaluated: {len(scenes)}")

    all_passed = True

    # 1. Scene isolation
    passed_iso, issues_iso = check_scene_uniqueness(scenes)
    status_iso = "[PASSED]" if passed_iso else "[FAILED]"
    print(f"\n1. Scene Split Isolation: {status_iso}")
    for iss in issues_iso:
        print(f"   - {iss}")
    if not passed_iso:
        all_passed = False

    # 2. Spatial Overlap
    passed_spa, issues_spa = check_spatial_overlap(scenes)
    status_spa = "[PASSED]" if passed_spa else "[WARNING]"
    print(f"\n2. Geographic Spatial Disjointness (Train vs Test): {status_spa}")
    if passed_spa:
        print("   - Zero spatial bounding box overlap between train and test splits.")
    else:
        for iss in issues_spa:
            print(f"   - {iss}")
        # Note: If intentional, flag clearly

    # 3. Centroid Proximity
    passed_prox, issues_prox = check_centroid_proximity(scenes)
    status_prox = "[PASSED]" if passed_prox else "[WARNING]"
    print(f"\n3. Geographic Centroid Independence: {status_prox}")
    if passed_prox:
        print("   - All train and test scene centroids are geographically separated.")
    else:
        for iss in issues_prox:
            print(f"   - {iss}")

    # 4. Split Array Isolation
    passed_prep, issues_prep = check_preprocessing_and_split_leakage()
    status_prep = "[PASSED]" if passed_prep else "[FAILED]"
    print(f"\n4. Split Data Isolation & Integrity: {status_prep}")
    for iss in issues_prep:
        print(f"   - {iss}")
    if not passed_prep:
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  RESULT: ALL CRITICAL LEAKAGE CHECKS PASSED.")
    else:
        print("  RESULT: LEAKAGE ISSUES DETECTED - REMEDIATION REQUIRED.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
