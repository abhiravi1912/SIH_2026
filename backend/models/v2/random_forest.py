"""
random_forest.py — V2 Random Forest Land-Cover Classifier
=========================================================
Scikit-learn based Random Forest classifier for pixel-level semantic
segmentation across 5 land-cover classes using 6 spectral features.

Features: [R, G, B, NIR, NDVI, NDWI]
Classes:
  0 = Bare land
  1 = Vegetation
  2 = Water
  3 = Road
  4 = Building
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseV2Classifier


class RandomForestV2(BaseV2Classifier):
    """
    V2 Random Forest Classifier for 6-band spectral land-cover classification.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=12
        Maximum depth of each tree. None allows unconstrained expansion.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {"sqrt", "log2", None} or float, default="sqrt"
        Number of features to consider when looking for the best split.
    random_state : int, default=42
        Seed used by the random number generator for reproducibility.
    n_jobs : int, default=-1
        Number of CPU cores to use during fitting and predicting (-1 = all).
    use_scaler : bool, default=False
        Whether to standardize features prior to training/inference.
        Random Forest is invariant to monotonic feature scaling, so False is standard.
    **kwargs : Any
        Additional keyword arguments forwarded to RandomForestClassifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        use_scaler: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name="RandomForestV2",
            use_scaler=use_scaler,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _build_model(self) -> RandomForestClassifier:
        """Instantiate underlying RandomForestClassifier."""
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", 12),
            min_samples_split=self.params.get("min_samples_split", 2),
            min_samples_leaf=self.params.get("min_samples_leaf", 1),
            max_features=self.params.get("max_features", "sqrt"),
            random_state=self.random_state,
            n_jobs=self.params.get("n_jobs", -1),
            **{
                k: v
                for k, v in self.params.items()
                if k
                not in [
                    "n_estimators",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "max_features",
                    "n_jobs",
                ]
            },
        )
