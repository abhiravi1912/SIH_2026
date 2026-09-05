"""
knn.py — V2 K-Nearest Neighbors (KNN) Land-Cover Classifier
===========================================================
Scikit-learn based KNeighborsClassifier for pixel-level semantic
segmentation across 5 land-cover classes using 6 spectral features.

Features: [R, G, B, NIR, NDVI, NDWI]
Classes:
  0 = Bare land
  1 = Vegetation
  2 = Water
  3 = Road
  4 = Building

Preprocessing & Scaling:
  KNN computes Euclidean/Minkowski distance in 6-dimensional feature space.
  Unscaled features will cause high-variance bands to dominate distance metrics.
  A reusable StandardScaler is fitted strictly on the training set and serialized
  with the model, guaranteeing consistent scaling during inference without test data leakage.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .base import BaseV2Classifier


class KNNV2(BaseV2Classifier):
    """
    V2 K-Nearest Neighbors (KNN) Classifier for 6-band spectral land-cover classification.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use for kneighbors queries.
    weights : {'uniform', 'distance'} or callable, default='distance'
        Weight function used in prediction. 'distance' weights points by the inverse of their
        distance, providing better continuous classification boundaries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.
    p : int, default=2
        Power parameter for the Minkowski metric (p=2 is standard Euclidean distance).
    metric : str or callable, default='minkowski'
        Distance metric to use for the tree.
    n_jobs : int, default=-1
        Number of parallel jobs to run for neighbor search (-1 = all cores).
    use_scaler : bool, default=True
        Whether to standardize features with StandardScaler. Default is True because
        distance-based algorithms are sensitive to feature variances and scales.
    **kwargs : Any
        Additional keyword arguments forwarded to KNeighborsClassifier.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        n_jobs: int = -1,
        use_scaler: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            name="KNNV2",
            use_scaler=use_scaler,
            random_state=42,  # KNN is deterministic, but kept for uniform API
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _build_model(self) -> KNeighborsClassifier:
        """Instantiate underlying KNeighborsClassifier."""
        return KNeighborsClassifier(
            n_neighbors=self.params.get("n_neighbors", 5),
            weights=self.params.get("weights", "distance"),
            algorithm=self.params.get("algorithm", "auto"),
            leaf_size=self.params.get("leaf_size", 30),
            p=self.params.get("p", 2),
            metric=self.params.get("metric", "minkowski"),
            n_jobs=self.params.get("n_jobs", -1),
            **{
                k: v
                for k, v in self.params.items()
                if k
                not in [
                    "n_neighbors",
                    "weights",
                    "algorithm",
                    "leaf_size",
                    "p",
                    "metric",
                    "n_jobs",
                ]
            },
        )
