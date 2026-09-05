"""
svm.py — V2 Support Vector Machine (SVM) Land-Cover Classifier
=============================================================
Scikit-learn based Support Vector Classifier (SVC) for pixel-level semantic
segmentation across 5 land-cover classes using 6 spectral features.

Features: [R, G, B, NIR, NDVI, NDWI]
Classes:
  0 = Bare land
  1 = Vegetation
  2 = Water
  3 = Road
  4 = Building

Preprocessing & Scaling:
  SVM with RBF kernel is sensitive to feature scale. A reusable StandardScaler
  is fitted strictly on the training set and automatically serialized alongside
  the model weights, ensuring exact preprocessing during inference without test data leakage.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.svm import SVC

from .base import BaseV2Classifier


class SVMV2(BaseV2Classifier):
    """
    V2 Support Vector Machine (SVM) Classifier for 6-band spectral land-cover classification.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is inversely proportional to C.
    kernel : str, default='rbf'
        Kernel type to be used in the algorithm ('rbf', 'linear', 'poly', 'sigmoid').
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    probability : bool, default=False
        Whether to enable probability estimates. Must be enabled prior to calling train()
        to support predict_proba(). Note: Training with probability=True uses 5-fold cross-validation
        internally and increases training wall-clock time significantly.
    random_state : int, default=42
        Controls pseudo random number generation for shuffling the data for probability estimates.
    use_scaler : bool, default=True
        Whether to standardize features with StandardScaler. Default is True because SVM RBF
        kernels require zero-mean unit-variance normalized feature space.
    **kwargs : Any
        Additional keyword arguments forwarded to sklearn.svm.SVC.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Any = "scale",
        probability: bool = False,
        random_state: int = 42,
        use_scaler: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            name="SVMV2",
            use_scaler=use_scaler,
            random_state=random_state,
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            **kwargs,
        )

    def _build_model(self) -> SVC:
        """Instantiate underlying Support Vector Classifier."""
        return SVC(
            C=self.params.get("C", 1.0),
            kernel=self.params.get("kernel", "rbf"),
            gamma=self.params.get("gamma", "scale"),
            probability=self.params.get("probability", False),
            random_state=self.random_state,
            **{
                k: v
                for k, v in self.params.items()
                if k not in ["C", "kernel", "gamma", "probability"]
            },
        )
