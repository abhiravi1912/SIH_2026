"""
xgboost_model.py — V2 XGBoost Land-Cover Classifier
===================================================
Extreme Gradient Boosting (XGBoost) classifier for pixel-level semantic
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

from .base import NUM_CLASSES, BaseV2Classifier

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


class XGBoostV2(BaseV2Classifier):
    """
    V2 XGBoost Classifier for 6-band spectral land-cover classification.

    Parameters
    ----------
    n_estimators : int, default=150
        Number of gradient boosted trees.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    learning_rate : float, default=0.10
        Boosting learning rate (shrinkage).
    subsample : float, default=0.80
        Subsample ratio of the training instances.
    colsample_bytree : float, default=0.80
        Subsample ratio of columns when constructing each tree.
    objective : str, default="multi:softprob"
        Multiclass objective function. "multi:softprob" outputs class probabilities.
    eval_metric : str, default="mlogloss"
        Evaluation metric for multiclass classification.
    random_state : int, default=42
        Random number seed for reproducible subsampling and tree building.
    n_jobs : int, default=-1
        Number of parallel threads used to run XGBoost (-1 = all available cores).
    use_scaler : bool, default=False
        Whether to standardize features prior to training/inference.
    **kwargs : Any
        Additional keyword arguments forwarded to xgboost.XGBClassifier.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.10,
        subsample: float = 0.80,
        colsample_bytree: float = 0.80,
        objective: str = "multi:softprob",
        eval_metric: str = "mlogloss",
        random_state: int = 42,
        n_jobs: int = -1,
        use_scaler: bool = False,
        **kwargs: Any,
    ):
        if not _HAS_XGBOOST:
            raise ImportError(
                "The 'xgboost' package is required for XGBoostV2. "
                "Install it using: pip install xgboost>=2.0.0"
            )

        super().__init__(
            name="XGBoostV2",
            use_scaler=use_scaler,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            eval_metric=eval_metric,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _build_model(self) -> xgb.XGBClassifier:
        """Instantiate underlying XGBClassifier with configured parameters."""
        return xgb.XGBClassifier(
            n_estimators=self.params.get("n_estimators", 150),
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.10),
            subsample=self.params.get("subsample", 0.80),
            colsample_bytree=self.params.get("colsample_bytree", 0.80),
            objective=self.params.get("objective", "multi:softprob"),
            num_class=NUM_CLASSES,
            eval_metric=self.params.get("eval_metric", "mlogloss"),
            random_state=self.random_state,
            n_jobs=self.params.get("n_jobs", -1),
            **{
                k: v
                for k, v in self.params.items()
                if k
                not in [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "objective",
                    "num_class",
                    "eval_metric",
                    "n_jobs",
                ]
            },
        )
