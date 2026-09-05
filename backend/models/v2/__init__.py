"""
backend.models.v2
=================
V2 Land-Cover Classification Model Suite for Satellite Change Intelligence.

Models:
-------
1. RandomForestV2  (backend.models.v2.random_forest)
2. SVMV2           (backend.models.v2.svm)
3. XGBoostV2       (backend.models.v2.xgboost_model)
4. KNNV2           (backend.models.v2.knn)

Shared Taxonomy:
----------------
0 = Bare land
1 = Vegetation
2 = Water
3 = Road
4 = Building

Features (6 dimensions):
------------------------
[R, G, B, NIR, NDVI, NDWI]
"""

from typing import Any, Dict, Type

from .base import (
    CLASS_MAP,
    FEATURE_NAMES,
    NUM_CLASSES,
    NUM_FEATURES,
    BaseV2Classifier,
    compute_spectral_features,
)
from .knn import KNNV2
from .random_forest import RandomForestV2
from .svm import SVMV2

# Graceful import for XGBoost in environments where xgboost is installed on-demand
try:
    from .xgboost_model import XGBoostV2
except ImportError:
    XGBoostV2 = None  # type: ignore

MODEL_REGISTRY: Dict[str, Type[BaseV2Classifier]] = {
    "rf": RandomForestV2,
    "random_forest": RandomForestV2,
    "svm": SVMV2,
    "knn": KNNV2,
}

if XGBoostV2 is not None:
    MODEL_REGISTRY["xgboost"] = XGBoostV2
    MODEL_REGISTRY["xgb"] = XGBoostV2


def get_classifier(name: str, **kwargs: Any) -> BaseV2Classifier:
    """
    Factory function to instantiate a V2 classifier by key.

    Parameters
    ----------
    name : str
        One of 'rf', 'random_forest', 'svm', 'xgboost', 'xgb', 'knn'.
    **kwargs : Any
        Hyperparameters passed to the classifier constructor.

    Returns
    -------
    classifier : BaseV2Classifier
        Configured V2 classifier instance.
    """
    key = name.lower().strip()
    if key in ["xgboost", "xgb"] and XGBoostV2 is None:
        raise ImportError(
            "XGBoost is not installed in the current environment. "
            "Please run: pip install xgboost>=2.0.0"
        )

    if key not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown classifier '{name}'. Available classifiers: {available}"
        )

    cls = MODEL_REGISTRY[key]
    return cls(**kwargs)


__all__ = [
    "CLASS_MAP",
    "FEATURE_NAMES",
    "NUM_CLASSES",
    "NUM_FEATURES",
    "BaseV2Classifier",
    "compute_spectral_features",
    "RandomForestV2",
    "SVMV2",
    "XGBoostV2",
    "KNNV2",
    "MODEL_REGISTRY",
    "get_classifier",
]
