"""
base.py — Core V2 Classifier Base Class and Feature Engineering Utilities
========================================================================
Defines the standard interface, feature extraction, scaling mechanism, and
persistence for all V2 land-cover semantic segmentation classifiers.

V1 Feature Specification:
-------------------------
Input features: [R, G, B, NIR, NDVI, NDWI] (6 dimensions per pixel)

Class Labels (5-class taxonomy):
--------------------------------
0: Bare land
1: Vegetation
2: Water
3: Road
4: Building
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Standard class mappings shared across all V2 classifiers
CLASS_MAP: Dict[int, str] = {
    0: "Bare land",
    1: "Vegetation",
    2: "Water",
    3: "Road",
    4: "Building",
}

# Standard feature order (6 dimensions)
FEATURE_NAMES: list[str] = ["R", "G", "B", "NIR", "NDVI", "NDWI"]
NUM_CLASSES: int = 5
NUM_FEATURES: int = 6


def compute_spectral_features(img_normalized: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Extracts the 6 standard spectral features from a 4-band normalized satellite image.

    Parameters
    ----------
    img_normalized : np.ndarray
        Array of shape (4, H, W) with float values in [0, 1].
        Band order: 0: Red, 1: Green, 2: Blue, 3: NIR.

    Returns
    -------
    features : np.ndarray
        Feature matrix of shape (H * W, 6) with columns [R, G, B, NIR, NDVI, NDWI].
    H : int
        Original image height.
    W : int
        Original image width.
    """
    if img_normalized.ndim != 3 or img_normalized.shape[0] < 4:
        raise ValueError(
            f"Expected img_normalized with shape (4, H, W), got {img_normalized.shape}"
        )

    R = img_normalized[0].astype(np.float32)
    G = img_normalized[1].astype(np.float32)
    B = img_normalized[2].astype(np.float32)
    N = img_normalized[3].astype(np.float32)

    # NDVI = (NIR - Red) / (NIR + Red + eps)
    ndvi = np.clip((N - R) / (N + R + 1e-5), -1.0, 1.0)
    # NDWI = (Green - NIR) / (Green + NIR + eps)
    ndwi = np.clip((G - N) / (G + N + 1e-5), -1.0, 1.0)

    H, W = R.shape
    features = np.stack(
        [R.ravel(), G.ravel(), B.ravel(), N.ravel(), ndvi.ravel(), ndwi.ravel()],
        axis=1,
    ).astype(np.float32)

    # Clean any NaN / Inf values
    np.nan_to_num(features, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return features, H, W


class BaseV2Classifier(ABC):
    """
    Abstract Base Class for all V2 Satellite Land-Cover Classifiers.

    Provides uniform interfaces for:
      - train(X_train, y_train)
      - predict(X)
      - predict_proba(X)
      - save_model(path)
      - load_model(path)
      - predict_image(img_normalized)
      - predict_image_with_proba(img_normalized)

    Ensures strict feature scaling integrity (scaler is fitted ONCE during
    train() and strictly transformed without re-fitting during inference).
    """

    def __init__(
        self,
        name: str,
        use_scaler: bool = False,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.name: str = name
        self.use_scaler: bool = use_scaler
        self.random_state: int = random_state
        self.params: Dict[str, Any] = kwargs
        self.model: Any = None
        self.scaler: Optional[StandardScaler] = StandardScaler() if use_scaler else None
        self._is_fitted: bool = False

    @abstractmethod
    def _build_model(self) -> Any:
        """Instantiate underlying estimator with configured hyperparameters."""
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> BaseV2Classifier:
        """
        Train classifier on feature matrix X_train and integer class labels y_train.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix of shape (N_samples, 6) where columns are [R, G, B, NIR, NDVI, NDWI].
        y_train : np.ndarray
            Target class vector of shape (N_samples,) with values in {0, 1, 2, 3, 4}.
        **kwargs : Any
            Additional model-specific training arguments (e.g. sample_weight).

        Returns
        -------
        self : BaseV2Classifier
            Fitted classifier instance.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32).ravel()

        if X_train.ndim != 2 or X_train.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"[{self.name}] Expected X_train shape (N, {NUM_FEATURES}), got {X_train.shape}"
            )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"[{self.name}] X_train ({len(X_train)}) and y_train ({len(y_train)}) sample counts mismatch."
            )

        # Preprocessing: Fit scaler only on training data if enabled
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_fit = self.scaler.fit_transform(X_train)
        else:
            self.scaler = None
            X_fit = X_train

        self.model = self._build_model()
        self._fit_estimator(X_fit, y_train, **kwargs)
        self._is_fitted = True
        return self

    def _fit_estimator(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Fit underlying estimator. Subclasses can override for custom fit calls."""
        self.model.fit(X, y, **kwargs)

    def _prepare_inference_features(self, X: np.ndarray) -> np.ndarray:
        """Validate shape and transform features using pre-fitted scaler."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError(f"[{self.name}] Model is not fitted. Call train() or load_model() first.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"[{self.name}] Expected X shape (N, {NUM_FEATURES}), got {X.shape}"
            )

        if self.use_scaler:
            if self.scaler is None:
                raise RuntimeError(f"[{self.name}] Scaler enabled but not fitted.")
            return self.scaler.transform(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for feature array X.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N_samples, 6).

        Returns
        -------
        predictions : np.ndarray
            Class labels of shape (N_samples,) with dtype uint8 (values 0–4).
        """
        X_ready = self._prepare_inference_features(X)
        preds = self.model.predict(X_ready)
        return np.asarray(preds, dtype=np.uint8).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for feature array X where supported.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N_samples, 6).

        Returns
        -------
        probabilities : np.ndarray
            Softmax / class probabilities of shape (N_samples, NUM_CLASSES) with dtype float32.
        """
        X_ready = self._prepare_inference_features(X)
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(
                f"[{self.name}] predict_proba is not supported by underlying model or "
                "probability estimation is disabled in hyperparameters."
            )
        proba = self.model.predict_proba(X_ready)
        return np.asarray(proba, dtype=np.float32)

    def predict_image(self, img_normalized: np.ndarray) -> np.ndarray:
        """
        Segment a full normalized satellite image and return 2D class map.

        Parameters
        ----------
        img_normalized : np.ndarray
            Array of shape (4, H, W).

        Returns
        -------
        seg_map : np.ndarray
            2D uint8 segmentation array of shape (H, W).
        """
        features, H, W = compute_spectral_features(img_normalized)
        preds = self.predict(features)
        return preds.reshape(H, W)

    def predict_image_with_proba(
        self, img_normalized: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment image and return both 2D class map and 3D probability cube.

        Returns
        -------
        seg_map : np.ndarray
            2D uint8 array of shape (H, W).
        proba_cube : np.ndarray
            3D float32 array of shape (H, W, 5).
        """
        features, H, W = compute_spectral_features(img_normalized)
        proba = self.predict_proba(features)
        preds = proba.argmax(axis=1).astype(np.uint8)
        return preds.reshape(H, W), proba.reshape(H, W, NUM_CLASSES)

    def save_model(self, path: str) -> str:
        """
        Save model, fitted scaler, and configuration metadata to disk.

        Parameters
        ----------
        path : str
            Target file path (typically ending in .joblib or .pkl).

        Returns
        -------
        saved_path : str
            Absolute path to saved model file.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError(f"[{self.name}] Cannot save an unfitted model.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        bundle = {
            "name": self.name,
            "use_scaler": self.use_scaler,
            "random_state": self.random_state,
            "params": self.params,
            "scaler": self.scaler,
            "model": self.model,
            "class_map": CLASS_MAP,
            "feature_names": FEATURE_NAMES,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(bundle, path)
        return os.path.abspath(path)

    def load_model(self, path: str) -> BaseV2Classifier:
        """
        Load model, fitted scaler, and configuration from disk.

        Parameters
        ----------
        path : str
            Path to saved bundle.

        Returns
        -------
        self : BaseV2Classifier
            Restored classifier instance.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[{self.name}] Model file not found at: {path}")

        bundle = joblib.load(path)
        self.name = bundle.get("name", self.name)
        self.use_scaler = bundle.get("use_scaler", self.use_scaler)
        self.random_state = bundle.get("random_state", self.random_state)
        self.params = bundle.get("params", self.params)
        self.scaler = bundle.get("scaler", None)
        self.model = bundle.get("model", None)
        self._is_fitted = bundle.get("is_fitted", True)
        return self

    @classmethod
    def from_saved(cls, path: str, **kwargs: Any) -> BaseV2Classifier:
        """Convenience constructor to instantiate and load a saved model directly."""
        instance = cls(**kwargs)
        instance.load_model(path)
        return instance
