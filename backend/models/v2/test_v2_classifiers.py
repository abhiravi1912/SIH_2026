"""
test_v2_classifiers.py — Basic Unit and Contract Tests for V2 Classifiers
========================================================================
Validates that RandomForestV2, SVMV2, KNNV2, and XGBoostV2 satisfy:
  1. Input shape (N, 6) -> Output shape (N,)
  2. Full image (4, H, W) -> Output shape (H, W)
  3. Preprocessing scaler fitting on train only and reuse on test
  4. Save and load persistence round-trip
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from backend.models.v2.base import (
    CLASS_MAP,
    FEATURE_NAMES,
    compute_spectral_features,
)
from backend.models.v2.knn import KNNV2
from backend.models.v2.random_forest import RandomForestV2
from backend.models.v2.svm import SVMV2
from backend.models.v2.xgboost_model import XGBoostV2, _HAS_XGBOOST


class TestV2Classifiers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tmp_dir = tempfile.mkdtemp()

        # Create dummy dataset for shape & interface verification (50 samples, 6 features)
        self.N_train = 100
        self.N_test = 30
        self.X_train = np.random.uniform(0.0, 1.0, (self.N_train, 6)).astype(np.float32)
        self.y_train = np.random.randint(0, 5, self.N_train).astype(np.int32)
        self.X_test = np.random.uniform(0.0, 1.0, (self.N_test, 6)).astype(np.float32)

        # Create dummy 4-band image (4, 32, 32)
        self.dummy_img = np.random.uniform(0.0, 1.0, (4, 32, 32)).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_spectral_features_shape(self):
        features, H, W = compute_spectral_features(self.dummy_img)
        self.assertEqual(H, 32)
        self.assertEqual(W, 32)
        self.assertEqual(features.shape, (32 * 32, 6))
        self.assertTrue(np.all(np.isfinite(features)))

    def _test_classifier_lifecycle(self, clf, supports_proba=True):
        # 1. Check training
        clf.train(self.X_train, self.y_train)
        self.assertTrue(clf._is_fitted)

        # 2. Check predict shape & values
        preds = clf.predict(self.X_test)
        self.assertEqual(preds.shape, (self.N_test,))
        self.assertTrue(np.all(np.isin(preds, [0, 1, 2, 3, 4])))

        # 3. Check predict_proba if supported
        if supports_proba:
            proba = clf.predict_proba(self.X_test)
            self.assertEqual(proba.shape, (self.N_test, 5))
            self.assertTrue(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4))

        # 4. Check full image inference
        seg_map = clf.predict_image(self.dummy_img)
        self.assertEqual(seg_map.shape, (32, 32))
        self.assertEqual(seg_map.dtype, np.uint8)

        # 5. Check save and load round-trip
        save_path = os.path.join(self.tmp_dir, f"{clf.name}.joblib")
        clf.save_model(save_path)
        self.assertTrue(os.path.isfile(save_path))

        loaded_clf = type(clf).from_saved(save_path)
        loaded_preds = loaded_clf.predict(self.X_test)
        np.testing.assert_array_equal(preds, loaded_preds)

    def test_random_forest(self):
        rf = RandomForestV2(n_estimators=10, max_depth=5, random_state=42)
        self._test_classifier_lifecycle(rf, supports_proba=True)

    def test_svm_no_proba(self):
        svm = SVMV2(C=1.0, kernel="rbf", probability=False, random_state=42)
        self._test_classifier_lifecycle(svm, supports_proba=False)
        with self.assertRaises(NotImplementedError):
            svm.predict_proba(self.X_test)

    def test_svm_with_proba(self):
        svm = SVMV2(C=1.0, kernel="rbf", probability=True, random_state=42)
        self._test_classifier_lifecycle(svm, supports_proba=True)

    def test_knn(self):
        knn = KNNV2(n_neighbors=3, weights="distance")
        self._test_classifier_lifecycle(knn, supports_proba=True)

    def test_xgboost_if_available(self):
        if not _HAS_XGBOOST:
            self.skipTest("xgboost is not installed in the current environment.")
        xgb_clf = XGBoostV2(n_estimators=10, max_depth=3, random_state=42)
        self._test_classifier_lifecycle(xgb_clf, supports_proba=True)

    def test_scaler_isolation(self):
        """Verify that scaler mean and variance are locked during inference and not altered."""
        knn = KNNV2(n_neighbors=3, use_scaler=True)
        knn.train(self.X_train, self.y_train)

        mean_before = knn.scaler.mean_.copy()
        var_before = knn.scaler.var_.copy()

        # Run multiple inference passes with different test distributions
        outlier_test = np.ones((20, 6), dtype=np.float32) * 999.0
        _ = knn.predict(outlier_test)

        # Scaler parameters must remain strictly identical
        np.testing.assert_array_equal(knn.scaler.mean_, mean_before)
        np.testing.assert_array_equal(knn.scaler.var_, var_before)


if __name__ == "__main__":
    unittest.main()
