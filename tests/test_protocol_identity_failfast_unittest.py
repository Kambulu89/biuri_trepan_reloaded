import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from core.algorithm_identity import FeatureSpace, identity_for
from core.evaluation_protocol import EvaluationProtocolGuard, PartitionRole
from core.feature_alignment import align_feature_spaces
from core.model_bundle import bundle_from_pipeline, validate_model_input
from core.residual_ontological_oracle import build_oof_residual_feature_matrices


class ProtocolIdentityFailFastTests(unittest.TestCase):
    def test_test_partition_is_locked_during_selection(self):
        guard = EvaluationProtocolGuard("case")
        guard.record_selection(PartitionRole.VALIDATION, "depth")
        with self.assertRaises(RuntimeError):
            guard.record_selection(PartitionRole.TEST, "retry")

    def test_algorithm_identity_is_native_c45_in_original_space(self):
        identity = identity_for("c45_j48")
        self.assertTrue(identity.canonical)
        self.assertEqual(identity.display_name, "C4.5-Nativo")
        self.assertIn("Gain Ratio", identity.implementation)
        self.assertEqual(identity.feature_space, FeatureSpace.ORIGINAL)

    def test_alignment_never_truncates_or_zero_pads(self):
        oracle = type("Oracle", (), {"n_features_in_": 2})()
        X = np.ones((4, 3))
        with self.assertRaises(ValueError):
            align_feature_spaces(oracle, X, feature_names=["a", "b", "c"])
        projected = align_feature_spaces(
            oracle, X, feature_names=["a", "b", "onto_c"],
            original_feature_names=["a", "b"], log=False,
        )
        np.testing.assert_array_equal(projected, X[:, :2])

    def test_bundle_rejects_reordered_names(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.array([0, 1] * 5)
        model = LogisticRegression().fit(X, y)
        bundle = bundle_from_pipeline("lr", model, ["a", "b"], "original")
        with self.assertRaises(ValueError):
            validate_model_input(bundle, X, feature_names=["b", "a"])

    def test_oof_has_exact_coverage_and_differs_from_insample(self):
        X, y = load_iris(return_X_y=True)
        enriched = np.column_stack([X, X[:, 0] > np.median(X[:, 0])])
        base = LogisticRegression(max_iter=500).fit(X, y)
        train_res, test_res, audit = build_oof_residual_feature_matrices(
            enriched, y, enriched[:12], base, [0, 1, 2, 3], [4], n_splits=5
        )
        self.assertEqual(audit["coverage_min"], 1)
        self.assertEqual(audit["coverage_max"], 1)
        self.assertEqual(train_res.shape, (150, 4))
        self.assertEqual(test_res.shape, (12, 4))
        self.assertFalse(np.allclose(train_res[:, :3], base.predict_proba(X)))


if __name__ == "__main__":
    unittest.main()
