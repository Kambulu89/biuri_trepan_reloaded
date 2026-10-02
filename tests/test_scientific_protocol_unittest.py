import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.mlp_optimizer import _pick_best_result
from core.onto_feature_selector import (
    fit_ontology_feature_selector,
    apply_ontology_feature_selection,
)
from core.residual_ontological_oracle import build_oof_residual_feature_matrices
from core.ontology_acceptance import evaluate_ontology_acceptance
from core.metrics_comparator import MetricsComparator
from core.scientific_validation import (
    paired_bootstrap_accuracy_difference,
    assess_pairwise_superiority,
)


class ScientificProtocolTests(unittest.TestCase):
    def test_model_selection_uses_cv_score_not_test_metrics(self):
        X = np.arange(24, dtype=float).reshape(12, 2)

        class Model:
            def __init__(self, value): self.value = value
            def predict(self, X): return np.arange(len(X)) % 2

        low_cv_high_test = {
            "model": Model(1), "best_score": 0.51,
            "test_metrics": {"accuracy": 1.0, "f1": 1.0},
        }
        high_cv_low_test = {
            "model": Model(2), "best_score": 0.75,
            "test_metrics": {"accuracy": 0.0, "f1": 0.0},
        }
        picked = _pick_best_result(
            [low_cv_high_test, high_cv_low_test], X, np.arange(12) % 2, None
        )
        self.assertIs(picked, high_cv_low_test)

    def test_feature_selection_fit_then_apply(self):
        rng = np.random.default_rng(3)
        X_train = rng.normal(size=(80, 5))
        y_train = (X_train[:, 3] > 0).astype(int)
        names = ["a", "b", "onto_noise", "onto_signal", "onto_noise2"]
        idx, selected, info = fit_ontology_feature_selector(
            X_train, y_train, names, ["a", "b"], top_k=1
        )
        X_test = rng.normal(size=(20, 5))
        applied = apply_ontology_feature_selection(X_test, idx)
        self.assertEqual(applied.shape[1], 3)
        self.assertIn("onto_signal", selected)
        self.assertEqual(info["fit_scope"], "training_only")

    def test_residual_training_uses_oof_probabilities(self):
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=4,
            n_redundant=0, random_state=5,
        )
        X_aug = np.column_stack([X, X[:, 0] ** 2])
        model = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(8,), max_iter=300, random_state=5
        ))
        model.fit(X[:80], y[:80])
        train_res, test_res, info = build_oof_residual_feature_matrices(
            X_aug[:80], y[:80], X_aug[80:], model,
            list(range(5)), [5], n_splits=4,
        )
        self.assertEqual(train_res.shape, (80, 3))
        self.assertEqual(test_res.shape, (20, 3))
        self.assertEqual(info["base_probability_source_train"], "out_of_fold")

    def test_acceptance_requires_all_primary_metrics(self):
        result = evaluate_ontology_acceptance(
            0.80, 0.70, 0.80, 0.81,
            balanced_accuracy_original=0.80,
            balanced_accuracy_onto=0.81,
            tolerance=0.01,
        )
        self.assertFalse(result["accepted"])

    def test_schema_mismatch_is_not_silently_padded(self):
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(random_state=1).fit(
            np.ones((6, 4)), np.arange(6) % 2
        )
        with self.assertRaises(ValueError):
            MetricsComparator()._prepare_reloaded_predict_matrix(
                tree, np.ones((3, 2))
            )

    def test_paired_superiority_guard(self):
        y = np.array([0, 1] * 50)
        candidate = y.copy()
        baseline = y.copy()
        baseline[:30] = 1 - baseline[:30]
        stats = paired_bootstrap_accuracy_difference(
            y, candidate, baseline, n_bootstrap=1000
        )
        self.assertGreater(stats["ci_lower"], 0)
        result = assess_pairwise_superiority(
            y, y, candidate, baseline, n_bootstrap=1000
        )
        self.assertEqual(result["verdict"], "superior")


if __name__ == "__main__":
    unittest.main()
