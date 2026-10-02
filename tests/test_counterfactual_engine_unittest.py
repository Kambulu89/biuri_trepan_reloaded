"""Testes sem dependência de pytest/PyQt para o motor contrafactual completo."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from counterfactuals.engine import (
    CounterfactualConstraints,
    CounterfactualEngine,
    FeatureSpaceAdapter,
    FeatureSpaceMismatchError,
    OntologyCFValidator,
    apply_rst_filter,
    evaluate_local_robustness,
)
from counterfactuals.export import export_counterfactual_result
from counterfactuals.service import (
    evaluate_transfer_from_session,
    generate_explanation_from_session,
)
from counterfactuals.transfer import (
    TransferCategory,
    aggregate_transfer_metrics,
    evaluate_transfer_protocol,
)
from tests.support.counterfactual_multidataset_runner import evaluate_six_dataset_test


class SumOracle:
    classes_ = np.array([0, 1])
    n_features_in_ = 2

    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        return (arr[:, 0] + arr[:, 1] > 0).astype(int)

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        score = 1.0 / (1.0 + np.exp(-4.0 * (arr[:, 0] + arr[:, 1])))
        return np.column_stack([1.0 - score, score])


class MockProperty:
    name = "income"
    label = ["income"]
    range = [float]


class MockOntology:
    def classes(self):
        return []

    def data_properties(self):
        return [MockProperty()]

    def object_properties(self):
        return []


class CounterfactualEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(5)
        cls.X = rng.uniform(-2, 2, size=(180, 2))
        cls.oracle = SumOracle()
        cls.y = cls.oracle.predict(cls.X)
        cls.names = ["income", "debt"]
        cls.tree = DecisionTreeClassifier(max_depth=4, random_state=5).fit(cls.X, cls.y)

    def constraints(self, **extra):
        config = {"feature_ranges": {"income": (-2, 2), "debt": (-2, 2)}}
        config.update(extra)
        return CounterfactualConstraints.from_data(self.X, self.names, config)

    def test_constraints_preserve_immutable_and_project_range(self):
        constraints = self.constraints(immutable_features=["income"])
        projected = constraints.project(np.array([-0.2, -0.1]), np.array([1.5, 9.0]))
        self.assertAlmostEqual(projected[0], -0.2)
        self.assertAlmostEqual(projected[1], 2.0)
        validation = constraints.validate(np.array([-0.2, -0.1]), projected)
        self.assertTrue(validation["plausible"])

    def test_ontology_validator_is_conservative(self):
        validator = OntologyCFValidator(MockOntology())
        result = validator.validate(
            np.array([1.0, 1.0]), np.array([1.5, 1.0]), self.names,
        )
        self.assertTrue(result["applicable"])
        self.assertTrue(result["consistent"])

    def test_feature_space_adapter_and_controlled_failure(self):
        vector = FeatureSpaceAdapter.adapt(
            np.array([1.0, 2.0]), ["a", "b"], ["b"],
        )
        np.testing.assert_allclose(vector, [2.0])
        with self.assertRaises(FeatureSpaceMismatchError):
            FeatureSpaceAdapter.adapt(np.array([1.0]), ["a"], ["a", "onto_a"])
        enriched = FeatureSpaceAdapter.adapt(
            np.array([1.5]), ["a"], ["a", "onto_a"], reference_target=[1.0, 7.0],
        )
        np.testing.assert_allclose(enriched, [1.5, 7.0])

    def test_all_generators_find_valid_counterfactuals(self):
        engine = CounterfactualEngine(
            self.oracle,
            self.X,
            self.names,
            y_reference=self.y,
            constraints=self.constraints(),
            global_tree=self.tree,
            seed=7,
        )
        instance = np.array([-0.20, -0.10])
        for method in ("DICE", "CLEAR", "COGS", "LORE-LOCAL", "LORE-GLOBAL"):
            with self.subTest(method=method):
                result = engine.generate(
                    instance,
                    desired_class=1,
                    method=method,
                    total_cfs=2,
                    robustness_samples=20,
                )
                self.assertEqual(result["status"], "success", result.get("warnings"))
                self.assertTrue(result["best_candidate"]["metrics"]["validity"])
                self.assertEqual(result["best_candidate"]["prediction"], 1)

    def test_metrics_robustness_and_rst(self):
        constraints = self.constraints()
        robust = evaluate_local_robustness(
            self.oracle,
            np.array([1.0, 1.0]),
            1,
            constraints,
            original=np.array([-1.0, -1.0]),
            n_perturbations=50,
            epsilon=0.01,
            seed=2,
        )
        self.assertGreaterEqual(robust["score"], 0.95)
        engine = CounterfactualEngine(
            self.oracle, self.X, self.names, y_reference=self.y,
            constraints=constraints, seed=4,
        )
        result = engine.generate(
            np.array([-0.2, -0.1]), desired_class=1, method="DICE",
            total_cfs=3, apply_rst=True, robustness_samples=10,
        )
        self.assertIn("rst_feature_scores", result)
        self.assertTrue(result["candidates"])

    def test_interactive_service_and_export(self):
        session = {
            "mlp_oracle": self.oracle,
            "mlp_original": self.oracle,
            "X_train_enc": self.X,
            "y_train_enc": self.y,
            "X_train_original": self.X,
            "y_train_original": self.y,
            "tree_a": self.tree,
            "tree_b": self.tree,
            "c45_tree": self.tree,
            "transformed_feature_names": self.names,
            "feature_names_original": self.names,
            "config": {"class_labels": {0: "no", 1: "yes"}},
        }
        negative_index = int(np.where(self.y == 0)[0][0])
        result = generate_explanation_from_session(
            session,
            {
                "target_model": "Trepan Original",
                "instance_index": negative_index,
                "desired_class": 1,
                "method": "AUTO",
                "total_cfs": 2,
                "robustness_samples": 10,
            },
        )
        self.assertEqual(result["status"], "success")
        with tempfile.TemporaryDirectory() as directory:
            paths = export_counterfactual_result(result, directory, stem="test_cf")
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "success")

    def test_transfer_protocol_strong_and_aggregates(self):
        session = {
            "dataset_name": "toy",
            "mlp_oracle": self.oracle,
            "mlp_original": self.oracle,
            "X_train_enc": self.X,
            "y_train_enc": self.y,
            "X_train_original": self.X,
            "y_train_original": self.y,
            "tree_a": self.oracle,
            "tree_b": self.oracle,
            "transformed_feature_names": self.names,
            "feature_names_original": self.names,
            "tree_a_feature_names": self.names,
            "tree_b_feature_names": self.names,
        }
        result = evaluate_transfer_protocol(
            session,
            methods=("DICE",),
            fraction=0.08,
            seed=3,
            robustness_samples=10,
        )
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["summary"]["valid_mlp_cf_rate"], 0.0)
        self.assertEqual(result["summary"]["no_transfer_rate"], 0.0)
        self.assertGreater(result["summary"]["strong_transfer_rate"], 0.0)
        self.assertIn("DICE", result["summary_by_method"])

        application_result = evaluate_transfer_from_session(
            session,
            {
                "transfer_methods": ("DICE",),
                "fraction": 0.08,
                "seed": 3,
                "robustness_samples": 10,
            },
        )
        self.assertEqual(application_result["protocol"], "P1-P8")
        self.assertEqual(application_result["scope"], "loaded_dataset_only")
        self.assertEqual(application_result["dataset"], "toy")
        self.assertNotIn("datasets", application_result)
        self.assertNotIn("global_summary", application_result)

        with self.assertRaises(ValueError):
            evaluate_transfer_from_session(session, {"datasets": ["a", "b"]})

        artificial = aggregate_transfer_metrics([
            {"category": TransferCategory.STRONG_TRANSFER.value, "cf_valid_mlp": True,
             "trepan_changed": True, "reloaded_changed": True,
             "trepan_aligned": True, "reloaded_aligned": True},
            {"category": TransferCategory.SKIPPED_NO_AGREEMENT.value},
        ])
        self.assertEqual(artificial["initial_agreement_rate"], 0.5)
        self.assertEqual(artificial["strong_transfer_rate"], 1.0)

    def test_transfer_rejects_unrecomputable_enriched_space(self):
        session = {
            "mlp_oracle": self.oracle,
            "mlp_original": self.oracle,
            "X_train_enc": self.X,
            "y_train_enc": self.y,
            "X_train_original": self.X,
            "y_train_original": self.y,
            "tree_a": self.oracle,
            "tree_b": self.oracle,
            "transformed_feature_names": self.names,
            "feature_names_original": self.names,
            "tree_a_feature_names": self.names,
            "tree_b_feature_names": [*self.names, "onto_income"],
            "X_tree_b_reference": np.column_stack([self.X, np.ones(len(self.X))]),
            "ontology_acceptance": {"accepted": False},
        }
        with self.assertRaises(FeatureSpaceMismatchError):
            evaluate_transfer_protocol(
                session, methods=("DICE",), fraction=0.02,
                robustness_samples=10,
            )

    def test_p9_requires_six_datasets(self):
        with self.assertRaises(ValueError):
            evaluate_six_dataset_test({"only_one": {}}, minimum_datasets=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
