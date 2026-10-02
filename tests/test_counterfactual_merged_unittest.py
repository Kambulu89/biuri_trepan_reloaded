"""Testes sem pytest/PyQt para a integração contrafactual entre versões."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from counterfactuals.cf_tree import build_counterfactual_tree
from counterfactuals.evaluation import evaluate_generation_result
from counterfactuals.export import export_counterfactual_result
from counterfactuals.global_rules import analyse_global_rules


class MergedCounterfactualCapabilitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_iris(return_X_y=True)
        cls.names = [f"feature_{index}" for index in range(cls.X.shape[1])]
        cls.oracle = LogisticRegression(max_iter=500, random_state=3).fit(cls.X, cls.y)
        cls.tree = DecisionTreeClassifier(max_depth=4, random_state=3).fit(
            cls.X, cls.oracle.predict(cls.X)
        )
        factual = cls.oracle.predict(cls.X[0:1])[0]
        candidates = []
        for row in cls.X:
            prediction = cls.oracle.predict(row.reshape(1, -1))[0]
            if prediction != factual:
                candidates.append({
                    "method": "DICE", "prediction": int(prediction),
                    "vector": row.tolist(), "changes": [],
                    "metrics": {"validity": True},
                })
            if len(candidates) == 4:
                break
        cls.generation = {
            "method": "DICE", "target_model": "Trepan Original",
            "dataset": "iris.arff", "original_instance": cls.X[0].tolist(),
            "feature_names": cls.names, "candidates": candidates,
        }

    def test_global_rules_are_complete_and_auditable(self):
        result = analyse_global_rules(
            self.tree, self.oracle, self.X, self.names, dataset_name="iris.arff"
        )
        self.assertEqual(result["scope"], "loaded_dataset_only")
        self.assertTrue(result["counterfactual_rules"])
        self.assertTrue(all(row["n_changes"] > 0 for row in result["counterfactual_rules"]))
        self.assertTrue(result["clusters"])
        self.assertTrue(result["for_audit"])
        self.assertTrue(result["for_fairness"])
        self.assertTrue(result["for_owl"])

    def test_formal_metrics_and_cf_tree(self):
        formal = evaluate_generation_result(
            self.generation, self.oracle, self.X, self.y, seed=4
        )
        self.assertEqual(len(formal["per_candidate"]), 4)
        self.assertIsNotNone(formal["per_candidate"][0]["probability_margin"])
        feature_before = self.tree.tree_.feature.copy()
        threshold_before = self.tree.tree_.threshold.copy()
        result = build_counterfactual_tree(
            self.oracle, self.X, self.generation, self.names,
            original_tree=self.tree, max_depth=3, dataset_name="iris.arff",
        )
        self.assertTrue(result["tree_rules"])
        self.assertEqual(result["aggregate_metrics"]["n_valid_counterfactuals"], 4)
        np.testing.assert_array_equal(feature_before, self.tree.tree_.feature)
        np.testing.assert_array_equal(threshold_before, self.tree.tree_.threshold)

    def test_runtime_tree_is_not_serialized(self):
        result = build_counterfactual_tree(
            self.oracle, self.X, self.generation, self.names, original_tree=self.tree
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_counterfactual_result(result, directory, stem="tree")
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        self.assertNotIn("_runtime_tree_model", payload)
        self.assertEqual(payload["result_type"], "counterfactual_tree")


if __name__ == "__main__":
    unittest.main()
