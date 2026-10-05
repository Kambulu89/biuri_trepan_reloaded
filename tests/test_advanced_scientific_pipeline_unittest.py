import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from validation.ablation_study import AblationConfig, run_dataset_ablation, summarize_ablation
from core.active_query_engine import ActiveQueryConfig, refine_with_active_queries
from validation.benchmark_ontologies import ensure_builtin_domain_ontologies
from core.biomedical_validation import validate_biomedical_model
from core.metrics_view_model import build_metrics_rows, claim_banner
from core.multiobjective_tree_selector import select_multiobjective_pruned_tree
from core.probabilistic_distillation import DistillationConfig, expand_soft_targets


class AdvancedScientificPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = load_iris(return_X_y=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.25, random_state=9, stratify=y
        )
        cls.oracle = LogisticRegression(max_iter=500).fit(cls.X_train, cls.y_train)

    def test_probability_distillation_preserves_distribution_signal(self):
        probs = self.oracle.predict_proba(self.X_train)
        hard = self.oracle.predict(self.X_train)
        X_out, y_out, weights, audit = expand_soft_targets(
            self.X_train, hard, probs, self.oracle.classes_,
            config=DistillationConfig(temperature=2.0, soft_weight=0.4),
        )
        self.assertGreater(len(X_out), len(self.X_train))
        self.assertEqual(len(X_out), len(y_out))
        self.assertEqual(len(X_out), len(weights))
        self.assertGreater(audit["mean_oracle_entropy"], 0)

    def test_active_and_pareto_selection_never_report_test_queries(self):
        hard = self.oracle.predict(self.X_train)
        tree = DecisionTreeClassifier(max_depth=3, random_state=2).fit(self.X_train, hard)
        w = np.ones(len(self.X_train))
        active, audit, X_a, y_a, w_a = refine_with_active_queries(
            tree, self.oracle, self.X_train, hard, w, self.X_train,
            config=ActiveQueryConfig(iterations=1, budget_per_iteration=12),
        )
        self.assertFalse(audit["test_queried"])
        self.assertEqual(audit["total_queries"], 12)
        selected, selection = select_multiobjective_pruned_tree(
            active, self.oracle, X_a, y_a, w_a, self.X_train,
            y_reference_true=self.y_train,
        )
        self.assertFalse(selection["test_used"])
        self.assertGreaterEqual(len(selection["pareto_candidates"]), 1)
        self.assertGreaterEqual(selected.get_n_leaves(), 1)

    def test_biomedical_guard_detects_patient_leakage(self):
        X, y = load_breast_cancer(return_X_y=True)
        model = LogisticRegression(max_iter=1000).fit(X[:400], y[:400])
        pred = model.predict(X[400:])
        report = validate_biomedical_model(
            y[400:], pred, y_proba=model.predict_proba(X[400:]),
            classes=model.classes_, patient_ids_test=np.arange(400, len(y)),
            patient_ids_train=np.r_[np.arange(400), 401],
            site_ids_test=np.repeat(["A", "B"], [85, 84]),
        )
        self.assertEqual(report["patient_split"], "leakage")
        self.assertIn("patient_overlap", report["clinical_claim_guard"]["blockers"])
        self.assertIn("auroc", report["discrimination"])

    def test_metrics_ui_identifies_oracle_and_claim_guard(self):
        results = {
            "precision": {"trepan_reloaded": {
                "accuracy": .8, "balanced_accuracy": .75,
                "f1_macro": .74, "precision": .79,
            }},
            "fidelity": {"trepan_reloaded": {
                "overall_fidelity": .9,
                "fidelity_reference": "MLP Residual Ontológico",
                "feature_space": "enriched",
            }},
            "scientific_validation": {"strong_superiority_claim_supported": False},
        }
        row = build_metrics_rows(results)[0]
        self.assertEqual(row["oracle"], "MLP Residual Ontológico")
        self.assertEqual(row["feature_space"], "enriched")
        self.assertEqual(claim_banner(results)["status"], "not_supported")

    def test_loaded_dataset_ablation_runs_all_variants(self):
        import importlib.util
        if importlib.util.find_spec("owlready2") is None:
            self.skipTest("dependência opcional owlready2 não instalada")
        dataset = load_iris()
        ontology_paths = ensure_builtin_domain_ontologies(
            Path(__file__).resolve().parents[1] / "data" / "benchmark_ontologies"
        )
        rows = run_dataset_ablation(
            dataset.data,
            dataset.target,
            feature_names=dataset.feature_names,
            ontology_path=ontology_paths["iris"],
            dataset_name="iris_test",
            config=AblationConfig(repeats=1, max_samples=150,
                                  active_iterations=1, active_budget=8),
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["test_role"] == "locked_final_test" for row in rows))
        summary = summarize_ablation(rows)
        self.assertIn("TREPAN Reloaded — sem OWL", summary)
        self.assertIn("TREPAN Reloaded — com OWL", summary)
        self.assertIn("C4.5-Nativo", summary)


if __name__ == "__main__":
    unittest.main()
