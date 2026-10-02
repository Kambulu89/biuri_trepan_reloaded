import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core.active_query_engine import ActiveQueryConfig, refine_with_active_queries
from core.component_ablation import (
    confirmatory_benchmark_gate, required_ablation_variants,
    run_development_ablation,
)
from core.hybrid_oracle import (
    FeatureProjectedOracle, WeightedHybridOracle, cross_fitted_probabilities,
    fit_validation_weighted_hybrid,
)
from core.semantic_utility_gate import SemanticUtilityConfig, SemanticUtilityGate
from core.trepan_reloaded_context import ORACLE_HYBRID, get_trepan_reloaded_context


def test_semantic_utility_gate_accepts_only_useful_nonduplicate_features():
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 80)
    base = rng.normal(size=(160, 2))
    semantic_signal = y + rng.normal(0, 0.08, len(y))
    X = np.c_[base, semantic_signal, base[:, 0], np.ones(len(y))]
    report = SemanticUtilityGate(SemanticUtilityConfig(
        cv_folds=4, min_predictive_gain=0.0, min_selection_stability=0.25,
    )).evaluate(
        X, y, ["x0", "x1", "onto_signal", "onto_duplicate", "onto_constant"],
        ["x0", "x1"],
    )
    assert report["accepted"] is True
    assert report["test_used"] is False
    assert report["selected_semantic_names"] == ["onto_signal"]
    rejected = {row["feature"]: row["reason"] for row in report["rejections"]}
    assert rejected["onto_duplicate"] == "deterministic_duplicate"
    assert rejected["onto_constant"] == "near_constant"


def test_semantic_gate_rejects_contaminated_ontology_without_scoring_test():
    X, y = make_classification(
        n_samples=80, n_features=3, n_redundant=0, random_state=3,
    )
    report = SemanticUtilityGate().evaluate(
        np.c_[X, X[:, 0] ** 2], y,
        ["a", "b", "c", "onto_d"], ["a", "b", "c"],
        quality_report={"accepted": False, "status": "CONTAMINATED_ONTOLOGY"},
    )
    assert report["accepted"] is False
    assert report["status"] in {"REJECT_LEAKAGE", "REJECT_NO_INFORMATIONAL_GAIN"}
    assert report["test_used"] is False


def test_hybrid_oracle_uses_validation_weights_and_exact_oof_probabilities():
    X, y = make_classification(
        n_samples=120, n_features=5, n_informative=4, n_redundant=0,
        random_state=11,
    )
    original = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
        X[:, :3], y,
    )
    projected = FeatureProjectedOracle(original, [0, 1, 2], X.shape[1])
    ontology = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    weights, audit = fit_validation_weighted_hybrid(
        {"original": projected, "ontology": ontology}, X, y,
    )
    assert np.isclose(sum(weights.values()), 1.0)
    assert audit["test_used"] is False
    oof, oof_audit = cross_fitted_probabilities(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
        cv_folds=4,
    )
    assert oof.shape == (len(X), 2)
    assert oof_audit["coverage_min"] == oof_audit["coverage_max"] == 1


def test_hybrid_context_carries_oof_teacher_probabilities():
    X, y = make_classification(n_samples=60, n_features=4, random_state=13)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    oof, _ = cross_fitted_probabilities(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
        cv_folds=3,
    )
    oracle = WeightedHybridOracle(
        {"ontology": model}, {"ontology": 1.0}, model.classes_, X.shape[1],
        training_oof_probabilities=oof,
    )
    split = {"X_train": X, "X_test": X[:10], "y_train": y, "y_test": y[:10]}
    ctx = get_trepan_reloaded_context(
        ontology_enabled=True, ontology_acceptance={"accepted": True},
        selected_oracle_label=ORACLE_HYBRID, selected_oracle=oracle,
        eval_split_enriched=split, feature_names_enriched=[f"f{i}" for i in range(4)],
    )
    assert ctx["feature_space"] == "enriched"
    assert np.array_equal(ctx["oof_teacher_probabilities"], oof)


def test_active_queries_report_real_label_and_semantic_strategies():
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=4, n_redundant=0,
        random_state=19,
    )
    oracle = LogisticRegression(max_iter=1000).fit(X, y)
    tree = DecisionTreeClassifier(max_depth=2, random_state=19).fit(X, oracle.predict(X))
    _, audit, *_ = refine_with_active_queries(
        tree, oracle, X, oracle.predict(X), np.ones(len(X)), X,
        original_oracle=oracle, ontological_oracle=oracle,
        semantic_feature_indices=[4], y_reference_true=y,
        config=ActiveQueryConfig(
            iterations=1, budget_per_iteration=8, pool_multiplier=2,
            random_state=19,
        ),
    )
    row = audit["trace"][0]
    assert "real_label_error_proxy_rate" in row
    assert "mean_minority_priority" in row
    assert "semantic_gap_rate" in row
    assert audit["test_queried"] is False


def test_development_ablation_reuses_folds_and_confirmatory_gate_blocks_fake_owl():
    X, y = make_classification(n_samples=60, n_features=4, random_state=23)
    seen = {}

    def evaluator(variant, fit_idx, val_idx, split_id):
        seen.setdefault(split_id, []).append((fit_idx.tolist(), val_idx.tolist()))
        return {
            "balanced_accuracy": 0.5, "macro_f1": 0.5,
            "minority_recall": 0.5, "accuracy": 0.5,
        }

    result = run_development_ablation(X, y, evaluator, folds=3, repeats=1)
    assert result["external_test_used"] is False
    assert result["same_folds_for_all_variants"] is True
    expected = len(required_ablation_variants())
    assert all(len(rows) == expected for rows in seen.values())
    for rows in seen.values():
        assert len({tuple(item[0]) for item in rows}) == 1
        assert len({tuple(item[1]) for item in rows}) == 1
    gate = confirmatory_benchmark_gate(
        tests_passed=True,
        ontology_catalog=[{
            "independent": False, "license": "internal", "sha256": "abc",
            "quality_status": "VALID_DOMAIN_ONTOLOGY",
        }],
    )
    assert gate["approved"] is False
    assert gate["reason"] == "no_independent_versioned_domain_ontology"


def test_ablation_rejects_any_evaluator_that_exposes_test_metrics():
    X, y = make_classification(
        n_samples=40, n_features=3, n_redundant=0, random_state=29,
    )

    def invalid(variant, fit_idx, val_idx, split_id):
        return {
            "balanced_accuracy": .5, "macro_f1": .5, "minority_recall": .5,
            "accuracy": .5, "test_accuracy": .9,
        }

    with pytest.raises(ValueError, match="teste externo"):
        run_development_ablation(X, y, invalid, folds=2, repeats=1)
