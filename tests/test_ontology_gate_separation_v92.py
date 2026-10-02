from __future__ import annotations

import numpy as np

from core.ontology_quality import OntologyQualityGate
from core.semantic_utility_gate import SemanticUtilityConfig, SemanticUtilityGate
from core.trepan_reloaded_extractor import TrepanReloadedExtractor


class DummyDataPropertyEntity:
    def __init__(self, name, labels=()):
        self.name = name
        self.label = list(labels)
        self.prefLabel = []
        self.altLabel = []


class DummyOntology:
    def __init__(self, properties):
        self._properties = list(properties)

    def classes(self):
        return []

    def data_properties(self):
        return list(self._properties)

    def object_properties(self):
        return []

    def individuals(self):
        return []


def _valid_quality_report(total=2, mapped=2):
    return {
        "accepted": True,
        "status": "VALID_DOMAIN_ONTOLOGY",
        "issues": [],
        "warnings": [],
        "metrics": {
            "feature_coverage": mapped / max(total, 1),
            "mapped_features": mapped,
            "total_features": total,
            "unique_mapped_entities": mapped,
            "generic_entity_ratio": 0.0,
        },
        "matches": [
            {
                "feature": f"f{i}",
                "entity_name": f"Entity{i}",
                "score": 1.0,
                "accepted": True,
                "reason": "accepted",
            }
            for i in range(mapped)
        ],
        "abox": {"accepted": True},
        "reasoner": {"executed": True, "consistent": True, "unsatisfiable_classes": []},
    }


def test_matching_is_token_order_agnostic_without_dataset_specific_aliases():
    ontology = DummyOntology([
        DummyDataPropertyEntity("WorstRadius", labels=["Worst Radius"]),
        DummyDataPropertyEntity("TextureMean", labels=["mean texture"]),
    ])
    gate = OntologyQualityGate(min_feature_coverage=1.0)
    report = gate.evaluate(
        ["radius_worst", "texture_mean"],
        ontology,
        reasoner_report={"executed": True, "consistent": True, "unsatisfiable_classes": []},
    )

    assert report.accepted is True
    assert report.status == "VALID_DOMAIN_ONTOLOGY"
    assert report.metrics["mapped_features"] == 2
    assert report.metrics["feature_coverage"] == 1.0
    assert all(row["accepted"] for row in report.matches)


def test_quality_rejection_is_not_misreported_as_no_informational_gain():
    X = np.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [1.3, 1.3],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    quality = _valid_quality_report(total=2, mapped=0)
    quality.update({
        "accepted": False,
        "status": "INSUFFICIENT_SCHEMA_COVERAGE",
        "issues": ["Cobertura de matching insuficiente."],
    })

    report = SemanticUtilityGate(SemanticUtilityConfig(cv_folds=2)).evaluate(
        X, y, ["base", "onto_candidate"], ["base"], quality_report=quality,
    )

    assert report["accepted"] is False
    assert report["status"] == "REJECT_ONTOLOGY_MATCHING"
    assert report["reason"] == "ontology_quality_gate_rejected"
    assert report["ontology_quality"]["accepted"] is False


def test_valid_ontology_without_derived_features_has_specific_status():
    X = np.array([[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    report = SemanticUtilityGate(SemanticUtilityConfig(cv_folds=2)).evaluate(
        X, y, ["base"], ["base"], quality_report=_valid_quality_report(total=1, mapped=1),
    )
    assert report["accepted"] is False
    assert report["status"] == "REJECT_NO_DERIVED_SEMANTIC_FEATURES"
    assert report["ontology_quality"]["accepted"] is True


def test_complexity_penalty_is_relative_to_full_selected_model_not_semantic_pool():
    gate = SemanticUtilityGate()
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    metrics = gate._metrics(
        y, pred,
        semantic_count=1,
        total_semantic=3,
        base_feature_count=30,
    )
    assert np.isclose(metrics["complexity_ratio"], 1.0 / 31.0)
    assert metrics["complexity_ratio"] < 0.05


def test_positive_raw_gain_blocked_by_penalty_gets_honest_status():
    gate = SemanticUtilityGate(SemanticUtilityConfig(min_predictive_gain=0.002))
    status, reason = gate._decision_status(
        accepted=False,
        complexity_excess=False,
        utility_gain=-0.001,
        balanced_accuracy_gain=0.004,
        macro_f1_gain=0.003,
        complexity_penalty_delta=0.006,
    )
    assert status == "REJECT_GAIN_BELOW_COMPLEXITY_COST"
    assert reason == "positive_predictive_gain_below_complexity_cost"


def test_mapping_statistics_prefer_quality_gate_report_over_legacy_empty_state():
    extractor = TrepanReloadedExtractor(ontology=object())
    extractor.feature_semantics = {}
    extractor.mapping_scores = {}
    extractor.unmapped_features = []
    extractor.ontology_quality_report = _valid_quality_report(total=5, mapped=5)

    stats = extractor.get_mapping_statistics()

    assert stats["features"]["total"] == 5
    assert stats["features"]["mapped"] == 5
    assert stats["features"]["unmapped"] == 0
    assert stats["features"]["source"] == "ontology_quality_gate"
    assert np.isclose(stats["features"]["avg_score"], 1.0)


def test_stage_status_separates_structural_owl_from_mlp_feature_gate():
    from core.ontology_stage_status import build_ontology_stage_status

    status = build_ontology_stage_status(
        _valid_quality_report(total=4, mapped=4),
        {
            "accepted": False,
            "ontology_feature_gate_accepted": False,
            "semantic_utility_status": "REJECT_GAIN_BELOW_COMPLEXITY_COST",
            "teacher_has_ontology": False,
        },
    )

    assert status["ontology_quality_accepted"] is True
    assert status["ontology_structural_available"] is True
    assert status["ontology_feature_engineering_accepted"] is False
    assert status["ontological_teacher_accepted"] is False
    assert status["trepan_semantic_use_allowed"] is True


def test_presenter_uses_quality_mapping_and_na_for_uncomputed_metrics():
    from gui.ontology_status_presenter import build_ontology_status_text

    text = build_ontology_status_text(
        _valid_quality_report(total=4, mapped=4),
        {
            "accepted": False,
            "ontology_feature_gate_accepted": False,
            "semantic_utility_status": "REJECT_GAIN_BELOW_COMPLEXITY_COST",
            "teacher_has_ontology": False,
            # métricas intencionalmente ausentes: devem ser N/A, nunca 0.0%
        },
        selected_oracle_label="MLP Original",
    )

    assert "Mapeamento OWL: 4/4 (100,0%)" in text
    assert "Ontologia estrutural: VÁLIDA" in text
    assert "Enriquecimento de features para o MLP: NÃO ACEITE" in text
    assert "Uso semântico no TREPAN Reloaded: DISPONÍVEL" in text
    assert "Accuracy MLP Original: N/A" in text
    assert "Accuracy MLP Original: 0,0%" not in text
    assert "Balanced Accuracy MLP Original: 0,0%" not in text



def test_reloaded_context_keeps_structural_owl_when_teacher_feature_gate_rejects():
    from core.trepan_reloaded_context import get_trepan_reloaded_context

    class Oracle:
        n_features_in_ = 2
        def predict(self, X):
            X = np.asarray(X)
            return (X[:, 0] > 0).astype(int)

    X_train = np.array([[-1.0, 0.0], [-0.5, 1.0], [0.5, 0.0], [1.0, 1.0]])
    X_test = np.array([[-0.7, 0.0], [0.7, 1.0]])
    y_train = np.array([0, 0, 1, 1])
    y_test = np.array([0, 1])
    split = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    oracle = Oracle()

    ctx = get_trepan_reloaded_context(
        ontology_enabled=True,
        ontology_acceptance={
            "accepted": False,
            "ontology_structural_available": True,
            "ontology_feature_gate_accepted": False,
        },
        selected_oracle_label="MLP Original",
        selected_oracle=oracle,
        mlp_original=oracle,
        eval_split_original=split,
        feature_names_original=["a", "b"],
    )

    assert ctx["oracle_key"] == "mlp_original"
    assert ctx["feature_space"] == "original"
    assert ctx["ontology_accepted"] is False
    assert ctx["ontology_structural_available"] is True
    assert ctx["use_ontology_semantic_pipeline"] is True


def test_historical_reloaded_uses_structural_mapping_in_original_feature_space():
    from core.trepan_reloaded_historical import TrepanReloadedClassifier

    class Oracle:
        n_features_in_ = 2
        classes_ = np.array([0, 1])
        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return (0.3 * X[:, 0] + X[:, 1] > 0).astype(int)
        def predict_proba(self, X):
            p = self.predict(X)
            return np.column_stack([1 - p, p]).astype(float)

    rng = np.random.default_rng(44)
    X = rng.normal(size=(140, 2))
    y = Oracle().predict(X)
    split = {
        "X_train": X[:100], "X_test": X[100:],
        "y_train": y[:100], "y_test": y[100:],
    }
    extractor = TrepanReloadedExtractor(ontology=object())
    extractor.ontology_quality_report = _valid_quality_report(total=2, mapped=2)
    extractor.register_quality_matches(["a", "b"], extractor.ontology_quality_report["matches"])
    extractor._training_limits = {
        "historical_min_sample": 120,
        "historical_max_queries": 500,
        "canonical_max_nodes": 7,
        "canonical_max_depth": 3,
        "canonical_m_of_n_max_n": 2,
        "random_state": 44,
    }
    # Evita reasoner neste teste unitário; a validade estrutural já é injectada acima.
    extractor._validate_semantic_coherence = lambda *a, **k: {
        "issues": [], "warnings": [], "total_nodes": 0, "validated_nodes": 0,
    }
    ctx = {
        "oracle": Oracle(),
        "oracle_name": "MLP Original",
        "X_train": split["X_train"], "X_test": split["X_test"],
        "y_train": split["y_train"], "y_test": split["y_test"],
        "feature_names": ["a", "b"],
        "feature_space": "original",
        "oracle_bundle": None,
        "mlp_original_ref": Oracle(),
        "mlp_model_onto": None,
        "use_ontology_semantic_pipeline": True,
    }

    extractor._extract_tree_in_oracle_space(ctx, ["não", "sim"], sample_size=400)

    assert isinstance(extractor.explainer_tree, TrepanReloadedClassifier)
    assert extractor.last_audit["ontology_structural_only"] is True
    assert extractor.last_audit["final_tree_family"] == "historical_trepan_reloaded"


def test_training_preset_uses_same_historical_budget_for_original_and_reloaded():
    from types import SimpleNamespace

    preset = SimpleNamespace(
        key="balanced",
        reloaded_fidelity_target=0.95,
        reloaded_fidelity_early_stop=0.97,
        reloaded_max_time_seconds=120,
        reloaded_sample_size=9999,
        trepan_sample_size=1200,
        trepan_max_queries=1500,
        trepan_max_nodes=17,
        trepan_max_depth=6,
        trepan_min_samples_leaf=3,
        hybrid_label_weight=0.6,
        hybrid_teacher_weight=0.3,
        hybrid_semantic_weight=0.1,
        canonical_trepan_enabled=True,
        canonical_m_of_n_max_n=3,
    )
    extractor = TrepanReloadedExtractor(ontology=object())
    extractor.apply_training_preset(preset)

    assert extractor._training_limits["sample_size"] == 1200
    assert extractor._training_limits["historical_max_queries"] == 1500
    assert extractor._training_limits["canonical_max_nodes"] == 17
    assert extractor._training_limits["canonical_max_depth"] == 6
    assert extractor._training_limits["historical_min_samples_leaf"] == 3
