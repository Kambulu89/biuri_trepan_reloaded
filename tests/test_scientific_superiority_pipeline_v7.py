import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core.active_query_engine import ActiveQueryConfig, refine_with_active_queries
from core.c45_j48_tree import C45Classifier
from core.canonical_trepan import CanonicalTrepanClassifier
from core.oracle_optimization import (
    OracleGateConfig, calibrate_estimator, compare_ontological_oracle_to_c45,
)
from core.plausible_counterfactuals import generate_plausible_boundary_counterfactuals
from core.probabilistic_distillation import DistillationConfig, expand_hybrid_targets
from core.soft_global_tree import (
    SoftDecisionTreeClassifier, select_global_surrogate,
)
from counterfactuals.cf_tree import build_counterfactual_tree
from counterfactuals.service import _resolve_interactive_context


def _oracle(X, y):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=4)),
    ]).fit(X, y)


def test_calibration_and_c45_gate_are_training_only():
    data = load_iris()
    X, y = data.data, data.target
    base = _oracle(X, y)
    calibrated, audit = calibrate_estimator(
        base, X, y, config=OracleGateConfig(folds=3),
    )
    assert audit["test_used"] is False
    assert calibrated.predict_proba(X[:5]).shape == (5, 3)
    gate = compare_ontological_oracle_to_c45(
        base, C45Classifier(min_samples_leaf=2), X, X, y,
        config=OracleGateConfig(folds=3),
    )
    assert gate["test_used"] is False
    assert gate["claim"] in {
        "superior_on_internal_oof", "noninferior_on_internal_oof",
        "ontological_oracle_rejected",
    }


def test_hybrid_distillation_keeps_true_and_soft_targets():
    X, y = make_classification(n_samples=80, n_features=5, random_state=2)
    oracle = _oracle(X, y)
    X_out, y_out, weights, audit = expand_hybrid_targets(
        X, y, oracle.predict_proba(X), oracle.classes_,
        semantic_confidence=np.linspace(0.2, 1.0, len(X)),
        config=DistillationConfig(max_expansion_factor=3),
    )
    assert len(X_out) > len(X)
    assert np.array_equal(y_out[: len(y)], y)
    assert np.all(weights >= 0)
    assert audit["method"] == "hybrid_true_soft_semantic_distillation"


def test_canonical_trepan_is_best_first_and_supports_m_of_n():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(260, 4))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    model = CanonicalTrepanClassifier(
        max_nodes=15, max_depth=4, max_n=3, beam_width=18,
        complexity_penalty=0.0, label_gain_weight=0.5,
        oracle_gain_weight=0.5, semantic_weight=0.0, stability_weight=0.0,
    ).fit(X, y, y_true=y, semantic_feature_indices=[0, 1])
    assert model.best_first_ is True
    assert model.get_n_leaves() >= 2
    assert any(item["n"] >= 2 for item in model.split_audit_)
    assert np.mean(model.predict(X) == y) >= 0.90


def test_queries_include_c45_disagreement_and_plausible_counterfactuals():
    X, y = make_classification(
        n_samples=160, n_features=6, n_informative=4, random_state=8,
    )
    oracle = _oracle(X, y)
    initial = DecisionTreeClassifier(max_depth=2, random_state=8).fit(X, oracle.predict(X))
    c45 = DecisionTreeClassifier(max_depth=1, random_state=9).fit(X, y)
    cf, cf_weights, cf_audit = generate_plausible_boundary_counterfactuals(oracle, X)
    assert cf_audit["test_used"] is False
    refined, audit, *_ = refine_with_active_queries(
        initial, oracle, X, oracle.predict(X), np.ones(len(X)), X,
        counterfactual_X=cf, counterfactual_weights=cf_weights,
        c45_baseline=c45, y_reference_true=y,
        config=ActiveQueryConfig(iterations=1, budget_per_iteration=20),
    )
    assert refined is not None
    assert "c45_disagreement_rate_pool" in audit["trace"][0]
    assert "mean_plausibility_pool" in audit["trace"][0]


def test_soft_global_tree_selects_without_external_test():
    X, y = make_classification(n_samples=140, n_features=5, random_state=3)
    oracle = _oracle(X, y)
    hard = DecisionTreeClassifier(max_depth=4, random_state=3).fit(X, oracle.predict(X))
    soft = SoftDecisionTreeClassifier(max_depth=4).fit(
        X, oracle.predict(X), teacher_probabilities=oracle.predict_proba(X),
    )
    selected, audit = select_global_surrogate(
        {"hard": hard, "soft": soft}, X, y, oracle,
    )
    assert selected is not None
    assert audit["test_used"] is False
    assert audit["selected"]["candidate"] in {"hard", "soft"}


def test_counterfactual_tree_is_bound_to_each_target_model():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(120, 3))
    model_a = DecisionTreeClassifier(max_depth=2, random_state=1).fit(
        X, (X[:, 0] > 0).astype(int),
    )
    model_b = DecisionTreeClassifier(max_depth=2, random_state=2).fit(
        X, (X[:, 1] > 0).astype(int),
    )
    original = X[0]
    candidates = []
    for vector in (X[1], X[2], X[3], X[4]):
        candidates.append({
            "vector": vector.tolist(),
            "metrics": {"validity": True, "plausibility": 0.8},
        })
    generation = {
        "original_instance": original.tolist(), "feature_names": ["a", "b", "c"],
        "candidates": candidates, "method": "test",
    }
    result_a = build_counterfactual_tree(
        model_a, X, generation, ["a", "b", "c"], model_name="A",
    )
    result_b = build_counterfactual_tree(
        model_b, X, generation, ["a", "b", "c"], model_name="B",
    )
    assert result_a["oracle_fingerprint"] != result_b["oracle_fingerprint"]
    assert result_a["tree_fingerprint"] != result_b["tree_fingerprint"]
    assert result_a["methodology"]["oracle_isolated_per_target"] is True


def test_counterfactual_service_uses_each_tree_as_its_own_oracle():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(90, 3))
    y = (X[:, 0] > 0).astype(int)
    original = DecisionTreeClassifier(max_depth=2, random_state=1).fit(X, y)
    c45 = DecisionTreeClassifier(max_depth=2, random_state=2).fit(
        X, (X[:, 1] > 0).astype(int),
    )
    reloaded = DecisionTreeClassifier(max_depth=2, random_state=3).fit(
        X, (X[:, 2] > 0).astype(int),
    )
    session = {
        "X_train_enc": X, "y_train_enc": y,
        "feature_names_original": ["a", "b", "c"],
        "mlp_original": _oracle(X, y),
        "tree_a": original, "tree_b": reloaded, "c45_tree": c45,
        "ontology_acceptance": {"accepted": False},
    }
    contexts = {
        target: _resolve_interactive_context(
            session, {"target_model": target},
        )
        for target in ("Trepan Original", "C4.5", "Trepan Reloaded")
    }
    assert contexts["Trepan Original"]["oracle"] is original
    assert contexts["C4.5"]["oracle"] is c45
    assert contexts["Trepan Reloaded"]["oracle"] is reloaded
    assert all(item["oracle_is_target_model"] for item in contexts.values())


def _fidelity_hierarchy_ontology(tag):
    from owlready2 import ConstrainedDatatype, DataProperty, Thing, get_ontology

    onto = get_ontology(f"http://test.org/hierarchy_{tag}#")
    with onto:
        class Measurement(Thing): pass
        class Vital(Measurement): pass
        class Cardiac(Vital): pass
        class Lab(Measurement): pass
        class Metabolic(Lab): pass
        for name, domain in (
            ("Heart_Rate", Cardiac), ("Blood_Pressure", Cardiac),
            ("Glucose_Level", Metabolic), ("Insulin_Level", Metabolic),
        ):
            type(name, (DataProperty,), {
                "namespace": onto, "domain": [domain],
                "range": [ConstrainedDatatype(float, min_inclusive=-4.0, max_inclusive=4.0)],
            })
    return onto


def _fidelity_hierarchy_run(seed):
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    from core.controlled_trepan_experiment import (
        ControlledTrepanConfig, fit_controlled_trepan_pair,
    )
    from core.ontology_semantic_graph import OntologySemanticGraph

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(700, 6))
    # Conceito 2-of-3 em que duas condições pertencem ao mesmo grupo OWL.
    votes = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.2) + (X[:, 2] > -0.1)
    y = (votes >= 2).astype(int)
    flip = rng.random(len(y)) < 0.06
    y[flip] = 1 - y[flip]
    # Grafias propositadamente diferentes da OWL: testa o matching tolerante.
    names = ["heart_rate", "BLOOD-PRESSURE", "glucose_level", "Insulin_Level", "noise_a", "noise_b"]
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    oracle = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier((16,), max_iter=800, random_state=seed)),
    ]).fit(X_tr, y_tr)
    graph = OntologySemanticGraph.from_ontology(
        _fidelity_hierarchy_ontology(seed),
        accepted_matches=[
            {"feature": "heart_rate", "entity_name": "http://test.org/x#Heart_Rate"},
            {"feature": "blood_pressure", "entity_name": "Blood_Pressure"},
            {"feature": "GLUCOSE_LEVEL", "entity_name": "glucose_level"},
            {"feature": "insulin_level", "entity_name": "Insulin_Level"},
        ],
    )
    pair = fit_controlled_trepan_pair(
        X_tr, y_tr, oracle=oracle, feature_names=names,
        config=ControlledTrepanConfig(
            max_nodes=15, max_depth=6, min_sample=500, max_queries=3000, random_state=seed,
        ),
        semantic_feature_groups=[graph.primary_group(name) for name in names],
        semantic_relatedness_matrix=np.asarray(graph.feature_relatedness_matrix(names)),
        ontology_graph=graph,
    )
    c45 = C45Classifier(min_samples_leaf=2, random_state=seed).fit(X_tr, y_tr)
    y_oracle = oracle.predict(X_te)
    fidelity = lambda model: float(np.mean(model.predict(X_te) == y_oracle))
    return fidelity(c45), fidelity(pair.original), fidelity(pair.reloaded), pair.reloaded


@pytest.mark.slow
def test_fidelity_hierarchy_c45_original_reloaded():
    runs = [_fidelity_hierarchy_run(seed) for seed in range(4)]
    for *_, reloaded in runs:
        summary = reloaded.semantic_audit_summary_
        assert summary["ontology_active"] is True
        assert summary["ontology_influenced_splits"] > 0
        assert np.any(reloaded.semantic_feature_depths_ > 0)
        # Membership queries projectadas para o domínio OWL antes do MLP.
        assert summary["query_projection_active"] is True
        assert summary["semantic_query_projection"]["enabled"] is True
    summaries = [run[-1].semantic_audit_summary_ for run in runs]
    # Agregado dos seeds: a semântica muda decisões e o EFSR aceita intervenções.
    assert np.mean([s["semantic_decision_impact"] for s in summaries]) > 0.0
    assert sum(s["error_focused_interventions_accepted"] for s in summaries) >= 1
    fidelity_c45, fidelity_trepan_original, fidelity_trepan_reloaded = (
        float(np.mean([run[i] for run in runs])) for i in range(3)
    )
    assert fidelity_c45 < fidelity_trepan_original < fidelity_trepan_reloaded
