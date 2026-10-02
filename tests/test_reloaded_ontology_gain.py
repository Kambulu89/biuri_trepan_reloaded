"""Gain_Reloaded, matching OWL tolerante e amostragem restrita pela ontologia."""
import logging

import numpy as np
import pytest
from owlready2 import ConstrainedDatatype, DataProperty, Thing, get_ontology

from core.ontology_semantic_graph import OntologySemanticGraph, normalize_ontology_name
from core.probabilistic_distillation import constrain_synthetic_samples
from core.semantic_utility_gate import _resolve_names
from core.trepan_original import Literal, MofNTest, TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


def _ontology(tag):
    onto = get_ontology(f"http://test.org/{tag}#")
    with onto:
        class Measurement(Thing): pass
        class Vital(Measurement): pass
        class Cardiac(Vital): pass
        class Heart_Rate(DataProperty):
            domain = [Cardiac]
            range = [ConstrainedDatatype(float, min_inclusive=0.0, max_inclusive=250.0)]
        class Weight(DataProperty):
            domain = [Measurement]
    return onto


def _graph(tag):
    return OntologySemanticGraph.from_ontology(_ontology(tag), accepted_matches=[
        {"feature": "heart_rate", "entity_name": "http://test.org/x#HEART_RATE"},
        {"feature": "weight", "entity_name": "weight"},
    ])


def test_normalize_ignores_case_underscores_and_uri_prefixes():
    expected = "meanradius"
    for raw in ("http://x.org/o#Mean_Radius", "ex:meanRadius", "MEAN_RADIUS",
                "http://x.org/onto/mean-radius"):
        assert normalize_ontology_name(raw) == expected


def test_graph_matching_is_tolerant_and_depth_follows_hierarchy():
    graph = _graph("depth")
    assert graph.is_active
    # IRI completo e caixa diferente ancoram no nó real do grafo.
    assert graph.feature_to_entity["heart_rate"] == "Heart_Rate"
    assert graph.entity_for_feature("Heart-Rate") == "Heart_Rate"
    # Heart_Rate -> Cardiac -> Vital -> Measurement (3); Weight -> Measurement (1).
    assert graph.get_feature_depth("HEART_RATE") == pytest.approx(1.0)
    assert graph.get_feature_depth("weight") == pytest.approx(1.0 / 3.0)
    assert graph.get_feature_depth("unmapped_column") == 0.0
    assert graph.feature_bounds("heart_rate") == (0.0, 250.0)


def test_failed_ontology_load_falls_back_to_original(tmp_path, caplog):
    broken = tmp_path / "broken.owl"
    broken.write_text("<rdf:RDF this is not valid xml", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        graph = OntologySemanticGraph.load(str(broken))
    assert graph.is_active is False
    assert graph.load_error
    assert graph.get_feature_depth("anything") == 0.0
    assert "Fallback para Trepan Original ativo" in caplog.text


def test_constrained_sampling_filters_and_fills_transparently():
    graph = _graph("sampling")
    names = ["heart_rate", "weight"]
    samples = np.array([[60.0, 1.0], [-5.0, 2.0], [300.0, 3.0], [120.0, 4.0]])
    kept, audit = constrain_synthetic_samples(
        samples, min_samples=2, ontology_graph=graph, feature_names=names,
    )
    assert kept.tolist() == [[60.0, 1.0], [120.0, 4.0]]
    assert audit["rejected"] == 2 and audit["fallback_filled"] == 0
    # Poucas válidas: completa com amostras estatísticas pela ordem original.
    kept, audit = constrain_synthetic_samples(
        samples, min_samples=3, ontology_graph=graph, feature_names=names,
    )
    assert kept.tolist() == [[60.0, 1.0], [120.0, 4.0], [-5.0, 2.0]]
    assert audit["fallback_filled"] == 1
    # OWL inactiva: amostras intactas.
    inactive = OntologySemanticGraph(load_error="boom")
    kept, audit = constrain_synthetic_samples(
        samples, min_samples=3, ontology_graph=inactive, feature_names=names,
    )
    assert np.array_equal(kept, samples) and audit["ontology_active"] is False


def test_utility_gate_resolves_names_tolerantly_but_not_ambiguously():
    resolved = _resolve_names(["Mean_Radius", "texture"], ["mean_radius", "TEXTURE", "onto_x"])
    assert resolved == {"Mean_Radius": "mean_radius", "texture": "TEXTURE"}
    assert _resolve_names(["a_b"], ["ab", "A_B_"])["a_b"] is None


@pytest.mark.parametrize("criterion", ["normalized_information_gain", "information_gain", "gain_ratio"])
def test_gain_reloaded_formula_matches_specification(criterion):
    model = TrepanReloadedClassifier(alpha=0.15, beta=0.10, gain_criterion=criterion)
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = 3
    model.semantic_feature_weights_ = np.ones(3)
    model.semantic_feature_groups_ = [None, None, None]
    model.semantic_relatedness_matrix_ = None
    model.semantic_feature_depths_ = np.array([0.5, 0.0, 1.0])
    model._current_error_profile_ = {"feature_error_scores": [0.2, 0.0, 0.8]}
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    mask = np.array([False, False, False, True, True, True, True, False])
    test = MofNTest(1, (Literal(0, 0.0, True),))
    ig = TrepanOriginalClassifier._split_selection_score(model, y, mask, test)
    if criterion == "gain_ratio":
        base = model._gain_ratio(ig, mask)
    elif criterion == "normalized_information_gain":
        # IG / H(y): H(y)=1 bit aqui, mas a normalização é pela entropia do nó.
        p = y.mean()
        base = ig / float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    else:
        base = ig
    expected = base + 0.15 * 0.5 + 0.10 * 0.2
    assert model._split_selection_score(y, mask, test) == pytest.approx(expected)
    # Sem informação, os bónus ontológicos não criam um split.
    assert model._split_selection_score(y, np.zeros(8, dtype=bool), test) == 0.0
    # alpha=beta=0 recupera exactamente o Information Gain do Original.
    model.alpha = model.beta = 0.0
    assert model._split_selection_score(y, mask, test) == pytest.approx(ig)


def test_inactive_ontology_reloaded_matches_original_tree():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(160, 4))
    y = ((X[:, 0] > 0) & (X[:, 1] > -0.2)).astype(int)
    kwargs = dict(max_nodes=7, max_depth=3, min_sample=200, max_queries=400, random_state=3)
    original = TrepanOriginalClassifier(**kwargs).fit(X, y)
    reloaded = TrepanReloadedClassifier(**kwargs).fit(
        X, y, ontology_graph=OntologySemanticGraph(load_error="missing"),
    )
    assert reloaded.ontology_active_ is False
    assert np.array_equal(original.predict(X), reloaded.predict(X))


def test_default_gain_parameters():
    model = TrepanReloadedClassifier()
    assert (model.alpha, model.beta) == (0.35, 0.20)
    assert model.gain_criterion == "normalized_information_gain"
    assert model.semantic_query_projection is True
    assert model.error_focus_fidelity_tolerance == pytest.approx(0.015)


def test_normalized_information_gain_is_bounded_by_node_entropy():
    model = TrepanReloadedClassifier()
    model.classes_ = np.array([0, 1])
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([False, False, False, True, True, True])
    ig = TrepanOriginalClassifier._split_selection_score(model, y, perfect, None)
    assert model._normalized_information_gain(ig, y) == pytest.approx(1.0)
    assert model._normalized_information_gain(0.0, np.zeros(4)) == 0.0


def test_domain_constraints_project_and_skip_unit_mismatch():
    graph = _graph("domain")
    names = ["heart_rate", "weight"]
    in_units = np.column_stack([np.linspace(40, 200, 50), np.linspace(1, 2, 50)])
    constraints = graph.domain_constraints(names, in_units)
    assert constraints.is_active
    projected = constraints.project(np.array([[-10.0, 5.0], [300.0, 6.0]]))
    assert projected.tolist() == [[0.0, 5.0], [250.0, 6.0]]
    assert constraints.sample_validity_mask(projected).all()
    # Coluna escalada (z-scores) não está nas unidades da OWL: não é recortada.
    scaled = np.column_stack([np.linspace(-2, 2, 50), np.linspace(1, 2, 50)])
    assert not graph.domain_constraints(names, scaled).is_active


def _efsr_model(tolerance):
    from core.trepan_original import Literal as L
    model = TrepanReloadedClassifier(
        error_focus_fidelity_tolerance=tolerance, error_focus_min_local_fidelity_gain=0.002,
    )
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = 2
    model.min_samples_leaf = 1
    model.semantic_feature_weights_ = np.ones(2)
    model.semantic_feature_groups_ = [None, None]
    model.semantic_relatedness_matrix_ = None
    # Só a feature 1 tem suporte ontológico: o teste semântico é mais coerente.
    model.semantic_feature_depths_ = np.array([0.0, 1.0])
    model._current_error_profile_ = {"eligible_for_semantic_refinement": True}
    return model, MofNTest(1, (L(0, 0.5, True),)), MofNTest(1, (L(1, 0.5, True),))


def test_efsr_accepts_small_fidelity_loss_only_with_coherence_gain():
    rng = np.random.default_rng(0)
    X = rng.random((400, 2))
    y = (X[:, 0] > 0.5).astype(int)
    # Feature 1 replica a 0 excepto em 4/400 linhas: perda local de 1% < epsilon.
    X[:, 1] = X[:, 0]
    X[:4, 1] = 1.0 - X[:4, 0]
    model, data_test, semantic_test = _efsr_model(0.015)
    result = model._evaluate_error_focused_intervention(
        X, y, data_test=data_test, semantic_test=semantic_test, disagreement_rate=0.2,
    )
    assert result["local_fidelity_gain"] == pytest.approx(-0.01)
    assert result["semantic_coherence_gain"] > 0
    assert result["accepted"] and result["accepted_within_tolerance"]
    # Com tolerância 0 recupera o gate estrito.
    strict, data_test, semantic_test = _efsr_model(0.0)
    result = strict._evaluate_error_focused_intervention(
        X, y, data_test=data_test, semantic_test=semantic_test, disagreement_rate=0.2,
    )
    assert not result["accepted"]
    # Sem ganho de coerência, nem uma perda pequena é tolerada.
    model, data_test, semantic_test = _efsr_model(0.015)
    model.semantic_feature_depths_ = np.array([1.0, 0.0])
    result = model._evaluate_error_focused_intervention(
        X, y, data_test=data_test, semantic_test=semantic_test, disagreement_rate=0.2,
    )
    assert not result["accepted"]
