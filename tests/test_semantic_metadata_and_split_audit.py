"""Metadata semântica, relatedness configurável, dupla contagem e auditoria por split do TREPAN Reloaded."""
import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_processor import OntologyProcessor
from core.ontology_quality import OntologyQualityGate
from core.ontology_semantic_graph import OntologySemanticGraph
from core.semantic_metadata import (
    RelatednessConfig, audit_double_counting, build_relatedness_matrix,
    build_semantic_metadata, semantic_split_weights,
)


def _setup(noise=0):
    onto = owlready2.World().get_ontology("http://test.org/meta.owl")
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        Morph = type("Morphology", (owlready2.Thing,), {})
        Other = type("Background", (owlready2.Thing,), {})
        for role in ("mean", "error", "worst"):
            p = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,),
                     {"range": [float], "domain": [Morph]})
            p.statisticRole = [role]; p.measurementFamily = ["Size"]
        type("hasBackgroundLevel", (owlready2.DataProperty,), {"range": [float], "domain": [Other]})
    rng = np.random.default_rng(0)
    mean = rng.uniform(5, 15, 120)
    X = pd.DataFrame({"hasSizeMean": mean, "hasSizeError": rng.uniform(.1, 1, 120),
                      "hasSizeWorst": mean * rng.uniform(1.1, 1.8, 120),
                      "hasBackgroundLevel": rng.normal(size=120)})
    q = OntologyQualityGate().evaluate(list(X.columns), onto, require_reasoner=False)
    proc = OntologyProcessor(onto).fit(X, accepted_matches=[m for m in q.matches if m["accepted"]], log=False)
    graph = OntologySemanticGraph.from_ontology(onto, accepted_matches=[m for m in q.matches if m["accepted"]])
    names = proc.output_features_
    meta = build_semantic_metadata(names, proc, graph, [m for m in q.matches if m["accepted"]])
    return onto, X, q, proc, graph, names, meta


def test_metadata_has_all_required_fields_and_provenance():
    *_, names, meta = _setup()
    assert [m["feature_name"] for m in meta] == names
    need = {"feature_name", "origin", "ontology_entity", "ontology_class", "ontology_family",
            "statistic_role", "source_features", "reasoner_inferred", "semantic_depth", "semantic_relations"}
    assert all(need <= set(m) for m in meta)
    derived = [m for m in meta if m["origin"] == "ontology"]
    original = [m for m in meta if m["origin"] == "original"]
    assert derived and len(original) == 4
    rel = next(m for m in derived if m["feature_name"].endswith("_relative_worst_delta"))
    assert set(rel["source_features"]) == {"hasSizeWorst", "hasSizeMean"}
    assert rel["ontology_family"] == "Size" and rel["statistic_role"] == "worst/mean"
    assert rel["knowledge_source"] == "ontology"
    # as originais herdam a família que as derivadas declaram
    assert next(m for m in original if m["feature_name"] == "hasSizeMean")["ontology_family"] == "Size"
    assert next(m for m in original if m["feature_name"] == "hasBackgroundLevel")["ontology_family"] is None


def test_relatedness_follows_documented_rules_and_is_symmetric():
    onto, X, q, proc, graph, names, meta = _setup()
    cfg = RelatednessConfig()
    R = build_relatedness_matrix(meta, graph, cfg)
    assert R.shape == (len(names),) * 2 and np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0)
    assert R.min() >= 0.0 and R.max() <= 1.0
    idx = {n: i for i, n in enumerate(names)}
    derived = "onto_Size_relative_worst_delta"
    assert R[idx[derived], idx["hasSizeWorst"]] == cfg.derived_to_source          # derivada <-> fonte
    assert R[idx["hasSizeMean"], idx["hasSizeError"]] >= cfg.same_family          # mesma família
    assert R[idx["hasSizeMean"], idx["hasBackgroundLevel"]] < cfg.same_superclass  # sem relação forte
    assert R[idx[derived], idx["hasBackgroundLevel"]] < cfg.same_superclass


def test_relatedness_values_are_configurable():
    *_, graph, names, meta = _setup()[0:0] or (None, None, None, None, None, None, None)
    onto, X, q, proc, graph, names, meta = _setup()
    low = build_relatedness_matrix(meta, graph, RelatednessConfig(derived_to_source=0.6, same_family=0.5, same_superclass=0.3))
    idx = {n: i for i, n in enumerate(names)}
    assert low[idx["onto_Size_relative_worst_delta"], idx["hasSizeWorst"]] == 0.6
    with pytest.raises(ValueError):
        RelatednessConfig(derived_to_source=0.3, same_family=0.8).validate()


def test_no_double_counting_derived_weight_never_exceeds_sources():
    *_, meta = _setup()
    w = semantic_split_weights(meta)
    assert audit_double_counting(meta, w) == []
    by = {m["feature_name"]: i for i, m in enumerate(meta)}
    for m in meta:
        if m["origin"] == "ontology":
            assert w[by[m["feature_name"]]] <= max(w[by[s]] for s in m["source_features"]) + 1e-12
    # um peso inflacionado para uma onto_* seria detetado
    bad = w.copy(); bad[by["onto_Size_family_contrast"]] = 2.5
    flagged = audit_double_counting(meta, bad)
    assert [f["feature"] for f in flagged] == ["onto_Size_family_contrast"]


def test_split_audit_records_selected_feature_entity_and_reason():
    from core.trepan_reloaded_historical import TrepanReloadedClassifier
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 4))
    y = (1.2 * X[:, 0] + 0.9 * X[:, 1] + rng.normal(0, .3, 500) > 0).astype(int)

    class Oracle:
        classes_ = np.array([0, 1])
        def predict(self, A): A = np.asarray(A, float); return (1.2 * A[:, 0] + 0.9 * A[:, 1] > 0).astype(int)

    names = ["a", "b", "c", "d"]
    rel = np.eye(4); rel[0, 1] = rel[1, 0] = 0.9
    model = TrepanReloadedClassifier(max_nodes=7, max_depth=3, min_sample=250, max_queries=1500, max_n=2,
                                     random_state=3).fit(
        X, oracle=Oracle(), feature_names=names,
        semantic_feature_weights=np.array([1.3, 1.3, 1.0, 1.0]),
        semantic_feature_groups=["G", "G", None, None],
        semantic_relatedness_matrix=rel,
        semantic_feature_entities=["EntA", "EntB", None, None])
    rows = model.semantic_split_audit_
    assert rows
    for r in rows:
        assert {"base_score", "final_score", "complexity_penalty", "selected_feature",
                "ontology_entity", "semantic_reason", "semantic_bonus"} <= set(r)
        assert r["complexity_penalty"] == 0.0 and r["complexity_penalty_note"].startswith("not_modelled")
        assert r["base_score"] == pytest.approx(r["information_gain"])
        assert r["final_score"] == pytest.approx(r["selection_score"])
        assert isinstance(r["semantic_reason"], str) and r["semantic_reason"]
    influenced = [r for r in rows if r["selected_feature"]]
    if influenced:
        assert all(set(r["selected_feature"]) <= set(names) for r in influenced)
        assert all(set(r["ontology_entity"] or []) <= {"EntA", "EntB"} for r in influenced)


def test_entities_length_is_validated():
    from core.trepan_reloaded_historical import TrepanReloadedClassifier
    X = np.random.default_rng(0).normal(size=(60, 3))
    with pytest.raises(ValueError):
        TrepanReloadedClassifier(max_nodes=3, max_queries=0, random_state=0).fit(
            X, y=(X[:, 0] > 0).astype(int), semantic_feature_entities=["only_one"])

