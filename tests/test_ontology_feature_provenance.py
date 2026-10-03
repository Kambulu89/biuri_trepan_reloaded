"""Proveniência, origem do conhecimento (OWL vs estatístico), filtros de novidade e resumo de geração."""
import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_processor import OntologyProcessor


def _onto(name="prov"):
    return owlready2.World().get_ontology(f"http://test.org/{name}.owl")


def _family_ontology(with_bounds=False, name="fam"):
    onto = _onto(name)
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        Measurement = type("Morphology", (owlready2.Thing,), {})
        props = {}
        for role in ("mean", "error", "worst"):
            attrs = {"range": [float], "domain": [Measurement]}
            props[role] = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,), attrs)
            props[role].statisticRole = [role]
            props[role].measurementFamily = ["Size"]
    return onto


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    mean = rng.uniform(5, 15, 120)
    return pd.DataFrame({
        "hasSizeMean": mean,
        "hasSizeError": rng.uniform(0.1, 1.0, 120),
        "hasSizeWorst": mean * rng.uniform(1.1, 1.8, 120),
    })


def test_relational_family_roles_discovered_from_ontology_not_names(frame):
    proc = OntologyProcessor(_family_ontology()).fit(frame, log=False)
    names = proc.last_engineering_stats["relational_names"]
    assert {"onto_Size_worst_minus_mean", "onto_Size_family_contrast",
            "onto_Size_normalized_error", "onto_Size_error_ratio",
            "onto_Size_relative_worst_delta"} <= set(names)
    for row in proc.feature_audit_:
        if row["kind"] == "relational":
            assert row["owl_origin"]["owl_entities_or_properties"]
            assert row["provenance"] == "owl_measurement_family_role"
            assert row["knowledge_source"] == "ontology"


def test_new_relations_are_bounded_and_formula_exact(frame):
    proc = OntologyProcessor(_family_ontology("b")).fit(frame, log=False)
    out = proc.transform(frame)
    m, w, e = frame["hasSizeMean"], frame["hasSizeWorst"], frame["hasSizeError"]
    np.testing.assert_allclose(out["onto_Size_family_contrast"], (w - m) / (w.abs() + m.abs()))
    np.testing.assert_allclose(out["onto_Size_normalized_error"], e.abs() / (m.abs() + w.abs()))
    assert out["onto_Size_family_contrast"].between(-1, 1).all()


def test_statistical_constraints_are_not_labelled_as_ontology_knowledge(frame):
    proc = OntologyProcessor(_family_ontology("c"), allow_train_calibrated_bounds=True).fit(frame, log=False)
    stats = proc.last_engineering_stats
    assert stats["constraint_features"] > 0
    assert all(spec["provenance"] == "training_quantile_not_owl"
               for spec in proc.feature_specs_ if spec["kind"] == "constraint")
    assert set(stats["statistical_feature_names"]) >= {
        s["name"] for s in proc.feature_specs_ if s["kind"] == "constraint"}
    assert not set(stats["statistical_feature_names"]) & set(stats["ontology_knowledge_feature_names"])


def test_without_calibration_no_statistical_constraints_are_invented(frame):
    proc = OntologyProcessor(_family_ontology("d")).fit(frame, log=False)
    assert proc.last_engineering_stats["constraint_features"] == 0
    assert proc.last_engineering_stats["statistical_feature_names"] == []


def test_owl_declared_bounds_are_ontology_knowledge(frame):
    onto = _family_ontology("e")
    # Faceta OWL padrão: rdfs:range = xsd:float[>= 0.0, <= 12.0]
    onto.hasSizeMean.range = [owlready2.ConstrainedDatatype(float, min_inclusive=0.0, max_inclusive=12.0)]
    proc = OntologyProcessor(onto).fit(frame, log=False)
    cons = [s for s in proc.feature_specs_ if s["kind"] == "constraint"]
    assert cons and all(s["provenance"] == "owl_explicit_bound" and s["knowledge_source"] == "ontology" for s in cons)


def test_generation_summary_counts_removed_features(frame):
    # espelho exato e quase-duplicado de uma coluna original
    frame = frame.copy()
    onto = _family_ontology("f")
    proc = OntologyProcessor(onto, near_duplicate_correlation=0.9).fit(frame, log=False)
    summary = proc.last_engineering_stats["generation_summary"]
    assert summary["generated"] == len(proc.feature_audit_)
    assert summary["retained"] == len(proc.feature_specs_)
    removed = sum(v for k, v in summary.items() if k.startswith("removed_"))
    assert summary["generated"] == summary["retained"] + removed
    assert summary["removed_near_duplicate"] >= 1  # limiar baixo força o caso


def test_constant_source_features_are_removed():
    onto = _family_ontology("g")
    n = 80
    df = pd.DataFrame({"hasSizeMean": np.full(n, 3.0), "hasSizeError": np.full(n, 0.5),
                       "hasSizeWorst": np.full(n, 4.0)})
    proc = OntologyProcessor(onto).fit(df, log=False)
    assert proc.feature_specs_ == []
    s = proc.last_engineering_stats["generation_summary"]
    assert s["retained"] == 0 and s["removed_constant"] >= 1


def test_reasoner_inferred_hierarchy_is_marked(frame):
    onto = _family_ontology("h")
    report = {"inferred_subclass_relations": [["hasSizeMean", "Morphology"]]}
    proc = OntologyProcessor(onto, reasoner_report=report).fit(frame, log=False)
    agg = [r for r in proc.feature_audit_ if r["kind"] == "hierarchical_aggregate"]
    assert agg and all(r["reasoner_inferred"] for r in agg)
    assert proc.last_engineering_stats["reasoner_inferred_feature_names"]
