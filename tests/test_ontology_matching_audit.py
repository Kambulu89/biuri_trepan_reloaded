"""Matching ARFF<->OWL auditável, ABox/TBox e ontologias genéricas (agnóstico ao dataset)."""
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_quality import OntologyQualityGate, mapping_entropy, taxonomy_depth


def _onto(name):
    return owlready2.World().get_ontology(f"http://test.org/{name}.owl")


def _dprop(onto, name, labels=()):
    with onto:
        prop = type(name, (owlready2.DataProperty,), {"range": [float]})
    for label in labels:
        prop.label.append(label)
    return prop


def _by_feature(decisions):
    return {d.feature: d for d in decisions}


def test_exact_match_records_runner_up_and_status():
    onto = _onto("exact")
    _dprop(onto, "alpha_measure"); _dprop(onto, "beta_measure"); _dprop(onto, "gamma_measure")
    d = _by_feature(OntologyQualityGate().match_features(["alpha_measure"], onto))["alpha_measure"]
    assert d.accepted and d.status == "ACCEPTED" and d.entity_name == "alpha_measure"
    assert d.runner_up_entity in {"beta_measure", "gamma_measure"}
    assert d.ambiguity_margin == pytest.approx(d.score - d.runner_up_score)


@pytest.mark.parametrize("feature,entity", [
    ("mean_radius", "MeanRadius"),       # snake_case vs CamelCase
    ("mean radius", "meanRadius"),       # espaços vs camelCase
    ("MEAN-RADIUS", "mean_radius"),      # hífen/maiúsculas
])
def test_lexical_conventions_match(feature, entity):
    onto = _onto("lex")
    _dprop(onto, entity); _dprop(onto, "unrelated_thing")
    d = OntologyQualityGate().match_features([feature], onto)[0]
    assert d.accepted and d.entity_name == entity


def test_alias_through_rdfs_label():
    onto = _onto("alias")
    _dprop(onto, "hasQty7", labels=["serum creatinine"]); _dprop(onto, "hasOther", labels=["height"])
    d = OntologyQualityGate().match_features(["serum creatinine"], onto)[0]
    assert d.accepted and d.entity_name == "hasQty7"


def test_ambiguous_match_is_rejected_and_flagged():
    onto = _onto("amb")
    _dprop(onto, "radius_left"); _dprop(onto, "radius_right")
    d = OntologyQualityGate().match_features(["radius"], onto)[0]
    assert not d.accepted and d.status == "REJECTED_AMBIGUOUS"
    assert d.runner_up_entity in {"radius_left", "radius_right"}


def test_low_score_is_rejected():
    onto = _onto("low")
    _dprop(onto, "zzzz")
    d = OntologyQualityGate().match_features(["completely_different"], onto)[0]
    assert not d.accepted and d.status == "REJECTED_LOW_SCORE"


def test_entity_collision_keeps_only_best_score():
    onto = _onto("coll")
    _dprop(onto, "area"); _dprop(onto, "perimeter")
    ds = _by_feature(OntologyQualityGate().match_features(["area", "area_x"], onto))
    assert ds["area"].accepted
    assert not ds["area_x"].accepted and ds["area_x"].status == "REJECTED_ENTITY_COLLISION"


def test_entity_collision_with_tie_rejects_all():
    onto = _onto("tie")
    _dprop(onto, "mean_radius"); _dprop(onto, "other")
    ds = OntologyQualityGate().match_features(["mean_radius", "meanradius"], onto)
    assert all(d.status == "REJECTED_ENTITY_COLLISION" for d in ds)


def test_property_preferred_over_class_on_tie_is_recorded():
    onto = _onto("tb")
    with onto:
        type("Radius", (owlready2.Thing,), {})
    _dprop(onto, "radius")
    d = OntologyQualityGate().match_features(["radius"], onto)[0]
    assert d.accepted and d.entity_type == "datatype_property" and d.tie_break == "datatype_property_over_class"


def _quality(onto, features, **kw):
    return OntologyQualityGate().evaluate(features, onto, require_reasoner=False, **kw)


def test_generic_ontology_is_rejected():
    onto = _onto("generic")
    for name in ("Feature", "Value", "Attribute", "Measurement"):
        with onto:
            type(name, (owlready2.Thing,), {})
    rep = _quality(onto, ["feature", "value", "attribute", "measurement"])
    assert not rep.accepted and rep.status == "GENERIC_ONTOLOGY"
    assert rep.metrics["generic_match_ratio"] == 1.0
    assert rep.metrics["semantic_specificity"] == 0.0


def test_specific_ontology_has_full_specificity_and_entropy():
    onto = _onto("specific")
    names = ["serum_sodium", "serum_potassium", "heart_rate", "body_temp"]
    for n in names:
        _dprop(onto, n)
    rep = _quality(onto, names)
    assert rep.accepted
    m = rep.metrics
    assert m["generic_match_ratio"] == 0.0 and m["mapping_entropy"] == pytest.approx(1.0)
    assert m["distinct_entity_ratio"] == 1.0
    assert m["knowledge_split"]["tbox"]["datatype_properties"] == 4
    assert m["knowledge_split"]["abox"]["individuals"] == 0


def test_abox_safe_for_controlled_vocabulary():
    onto = _onto("abox_safe")
    with onto:
        Sector = type("Sector", (owlready2.Thing,), {})
        Sector("Private"); Sector("Public")
    rep = OntologyQualityGate().audit_abox(onto)
    assert rep["accepted"] and rep["status"] == "SAFE"


def test_abox_rejects_dataset_records():
    onto = _onto("abox_rec")
    with onto:
        Rec = type("Record", (owlready2.Thing,), {})
        Rec("row_1"); Rec("row_2")
    rep = OntologyQualityGate().audit_abox(onto)
    assert not rep["accepted"] and rep["status"] == "REJECT_DATASET_RECORDS_IN_ABOX"


def test_abox_rejects_forbidden_test_identifiers_without_dropping_vocabulary():
    onto = _onto("abox_leak")
    with onto:
        Sector = type("Sector", (owlready2.Thing,), {})
        Sector("Private"); Sector("case_9917")
    rep = OntologyQualityGate().audit_abox(onto, forbidden_test_identifiers=["case_9917"])
    assert rep["status"] == "REJECT_TEST_INSTANCE_LEAKAGE"
    assert "case_9917" in rep["test_identifier_collisions"]
    assert "Private" in rep["controlled_vocabulary_individuals"]


def test_mapping_entropy_and_depth_helpers():
    assert mapping_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)
    assert mapping_entropy(["a", "a", "a", "a"]) == pytest.approx(0.0)
    assert mapping_entropy([]) == 0.0
    onto = _onto("depth")
    with onto:
        A = type("A", (owlready2.Thing,), {}); B = type("B", (A,), {}); type("C", (B,), {})
    assert taxonomy_depth(list(onto.classes())) == 3


def test_invalid_owl_file_fails_loudly(tmp_path):
    from core.production_training import _load_ontology
    bad = tmp_path / "bad.owl"; bad.write_text("<rdf:RDF this is not owl", encoding="utf-8")
    with pytest.raises(Exception):
        _load_ontology(bad)
    with pytest.raises(Exception):
        _load_ontology(tmp_path / "missing.owl")


def test_quality_reports_reasoner_flag_from_report():
    onto = _onto("rsn")
    for n in ("p1_value", "p2_value", "p3_value"):
        _dprop(onto, n)
    rep = OntologyQualityGate().evaluate(
        ["p1_value", "p2_value", "p3_value"], onto,
        reasoner_report={"executed": True, "consistent": True, "reasoner_used": True},
    )
    assert rep.accepted and rep.metrics["reasoner_used"] is True and rep.metrics["reasoner_consistent"] is True
    inconsistent = OntologyQualityGate().evaluate(
        ["p1_value", "p2_value", "p3_value"], onto,
        reasoner_report={"executed": True, "consistent": False},
    )
    assert inconsistent.status == "LOGICALLY_INCONSISTENT"
