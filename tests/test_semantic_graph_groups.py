"""Regressão: o tipo de dados do rdfs:range não é um grupo semântico."""
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_semantic_graph import OntologySemanticGraph


def test_datatype_range_is_not_a_semantic_group():
    """Regressão: features float sem relação não podem partilhar o 'grupo' do tipo de dados."""
    onto = owlready2.World().get_ontology("http://test.org/groups.owl")
    with onto:
        A = type("ClassA", (owlready2.Thing,), {}); B = type("ClassB", (owlready2.Thing,), {})
        type("hasAlpha", (owlready2.DataProperty,), {"range": [float], "domain": [A]})
        type("hasBeta", (owlready2.DataProperty,), {"range": [float], "domain": [A]})
        type("hasGamma", (owlready2.DataProperty,), {"range": [float], "domain": [B]})
    matches = [{"feature": n, "entity_name": n, "accepted": True} for n in ("hasAlpha", "hasBeta", "hasGamma")]
    g = OntologySemanticGraph.from_ontology(onto, accepted_matches=matches)
    assert g.primary_group("hasAlpha") == "ClassA" == g.primary_group("hasBeta")
    assert g.primary_group("hasGamma") == "ClassB"
    for feature in ("hasAlpha", "hasBeta", "hasGamma"):
        assert all("float" not in grp and grp != "DatatypeProperty" for grp in g.feature_groups[feature])


def test_feature_without_class_context_has_no_shared_group():
    onto = owlready2.World().get_ontology("http://test.org/nogroup.owl")
    with onto:
        type("hasX", (owlready2.DataProperty,), {"range": [float]})
        type("hasY", (owlready2.DataProperty,), {"range": [float]})
    matches = [{"feature": n, "entity_name": n, "accepted": True} for n in ("hasX", "hasY")]
    g = OntologySemanticGraph.from_ontology(onto, accepted_matches=matches)
    assert g.primary_group("hasX") != g.primary_group("hasY")
