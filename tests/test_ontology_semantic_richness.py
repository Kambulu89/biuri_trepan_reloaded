"""Gate de riqueza semântica: mede conhecimento além da taxonomia, sem alterar a aceitação."""
from pathlib import Path

import pytest

from core.ontology_quality import OntologyQualityGate, semantic_richness

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark_ontologies"


class _Entity:
    def __init__(self, name, **attrs):
        self.name = name
        self.label = [name]
        self.prefLabel = []
        self.altLabel = []
        self.__dict__.update(attrs)


class _Restriction:
    property = object()
    type = 1


def test_empty_entities_are_poor():
    r = semantic_richness({"classes": [], "datatype_properties": [], "object_properties": []})
    assert r["level"] == "poor"
    assert r["knowledge_sources"] == []


def test_object_property_and_restriction_are_counted():
    cls = _Entity("Patient", is_a=[_Restriction()])
    r = semantic_richness({
        "classes": [cls],
        "datatype_properties": [],
        "object_properties": [_Entity("hasFinding")],
    })
    assert r["object_property_count"] == 1 and r["restriction_count"] == 1
    assert r["knowledge_sources"] == ["object_properties", "restrictions"]
    assert r["level"] == "rich"


def test_declared_bounds_make_it_limited():
    prop = _Entity("hasAge", minInclusive=0, maxInclusive=120)
    r = semantic_richness({"classes": [], "datatype_properties": [prop], "object_properties": []})
    assert r["bounded_datatype_properties"] == 1
    assert r["level"] == "limited"


def test_roles_need_both_family_and_role():
    only_role = _Entity("a", statisticRole="mean")
    r = semantic_richness({"classes": [], "datatype_properties": [only_role], "object_properties": []})
    assert "statistic_roles" not in r["knowledge_sources"]
    both = _Entity("b", statisticRole="mean", measurementFamily="area")
    r = semantic_richness({"classes": [], "datatype_properties": [both], "object_properties": []})
    assert "statistic_roles" in r["knowledge_sources"]


def test_broken_entities_do_not_raise():
    class Bad:
        name = "bad"
        @property
        def is_a(self):
            raise RuntimeError("boom")
    r = semantic_richness({"classes": [Bad()], "datatype_properties": [], "object_properties": []})
    assert r["level"] == "poor"


owlready2 = pytest.importorskip("owlready2")


def _evaluate(name, features):
    onto = owlready2.World().get_ontology(str((BENCH / f"{name}.owl").resolve())).load()
    return OntologyQualityGate().evaluate(features, onto, require_reasoner=False)


def test_flat_benchmark_ontology_is_flagged_poor_but_still_accepted():
    rep = _evaluate("iris", [
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)",
    ])
    assert rep.accepted and rep.status == "VALID_DOMAIN_ONTOLOGY"  # comportamento inalterado
    assert rep.metrics["semantic_richness"]["level"] == "poor"
    assert any("semanticamente pobre" in w for w in rep.warnings)


def test_breast_cancer_roles_are_detected():
    sklearn_datasets = pytest.importorskip("sklearn.datasets")
    names = [str(c) for c in sklearn_datasets.load_breast_cancer(as_frame=True).data.columns]
    rep = _evaluate("breast_cancer", names)
    richness = rep.metrics["semantic_richness"]
    assert "statistic_roles" in richness["knowledge_sources"]
    assert richness["level"] == "limited"
    assert not any("semanticamente pobre" in w for w in rep.warnings)
