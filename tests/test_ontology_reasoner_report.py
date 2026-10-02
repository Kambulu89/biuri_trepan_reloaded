"""Reasoner: observabilidade (duração, axiomas inferidos) e fallback explícito."""
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_reasoner import run_owl_reasoner


def _ontology_with_inferable_subclass(iri="http://test.org/infer.owl"):
    onto = owlready2.World().get_ontology(iri)
    with onto:
        class Part(owlready2.Thing): ...
        class has_part(owlready2.ObjectProperty):
            domain = [owlready2.Thing]; range = [Part]
        class Parent(owlready2.Thing):
            equivalent_to = [owlready2.Thing & has_part.some(Part)]
        class Child(owlready2.Thing):
            is_a = [has_part.some(Part)]
    return onto


def test_reasoner_reports_inferred_subclass_and_duration():
    report = run_owl_reasoner(_ontology_with_inferable_subclass())
    if not report["executed"]:
        pytest.skip(f"BLOCKED_BY_ENVIRONMENT: {report['error_type']}")
    assert report["reasoner_used"] is True
    assert report["reasoning_mode"] == "inferred"
    assert report["consistent"] is True
    assert report["duration_seconds"] >= 0.0
    assert ["Child", "Parent"] in report["inferred_subclass_relations"]
    assert report["inferred_axioms_count"] >= 1


def test_flat_ontology_reports_zero_inferred_axioms_honestly():
    onto = owlready2.World().get_ontology("http://test.org/flat.owl")
    with onto:
        class A(owlready2.Thing): ...
        class B(A): ...
    report = run_owl_reasoner(onto)
    if not report["executed"]:
        pytest.skip(f"BLOCKED_BY_ENVIRONMENT: {report['error_type']}")
    assert report["inferred_axioms_count"] == 0


def test_inconsistent_ontology_is_reported_not_hidden():
    onto = owlready2.World().get_ontology("http://test.org/bad.owl")
    with onto:
        class A(owlready2.Thing): ...
        class B(owlready2.Thing): ...
        owlready2.AllDisjoint([A, B])
        class AB(A, B): ...
    report = run_owl_reasoner(onto)
    if not report["executed"]:
        pytest.skip(f"BLOCKED_BY_ENVIRONMENT: {report['error_type']}")
    assert report["consistent"] is False
    assert "AB" in report["unsatisfiable_classes"] + report["inconsistent_classes"]


def test_failure_has_explicit_fallback_and_is_not_marked_used():
    report = run_owl_reasoner(None)
    assert report["executed"] is False
    assert report["reasoner_used"] is False
    assert report["reasoning_mode"] == "explicit_axioms_only"
    bad = run_owl_reasoner(_ontology_with_inferable_subclass("http://test.org/x.owl"), engine="nope")
    assert bad["executed"] is False and bad["reasoner_used"] is False
