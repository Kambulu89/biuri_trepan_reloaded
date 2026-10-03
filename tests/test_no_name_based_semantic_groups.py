"""Nenhum grupo semântico é adivinhado por palavras no nome da feature (agnóstico ao domínio)."""
import ast
from pathlib import Path

import pytest

from core.trepan_reloaded_extractor import TrepanReloadedExtractor

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", [
    "medical_history", "health_score", "financial_risk", "money_spent", "social_ties",
    "system_load", "tech_level", "anything_at_all", "x1",
])
def test_group_is_never_inferred_from_the_feature_name(name):
    assert TrepanReloadedExtractor._group_by_semantics(None, name) == "general"


def test_domain_keyword_lists_are_gone_from_group_logic():
    tree = ast.parse((ROOT / "core" / "trepan_reloaded_extractor.py").read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_group_by_semantics")
    literals = {n.value for n in ast.walk(func) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not literals & {"medical", "health", "financial", "finance", "money", "social", "technical", "tech", "system"}


def test_groups_still_come_from_the_ontology_graph():
    owlready2 = pytest.importorskip("owlready2")
    from core.ontology_semantic_graph import OntologySemanticGraph
    onto = owlready2.World().get_ontology("http://test.org/g.owl")
    with onto:
        Medical = type("ClinicalMeasure", (owlready2.Thing,), {})
        type("hasValueOne", (owlready2.DataProperty,), {"range": [float], "domain": [Medical]})
    g = OntologySemanticGraph.from_ontology(
        onto, accepted_matches=[{"feature": "value_one", "entity_name": "hasValueOne", "accepted": True}])
    assert g.primary_group("value_one") == "ClinicalMeasure"
