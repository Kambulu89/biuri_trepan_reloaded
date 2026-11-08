"""Regressão: a seleção ontológica pode devolver nomes fora da ordem original dos specs; o esquema
congelado do OntologyProcessor tem de continuar a coincidir com o que transform() produz (>= 2 features)."""
import numpy as np
import pandas as pd

from core.onto_feature_selector import align_processor_to_selection
from core.ontology_processor import OntologyProcessor


class _Fitted(OntologyProcessor):
    """Processor ajustado à mão com 3 agregados em ordem A, B, C (sem OWL real: só o contrato de esquema)."""

    def __init__(self, cols, groups):
        super().__init__(ontology=object())
        self.input_features_ = list(cols)
        self.feature_specs_ = [
            {"kind": "hierarchical_aggregate", "name": f"onto_{g}_aggregate", "sources": list(src),
             "centers": {c: 0.0 for c in src}, "scales": {c: 1.0 for c in src}, "method": "standardized_mean"}
            for g, src in groups.items()
        ]
        self.output_features_ = list(cols) + [s["name"] for s in self.feature_specs_]
        self.is_fitted_ = True


def _frame(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))


def test_selection_in_different_order_keeps_schema_consistent():
    groups = {"A": "ab", "B": "cd", "C": "ef"}
    proc = _Fitted(list("abcdef"), groups)
    selected = list("abcdef") + ["onto_C_aggregate", "onto_A_aggregate"]  # ordem de seleção != ordem dos specs
    align_processor_to_selection(proc, selected)
    out = proc.transform(_frame())
    assert list(out.columns) == proc.output_features_ == selected
    assert [s["name"] for s in proc.feature_specs_] == ["onto_C_aggregate", "onto_A_aggregate"]


def test_unpatched_order_would_diverge():
    """Documenta a falha original: filtrar mantendo a ordem dos specs + output_features_ na ordem de seleção."""
    groups = {"A": "ab", "B": "cd", "C": "ef"}
    proc = _Fitted(list("abcdef"), groups)
    selected = list("abcdef") + ["onto_C_aggregate", "onto_A_aggregate"]
    keep = set(selected)
    proc.feature_specs_ = [s for s in proc.feature_specs_ if s["name"] in keep]
    proc.output_features_ = selected
    try:
        proc.transform(_frame())
    except Exception as exc:  # OntologySchemaError
        assert "schema" in str(exc).lower() or "esquema" in str(exc).lower()
    else:
        raise AssertionError("esperava OntologySchemaError com a ordem antiga")


def test_single_selected_feature_unchanged():
    proc = _Fitted(list("abcdef"), {"A": "ab", "B": "cd"})
    selected = list("abcdef") + ["onto_B_aggregate"]
    align_processor_to_selection(proc, selected)
    assert list(proc.transform(_frame()).columns) == selected
