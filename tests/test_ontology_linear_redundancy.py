"""Auditoria (e descarte opt-in) de features onto_* que são combinações lineares das fontes."""
from pathlib import Path

import pytest

owlready2 = pytest.importorskip("owlready2")
datasets = pytest.importorskip("sklearn.datasets")

from core.ontology_processor import OntologyProcessor

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark_ontologies"


def _frame(loader):
    X = loader(as_frame=True).data.copy()
    X.columns = [str(c) for c in X.columns]
    return X


def _onto(name):
    return owlready2.World().get_ontology(str((BENCH / f"{name}.owl").resolve())).load()


def _fit(name, loader, **kwargs):
    return OntologyProcessor(_onto(name), **kwargs).fit(_frame(loader), log=False)


def test_default_behaviour_keeps_aggregates_and_audits_them():
    p = _fit("iris", datasets.load_iris)
    assert len(p.feature_specs_) == 2  # inalterado: nada é descartado por omissão
    rows = [r for r in p.feature_audit_ if r["accepted"]]
    assert rows and all(r["linear_redundancy_r2"] > 0.999 for r in rows)
    stats = p.last_engineering_stats
    assert sorted(stats["linear_redundant_features"]) == sorted(p.output_features_[4:])


def test_drop_linear_redundant_removes_pure_aggregates():
    p = _fit("iris", datasets.load_iris, drop_linear_redundant=True)
    assert p.feature_specs_ == []
    assert p.output_features_ == p.input_features_
    reasons = {r["rejection_reason"] for r in p.feature_audit_}
    assert reasons == {"linear_combination_of_sources"}


def test_drop_keeps_nonlinear_relational_features():
    p = _fit("breast_cancer", datasets.load_breast_cancer, drop_linear_redundant=True)
    kept = [r for r in p.feature_audit_ if r["accepted"]]
    assert kept, "rácios/deltas relativos não lineares devem sobreviver"
    assert all(r["linear_redundancy_r2"] < p.linear_redundancy_threshold for r in kept)
    assert all(r["derivation_rule"].count("/") >= 1 for r in kept)
    dropped = {r["feature"] for r in p.feature_audit_
               if r["rejection_reason"] == "linear_combination_of_sources"}
    assert any(name.endswith("_worst_minus_mean") for name in dropped)
    assert any(name.endswith("_aggregate") for name in dropped)


def test_transform_matches_fitted_schema_when_dropping():
    X = _frame(datasets.load_breast_cancer)
    p = OntologyProcessor(_onto("breast_cancer"), drop_linear_redundant=True).fit(X, log=False)
    assert list(p.transform(X).columns) == p.output_features_


def test_non_linear_specs_report_no_r2():
    p = OntologyProcessor(_onto("iris"))
    assert p._linear_redundancy(_frame(datasets.load_iris), {"kind": "constraint", "source": "x"}, None) is None
