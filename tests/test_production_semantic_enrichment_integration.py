"""O pipeline headless de produção avalia o enriquecimento sem usar o teste e sem regressões."""
import json

import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.production_training import train_production_dataframe
from core.semantic_enrichment import EnrichmentConfig
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig

FAST_TREPAN = ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1, tune_structure=False, max_capacity_candidates=1, max_semantic_candidates=1)
FAST_ENRICH = EnrichmentConfig(cv_folds=3, tuning_candidates=2, tuning_inner_folds=2, max_iter=120, n_bootstrap=100)


def _onto():
    onto = owlready2.World().get_ontology("http://test.org/prod.owl")
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        Morph = type("Morphology", (owlready2.Thing,), {})
        for role in ("mean", "error", "worst"):
            p = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,),
                     {"range": [float], "domain": [Morph]})
            p.statisticRole = [role]; p.measurementFamily = ["Size"]
        Other = type("Background", (owlready2.Thing,), {})
        for i in range(6):
            type(f"hasBackgroundVariable{i}Level", (owlready2.DataProperty,), {"range": [float], "domain": [Other]})
    return onto


def _frame(n=140, seed=4):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5, 15, n); delta = rng.uniform(0, 1, n)
    df = pd.DataFrame({"hasSizeMean": mean, "hasSizeError": rng.uniform(.1, 1, n), "hasSizeWorst": mean * (1 + delta)})
    for i in range(6):
        df[f"hasBackgroundVariable{i}Level"] = rng.normal(size=n)
    df["target"] = np.where(delta > np.median(delta), "yes", "no")
    return df


def test_default_run_is_unchanged_and_has_no_enrichment(tmp_path):
    rep = train_production_dataframe(_frame(), target="target", out_dir=tmp_path, seed=4, ontology=_onto(),
                                     require_reasoner=False, scientific_tuning=False)
    ev = rep["evaluation"]
    assert ev["semantic_enrichment"] is None
    assert ev["ontology_stage_status"]["semantic_trepan_available"] is True
    assert ev["ontology_stage_status"]["mlp_enrichment_status"] == "NOT_EVALUATED"


def test_enrichment_runs_on_training_only_and_reports_independent_states(tmp_path):
    df = _frame()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=4, ontology=_onto(),
                                     require_reasoner=False, scientific_tuning=False,
                                     semantic_enrichment=FAST_ENRICH)
    enr = rep["evaluation"]["semantic_enrichment"]
    assert enr["test_used"] is False
    assert enr["n_samples"] == len(rep["split"]["train_indices"]) < len(df)  # só treino de desenvolvimento
    assert enr["decision"] in {
        "ACCEPT_SIGNIFICANT_GAIN", "ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN", "ACCEPT_PARTIAL_FEATURE_SET",
        "REJECT_NO_NOVEL_FEATURES", "REJECT_UNSTABLE_FEATURES", "REJECT_DEGRADATION",
        "REJECT_NO_INFORMATIONAL_GAIN"}
    st = rep["evaluation"]["ontology_stage_status"]
    assert st["mlp_enrichment_status"] == enr["decision"]
    assert st["semantic_mlp_accepted"] == enr["semantic_mlp_accepted"]
    assert st["semantic_trepan_available"] is True  # independente da decisão do MLP
    saved = json.loads((tmp_path / "production_report.json").read_text(encoding="utf-8"))
    assert saved["evaluation"]["semantic_enrichment"]["decision"] == enr["decision"]


def test_enrichment_failure_never_breaks_production_training(tmp_path, monkeypatch):
    import core.production_training as pt
    monkeypatch.setattr(pt, "evaluate_semantic_enrichment",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = train_production_dataframe(_frame(), target="target", out_dir=tmp_path, seed=4, ontology=_onto(),
                                     require_reasoner=False, scientific_tuning=False,
                                     semantic_enrichment=FAST_ENRICH)
    enr = rep["evaluation"]["semantic_enrichment"]
    assert enr["decision"] == "NOT_EVALUATED" and "boom" in enr["decision_reason"]
    assert enr["semantic_mlp_accepted"] is False
    assert (tmp_path / "production_bundle.joblib").exists()


def test_declared_family_raises_relatedness_between_family_members(tmp_path, monkeypatch):
    captured = {}
    import core.production_training as pt
    original = pt.fit_controlled_trepan_pair

    def spy(*a, **k):
        captured["matrix"] = np.asarray(k["semantic_relatedness_matrix"])
        captured["entities"] = k["semantic_feature_entities"]
        return original(*a, **k)

    monkeypatch.setattr(pt, "fit_controlled_trepan_pair", spy)
    train_production_dataframe(_frame(), target="target", out_dir=tmp_path, seed=4, ontology=_onto(),
                               require_reasoner=False, scientific_tuning=False)
    m, ents = captured["matrix"], captured["entities"]
    assert any(e for e in ents)
    names = ["hasSizeMean", "hasSizeError", "hasSizeWorst"]
    idx = [i for i, e in enumerate(ents) if e in names]
    assert len(idx) == 3
    for a in idx:
        for b in idx:
            if a != b:
                assert m[a, b] >= 0.75  # mesma família declarada na OWL
