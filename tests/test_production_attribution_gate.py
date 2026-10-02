"""Gate de atribuição no pipeline de produção: o Reloaded só diverge do Original se o ganho vier da ontologia."""
import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.production_training import train_production_dataframe
from core.semantic_attribution import AttributionConfig
from core.semantic_enrichment import EnrichmentConfig

ENRICH = EnrichmentConfig(cv_folds=3, tuning_candidates=2, tuning_inner_folds=2, max_iter=150, n_bootstrap=200, random_state=5)
GATE = AttributionConfig(controls=2, inner_folds=2, random_state=5)


def _family_onto(noise, name):
    onto = owlready2.World().get_ontology(f"http://test.org/{name}.owl")
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        M = type("Morphology", (owlready2.Thing,), {}); B = type("Background", (owlready2.Thing,), {})
        for role in ("mean", "error", "worst"):
            p = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,), {"range": [float], "domain": [M]})
            p.statisticRole = [role]; p.measurementFamily = ["Size"]
        for i in range(noise):
            type(f"hasBackgroundVariable{i:02d}Level", (owlready2.DataProperty,), {"range": [float], "domain": [B]})
    return onto


def _hidden_relation(n=420, noise=20, seed=2):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5, 15, n); delta = rng.uniform(0, 1, n)
    df = pd.DataFrame({"hasSizeMean": mean, "hasSizeError": rng.uniform(.1, 1, n), "hasSizeWorst": mean * (1 + delta)})
    for i in range(noise):
        df[f"hasBackgroundVariable{i:02d}Level"] = rng.normal(size=n)
    y = (delta > np.median(delta)).astype(int)
    df["target"] = np.where(np.where(rng.random(n) < .03, 1 - y, y) == 1, "yes", "no")
    return df


def _uninformative(n=240, seed=1):
    """A ontologia agrupa atributos sem relação com a classe (a classe depende de uma só coluna)."""
    onto = owlready2.World().get_ontology("http://test.org/uninf.owl")
    with onto:
        A = type("GroupA", (owlready2.Thing,), {}); B = type("GroupB", (owlready2.Thing,), {})
        for i in range(4):
            type(f"hasAlpha{i}", (owlready2.DataProperty,), {"range": [float], "domain": [A]})
            type(f"hasBeta{i}", (owlready2.DataProperty,), {"range": [float], "domain": [B]})
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"hasAlpha{i}": rng.normal(size=n) for i in range(4)} | {f"hasBeta{i}": rng.normal(size=n) for i in range(4)})
    df["target"] = np.where(df["hasAlpha0"] + 0.8 * df["hasBeta1"] > 0, "y", "n")
    return df, onto


def test_informative_ontology_is_attributed_in_the_augmented_space(tmp_path):
    df = _hidden_relation()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=5, ontology=_family_onto(20, "g1"),
                                     require_reasoner=False, scientific_tuning=False,
                                     semantic_enrichment=ENRICH, semantic_attribution=GATE)
    ev = rep["evaluation"]
    att = ev["semantic_attribution"]
    assert ev["semantic_teacher"]["teacher"] == "mlp_semantic"
    assert att["selected_mode"] == "augmented" and att["status"] == "ATTRIBUTED_TO_ONTOLOGY"
    cand = att["candidates"][0]
    assert cand["mode"] == "augmented" and cand["attributed"] and cand["test_used"] is False
    assert cand["mean_gain_over_controls"] > 0 and cand["wins"] > cand["losses"]
    assert ev["reloaded_feature_space"]["space"] == "augmented" and rep["manifest"]["reloaded_mode"] == "augmented"
    # O teste final só é RELATADO (nunca decide). Um controlo aleatório pode, por acaso, incluir uma
    # coluna informativa, por isso o sinal numa só amostra pequena é ruído; a evidência é agregada
    # em várias sementes (scripts/run_reloaded_attribution.py).
    t = att["test_attribution"]
    assert t["scope"] == "final_test_reported_only" and len(t["controls"]) == GATE.controls
    assert {"attributable_balanced_accuracy", "attributable_oracle_fidelity"} <= set(t)
    assert all(np.isfinite(v) for v in (t["attributable_balanced_accuracy"], t["attributable_oracle_fidelity"]))


def test_uninformative_ontology_degrades_to_original_with_no_spurious_gain(tmp_path):
    df, onto = _uninformative()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=4, ontology=onto,
                                     require_reasoner=False, scientific_tuning=False, semantic_attribution=GATE)
    ev = rep["evaluation"]
    att = ev["semantic_attribution"]
    assert att["status"] == "NOT_ATTRIBUTED" and att["selected_mode"] == "neutral"
    assert all(not c["attributed"] for c in att["candidates"])
    assert rep["manifest"]["reloaded_mode"] == "neutral"
    assert ev["reloaded_feature_space"]["reason"] == "semantics_not_attributable_to_ontology"
    # Reloaded == Original (mesmo professor, semente, orçamento): nenhum ganho nem perda reivindicados
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "oracle_fidelity", "nodes", "depth"):
        assert ev["models"]["original"][key] == ev["models"]["reloaded"][key], key
    assert ev["comparison"]["delta_balanced_accuracy"] == 0.0 and ev["comparison"]["delta_oracle_fidelity"] == 0.0
    assert "test_attribution" not in att


def test_gate_is_deterministic(tmp_path):
    df, onto = _uninformative()
    a = train_production_dataframe(df, target="target", out_dir=tmp_path / "a", seed=4, ontology=onto,
                                   require_reasoner=False, scientific_tuning=False, semantic_attribution=GATE)
    df2, onto2 = _uninformative()
    b = train_production_dataframe(df2, target="target", out_dir=tmp_path / "b", seed=4, ontology=onto2,
                                   require_reasoner=False, scientific_tuning=False, semantic_attribution=GATE)
    sa, sb = a["evaluation"]["semantic_attribution"], b["evaluation"]["semantic_attribution"]
    assert sa["selected_mode"] == sb["selected_mode"]
    assert [c["per_fold"] for c in sa["candidates"]] == [c["per_fold"] for c in sb["candidates"]]


def test_gate_is_opt_in_and_default_behaviour_is_unchanged(tmp_path):
    df, onto = _uninformative()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=4, ontology=onto,
                                     require_reasoner=False, scientific_tuning=False)
    ev = rep["evaluation"]
    assert "semantic_attribution" not in ev
    assert ev["reloaded_feature_space"]["mode"] == "original_space"
    assert rep["manifest"]["reloaded_mode"] == "original_space"


def test_without_ontology_gate_is_skipped(tmp_path):
    df, _ = _uninformative()
    rep = train_production_dataframe(df, target="target", out_dir=tmp_path, seed=4,
                                     scientific_tuning=False, semantic_attribution=GATE)
    assert "semantic_attribution" not in rep["evaluation"] and rep["manifest"]["reloaded_mode"] == "none"
