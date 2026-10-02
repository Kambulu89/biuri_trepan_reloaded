"""Professor MLP+OWL para o TREPAN: seleção rigorosa, reconstrução a partir de Z, serialização e integração."""
import joblib
import numpy as np
import pandas as pd
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.production_inference import load_production_bundle
from core.production_training import train_production_dataframe
from core.semantic_enrichment import EnrichmentConfig, evaluate_semantic_enrichment
from core.semantic_teacher import (
    SemanticTeacher, build_semantic_teacher, decide_semantic_teacher, detach_ontology, teacher_inputs_available,
)
from core.ontology_quality import OntologyQualityGate
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig

CFG = EnrichmentConfig(cv_folds=3, tuning_candidates=2, tuning_inner_folds=2, max_iter=150, n_bootstrap=200, random_state=5)
FAST_TREPAN = ScientificTrepanSearchConfig(cv_folds=2, max_capacity_candidates=1, max_semantic_candidates=1)


def _onto(noise=0, name="teach"):
    onto = owlready2.World().get_ontology(f"http://test.org/{name}.owl")
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        Morph = type("Morphology", (owlready2.Thing,), {})
        Bg = type("Background", (owlready2.Thing,), {})
        for role in ("mean", "error", "worst"):
            p = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,), {"range": [float], "domain": [Morph]})
            p.statisticRole = [role]; p.measurementFamily = ["Size"]
        for i in range(noise):
            type(f"hasBackgroundVariable{i:02d}Level", (owlready2.DataProperty,), {"range": [float], "domain": [Bg]})
    return onto


def _frame(n, noise, signal="relation", seed=2):
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5, 15, n); delta = rng.uniform(0, 1, n)
    df = pd.DataFrame({"hasSizeMean": mean, "hasSizeError": rng.uniform(.1, 1, n), "hasSizeWorst": mean * (1 + delta)})
    for i in range(noise):
        df[f"hasBackgroundVariable{i:02d}Level"] = rng.normal(size=n)
    base = delta if signal == "relation" else mean - mean.mean()
    y = (base > np.median(base)).astype(int)
    df["target"] = np.where(np.where(rng.random(n) < .03, 1 - y, y) == 1, "yes", "no")
    return df


def _result(n=300, noise=20, signal="relation", name="t"):
    df = _frame(n, noise, signal); onto = _onto(noise, name)
    X = df.drop(columns="target")
    q = OntologyQualityGate().evaluate(list(X.columns), onto, require_reasoner=False)
    return X, df["target"].to_numpy(), evaluate_semantic_enrichment(X, df["target"].to_numpy(), onto,
                                                                    quality_report=q, config=CFG)


# ------------------------------------------------------------------ política de seleção ---
@pytest.mark.parametrize("report,min_ev,use,why", [
    (None, "strong", False, "no_enrichment_report"),
    ({"semantic_mlp_accepted": False, "decision": "REJECT_DEGRADATION"}, "strong", False, "enrichment_not_accepted"),
    ({"semantic_mlp_accepted": True, "evidence_strength": "weak", "selected_semantic_features": ["a"]}, "strong", False, "evidence_weak"),
    ({"semantic_mlp_accepted": True, "evidence_strength": "weak", "selected_semantic_features": ["a"]}, "weak", True, "accepted"),
    ({"semantic_mlp_accepted": True, "evidence_strength": "strong", "selected_semantic_features": ["a"]}, "strong", True, "accepted"),
    ({"semantic_mlp_accepted": True, "evidence_strength": "strong", "selected_semantic_features": []}, "strong", False, "no_selected"),
    ({"semantic_mlp_accepted": True, "evidence_strength": "strong", "selected_semantic_features": ["a"]}, None, False, "disabled"),
])
def test_teacher_policy(report, min_ev, use, why):
    d = decide_semantic_teacher(report, min_ev)
    assert d.use is use and why in d.reason


def test_invalid_min_evidence_is_rejected():
    with pytest.raises(ValueError):
        decide_semantic_teacher({"semantic_mlp_accepted": True}, "maybe")


# ------------------------------------------------------------------ construção do professor ---
def test_teacher_reconstructs_enriched_space_from_model_space_and_matches_pipeline():
    X, y, res = _result(name="t1")
    assert res.report["semantic_mlp_accepted"], res.report["decision"]
    names = list(X.columns)
    teacher = build_semantic_teacher(res, X.to_numpy(float), y, names, config=CFG, seed=5)
    assert isinstance(teacher, SemanticTeacher) and teacher.processor.ontology is None  # destacada da OWL
    assert set(teacher.selected_features) == set(res.selected_features)
    enriched = res.transform(X)
    assert list(teacher._enriched(X.to_numpy(float)).columns) == list(enriched.columns)
    np.testing.assert_allclose(teacher._enriched(X.to_numpy(float)).to_numpy(), enriched.to_numpy())
    pred = teacher.predict(X.to_numpy(float))
    assert set(pred) <= {"yes", "no"} and list(teacher.classes_) == ["no", "yes"]
    proba = teacher.predict_proba(X.to_numpy(float))
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)
    assert (pred == y).mean() > 0.8


def test_teacher_is_picklable_and_equal_after_roundtrip(tmp_path):
    X, y, res = _result(name="t2")
    teacher = build_semantic_teacher(res, X.to_numpy(float), y, list(X.columns), config=CFG, seed=5)
    joblib.dump(teacher, tmp_path / "t.joblib")
    again = joblib.load(tmp_path / "t.joblib")
    np.testing.assert_array_equal(teacher.predict(X.to_numpy(float)), again.predict(X.to_numpy(float)))


def test_teacher_is_deterministic_for_the_same_seed():
    X, y, res = _result(name="t3")
    a = build_semantic_teacher(res, X.to_numpy(float), y, list(X.columns), config=CFG, seed=5)
    b = build_semantic_teacher(res, X.to_numpy(float), y, list(X.columns), config=CFG, seed=5)
    np.testing.assert_array_equal(a.predict_proba(X.to_numpy(float)), b.predict_proba(X.to_numpy(float)))


def test_unavailable_when_inputs_are_not_columns_of_model_space():
    X, y, res = _result(name="t4")
    d = teacher_inputs_available(res.processor, ["other_a", "other_b"])
    assert d.use is False and d.reason == "inputs_not_reconstructable_from_model_space"
    with pytest.raises(ValueError):
        build_semantic_teacher(res, X.to_numpy(float), y, ["other"] * X.shape[1], config=CFG, seed=5)


def test_detach_does_not_mutate_the_original_processor():
    X, y, res = _result(name="t5")
    clone = detach_ontology(res.processor)
    assert clone.ontology is None and res.processor.ontology is not None
    assert clone.feature_specs_ == res.processor.feature_specs_


def test_teacher_rejects_wrong_matrix_shape():
    X, y, res = _result(name="t6")
    teacher = build_semantic_teacher(res, X.to_numpy(float), y, list(X.columns), config=CFG, seed=5)
    with pytest.raises(ValueError):
        teacher.predict(np.zeros((3, 2)))


# ------------------------------------------------------------------ produção ---
def _produce(tmp_path, df, noise, name, **kw):
    return train_production_dataframe(df, target="target", out_dir=tmp_path, seed=5, ontology=_onto(noise, name),
                                      require_reasoner=False, scientific_tuning=False,
                                      semantic_enrichment=CFG, **kw)


def test_production_uses_semantic_teacher_for_both_arms_when_evidence_is_strong(tmp_path, monkeypatch):
    import core.production_training as pt
    seen = {}
    original = pt.fit_controlled_trepan_pair

    def spy(*a, **k):
        seen["oracle"] = k["oracle"]
        return original(*a, **k)
    monkeypatch.setattr(pt, "fit_controlled_trepan_pair", spy)
    rep = _produce(tmp_path, _frame(420, 20), 20, "p1")
    ev = rep["evaluation"]
    assert ev["semantic_enrichment"]["evidence_strength"] == "strong", ev["semantic_enrichment"]["decision"]
    assert ev["semantic_teacher"]["teacher"] == "mlp_semantic"
    assert isinstance(seen["oracle"], SemanticTeacher)                      # o MESMO professor para os dois braços
    assert ev["experiment_audit"]["reloaded_oracle_adapter"] == "identity_same_oracle"
    assert "mlp_semantic" in ev["models"] and "mlp_original" in ev["models"]
    assert ev["ontology_stage_status"]["semantic_teacher_used"] is True
    assert rep["manifest"]["teacher_id"] == "mlp_semantic" and rep["manifest"]["test_used_for_selection"] is False
    # o bundle guarda o professor e a inferência consegue usá-lo
    predictor = load_production_bundle(tmp_path)
    sample = _frame(420, 20).drop(columns="target").iloc[:6]
    assert len(predictor.predict(sample, model="mlp_semantic")) == 6
    assert len(predictor.predict(sample, model="trepan_reloaded")) == 6


def test_production_keeps_original_teacher_when_enrichment_is_not_accepted(tmp_path):
    rep = _produce(tmp_path, _frame(160, 0, signal="mean_only"), 0, "p2")
    ev = rep["evaluation"]
    assert ev["semantic_teacher"]["teacher"] == "mlp_original"
    assert "mlp_semantic" not in ev["models"]
    assert ev["ontology_stage_status"]["semantic_teacher_used"] is False
    assert ev["ontology_stage_status"]["semantic_trepan_available"] is True  # TREPAN continua com semântica
    assert load_production_bundle(tmp_path).semantic_teacher is None
    with pytest.raises(Exception):
        load_production_bundle(tmp_path).predict(_frame(160, 0).drop(columns="target").iloc[:2], model="mlp_semantic")


def test_teacher_can_be_disabled_even_with_strong_evidence(tmp_path):
    rep = _produce(tmp_path, _frame(420, 20), 20, "p3", use_semantic_teacher=False)
    assert rep["evaluation"]["semantic_teacher"]["teacher"] == "mlp_original"
    assert rep["evaluation"]["semantic_teacher"]["reason"] == "semantic_teacher_disabled"


def test_no_enrichment_requested_means_original_teacher_and_unchanged_behaviour(tmp_path):
    rep = train_production_dataframe(_frame(160, 0), target="target", out_dir=tmp_path, seed=5, ontology=_onto(0, "p4"),
                                     require_reasoner=False, scientific_tuning=False)
    assert rep["evaluation"]["semantic_teacher"]["reason"] == "semantic_enrichment_not_requested"
    assert rep["manifest"]["teacher_id"] == "mlp_original"


def test_teacher_build_failure_falls_back_to_original_without_hiding_it(tmp_path, monkeypatch):
    import core.production_training as pt
    monkeypatch.setattr(pt, "build_semantic_teacher", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = _produce(tmp_path, _frame(420, 20), 20, "p5")
    st = rep["evaluation"]["semantic_teacher"]
    assert st["teacher"] == "mlp_original" and "teacher_build_failed:RuntimeError: boom" in st["reason"]
    assert (tmp_path / "production_bundle.joblib").exists()
