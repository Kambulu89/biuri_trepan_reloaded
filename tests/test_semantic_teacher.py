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
    SemanticTeacher, build_semantic_teacher, decide_semantic_teacher, detach_ontology, make_consistency_projector,
    teacher_inputs_available,
)
from core.ontology_quality import OntologyQualityGate
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig

CFG = EnrichmentConfig(cv_folds=3, tuning_candidates=2, tuning_inner_folds=2, max_iter=150, n_bootstrap=200, random_state=5)
FAST_TREPAN = ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1, tune_structure=False, max_capacity_candidates=1, max_semantic_candidates=1)


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
    assert isinstance(seen["oracle"].model, SemanticTeacher)               # o MESMO professor (congelado) para os dois braços
    assert ev["oracle_contract"]["single_oracle_for_all_trees"] and ev["oracle_contract"]["oracle"]["builder"] == "semantic_teacher"
    assert ev["experiment_audit"]["same_oracle"] is True                  # mesmo professor nos dois braços
    assert ev["experiment_audit"]["reloaded_oracle_adapter"] == "OriginalOracleProjection"
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


# ------------------------------------------------------- Reloaded em espaço aumentado ---
def _teacher_and_data(name):
    X, y, res = _result(name=name)
    teacher = build_semantic_teacher(res, X.to_numpy(float), y, list(X.columns), config=CFG, seed=5)
    return X.to_numpy(float), y, res, teacher


def test_augment_appends_selected_features_computed_from_original_columns():
    Z, y, res, teacher = _teacher_and_data("a1")
    aug = teacher.augment(Z)
    k = len(teacher.selected_features)
    assert aug.shape == (len(Z), Z.shape[1] + k) and k > 0
    np.testing.assert_array_equal(aug[:, :Z.shape[1]], Z)
    np.testing.assert_allclose(aug[:, Z.shape[1]:], teacher.semantic_columns(Z).to_numpy(float))


def test_consistency_projector_recomputes_semantic_columns_and_keeps_original_ones():
    Z, y, res, teacher = _teacher_and_data("a2")
    n = Z.shape[1]
    proj = make_consistency_projector(teacher, n)
    rng = np.random.default_rng(0)
    # consulta "sintética" com colunas semânticas amostradas de forma independente (incoerente)
    raw = np.hstack([Z[rng.integers(0, len(Z), 50)], rng.normal(size=(50, len(teacher.selected_features))) * 99])
    out = proj(raw)
    np.testing.assert_array_equal(out[:, :n], raw[:, :n])                       # originais intactas
    np.testing.assert_allclose(out[:, n:], teacher.semantic_columns(raw[:, :n]).to_numpy(float))
    assert not np.allclose(out[:, n:], raw[:, n:])                              # e a incoerência foi corrigida
    assert out.shape == raw.shape and proj(out).shape == raw.shape
    np.testing.assert_allclose(proj(out), out)                                  # idempotente


def test_extend_semantic_inputs_has_no_double_counting_and_documented_relatedness():
    from core.semantic_metadata import RelatednessConfig, audit_double_counting, extend_semantic_inputs
    Z, y, res, teacher = _teacher_and_data("a3")
    names = list(res.input_features)
    n = len(names)
    weights = np.linspace(1.0, 1.3, n); groups = ["G"] * n; entities = [f"E{i}" for i in range(n)]
    base = np.eye(n); base[0, 1] = base[1, 0] = 0.8
    cfg = RelatednessConfig()
    n2, w2, g2, e2, m2, rec = extend_semantic_inputs(names, weights, groups, entities, base, teacher.processor,
                                                      teacher.selected_features, cfg)
    k = len(teacher.selected_features)
    assert len(n2) == len(w2) == len(g2) == len(e2) == n + k and m2.shape == (n + k, n + k)
    assert np.allclose(m2, m2.T) and np.allclose(np.diag(m2), 1.0) and m2.min() >= 0 and m2.max() <= 1
    np.testing.assert_array_equal(m2[:n, :n], base)            # o bloco original não muda
    np.testing.assert_array_equal(w2[:n], weights)
    idx = {nm: i for i, nm in enumerate(n2)}
    for r in rec:
        d = idx[r["feature"]]
        assert w2[d] <= max(w2[idx[s]] for s in r["sources"]) + 1e-12   # sem dupla contagem
        for s in r["sources"]:
            assert m2[d, idx[s]] == cfg.derived_to_source
    meta = [{"feature_name": nm, "origin": "ontology" if i >= n else "original", "ontology_entities": [],
             "source_features": next((r["sources"] for r in rec if r["feature"] == nm), [])}
            for i, nm in enumerate(n2)]
    assert audit_double_counting(meta, w2) == []


def test_production_reloaded_arm_uses_augmented_space_with_consistent_queries(tmp_path, monkeypatch):
    import core.production_training as pt
    seen = {}
    original = pt.fit_controlled_trepan_pair

    def spy(*a, **k):
        seen.update(k)
        return original(*a, **k)
    monkeypatch.setattr(pt, "fit_controlled_trepan_pair", spy)
    df = _frame(420, 20)
    rep = _produce(tmp_path, df, 20, "r1")
    ev = rep["evaluation"]
    assert ev["semantic_teacher"]["teacher"] == "mlp_semantic"
    space = ev["reloaded_feature_space"]
    assert space["space"] == "augmented" and space["semantic_features"]
    n = df.shape[1] - 1
    assert space["n_features"] == n + len(space["semantic_features"])
    assert seen["original_feature_indices"] == tuple(range(n)) and seen["query_projector"] is not None
    assert seen["reloaded_X_train"].shape[1] == space["n_features"]
    # braço Original mantém o espaço original; o protocolo continua pareado e sem espelho falso
    audit = ev["experiment_audit"]
    assert audit["base_feature_count"] == n and audit["reloaded_feature_count"] == space["n_features"]
    assert audit["same_oracle"] is True and audit["semantic_effect_mirror_applied"] is False
    assert audit["config"]["mirror_when_no_semantic_effect"] is False
    assert rep["manifest"]["reloaded_feature_space"] == "augmented" and rep["manifest"]["test_used_for_selection"] is False
    # a inferência reconstrói o espaço aumentado a partir do bundle
    predictor = load_production_bundle(tmp_path)
    sample = df.drop(columns="target").iloc[:8]
    z = predictor._z(sample)
    direct = predictor.reloaded.predict(predictor.semantic_teacher.augment(z))
    np.testing.assert_array_equal(predictor.predict(sample, model="trepan_reloaded"), direct)
    assert predictor.explain(sample.iloc[[0]])["model"] == "trepan_reloaded"
    assert len(predictor.predict(sample, model="trepan_original")) == 8


def test_augmentation_can_be_disabled_and_keeps_the_original_space(tmp_path):
    rep = _produce(tmp_path, _frame(420, 20), 20, "r2", augment_reloaded_space=False)
    ev = rep["evaluation"]
    assert ev["semantic_teacher"]["teacher"] == "mlp_semantic"      # professor semântico, mas...
    assert ev["reloaded_feature_space"]["space"] == "original"
    assert ev["experiment_audit"]["reloaded_feature_count"] == ev["experiment_audit"]["base_feature_count"]
    assert rep["manifest"]["reloaded_feature_space"] == "original"


def test_reloaded_stays_in_original_space_when_teacher_is_not_used(tmp_path):
    rep = _produce(tmp_path, _frame(160, 0, signal="mean_only"), 0, "r3")
    assert rep["evaluation"]["reloaded_feature_space"]["space"] == "original"
    assert rep["manifest"]["reloaded_feature_space"] == "original"
