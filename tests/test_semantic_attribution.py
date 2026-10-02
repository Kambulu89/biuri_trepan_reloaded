"""Gate de atribuição à ontologia e controlo de features derivadas aleatórias."""
import numpy as np
import pandas as pd
import pytest

from core.semantic_attribution import (
    ATTRIBUTED, NOT_ATTRIBUTED, NOT_ATTRIBUTED_TIES, NOT_EVALUATED, AttributionConfig, evaluate_attribution,
)


class _Oracle:
    def predict(self, Z):
        return (np.asarray(Z)[:, 0] > 0).astype(int)


class _Const:
    def __init__(self, fn): self.fn = fn
    def predict(self, Z): return self.fn(np.asarray(Z))


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3)); y = (Z[:, 0] > 0).astype(int)
    return Z, y


def _variant(fn):
    return lambda Ztr, ytr: _Const(fn)


PERFECT = lambda Z: (Z[:, 0] > 0).astype(int)
NOISY = lambda Z: (Z[:, 1] > 0).astype(int)  # independente do professor


def test_attributed_when_real_beats_controls_clearly():
    Z, y = _data()
    rep = evaluate_attribution(Z, y, _Oracle(), _variant(PERFECT), lambda k: _variant(NOISY))
    assert rep["attributed"] and rep["status"] == ATTRIBUTED
    assert rep["mean_gain_over_controls"] > 0.2 and rep["wins"] == rep["comparisons"] and rep["ties"] == 0
    assert rep["test_used"] is False and rep["scope"] == "training_internal_cv_only"


def test_all_ties_is_not_attribution():
    Z, y = _data()
    rep = evaluate_attribution(Z, y, _Oracle(), _variant(PERFECT), lambda k: _variant(PERFECT))
    assert not rep["attributed"] and rep["status"] == NOT_ATTRIBUTED_TIES and rep["ties"] == rep["comparisons"]


def test_real_worse_than_controls_is_rejected():
    Z, y = _data()
    rep = evaluate_attribution(Z, y, _Oracle(), _variant(NOISY), lambda k: _variant(PERFECT))
    assert not rep["attributed"] and rep["status"] == NOT_ATTRIBUTED and rep["mean_gain_over_controls"] < 0


def test_ties_count_against_so_a_single_win_is_not_attribution():
    """Regressão: antes, 1 vitória + 8 empates passava (1/1 entre não empatadas = 100%)."""
    Z, y = _data(300)
    calls = {"n": 0}

    def control(k):
        def make(Ztr, ytr):
            calls["n"] += 1
            # idêntico ao real exceto na 1.ª comparação, em que é pior
            if calls["n"] == 1:
                return _Const(lambda A: np.where(np.arange(len(A)) % 3 == 0, 1 - PERFECT(A), PERFECT(A)))
            return _Const(PERFECT)
        return make
    rep = evaluate_attribution(Z, y, _Oracle(), _variant(PERFECT), control, AttributionConfig(controls=3))
    assert rep["wins"] == 1 and rep["losses"] == 0 and rep["ties"] == rep["comparisons"] - 1
    assert rep["mean_gain_over_controls"] > 0               # a média é positiva...
    assert rep["win_fraction"] == pytest.approx(1 / rep["comparisons"])
    assert not rep["attributed"] and rep["status"] == NOT_ATTRIBUTED   # ...mas não há evidência


def test_failing_real_variant_is_not_evaluated_instead_of_crashing():
    Z, y = _data()

    def boom(Ztr, ytr):
        raise RuntimeError("membership queries")
    rep = evaluate_attribution(Z, y, _Oracle(), boom, lambda k: _variant(NOISY))
    assert rep["status"] == NOT_EVALUATED and rep["attributed"] is False and "RuntimeError" in rep["reason"]


def test_failing_controls_are_skipped_not_counted_for_the_real():
    Z, y = _data()

    def bad_control(k):
        def make(Ztr, ytr):
            if k % 2 == 0:
                raise RuntimeError("fail")
            return _Const(NOISY)
        return make
    rep = evaluate_attribution(Z, y, _Oracle(), _variant(PERFECT), bad_control, AttributionConfig(controls=2))
    assert rep["failed_controls"] > 0 and rep["comparisons"] > 0 and rep["attributed"]
    all_fail = evaluate_attribution(Z, y, _Oracle(), _variant(PERFECT), lambda k: (lambda a, b: (_ for _ in ()).throw(RuntimeError())))
    assert all_fail["status"] == NOT_EVALUATED and all_fail["reason"] == "no_control_could_be_fitted"


def test_positive_mean_but_losing_most_comparisons_is_rejected():
    Z, y = _data(300)
    state = {"i": 0}

    def control(k):
        # um controlo muito fraco e vários ligeiramente melhores que o real
        def make(Ztr, ytr):
            state["i"] += 1
            if k % 3 == 0:
                return _Const(lambda A: np.zeros(len(A), dtype=int))                       # péssimo
            return _Const(lambda A: np.where(np.arange(len(A)) % 25 == 0, 1 - PERFECT(A), PERFECT(A)))  # ~96%
        return make
    real = _variant(lambda A: np.where(np.arange(len(A)) % 12 == 0, 1 - PERFECT(A), PERFECT(A)))      # ~92%
    rep = evaluate_attribution(Z, y, _Oracle(), real, control, AttributionConfig(controls=3))
    assert rep["mean_gain_over_controls"] > 0 and rep["win_fraction"] < 0.6
    assert not rep["attributed"]


def test_thresholds_are_configurable():
    Z, y = _data()
    strict = AttributionConfig(min_gain=0.99)
    rep = evaluate_attribution(Z, y, _Oracle(), _variant(PERFECT), lambda k: _variant(NOISY), strict)
    assert not rep["attributed"] and rep["wins"] == rep["comparisons"]  # vence tudo mas não excede min_gain


def test_too_few_samples_per_class_is_not_evaluated():
    Z = np.random.default_rng(0).normal(size=(5, 3)); y = np.array([0, 0, 0, 0, 1])
    rep = evaluate_attribution(Z, y, _Oracle(), _variant(PERFECT), lambda k: _variant(NOISY))
    assert rep["status"] == NOT_EVALUATED and rep["attributed"] is False


def test_gate_is_deterministic_and_never_receives_test_rows():
    Z, y = _data()
    seen = []

    def real(Ztr, ytr):
        seen.append(len(Ztr)); return _Const(PERFECT)
    a = evaluate_attribution(Z, y, _Oracle(), real, lambda k: _variant(NOISY))
    b = evaluate_attribution(Z, y, _Oracle(), real, lambda k: _variant(NOISY))
    assert a == b and all(n < len(Z) for n in seen)  # só subconjuntos internos do treino


# ------------------------------------------------------- controlo de features aleatórias ---
owlready2 = pytest.importorskip("owlready2")

from core.ontology_quality import OntologyQualityGate
from core.semantic_attribution import make_random_derived_teacher
from core.semantic_enrichment import EnrichmentConfig, evaluate_semantic_enrichment
from core.semantic_teacher import build_semantic_teacher

CFG = EnrichmentConfig(cv_folds=3, tuning_candidates=2, tuning_inner_folds=2, max_iter=150, n_bootstrap=200, random_state=5)


def _teacher(name="ctl"):
    onto = owlready2.World().get_ontology(f"http://test.org/{name}.owl")
    with onto:
        type("statisticRole", (owlready2.AnnotationProperty,), {})
        type("measurementFamily", (owlready2.AnnotationProperty,), {})
        M = type("Morphology", (owlready2.Thing,), {}); B = type("Background", (owlready2.Thing,), {})
        for role in ("mean", "error", "worst"):
            p = type(f"hasSize{role.capitalize()}", (owlready2.DataProperty,), {"range": [float], "domain": [M]})
            p.statisticRole = [role]; p.measurementFamily = ["Size"]
        for i in range(20):
            type(f"hasBackgroundVariable{i:02d}Level", (owlready2.DataProperty,), {"range": [float], "domain": [B]})
    rng = np.random.default_rng(2)
    n = 300
    mean = rng.uniform(5, 15, n); delta = rng.uniform(0, 1, n)
    X = pd.DataFrame({"hasSizeMean": mean, "hasSizeError": rng.uniform(.1, 1, n), "hasSizeWorst": mean * (1 + delta)})
    for i in range(20):
        X[f"hasBackgroundVariable{i:02d}Level"] = rng.normal(size=n)
    y = np.where(delta > np.median(delta), "yes", "no")
    q = OntologyQualityGate().evaluate(list(X.columns), onto, require_reasoner=False)
    res = evaluate_semantic_enrichment(X, y, onto, quality_report=q, config=CFG)
    assert res.report["semantic_mlp_accepted"]
    return build_semantic_teacher(res, X.to_numpy(float), y, list(X.columns), config=CFG, seed=5), X.to_numpy(float)


def test_random_derived_control_keeps_shape_and_original_columns():
    teacher, Z = _teacher("c1")
    ctrl = make_random_derived_teacher(teacher, Z, seed=1)
    real_aug, ctrl_aug = teacher.augment(Z), ctrl.augment(Z)
    assert real_aug.shape == ctrl_aug.shape and ctrl.selected_features == teacher.selected_features
    np.testing.assert_array_equal(ctrl_aug[:, :Z.shape[1]], Z)
    assert np.isfinite(ctrl_aug).all()
    assert not np.allclose(real_aug[:, Z.shape[1]:], ctrl_aug[:, Z.shape[1]:])


def test_random_derived_control_drops_ontology_meaning_and_is_reproducible():
    teacher, Z = _teacher("c2")
    a = make_random_derived_teacher(teacher, Z, seed=1); b = make_random_derived_teacher(teacher, Z, seed=1)
    c = make_random_derived_teacher(teacher, Z, seed=2)
    np.testing.assert_array_equal(a.augment(Z), b.augment(Z))
    assert not np.array_equal(a.augment(Z), c.augment(Z))
    sel = set(teacher.selected_features)
    for spec in a.processor.feature_specs_:
        if spec["name"] in sel:
            assert spec["provenance"] == "random_control" and not spec.get("family") and not spec.get("owl_entities")
    assert a.audit_["control"] == "random_derived_features"
    assert all(s.get("family") for s in teacher.processor.feature_specs_ if s["name"] in sel and s["kind"] == "relational")  # o real não foi alterado


def test_random_control_statistics_are_train_only():
    teacher, Z = _teacher("c3")
    ctrl = make_random_derived_teacher(teacher, Z[:200], seed=3)
    agg = [s for s in ctrl.processor.feature_specs_ if s["name"] in set(teacher.selected_features)
           and s["kind"] == "hierarchical_aggregate"]
    frame = pd.DataFrame(Z[:200][:, teacher.z_indices], columns=teacher.input_columns)
    for spec in agg:
        for col in spec["sources"]:
            assert spec["centers"][col] == pytest.approx(float(frame[col].mean()))   # só as 200 linhas de treino
    changed = Z.copy(); changed[200:] *= 1000.0                                         # linhas "de teste" alteradas
    np.testing.assert_array_equal(make_random_derived_teacher(teacher, changed[:200], seed=3).augment(Z[:5]),
                                  ctrl.augment(Z[:5]))
