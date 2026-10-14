"""Amostragem ontologicamente COERENTE (EXPERIMENTAL, desligada por omissão; Doc. 22).

Distingue COERÊNCIA ALGÉBRICA (derivadas = transformador(originais), garantida por construção) de PLAUSIBILIDADE ESTATÍSTICA das originais
(não garantida: o modelo marginal do Original continua a ser o mesmo)."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from core.trepan_original import ConstraintSet, FeatureDistributionModel, TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier, _OriginalSpaceModel
from tests.eq_support import BUDGET, SpyOracle, ClosedFormOracle, digest

P = 6                      # colunas originais
NAMES = [f"a{i}" for i in range(P)] + ["d_mean01", "d_prod23"]


class LiftedOracle:
    """Oráculo que só vê as colunas originais (como ``OriginalOracleProjection``)."""
    def __init__(self, inner):
        self.inner, self.classes_ = inner, inner.classes_

    def predict(self, X):
        return self.inner.predict(np.asarray(X)[:, :P])

    def predict_proba(self, X):
        return self.inner.predict_proba(np.asarray(X)[:, :P])


def make(seed, n=240):
    rng = np.random.default_rng(seed)
    Xo = rng.normal(size=(n, P))
    mu, sd = Xo[:, :2].mean(0), Xo[:, :2].std(0)

    def derive(Z):
        Z = np.asarray(Z, dtype=float)
        o = Z[:, :P]
        return np.column_stack([o, ((o[:, 0] - mu[0]) / sd[0] + (o[:, 1] - mu[1]) / sd[1]) / 2.0, o[:, 2] * o[:, 3]])
    return Xo, derive, ClosedFormOracle(P, 3, seed + 1)


def fit_tree(seed, coherent, *, enriched=True, kw=None, derive_override=None):
    Xo, derive, base = make(seed)
    X = derive(Xo) if enriched else Xo
    oracle = LiftedOracle(base) if enriched else base
    spy = SpyOracle(oracle)
    m = TrepanReloadedClassifier(**BUDGET, random_state=seed, **(kw or {}))
    fit_kw = dict(query_projector=derive_override or derive)
    if coherent:
        fit_kw["coherent_original_indices"] = list(range(P))
    m.fit(X, oracle=spy, feature_names=NAMES if enriched else NAMES[:P], **fit_kw)
    return m, spy, X, derive


NEUTRAL = dict(error_focused_refinement=False, alpha=0.0, beta=0.0, mirror_when_no_semantic_effect=False)


def test_default_is_off():
    assert TrepanReloadedClassifier().coherent_sampling is False


def test_incompatible_with_b2c_and_requires_inputs():
    with pytest.raises(ValueError):
        TrepanReloadedClassifier(coherent_sampling=True, statistical_fallback=True)
    Xo, derive, base = make(0)
    m = TrepanReloadedClassifier(**BUDGET, coherent_sampling=True, random_state=0)
    with pytest.raises(ValueError):
        m.fit(derive(Xo), oracle=LiftedOracle(base), feature_names=NAMES, query_projector=derive)       # falta coherent_original_indices


@pytest.mark.parametrize("seed", [0, 3])
def test_flag_has_no_effect_without_effective_ontology(seed):
    Xo, _, base = make(seed)
    so, sr = SpyOracle(base), SpyOracle(base)
    o = TrepanOriginalClassifier(**BUDGET, random_state=seed).fit(Xo, oracle=so, feature_names=NAMES[:P])
    r = TrepanReloadedClassifier(**BUDGET, random_state=seed, coherent_sampling=True).fit(Xo, oracle=sr, feature_names=NAMES[:P])
    assert r.ontology_state_ == "S0" and digest(o.export_rules()) == digest(r.export_rules()) and so.stream_digest() == sr.stream_digest()


@pytest.mark.parametrize("seed", [0, 3, 5])
def test_identity_transformer_reproduces_original_bitwise(seed):
    """Cópia fiel de ``_decision_sample``: com transformador identidade (sem derivadas) e camada semântica neutra, árvore, previsões e FLUXO AO ORÁCULO
    ≡ Original (os sorteios no espaço original coincidem com os do Original)."""
    Xo, _, base = make(seed)

    import hashlib

    class PredictOnly:                      # regista só as chamadas ``predict`` (o perfil de erro do Reloaded pede ``predict_proba`` em linhas conhecidas)
        classes_ = base.classes_
        def __init__(self):
            self.h, self.n = hashlib.sha256(), 0
        def predict(self, X):
            a = np.ascontiguousarray(np.asarray(X, float)); self.h.update(str(a.shape).encode() + a.tobytes()); self.n += 1
            return base.predict(X)
    so, sr = PredictOnly(), PredictOnly()
    o = TrepanOriginalClassifier(**BUDGET, random_state=seed).fit(Xo, oracle=so, feature_names=NAMES[:P])
    r = TrepanReloadedClassifier(**BUDGET, random_state=seed, coherent_sampling=True, **NEUTRAL)
    r.fit(Xo, oracle=sr, feature_names=NAMES[:P], query_projector=lambda Z: np.asarray(Z, float), coherent_original_indices=list(range(P)))
    assert r.ontology_state_ == "S3"
    assert digest(o.export_rules()) == digest(r.export_rules())
    assert np.array_equal(o.predict(Xo), r.predict(Xo))
    assert so.h.hexdigest() == sr.h.hexdigest() and so.n == sr.n          # consultas ``predict`` idênticas, byte a byte
    assert r.coherent_sampling_audit_["rows"] == int(r.membership_queries_) > 0


def _collect(model_cls, seed, coherent):
    seen = []

    class Cap(model_cls):
        def _draw_membership_queries(self, model, n, constraints, node):
            out = super()._draw_membership_queries(model, n, constraints, node)
            seen.append((np.asarray(out), node.constraints))
            return out
    Xo, derive, base = make(seed)
    m = Cap(**BUDGET, random_state=seed, **NEUTRAL, coherent_sampling=coherent)
    m.fit(derive(Xo), oracle=LiftedOracle(base), feature_names=NAMES, query_projector=derive,
          **({"coherent_original_indices": list(range(P))} if coherent else {}))
    return seen, derive


def test_coherent_queries_are_algebraically_exact_and_respect_path_constraints():
    for seed in (0, 1, 2):
        seen, derive = _collect(TrepanReloadedClassifier, seed, True)
        assert seen
        for q, cons in seen:
            assert np.array_equal(q, derive(q))                  # violação algébrica = 0 (exatamente)
            assert cons.accepts(q).all()                         # restrições do caminho (também sobre derivadas) cumpridas


def test_original_marginal_sampling_in_enriched_space_violates_relations():
    """Diagnóstico (não é B2/B3): o sorteio marginal do Original no espaço enriquecido quebra as relações determinísticas."""
    Xo, derive, _ = make(0)
    Z = derive(Xo)
    q = FeatureDistributionModel(random_state=0).fit(Z).draw(500, ConstraintSet())
    viol = np.abs(q - derive(q)).max(axis=1) > 1e-9
    assert viol.mean() > 0.9


def test_original_space_model_draws_equal_originals_draws():
    Xo, derive, _ = make(0)
    Z = derive(Xo)
    ref = FeatureDistributionModel(random_state=7).fit(Xo).draw(300, ConstraintSet())
    got = _OriginalSpaceModel(7, list(range(P)), Z.shape[1], derive).fit(Z).draw(300, ConstraintSet())
    assert np.array_equal(got[:, :P], ref) and np.array_equal(got, derive(got))


def test_deterministic_seed_dependent_and_picklable():
    a, _, X, _ = fit_tree(3, True, kw=dict(alpha=0.35, beta=0.2, mirror_when_no_semantic_effect=False))
    b, _, _, _ = fit_tree(3, True, kw=dict(alpha=0.35, beta=0.2, mirror_when_no_semantic_effect=False))
    assert digest(a.export_rules()) == digest(b.export_rules()) and a.membership_queries_ == b.membership_queries_
    assert a._coherent_train_X is None and a._coherent_global_model is None
    r2 = pickle.loads(pickle.dumps(a))
    assert np.array_equal(r2.predict(X), a.predict(X))


def test_off_is_bitwise_previous_behaviour():
    a, sa, _, _ = fit_tree(3, False, kw=dict(alpha=0.35, beta=0.2, mirror_when_no_semantic_effect=False))
    b, sb, _, _ = fit_tree(3, False, kw=dict(alpha=0.35, beta=0.2, mirror_when_no_semantic_effect=False, coherent_sampling=False))
    assert digest(a.export_rules()) == digest(b.export_rules()) and sa.stream_digest() == sb.stream_digest()


def test_marginal_sampling_does_not_preserve_joint_categorical_constraints():
    """Limite herdado do Original (NÃO corrigido por esta variante): colunas one-hot amostradas marginalmente produzem combinações inválidas.
    A variante coerente garante relações algébricas das derivadas, não a validade conjunta das categorias."""
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 3, size=400)
    Xo = np.eye(3)[cat]
    q = FeatureDistributionModel(random_state=0).fit(Xo).draw(1000, ConstraintSet())
    invalid = (q.sum(axis=1) != 1.0).mean()
    assert invalid > 0.3
    model = _OriginalSpaceModel(0, [0, 1, 2], 3, lambda Z: np.asarray(Z, float)).fit(Xo)
    assert (model.draw(1000, ConstraintSet()).sum(axis=1) != 1.0).mean() > 0.3          # mesma limitação, declarada


def test_integration_owl_and_mlp_on_a_dev_dataset_zero_algebraic_violations():
    """Integração com OWL real + MLP congelado (dataset DEV; sem lógica por dataset): as consultas da variante coerente têm violação algébrica 0 e o
    oráculo recebe o mesmo espaço de entrada que nos braços comparáveis. Salta se o ambiente OWL (Java/reasoner) não estiver disponível."""
    from sklearn.neural_network import MLPClassifier
    from core.benchmark.semantic import OwlSemanticProvider
    from core.controlled_trepan_experiment import OriginalOracleProjection
    from validation.benchmark import datasets as ds_mod
    spec = ds_mod.REGISTRY["iris"]
    ds, _t, _c = spec.load()
    tr = np.arange(0, len(ds.X), 2)
    Xtr, ytr = ds.X[tr], ds.y[tr]
    ctx = OwlSemanticProvider(str(spec.ontology_path)).build(Xtr, ds.feature_names, 0)
    if not ctx.available:
        pytest.skip("contexto OWL indisponível: " + ctx.reason)
    mlp = MLPClassifier((8,), max_iter=300, random_state=0).fit(Xtr, ytr)
    oracle = OriginalOracleProjection(mlp, ctx.orig_idx)
    seen = []

    class Cap(TrepanReloadedClassifier):
        def _draw_membership_queries(self, model, n, constraints, node):
            out = super()._draw_membership_queries(model, n, constraints, node)
            seen.append(np.asarray(out))
            return out
    budget = dict(BUDGET)
    Z = ctx.enrich(Xtr)
    t = Cap(**budget, random_state=0, coherent_sampling=True, mirror_when_no_semantic_effect=False)
    t.fit(Z, oracle=oracle, feature_names=ctx.enriched_names, semantic_feature_weights=ctx.weights, semantic_feature_groups=ctx.groups,
          semantic_relatedness_matrix=ctx.relatedness, semantic_feature_depths=ctx.depths, query_projector=ctx.query_projector(),
          coherent_original_indices=list(ctx.orig_idx))
    assert seen and t.ontology_state_ == "S3"
    for q in seen:
        assert q.shape[1] == Z.shape[1]
        assert np.array_equal(q, ctx.enrich(q[:, ctx.orig_idx]))
    assert np.array_equal(oracle.predict(Z), mlp.predict(Xtr))
