"""Amostrador C (conjunta empírica + perturbação controlada; EXPERIMENTAL, desligado por omissão; Doc. 26)."""
from __future__ import annotations

import numpy as np
import pytest

import core.trepan_original as TO
import core.trepan_reloaded_historical as TR
from core.query_sampling import JointEmpiricalDistributionModel, infer_schema, joint_sampler_context, make_joint_model_class
from core.trepan_original import ConstraintSet, FeatureDistributionModel, TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from tests.eq_support import BUDGET, ClosedFormOracle, SpyOracle, fingerprint, make_problem
from validation.experimental.mixed_data import make_mixed

SEEDS = (3, 11)


def _mixed(seed=0):
    X, y, names, transform, oracle = make_mixed(seed)
    return X, names, oracle


def test_off_by_default_and_restored_after_context():
    orig = (TO.FeatureDistributionModel, TR.FeatureDistributionModel)
    assert orig[0] is FeatureDistributionModel and orig[1] is FeatureDistributionModel
    with joint_sampler_context() as sh:
        assert issubclass(TO.FeatureDistributionModel, JointEmpiricalDistributionModel)
        assert TO.FeatureDistributionModel is TR.FeatureDistributionModel
    assert (TO.FeatureDistributionModel, TR.FeatureDistributionModel) == orig


def test_restored_even_on_exception():
    try:
        with joint_sampler_context():
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert TO.FeatureDistributionModel is FeatureDistributionModel


def test_default_original_unchanged_by_importing_sampler():
    X, names, oracle = make_problem(5)
    a, sa = _fit(TrepanOriginalClassifier, X, names, oracle, 5, joint=False)
    b, sb = _fit(TrepanOriginalClassifier, X, names, oracle, 5, joint=False)
    assert fingerprint(a, sa, X) == fingerprint(b, sb, X)


def _fit(cls, X, names, oracle, seed, *, joint, replica=0, kw=None):
    spy = SpyOracle(oracle)
    m = cls(**BUDGET, random_state=seed, **(kw or {}))
    if joint:
        with joint_sampler_context(replica=replica):
            m.fit(X, oracle=spy, feature_names=names)
    else:
        m.fit(X, oracle=spy, feature_names=names)
    return m, spy


@pytest.mark.parametrize("seed", SEEDS)
def test_deterministic_same_config_and_seed(seed):
    X, names, oracle = make_problem(seed)
    a, sa = _fit(TrepanOriginalClassifier, X, names, oracle, seed, joint=True)
    b, sb = _fit(TrepanOriginalClassifier, X, names, oracle, seed, joint=True)
    assert fingerprint(a, sa, X) == fingerprint(b, sb, X)


def test_replica_changes_only_sampler_stream():
    X, names, oracle = make_problem(4)
    a, sa = _fit(TrepanOriginalClassifier, X, names, oracle, 4, joint=True, replica=0)
    b, sb = _fit(TrepanOriginalClassifier, X, names, oracle, 4, joint=True, replica=1)
    assert sa.h.hexdigest() != sb.h.hexdigest()                        # consultas diferentes
    assert a.membership_queries_ <= a.max_queries and b.membership_queries_ <= b.max_queries      # mesmo orçamento (a árvore pode diferir)
    c, sc = _fit(TrepanOriginalClassifier, X, names, oracle, 4, joint=True, replica=1)
    assert sb.h.hexdigest() == sc.h.hexdigest()                        # mas cada réplica é determinística


def test_differs_from_marginal_sampler():
    X, names, oracle = make_problem(6)
    a, sa = _fit(TrepanOriginalClassifier, X, names, oracle, 6, joint=False)
    b, sb = _fit(TrepanOriginalClassifier, X, names, oracle, 6, joint=True)
    assert sa.h.hexdigest() != sb.h.hexdigest()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ["continuous", "mixed"])
def test_reloaded_without_ontology_equals_original_under_sampler_c(seed, kind):
    """B ≡ F: contrato EQ sob o novo amostrador (mesma classe, sementes e fluxo)."""
    X, names, oracle = make_problem(seed, kind=kind)
    o, so = _fit(TrepanOriginalClassifier, X, names, oracle, seed, joint=True)
    r, sr = _fit(TrepanReloadedClassifier, X, names, oracle, seed, joint=True, kw=dict(mirror_when_no_semantic_effect=False))
    assert so.h.hexdigest() == sr.h.hexdigest()                        # fluxo de consultas ao oráculo idêntico
    assert so.sequence == sr.sequence
    assert np.array_equal(o.predict(X), r.predict(X))
    assert o.membership_queries_ == r.membership_queries_


def test_categorical_structure_preserved_and_no_copies():
    X, names, oracle = _mixed(0)
    cls, sh = make_joint_model_class()
    m = cls(random_state=1).fit(X)
    Q = m.draw(1500)
    s = infer_schema(X)
    assert s.onehot_groups                                              # o esquema reconhece os grupos one-hot
    for g in s.onehot_groups:
        assert np.all(Q[:, g].sum(axis=1) == 1.0)                       # exclusividade preservada
        assert set(np.unique(Q[:, g])) <= {0.0, 1.0}
    for j in s.binary:
        assert set(np.unique(Q[:, j])) <= {0.0, 1.0}
    for j in s.discrete:
        assert set(np.unique(Q[:, j])) <= set(s.vocab[j])               # vocabulário autorizado
    rows = {tuple(r) for r in X}
    assert np.mean([tuple(r) in rows for r in Q]) < 0.02                # sem cópia excessiva
    for j in s.continuous:
        amp = s.hi[j] - s.lo[j]
        assert Q[:, j].min() >= s.lo[j] - 0.1 * amp - 1e-9 and Q[:, j].max() <= s.hi[j] + 0.1 * amp + 1e-9


def test_all_categorical_data_still_perturbs():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 3, 300)
    X = np.column_stack([np.eye(3)[cat], rng.integers(0, 2, 300), rng.integers(0, 5, 300)]).astype(float)
    cls, sh = make_joint_model_class()
    Q = cls(random_state=0).fit(X).draw(800)
    assert sh["audit"]["forced"] > 0                                    # perturbação forçada quando não há contínuas
    assert len({tuple(r) for r in Q}) >= 20                             # cobre o suporte (30 combinações possíveis; coincidir com linhas reais é inevitável aqui)


def test_path_constraints_and_local_models_share_schema():
    X, names, oracle = make_problem(8, kind="mixed")
    spy = SpyOracle(oracle)
    m = TrepanOriginalClassifier(**BUDGET, random_state=8)
    with joint_sampler_context() as sh:
        m.fit(X, oracle=spy, feature_names=names)
    assert sh["audit"]["models"] >= 2                                   # global + locais
    assert all(n.stats.get("queries_violating_path", 0) == 0 for n in m.iter_nodes() if hasattr(n, "stats"))


def test_deterministic_and_picklable_model():
    import pickle
    X, _, _ = _mixed(1)
    cls, _ = make_joint_model_class()
    a = cls(random_state=3).fit(X)
    b = cls(random_state=3).fit(X)
    assert np.array_equal(a.draw(200), b.draw(200))
    pickle.loads(pickle.dumps(a))


def test_onehot_groups_recovered_with_rare_cross_attribute_exclusivity():
    """Níveis raros de atributos DIFERENTES que nunca coincidem por acaso não podem fundir grupos; cada atributo é uma partição exata."""
    rng = np.random.default_rng(0)
    n, levels = 600, [3, 4, 6, 2, 5, 8]
    cols, truth, start = [], [], 0
    for L in levels:
        p = rng.dirichlet(np.ones(L) * 0.4) * 0.8 + 0.2 / L          # todos os níveis presentes (colunas constantes não são binárias)
        c = rng.choice(L, size=n, p=p)
        cols.append(np.eye(L)[c])
        truth.append(list(range(start, start + L)))
        start += L
    X = np.hstack(cols + [rng.normal(size=(n, 2))])
    s = infer_schema(X)
    assert sorted(map(tuple, s.onehot_groups)) == sorted(map(tuple, truth))
    cls, _ = make_joint_model_class()
    Q = cls(random_state=0).fit(X).draw(1000)
    for g in truth:
        assert np.all(Q[:, g].sum(axis=1) == 1.0)


def test_independent_rare_binaries_not_grouped():
    rng = np.random.default_rng(1)
    X = (rng.random((500, 5)) < 0.03).astype(float)
    assert infer_schema(X).onehot_groups == []
