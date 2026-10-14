"""Amostradores experimentais de consultas: esquema inferido, validade categórica, coerência das derivadas, restrições, determinismo (Doc. 24)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.experimental.mixed_data import make_mixed
from validation.experimental.query_samplers import (DomainConstrained, JointEmpiricalSampler, MarginalSampler, NeighborhoodSampler, infer_schema)


@pytest.fixture(scope="module")
def data():
    return make_mixed(0)


def test_schema_inference_is_generic_and_correct(data):
    X, _, names, _, _ = data
    s = infer_schema(X)
    assert sorted(s.onehot_groups) == [[3, 4, 5], [6, 7, 8, 9]]
    assert s.binary == [10, 11] and set(s.discrete) == {12, 13} and s.continuous == [0, 1, 2]
    assert infer_schema(X, ordinal_columns=[12]).ordinal == [12]


def test_independent_rare_binaries_are_not_misread_as_onehot():
    r = np.random.default_rng(0)
    X = np.column_stack([(r.random(500) < 0.05).astype(float) for _ in range(4)] + [r.normal(size=500)])
    assert infer_schema(X).onehot_groups == []


def _valid(X, rows, s):
    ok_oh = all(np.all(rows[:, g].sum(1) == 1.0) for g in s.onehot_groups)
    ok_vocab = all(np.isin(rows[:, j], s.vocab[j]).all() for j in s.discrete)
    ok_bin = all(np.isin(rows[:, j], [0.0, 1.0]).all() for j in s.binary)
    return ok_oh, ok_vocab, ok_bin


def test_marginal_historical_violates_joint_categories_others_respect_them(data):
    X, _, _, T, _ = data
    s = infer_schema(X)
    A = MarginalSampler(X, random_state=0).draw(800)
    assert not _valid(X, A, s)[0]                                            # limitação herdada do Original
    for smp in (MarginalSampler(X, joint_groups=True, random_state=0), JointEmpiricalSampler(X, random_state=0), NeighborhoodSampler(X, random_state=0)):
        assert all(_valid(X, smp.draw(800), s)), type(smp).__name__
    assert _valid(X, A, s)[1] and _valid(X, A, s)[2]                         # vocabulário e binárias válidos mesmo no marginal


def test_derived_features_are_exactly_recomputed(data):
    X, _, _, T, _ = data
    for smp in (MarginalSampler(X, transform=T, random_state=0), JointEmpiricalSampler(X, transform=T, random_state=0), NeighborhoodSampler(X, transform=T, random_state=0)):
        q = smp.draw(500)
        assert q.shape[1] == 16 and np.array_equal(q, T(q[:, :14]))


def test_historical_enriched_marginal_breaks_derived_relations(data):
    X, _, _, T, _ = data
    q = MarginalSampler(X, transform=T, on_enriched=True, random_state=0).draw(500)
    assert (np.abs(q[:, 14:] - T(q[:, :14])[:, 14:]).max(axis=1) > 1e-9).mean() > 0.9


def test_path_constraints_and_explicit_domain_constraints_are_respected(data):
    X, _, _, T, _ = data

    class C:
        def accepts(self, A):
            return A[:, 0] > 0.0
    for smp in (MarginalSampler(X, transform=T, random_state=0), JointEmpiricalSampler(X, transform=T, random_state=0), NeighborhoodSampler(X, transform=T, random_state=0)):
        assert (smp.draw(300, C())[:, 0] > 0).all()
    # restrição de domínio explícita (axioma: categoria n3_c ativa ⇒ z2 ≥ 0)
    ax = lambda A: ~((A[:, 5] == 1.0) & (A[:, 2] < 0.0))
    raw = MarginalSampler(X, joint_groups=True, random_state=1)
    assert (~ax(raw.draw(2000))).mean() > 0.05                                  # sem a restrição há violações
    dc = DomainConstrained(MarginalSampler(X, joint_groups=True, random_state=1), [ax])
    assert ax(dc.draw(2000)).all() and 0 < dc.acceptance_rate < 1


def test_determinism_and_seed_dependence(data):
    X, _, _, T, _ = data
    a = JointEmpiricalSampler(X, transform=T, random_state=3).draw(200)
    assert np.array_equal(a, JointEmpiricalSampler(X, transform=T, random_state=3).draw(200))
    assert not np.array_equal(a, JointEmpiricalSampler(X, transform=T, random_state=4).draw(200))


def test_joint_empirical_and_neighborhood_do_not_just_copy_real_rows(data):
    X, _, _, _, _ = data
    cont = [0, 1, 2]
    real = {tuple(np.round(r[cont], 9)) for r in X}
    for smp in (JointEmpiricalSampler(X, random_state=0), NeighborhoodSampler(X, random_state=0)):
        q = smp.draw(500)
        copies = np.mean([tuple(np.round(r[cont], 9)) in real for r in q])
        assert copies < 0.02, (type(smp).__name__, copies)                       # explora regiões novas, não só copia linhas
        assert len({tuple(np.round(r, 6)) for r in q}) > 450                     # diversidade
