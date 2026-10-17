"""Controlos ontológicos do Doc. 29: permutação acoplada, grupos aleatórios desacoplados, porta de validade, retenção de grupos, escala e dimensão preservadas."""
from __future__ import annotations

import numpy as np
import pytest

from core.benchmark.semantic import (ConventionalFeatureProvider, DecoupledRandomOwlProvider, GroupSemanticProvider, NamePermutedOwlProvider, build_valid_control, chance_retention,
                                     column_groups, control_validity, group_retention, scale_strata, select_destructive_permutation)

NAMES = [f"f{i}" for i in range(12)]
GROUPS = {"G1": NAMES[:4], "G2": NAMES[4:7], "G3": NAMES[7:12]}


def data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)) * np.array([1, 1, 1, 1, 10, 10, 10, 100, 100, 100, 100, 100])
    return X + rng.normal(size=(n, 1)) * 0.3


def base():
    return GroupSemanticProvider(GROUPS)


def test_retention_and_chance_definition():
    g = ["a", "a", "b", "b", None]
    assert group_retention(g, np.array([0, 1, 2, 3, 4])) == 1.0                   # identidade retém tudo
    assert group_retention(g, np.array([2, 3, 0, 1, 4])) == 1.0                   # trocar blocos inteiros mantém co-agrupamento
    assert group_retention(g, np.array([0, 2, 1, 3, 4])) == 0.0
    assert chance_retention(["a", "a", "b", "b", None]) == pytest.approx((2 + 2) / (5 * 4))


def test_selected_permutation_is_deterministic_structural_only_and_not_better_than_chance():
    g = column_groups(base().build(data(), NAMES))
    p1, m1 = select_destructive_permutation(g, 123)
    p2, m2 = select_destructive_permutation(g, 123)
    assert np.array_equal(p1, p2) and m1 == m2 and m1["met_criterion"] and m1["retention"] <= m1["chance_retention"] + 1e-12 and m1["fixed_point_fraction"] < 1.0
    p3, _ = select_destructive_permutation(g, 9999)
    assert not np.array_equal(p1, p3)


def test_scale_strata_permutes_only_within_similar_scales():
    X = data()
    st = scale_strata(X, min_block=4)
    assert len(set(st)) == 3
    g = column_groups(base().build(X, NAMES))
    p, _ = select_destructive_permutation(g, 5, strata=st)
    assert all(st[p[i]] == st[i] for i in range(12))


def test_name_permuted_control_preserves_dimension_group_sizes_and_scale_and_destroys_correspondence():
    X = data()
    b = base()
    real = b.build(X, NAMES)
    P = NamePermutedOwlProvider(b, 77).build(X, NAMES, real_ctx=real)
    assert P.available and len(P.onto_idx) == len(real.onto_idx) and P.n_features == real.n_features
    assert sorted(g for g in column_groups(P) if g) == sorted(g for g in column_groups(real) if g)    # mesma distribuição de tamanhos de grupo
    assert P.base_names == NAMES and P.enriched_names[:12] == NAMES
    v = control_validity(real, P, X)
    assert v["valid"], v
    assert column_groups(P) != column_groups(real)                                                    # correspondência coluna↔grupo destruída
    # a composição de cada derivada mudou (acoplada à nova rotulagem): a derivada do grupo G1 já não é a média das colunas f0..f3
    d_real, d_perm = real.enrich(X)[:, 12:], P.enrich(X)[:, 12:]
    assert not np.allclose(d_real, d_perm)
    assert np.abs(d_perm.mean(0)).max() < 1e-8 and d_perm.std(0).max() < 2.0                          # escala correta (reajustado ao treino), sem artefactos


def test_random_groups_control_is_decoupled_and_independent_of_the_coupled_one():
    X = data()
    b = base()
    real = b.build(X, NAMES)
    P = NamePermutedOwlProvider(b, 77).build(X, NAMES, real_ctx=real)
    D = DecoupledRandomOwlProvider(b, 77, 4242).build(X, NAMES, real_ctx=real)
    assert D.available and len(D.onto_idx) == len(real.onto_idx)
    assert np.allclose(P.enrich(X), D.enrich(X))                                                      # mesma composição (permutação A igual)
    assert D.groups != P.groups or D.structure_signature() != P.structure_signature()                 # rótulos independentes (permutação B)
    assert sorted(g for g in D.groups[:12] if g) == sorted(g for g in real.groups[:12] if g)           # tamanhos de grupo preservados


def test_validity_gate_rejects_wrong_dimension_and_scale_and_fallback_is_explicit():
    X = data()
    b = base()
    real = b.build(X, NAMES)

    class Broken:
        def build(self, X_train, names, seed=0):
            c = b.build(X_train, names, seed)
            if names != NAMES:                                     # permutações produzem uma derivada a menos e escalas absurdas
                tr = c.transform
                c.transform = lambda Z: tr(Z)[:, :-1] * 1e9
                c.enriched_names = c.enriched_names[:-1]
                c.onto_idx = c.onto_idx[:-1]
            return c
    v = control_validity(real, Broken().build(X, list(reversed(NAMES))), X)
    assert not v["valid"] and any("dimensão" in r for r in v["reasons"])
    ctx, meta = build_valid_control("permuted", Broken(), real, X, NAMES, 0, 11, 0, max_attempts=3)
    assert meta["variant"] == "structure_only_fallback" and len(meta["attempts"]) == 3 and all(not a["valid"] for a in meta["attempts"])
    assert ctx.info["control_variant"] == "structure_only_fallback"
    assert control_validity(real, ctx, X)["valid"]                                                    # o fallback tem dimensão e escala exatas
    ok_ctx, ok_meta = build_valid_control("permuted", b, real, X, NAMES, 0, 11, 0)
    assert ok_meta["variant"] == "full" and ok_ctx.info["control_variant"] == "full"


def test_conventional_control_has_no_semantic_structure_and_documents_dimension():
    X = data()
    c = ConventionalFeatureProvider(n_target=3).build(X, NAMES)
    assert c.available and np.allclose(c.weights, 1.0) and all(g is None for g in c.groups)
    assert 1 <= len(c.onto_idx) <= 3


def test_different_draws_give_different_structures():
    X = data()
    b = base()
    real = b.build(X, NAMES)
    sigs = {NamePermutedOwlProvider(b, s).build(X, NAMES, real_ctx=real).structure_signature() for s in (1, 2, 3, 4)}
    assert len(sigs) >= 3
