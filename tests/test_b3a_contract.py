"""Contrato de compatibilidade semântica de B3-a (desenho; Doc. 27): propriedades que qualquer implementação futura tem de cumprir."""
from __future__ import annotations

import numpy as np

from validation.experimental.b3a_contract import build_compatibility, space_stats


def _rel(n, pairs):
    r = np.eye(n)
    for i, j, v in pairs:
        r[i, j] = r[j, i] = v
    return r


def test_open_world_unmapped_features_are_never_restricted():
    c = build_compatibility(["A", None, "B"], _rel(3, []))
    assert c.allowed[0, 1] and c.allowed[1, 2]                 # sem conhecimento ⇒ sem restrição
    assert not c.allowed[0, 2]                                  # ambas mapeadas, grupos distintos, sem relação


def test_same_group_and_relation_allow_and_matrix_is_symmetric():
    c = build_compatibility(["A", "A", "B", "C"], _rel(4, [(2, 3, 0.5), (0, 3, 0.1)]))
    assert c.allowed[0, 1] and c.allowed[2, 3] and not c.allowed[0, 3]
    assert np.array_equal(c.allowed, c.allowed.T) and c.allowed.diagonal().all()


def test_general_group_label_counts_as_unmapped():
    c = build_compatibility(["general", "A"], _rel(2, []))
    assert not c.mapped[0] and c.allowed[0, 1]


def test_no_transitive_closure():
    c = build_compatibility(["A", "B", "C"], _rel(3, [(0, 1, 0.5), (1, 2, 0.5)]))
    assert c.allowed[0, 1] and c.allowed[1, 2] and not c.allowed[0, 2]


def test_permuting_group_labels_preserves_space_size_exactly():
    """Controlo com 'mesmas oportunidades de pesquisa': permutar a atribuição feature→grupo mantém os tamanhos dos grupos ⇒ mesma fração de pares/subconjuntos."""
    rng = np.random.default_rng(0)
    groups = ["A"] * 4 + ["B"] * 3 + ["C"] * 5
    base = space_stats(build_compatibility(groups, _rel(12, [])))
    for _ in range(5):
        perm = list(rng.permutation(groups))
        s = space_stats(build_compatibility(perm, _rel(12, [])))
        assert s["allowed_pair_fraction"] == base["allowed_pair_fraction"]
        assert s["allowed_3_subset_fraction"] == base["allowed_3_subset_fraction"]


def test_deterministic():
    g = ["A", "B", "A", None]
    r = _rel(4, [(0, 1, 0.4)])
    a, b = build_compatibility(g, r), build_compatibility(g, r)
    assert np.array_equal(a.allowed, b.allowed) and a.reason == b.reason
