"""Testes da variante EXPERIMENTAL de desempate determinístico (Doc. 19). Isolada: o Original canónico não é tocado."""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from core.trepan_original import Literal, MofNTest, TrepanOriginalClassifier
from tests.eq_support import BUDGET, SpyOracle, digest, make_problem
from validation.experimental.deterministic_tiebreak import (DeterministicTieBreakTrepan, canonical_signature, complement, signature)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fit(cls, seed, kw=None):
    X, names, oracle = make_problem(seed, p=8, k=3)
    spy = SpyOracle(oracle)
    m = cls(**BUDGET, random_state=seed, **(kw or {}))
    m.fit(X, oracle=spy, feature_names=names)
    return m, spy, X


def _rules(m):
    return digest(m.export_rules())


@pytest.mark.parametrize("seed", [0, 3, 5])
def test_disabled_equals_original_bitwise(seed):
    o, so, X = _fit(TrepanOriginalClassifier, seed)
    v, sv, _ = _fit(lambda **k: DeterministicTieBreakTrepan(score_digits=None, **k), seed)
    assert _rules(o) == _rules(v)
    assert np.array_equal(o.predict(X), v.predict(X))
    assert so.stream_digest() == sv.stream_digest()
    assert so.sequence == sv.sequence


def test_complement_is_same_partition_and_same_canonical_signature():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    t = MofNTest(2, (Literal(0, 0.1, True), Literal(2, -0.3, False), Literal(4, 0.5, True)))
    c = complement(t)
    assert np.array_equal(t.evaluate(X), ~c.evaluate(X))
    assert canonical_signature(t) == canonical_signature(c)
    assert signature(t) != signature(c)
    assert signature(complement(c)) == signature(t)


def _noisy(monkeypatch, cls, noise_seed):
    """Injeta ruído de 1–2 ulp determinístico por (máscara) na pontuação — simula diferenças de libm/BLAS entre plataformas."""
    orig = cls._split_selection_score

    def noisy(self, y, mask, test=None):
        s = orig(self, y, mask, test)
        h = hash((noise_seed, mask.tobytes())) & 3
        for _ in range(h):
            s = float(np.nextafter(s, np.inf))
        return s
    monkeypatch.setattr(cls, "_split_selection_score", noisy)


def test_variant_invariant_to_ulp_noise(monkeypatch):
    base, _, _ = _fit(DeterministicTieBreakTrepan, 3)
    ref = _rules(base)
    for ns in (1, 2, 3):
        with monkeypatch.context() as mp:
            _noisy(mp, DeterministicTieBreakTrepan, ns)
            m, _, _ = _fit(DeterministicTieBreakTrepan, 3)
        assert _rules(m) == ref


def test_variant_changes_only_selection_not_rng_use():
    # a variante não consome RNG próprio: com a mesma árvore, o fluxo ao oráculo é idêntico bit a bit
    o, so, _ = _fit(TrepanOriginalClassifier, 3)
    v, sv, _ = _fit(DeterministicTieBreakTrepan, 3)
    if _rules(o) == _rules(v):
        assert so.stream_digest() == sv.stream_digest()
    assert sv.calls["predict"] >= 1 and sv.calls["proba"] >= 0


def test_variant_is_isolated_from_core():
    for p in (ROOT / "core").rglob("*.py"):
        assert not re.search(r"validation\.experimental|DeterministicTieBreak", p.read_text(encoding="utf-8")), p


def test_cross_platform_digest_informational():
    """Informativo (não falha): emite a impressão digital do Original e da variante em 4 problemas fixos para comparar Linux × Windows
    no resumo de avisos do pytest (Doc. 19 §8). A igualdade entre plataformas NÃO é afirmada aqui."""
    import hashlib
    import warnings
    parts = []
    for seed in (0, 3, 5, 7):
        o, _, _ = _fit(TrepanOriginalClassifier, seed)
        v, _, _ = _fit(DeterministicTieBreakTrepan, seed)
        parts.append(f"s{seed}:O={_rules(o)[:10]}:V={_rules(v)[:10]}")
    warnings.warn("TIEBREAK_XPLAT_DIGEST " + " ".join(parts) + " all=" + hashlib.sha256("".join(parts).encode()).hexdigest()[:12], UserWarning)
