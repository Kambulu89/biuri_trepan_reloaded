"""B2-C (EXPERIMENTAL, desligado por omissão): fallback estatístico local + corrente RNG semântica separada (Doc. 20).

Distingue sempre três coisas (nunca declara equivalência estrutural global só porque houve fallback num nó):
  (1) fallback estatístico DENTRO do contexto atual; (2) reprodução efetiva da decisão do Original (provada contra um fit Original isolado);
  (3) impossibilidade de presumir reprodução quando o contexto já divergiu.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from tests.eq_support import BUDGET, SpyOracle, digest, make_problem, scenario_fit_kwargs

P = 8
INERT = dict(error_focus_min_local_fidelity_gain=10.0, error_focus_fidelity_tolerance=0.0)   # gate que rejeita sempre ⇒ camada semântica inerte
NOMIRROR = dict(mirror_when_no_semantic_effect=False)


def _fit(cls, seed, scenario="combined", kw=None, problem_kw=None):
    X, names, oracle = make_problem(seed, p=P, k=3, **(problem_kw or {}))
    spy = SpyOracle(oracle)
    m = cls(**BUDGET, random_state=seed, **(kw or {}))
    extra = scenario_fit_kwargs(scenario, P) if cls is TrepanReloadedClassifier else {}
    m.fit(X, oracle=spy, feature_names=names, **extra)
    return m, spy, X


def _rules(m):
    return digest(m.export_rules())


def test_default_is_off():
    m = TrepanReloadedClassifier()
    assert m.statistical_fallback is False and m.b2c_semantic_node_priority is False


@pytest.mark.parametrize("seed", [0, 3])
def test_flag_has_no_effect_without_effective_ontology(seed):
    """S0: B2-C ligado sem ontologia ≡ Original (bit a bit, incluindo o fluxo ao oráculo)."""
    o, so, X = _fit(TrepanOriginalClassifier, seed)
    r, sr, _ = _fit(TrepanReloadedClassifier, seed, "none", dict(statistical_fallback=True))
    assert _rules(o) == _rules(r) and np.array_equal(o.predict(X), r.predict(X))
    assert so.stream_digest() == sr.stream_digest() and so.sequence == sr.sequence
    assert r.ontology_state_ == "S0" and r.b2c_audit_ == []


@pytest.mark.parametrize("seed", [0, 3, 5])
@pytest.mark.parametrize("scenario", ["combined", "weights", "projector"])
def test_inert_semantic_layer_reproduces_original_portion(seed, scenario):
    """S3 com gate que rejeita sempre: árvore, previsões e consultas da porção Original ≡ Original isolado; a camada semântica é só auditada."""
    o, so, X = _fit(TrepanOriginalClassifier, seed)
    r, sr, _ = _fit(TrepanReloadedClassifier, seed, scenario, dict(statistical_fallback=True, **INERT, **NOMIRROR))
    assert r.ontology_state_ == "S3"
    assert _rules(o) == _rules(r) and np.array_equal(o.predict(X), r.predict(X))
    assert int(o.membership_queries_) == int(r.membership_queries_)
    acc = r.oracle_accounting_["b2c_split_attribution"]
    assert acc["original"]["predict"] == {"calls": so.calls["predict"], "samples": so.samples["predict"]}
    assert acc["original"]["predict_proba"]["calls"] == 0                       # a porção Original nunca usa predict_proba
    assert r.oracle_accounting_["original_portion_executed"] is True
    tot = r.oracle_accounting_
    assert tot["total_oracle_samples"] == tot["original_oracle_samples"] + tot["semantic_oracle_samples"] - tot["reused_oracle_samples"]
    assert tot["total_oracle_samples"] == sr.samples["predict"] + sr.samples["proba"]    # nada oculto: tudo o que o oráculo recebeu está atribuído
    assert all(not row["accepted"] for row in r.b2c_audit_)


def test_semantic_rng_is_separate_from_original_stream():
    """Cenários com amostragem semântica muito diferente (projetor/pesos/combinado) deixam a árvore-base inalterada ⇒ rng_semantic ≠ rng_original."""
    refs = {_rules(_fit(TrepanReloadedClassifier, 3, sc, dict(statistical_fallback=True, **INERT, **NOMIRROR))[0])
            for sc in ("combined", "weights", "projector", "depths")}
    o = _rules(_fit(TrepanOriginalClassifier, 3)[0])
    assert refs == {o}
    m1, _, _ = _fit(TrepanReloadedClassifier, 3, "combined", dict(statistical_fallback=True, semantic_active_pool_multiplier=2, **INERT, **NOMIRROR))
    assert _rules(m1) == o


def test_semantic_stream_is_deterministic_and_depends_on_seed():
    kw = dict(statistical_fallback=True, **NOMIRROR)
    a, _, _ = _fit(TrepanReloadedClassifier, 3, "combined", kw)
    b, _, _ = _fit(TrepanReloadedClassifier, 3, "combined", kw)
    assert digest(a.b2c_audit_) == digest(b.b2c_audit_) and _rules(a) == _rules(b)


def test_seed_derivation_is_stable_and_documented():
    m = TrepanReloadedClassifier(random_state=42)
    class N: node_id = 3
    got = m._b2c_semantic_rng_seed(N()).generate_state(2).tolist()
    assert got == np.random.SeedSequence([42, 0xB2C, 3]).generate_state(2).tolist()
    N.node_id = 4
    assert m._b2c_semantic_rng_seed(N()).generate_state(2).tolist() != got


def test_fallback_classes_and_effective_reproduction():
    """(1) todo o fallback é estatístico no contexto atual; (2) onde as condições são cumpridas, a escolha estatística É a do Original isolado;
    (3) depois da primeira divergência a reprodução não é presumida (classe própria) e nada afirma equivalência global."""
    seen_met = seen_div = False
    for seed in (0, 1, 2, 3, 4, 5):
        o, _, _ = _fit(TrepanOriginalClassifier, seed)
        r, _, _ = _fit(TrepanReloadedClassifier, seed, "combined", dict(statistical_fallback=True, **NOMIRROR))
        orig_split = {row["node_id"]: row["test"] for row in o.split_audit_}
        first_div = None
        for row in r.b2c_audit_:
            assert row["base_stream_preserved"] is True and row["node_priority"] == "original"
            assert row["ontology_state"] == "S3"
            if row["fallback_used"]:
                assert row["chosen_candidate_id"] == "statistical" and row["tree_complexity_delta"] == 0 and row["accepted"] is False
            if row["context_class"] == "original_reproduction_conditions_met":
                seen_met = True
                assert first_div is None
                if row["node_id"] in orig_split:                       # reprodução efetiva (prova contra o Original isolado)
                    stat = row["candidate_ids"]["statistical"][len("stat:"):]
                    assert stat == orig_split[row["node_id"]]
            else:
                seen_div = True
                assert first_div is not None
            if row["accepted"] and first_div is None:
                first_div = row["node_id"]
        if first_div is None:
            assert _rules(o) == _rules(r)                              # sem nenhuma divisão semântica aceite a árvore coincide
        else:
            # NÃO se declara equivalência global: a árvore pode divergir do Original após a primeira divisão semântica aceite
            assert r.b2c_audit_[[x["node_id"] for x in r.b2c_audit_].index(first_div)]["accepted"]
    assert seen_met


def test_accepted_semantic_splits_are_valid_in_base_context():
    for seed in (0, 1, 2, 3):
        r, _, _ = _fit(TrepanReloadedClassifier, seed, "combined", dict(statistical_fallback=True, **NOMIRROR))
        assert r.node_count_ <= BUDGET["max_nodes"]
        assert all(row["tree_complexity_delta"] <= BUDGET["max_n"] - 1 for row in r.b2c_audit_)
        assert len(r.b2c_audit_) == len(r.split_audit_)


def test_semantic_node_priority_is_a_separate_experimental_factor():
    r, _, _ = _fit(TrepanReloadedClassifier, 3, "combined", dict(statistical_fallback=True, b2c_semantic_node_priority=True, **NOMIRROR))
    assert all(row["node_priority"] == "semantic" and row["context_class"] == "context_diverged_not_reproducible" for row in r.b2c_audit_)


def test_b2c_off_is_bitwise_previous_behaviour():
    a, sa, _ = _fit(TrepanReloadedClassifier, 3, "combined", NOMIRROR)
    b, sb, _ = _fit(TrepanReloadedClassifier, 3, "combined", dict(statistical_fallback=False, **NOMIRROR))
    assert _rules(a) == _rules(b) and sa.stream_digest() == sb.stream_digest() and a.b2c_audit_ == []
