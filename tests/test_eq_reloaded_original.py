"""Contrato de equivalência EQ (Fase 1a): Reloaded sem enriquecimento ontológico efetivo ≡ TREPAN Original (Doc. 7, E1–E9).

Tudo sintético e dataset-agnóstico. O oráculo-espião regista o fluxo exato de entradas (método, forma, bytes).
"""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eq_support import (BUDGET, audit_views, compare_golden, diff_views, ClosedFormOracle, SpyOracle, digest, fingerprint, fit_model, make_problem, portable_audit_view,
                        scenario_fit_kwargs)
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier

# Allowlist de diferenças algorítmicas autorizadas: VAZIA por contrato (CLAUDE.md §2b). Não acrescentar sem decisão explícita e versionada.
ALGORITHMIC_DIFFERENCE_ALLOWLIST: tuple = ()
# Campos de relato, fora de E1–E9, SEM efeito em decisões/consultas/RNG/previsões (D-15). Lista separada da allowlist acima.
NON_SCIENTIFIC_REPORTING_FIELDS = ("semantic_split_audit_", "error_region_audit_", "semantic_audit_summary_",
                                   "pre_mirror_semantic_audit_", "pre_mirror_semantic_split_audit_", "oracle_accounting_")

NON_EFFECTIVE = ("none", "depths_zero", "groups_unique")
EFFECTIVE = ("depths", "weights", "groups", "relatedness", "projector", "combined")
GOLDEN = json.loads((ROOT / "tests" / "golden" / "golden_effective.json").read_text())["entries"]
# Suplemento portável (auditorias com 8 algarismos significativos), gerado com o código pré-correção (blob 0053b3d): as auditorias com floats
# diferem no último bit entre plataformas (CI Windows), tal como os bytes das amostras sintéticas enviadas ao oráculo (hash com entradas
# arredondadas); as saídas discretas (regras, previsões, contagens, fluxo ao oráculo) são sempre exatas.
GOLDEN_PORTABLE = json.loads((ROOT / "tests" / "golden" / "golden_effective_portable.json").read_text())["entries"]
SIG = 8
# Vistas de referência (Linux) das auditorias, só para diagnóstico de diferenças entre plataformas (não decidem passa/falha).
DIAG_VIEWS = json.loads((ROOT / "tests" / "golden" / "diag_reference_views.json").read_text())
FLAGS = {"default": {}, "no_mirror": {"mirror_when_no_semantic_effect": False},
         "no_efsr": {"error_focused_refinement": False, "mirror_when_no_semantic_effect": False}}
E_FIELDS = ("rules", "predictions", "membership_queries", "oracle_query_count", "n_nodes", "oracle_stream", "oracle_calls",
            "oracle_samples", "oracle_sequence", "split_audit", "expansion_order")


def test_allowlist_is_empty():
    assert ALGORITHMIC_DIFFERENCE_ALLOWLIST == ()
    assert not set(NON_SCIENTIFIC_REPORTING_FIELDS) & set(ALGORITHMIC_DIFFERENCE_ALLOWLIST)


def _pair(seed, p, k, kind, scenario, flag="default", extra=None):
    X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
    Xtr, Xte = X[:200], X[200:]
    mo, so = fit_model(TrepanOriginalClassifier, {}, Xtr, names, oracle, seed)
    kw = dict(FLAGS[flag], **(extra or {}))
    mr, sr = fit_model(TrepanReloadedClassifier, kw, Xtr, names, oracle, seed, fit_kw=scenario_fit_kwargs(scenario, p))
    return (mo, so, fingerprint(mo, so, Xte)), (mr, sr, fingerprint(mr, sr, Xte))


def test_ontology_effective_predicate_truth_table():
    p = 6
    expected = {"none": False, "depths_zero": False, "groups_unique": False, "depths": True, "weights": True,
                "groups": True, "relatedness": True, "projector": True, "combined": True}
    X, names, oracle = make_problem(5, p=p, k=3)
    for sc, exp in expected.items():
        m = TrepanReloadedClassifier(**BUDGET, random_state=5)
        m.fit(X[:200], oracle=oracle, feature_names=names, **scenario_fit_kwargs(sc, p))
        assert m.ontology_effective_ is exp, sc
        assert m._ontology_effective() is exp


def test_alpha_zero_makes_depths_ineffective():
    p = 6
    X, names, oracle = make_problem(5, p=p, k=3)
    m = TrepanReloadedClassifier(**BUDGET, random_state=5, alpha=0.0)
    m.fit(X[:200], oracle=oracle, feature_names=names, **scenario_fit_kwargs("depths", p))
    assert m.ontology_effective_ is False


@pytest.mark.parametrize("scenario", NON_EFFECTIVE)
def test_eq_dispatch_never_calls_predict_proba(scenario):
    X, names, oracle = make_problem(7, p=8, k=3)
    m, spy = fit_model(TrepanReloadedClassifier, {}, X[:200], names, oracle, 7,
                       fit_kw=scenario_fit_kwargs(scenario, 8), forbid_proba=True)
    assert spy.calls["proba"] == 0


@pytest.mark.parametrize("flag", list(FLAGS))
@pytest.mark.parametrize("scenario", NON_EFFECTIVE)
@pytest.mark.parametrize("seed", [3, 11, 29, 41, 97])
@pytest.mark.parametrize("problem", [("continuous", 8, 3), ("mixed", 6, 2), ("continuous", 10, 4)])
def test_eq_oracle_stream_and_outputs_identical(problem, seed, scenario, flag):
    kind, p, k = problem
    (_, _, fo), (mr, _, fr) = _pair(seed, p, k, kind, scenario, flag)
    for f in E_FIELDS:
        assert fo[f] == fr[f], f
    # Relato inativo (D-15): sem EFSR, sem intervenções, sem projeção
    assert mr.error_region_audit_ == []
    assert mr.error_focused_interventions_attempted_ == 0
    assert mr.semantic_projected_query_count_ == 0
    assert mr.semantic_effect_mirror_applied_ is False
    assert not any(r.get("ontology_influenced") for r in mr.semantic_split_audit_)


def test_ontology_present_but_inactive_graph_is_equivalent():
    class RejectedGraph:   # ontologia rejeitada pelo quality gate / sem atividade
        is_active = False

    X, names, oracle = make_problem(13, p=7, k=3)
    mo, so = fit_model(TrepanOriginalClassifier, {}, X[:200], names, oracle, 13)
    mr, sr = fit_model(TrepanReloadedClassifier, {}, X[:200], names, oracle, 13, fit_kw={"ontology_graph": RejectedGraph()},
                       forbid_proba=True)
    fo, fr = fingerprint(mo, so, X[200:]), fingerprint(mr, sr, X[200:])
    assert all(fo[f] == fr[f] for f in E_FIELDS)


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_metamorphic_column_permutation_preserves_equivalence(seed):
    X, names, oracle = make_problem(seed, p=8, k=3)
    perm = np.random.default_rng(seed).permutation(8)
    Xp = X[:, perm]

    class Perm:
        def __init__(self, inner): self.inner, self.classes_ = inner, inner.classes_
        def predict(self, Z): return self.inner.predict(np.asarray(Z)[:, np.argsort(perm)])
        def predict_proba(self, Z): return self.inner.predict_proba(np.asarray(Z)[:, np.argsort(perm)])

    o = Perm(oracle)
    mo, so = fit_model(TrepanOriginalClassifier, {}, Xp[:200], [names[i] for i in perm], o, seed)
    mr, sr = fit_model(TrepanReloadedClassifier, {}, Xp[:200], [names[i] for i in perm], o, seed, forbid_proba=True)
    fo, fr = fingerprint(mo, so, Xp[200:]), fingerprint(mr, sr, Xp[200:])
    assert all(fo[f] == fr[f] for f in E_FIELDS)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_property_random_budgets_and_shapes(seed):
    rng = np.random.default_rng(1000 + seed)
    p, k = int(rng.integers(3, 13)), int(rng.integers(2, 6))
    n = int(rng.integers(60, 260))
    X = rng.normal(size=(n, p))
    if seed % 2:                        # discretas/binárias/constante
        X[:, 0] = (X[:, 0] > 0).astype(float)
        X[:, 1] = np.round(X[:, 1] * 3)
        X[:, p - 1] = 2.0
    oracle = ClosedFormOracle(p, k, seed + 50)
    names = [f"v{i}" for i in range(p)]
    budget = dict(min_sample=int(rng.integers(60, 400)), max_queries=int(rng.integers(300, 1500)), max_nodes=int(rng.integers(3, 12)),
                  max_depth=int(rng.integers(2, 6)), max_n=int(rng.integers(1, 4)), beam_width=int(rng.integers(1, 4)),
                  min_samples_leaf=int(rng.integers(1, 5)), max_features_per_node=min(p, 8),
                  purity_epsilon=float(rng.choice([0.0, 0.05, 0.1])), random_state=seed)
    outs = []
    for cls, forbid in ((TrepanOriginalClassifier, False), (TrepanReloadedClassifier, True)):
        spy = SpyOracle(oracle, forbid_proba=forbid)
        m = cls(**budget).fit(X, oracle=spy, feature_names=names)
        outs.append(fingerprint(m, spy, X))
    assert all(outs[0][f] == outs[1][f] for f in E_FIELDS)


@pytest.mark.parametrize("scenario", EFFECTIVE)
def test_effective_behaviour_unchanged_golden(scenario):
    """Com ontologia efetiva o comportamento é o de main@1789c17 (âncora gerada antes da correção), em todas as plataformas (comparação
    portável). Única diferença legítima: retirada das chamadas byte-idênticas no mirror (V2, ver ``compare_golden``)."""
    kind, p, k, seed = "continuous", 8, 3, 3
    X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
    for flag, kw in FLAGS.items():
        m, spy = fit_model(TrepanReloadedClassifier, kw, X[:200], names, oracle, seed, fit_kw=scenario_fit_kwargs(scenario, p))
        key = f"{kind}-p{p}-k{k}-s{seed}|{scenario}|{flag}"
        got = fingerprint(m, spy, X[200:], SIG)
        diffs = compare_golden(got, GOLDEN_PORTABLE[key], m)
        if diffs:      # diagnóstico: campos de auditoria que diferem da vista de referência (Linux)
            ref = DIAG_VIEWS.get(key)
            detail = diff_views(ref, audit_views(m)) if ref else [portable_audit_view(m)]
            assert not diffs, (scenario, flag, diffs, detail)


def _legacy_class():
    import importlib.util
    path = ROOT / "tests" / "golden" / "legacy" / "trepan_reloaded_main_1789c17.py"
    spec = importlib.util.spec_from_file_location("trepan_reloaded_main_1789c17", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrepanReloadedClassifier


@pytest.mark.parametrize("seed", [3, 11])
@pytest.mark.parametrize("scenario", EFFECTIVE)
def test_effective_behaviour_identical_to_frozen_main_same_host(scenario, seed):
    """DIFERENCIAL, bit-a-bit e no mesmo anfitrião: com ontologia efetiva o código atual ≡ cópia congelada de main@1789c17 (independente de
    hashes guardados, do kernel BLAS e das instruções SIMD). Única diferença legítima: retirada de chamadas byte-idênticas no mirror (V2)."""
    Legacy = _legacy_class()
    X, names, oracle = make_problem(seed, p=8, k=3)
    for flag, kw in FLAGS.items():
        fk = scenario_fit_kwargs(scenario, 8)
        ml, sl = fit_model(Legacy, kw, X[:200], names, oracle, seed, fit_kw=fk)
        mn, sn = fit_model(TrepanReloadedClassifier, kw, X[:200], names, oracle, seed, fit_kw=fk)
        diffs = compare_golden(fingerprint(mn, sn, X[200:]), fingerprint(ml, sl, X[200:]), mn)
        assert not diffs, (scenario, seed, flag, diffs)


@pytest.mark.skipif(os.environ.get("BIURI_EXACT_GOLDEN") != "1", reason="âncora exata guardada: opt-in (BIURI_EXACT_GOLDEN=1); o gate é o teste diferencial no mesmo anfitrião")
def test_effective_behaviour_unchanged_golden_exact_pinned_numeric_env():
    """Âncora EXATA (bit-a-bit) num subprocesso com ambiente numérico fixado: independente do kernel BLAS do anfitrião."""
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "golden" / "check_exact.py")], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1500:]


def test_golden_covers_all_scenarios_and_original_baseline_unchanged():
    kind, p, k, seed = "mixed", 6, 2, 11
    X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
    m, spy = fit_model(TrepanOriginalClassifier, {}, X[:200], names, oracle, seed)
    key = f"{kind}-p{p}-k{k}-s{seed}|original"
    assert fingerprint(m, spy, X[200:], SIG) == GOLDEN_PORTABLE[key]
    assert {k_.split("|")[1] for k_ in GOLDEN if not k_.endswith("|original")} == set(NON_EFFECTIVE) | set(EFFECTIVE)


MIRROR_SCENARIOS = ("projector", "groups", "relatedness", "weights", "combined", "depths")


def _mirrored_pair(scenario, seed, reuse, p=8, k=3, kind="continuous"):
    """Original isolado vs Reloaded com mirror (espião no oráculo REAL, abaixo de qualquer invólucro)."""
    X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
    mo, so = fit_model(TrepanOriginalClassifier, {}, X[:200], names, oracle, seed)
    old = TrepanReloadedClassifier.REUSE_IDENTICAL_ORACLE_CALLS
    TrepanReloadedClassifier.REUSE_IDENTICAL_ORACLE_CALLS = reuse
    try:
        mr, sr = fit_model(TrepanReloadedClassifier, {}, X[:200], names, oracle, seed, fit_kw=scenario_fit_kwargs(scenario, p))
    finally:
        TrepanReloadedClassifier.REUSE_IDENTICAL_ORACLE_CALLS = old
    return (mo, so), (mr, sr), X


@pytest.mark.parametrize("seed", [3, 11, 29])
@pytest.mark.parametrize("scenario", MIRROR_SCENARIOS)
def test_v2_mirror_final_model_equals_original(scenario, seed):
    (mo, so), (mr, sr), X = _mirrored_pair(scenario, seed, True)
    if not mr.semantic_effect_mirror_applied_:
        pytest.skip("sem mirror neste cenário/seed (efeito semântico mensurável: comportamento ontológico preservado pelas âncoras)")
    assert mo.export_rules() == mr.export_rules()
    assert np.array_equal(mo.predict(X[200:]), mr.predict(X[200:]))
    assert int(mo.membership_queries_) == int(mr.membership_queries_)
    assert mr.oracle_info_["oracle_type"] == mo.oracle_info_["oracle_type"]


@pytest.mark.parametrize("seed", [3, 11, 29])
@pytest.mark.parametrize("scenario", MIRROR_SCENARIOS)
def test_v2_original_portion_of_stream_is_exactly_the_original_stream(scenario, seed):
    """Sem reutilização, o fluxo da reexecução do Original (sufixo do fluxo total) é IDÊNTICO ao do Original isolado; com reutilização
    é o mesmo fluxo menos as chamadas byte-idênticas já servidas (contabilidade). Total = tentativa semântica + Original − reutilizado."""
    (mo, so), (mr0, sr0), _ = _mirrored_pair(scenario, seed, False)
    (_, _), (mr1, sr1), _ = _mirrored_pair(scenario, seed, True)
    if not mr0.semantic_effect_mirror_applied_:
        pytest.skip("sem mirror neste cenário/seed")
    n_o = len(so.sequence)
    assert sr0.sequence[-n_o:] == so.sequence                                   # sufixo = Original isolado (formas, ordem, métodos)
    assert so.calls["predict"] == n_o and so.calls["proba"] == 0
    acc = mr1.oracle_accounting_
    reused = acc["reused_identical"]
    assert reused["calls"] >= 1 and reused["samples"] == 200                    # a rotulagem inicial do treino
    # espião no oráculo real: total = tentativa + rerun (o que chegou ao oráculo)
    assert sr1.samples["predict"] == acc["semantic_attempt"]["predict"]["samples"] + acc["original_rerun"]["predict"]["samples"]
    assert sr1.samples["proba"] == acc["semantic_attempt"]["predict_proba"]["samples"]
    assert acc["original_rerun"]["predict_proba"]["calls"] == 0                   # o Original nunca consulta predict_proba
    # equivalência exata ao Original: rerun + reutilizado == Original isolado
    assert acc["original_rerun"]["predict"]["samples"] + reused["samples"] == so.samples["predict"]
    assert acc["original_rerun"]["predict"]["calls"] + reused["calls"] == so.calls["predict"]
    # a reutilização elimina exatamente o que o espião deixou de ver
    assert sr0.samples["predict"] - sr1.samples["predict"] == reused["samples"]
    assert sr0.samples["proba"] == sr1.samples["proba"]                           # a tentativa semântica não muda
    assert mr0.export_rules() == mr1.export_rules()


def test_v2_replay_oracle_reuses_only_byte_identical_predict_calls():
    from core.trepan_reloaded_historical import _ExactCallLedger, _ExactReplayOracle
    oracle = ClosedFormOracle(5, 3, 1)
    X = np.random.default_rng(0).normal(size=(40, 5))
    spy = SpyOracle(oracle)
    ledger = _ExactCallLedger(spy)
    y = ledger.predict(X)
    replay = _ExactReplayOracle(ledger)
    assert np.array_equal(replay.predict(X), y) and replay.reused == {"calls": 1, "samples": 40}
    assert spy.calls["predict"] == 1                                              # servido sem nova consulta
    assert replay.predict(X.copy()) is not None and replay.reused["calls"] == 2    # cópia byte-idêntica
    # semelhante mas NÃO equivalente: 1 ulp, outra ordem, outro subconjunto, outro dtype/forma -> vão ao oráculo
    X2 = X.copy(); X2[0, 0] = np.nextafter(X2[0, 0], np.inf)
    replay.predict(X2); replay.predict(X[::-1]); replay.predict(X[:20]); replay.predict(X.astype(np.float32).astype(float))
    assert replay.reused["calls"] == 2 and spy.calls["predict"] == 5
    # predict_proba nunca vem da cache e nunca se confunde com predict
    replay.predict_proba(X); replay.predict_proba(X)
    assert spy.calls["proba"] == 2 and replay.reused["calls"] == 2
    # oráculo diferente => ledger diferente => nada partilhado
    other = ClosedFormOracle(5, 3, 2)
    assert _ExactReplayOracle(_ExactCallLedger(other)).predict(X).tolist() == other.predict(X).tolist()
    # a resposta devolvida é uma cópia (alterá-la não corrompe a cache)
    r = replay.predict(X); r[:] = -1
    assert np.array_equal(replay.predict(X), y)


def test_v2_ledger_does_not_alter_observed_oracle_surface():
    """O invólucro só observa: ``predict_proba`` só existe se o oráculo o tiver; atributos são reencaminhados."""
    from core.trepan_reloaded_historical import _ExactCallLedger

    class NoProba:
        classes_ = np.arange(2)
        def predict(self, X): return np.zeros(len(X), dtype=int)

    assert not hasattr(_ExactCallLedger(NoProba()), "predict_proba")
    assert hasattr(_ExactCallLedger(ClosedFormOracle(3, 2, 0)), "predict_proba")
    assert _ExactCallLedger(ClosedFormOracle(3, 2, 0)).classes_.tolist() == [0, 1]


def test_v2_no_response_cache_without_mirror_or_without_effective_ontology():
    """Sem mirror possível o observador só conta (não guarda respostas nem faz hashing): nada para reutilizar."""
    X, names, oracle = make_problem(5, p=8, k=3)
    for kw, sc in (({"mirror_when_no_semantic_effect": False}, "combined"), ({}, "none"), ({}, "depths_zero")):
        m, spy = fit_model(TrepanReloadedClassifier, kw, X[:200], names, oracle, 5, fit_kw=scenario_fit_kwargs(sc, 8))
        assert m.oracle_accounting_["reused_oracle_calls"] == 0 and m.oracle_accounting_["reused_oracle_samples"] == 0


def test_v2_oracle_not_shared_across_different_oracles_and_seeds_reproducible():
    outs = []
    for _ in range(2):
        (_, _), (mr, sr), _ = _mirrored_pair("projector", 3, True)
        outs.append((mr.oracle_accounting_, sr.stream_digest()))
    assert outs[0] == outs[1]                                                     # reprodutível
    X, names, oracle = make_problem(3, p=8, k=3)
    other = ClosedFormOracle(8, 3, 777)
    m1, _ = fit_model(TrepanReloadedClassifier, {}, X[:200], names, oracle, 3, fit_kw=scenario_fit_kwargs("projector", 8))
    m2, _ = fit_model(TrepanReloadedClassifier, {}, X[:200], names, other, 3, fit_kw=scenario_fit_kwargs("projector", 8))
    o2, _ = fit_model(TrepanOriginalClassifier, {}, X[:200], names, other, 3)
    assert m2.export_rules() == o2.export_rules()                                 # com outro oráculo continua ≡ Original (mirror)


# ---------------------------------------------------------------------------------------------------------------------------------
# Contrato EQ v2.0 (S0–S3): substitui o xfail da V2 por verificações positivas
# ---------------------------------------------------------------------------------------------------------------------------------
def _acc_matches_spy(m, spy):
    a = m.oracle_accounting_
    assert a["total_oracle_calls"] == sum(spy.calls.values())
    assert a["total_oracle_samples"] == sum(spy.samples.values())
    assert a["total_oracle_samples"] == a["original_oracle_samples"] + a["semantic_oracle_samples"] - a["reused_oracle_samples"] or a["ontology_state"] != "S3"


class _RejectedGraph:
    is_active = False


def test_contract_state_classification_S0_S1_S2_S3():
    X, names, oracle = make_problem(5, p=8, k=3)
    cases = {
        "S0": {},
        "S1": {"ontology_graph": _RejectedGraph()},
        "S2": scenario_fit_kwargs("depths_zero", 8),          # estrutura fornecida mas sem mecanismo executável
        "S3": scenario_fit_kwargs("weights", 8),
    }
    for state, fk in cases.items():
        m, spy = fit_model(TrepanReloadedClassifier, {}, X[:200], names, oracle, 5, fit_kw=fk)
        assert m.ontology_state_ == state and m.oracle_accounting_["ontology_state"] == state
        assert m.oracle_accounting_["eq_contract_version"] == TrepanReloadedClassifier.EQ_CONTRACT_VERSION == "2.0-S0S3"
        assert m.oracle_accounting_["semantic_attempt_executed"] is (state == "S3")


@pytest.mark.parametrize("seed", [3, 11, 29])
@pytest.mark.parametrize("state_kw", ["S0", "S1", "S2"])
def test_contract_S0_S1_S2_full_equivalence_and_zero_semantic_cost(state_kw, seed):
    """S0–S2: equivalência INTEGRAL E1–E9 (fluxo de consultas incluído) e custo semântico nulo."""
    fk = {"S0": {}, "S1": {"ontology_graph": _RejectedGraph()}, "S2": scenario_fit_kwargs("groups_unique", 8)}[state_kw]
    X, names, oracle = make_problem(seed, p=8, k=3)
    mo, so = fit_model(TrepanOriginalClassifier, {}, X[:200], names, oracle, seed)
    mr, sr = fit_model(TrepanReloadedClassifier, {}, X[:200], names, oracle, seed, fit_kw=fk, forbid_proba=True)
    fo, fr = fingerprint(mo, so, X[200:]), fingerprint(mr, sr, X[200:])
    assert all(fo[f] == fr[f] for f in E_FIELDS)
    a = mr.oracle_accounting_
    assert a["semantic_attempt_executed"] is False and a["semantic_oracle_calls"] == 0 and a["semantic_oracle_samples"] == 0
    assert a["semantic_attempt_status"] == "NO_SEMANTIC_ATTEMPT"
    assert a["total_oracle_calls"] == sum(so.calls.values()) and a["total_oracle_samples"] == sum(so.samples.values())
    _acc_matches_spy(mr, sr)


@pytest.mark.parametrize("seed", [3, 11, 29])
@pytest.mark.parametrize("scenario", MIRROR_SCENARIOS)
def test_contract_S3_original_portion_equivalent_and_attempt_fully_accounted(scenario, seed):
    """S3 com mirror: porção Original ≡ Original isolado (regras, previsões, fluxo), modelo final ≡ Original, e TODAS as consultas
    adicionais atribuídas à tentativa semântica; o total contabilizado coincide com o que o oráculo recebeu (sem alterações ocultas)."""
    (mo, so), (mr, sr), X = _mirrored_pair(scenario, seed, True)
    a = mr.oracle_accounting_
    assert a["ontology_state"] == "S3" and a["semantic_attempt_executed"] is True
    if not mr.semantic_effect_mirror_applied_:
        pytest.skip("efeito semântico estrutural observado neste cenário/seed (sem mirror): coberto pelo teste S3 sem mirror")
    assert a["semantic_effect_observed"] is False and a["semantic_attempt_status"] == "SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN"
    assert mo.export_rules() == mr.export_rules() and np.array_equal(mo.predict(X[200:]), mr.predict(X[200:]))
    # porção Original: pedidos (enviados + reutilizados) == Original isolado, em chamadas e amostras
    assert a["original_portion_executed"] is True
    assert a["original_oracle_calls"] == sum(so.calls.values()) and a["original_oracle_samples"] == sum(so.samples.values())
    # tudo o que excede o Original é tentativa semântica (nenhuma consulta fora das duas porções)
    assert a["semantic_oracle_calls"] == a["semantic_attempt"]["predict"]["calls"] + a["semantic_attempt"]["predict_proba"]["calls"]
    assert a["total_oracle_calls"] == a["original_oracle_calls"] + a["semantic_oracle_calls"] - a["reused_oracle_calls"]
    _acc_matches_spy(mr, sr)
    assert a["original_rerun"]["predict_proba"]["calls"] == 0            # o Original nunca consulta predict_proba


@pytest.mark.parametrize("scenario", ["weights", "groups", "combined"])
def test_contract_S3_without_mirror_attempt_is_whole_pass_and_effect_flagged(scenario):
    X, names, oracle = make_problem(3, p=8, k=3)
    kw = {"mirror_when_no_semantic_effect": False}
    m, spy = fit_model(TrepanReloadedClassifier, kw, X[:200], names, oracle, 3, fit_kw=scenario_fit_kwargs(scenario, 8))
    a = m.oracle_accounting_
    assert a["ontology_state"] == "S3" and a["semantic_attempt_executed"] is True
    assert a["original_portion_executed"] is False and a["original_oracle_samples"] == 0
    assert a["semantic_oracle_samples"] == a["total_oracle_samples"] == sum(spy.samples.values())
    assert a["semantic_effect_observed"] == bool(m.semantic_audit_summary()["ontology_influenced_splits"]
                                                  or m.semantic_audit_summary()["error_focused_interventions_accepted"])
    _acc_matches_spy(m, spy)


def test_contract_gain_is_not_inferred_from_attempt():
    """Tentativa executada ≠ ganho: com árvores idênticas ao Original o estado é SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN; empates não são ganho."""
    from validation.benchmark.semantic_cost import cost_report, efficiency_report, no_effect_rate, semantic_gain_report
    (mo, so), (mr, sr), X = _mirrored_pair("projector", 3, True)
    orc = ClosedFormOracle(8, 3, 4).predict(X[200:])                                      # oráculo do problema (make_problem: seed+1)
    y = orc                                                                               # só para exercitar o cálculo
    g = semantic_gain_report(mr, mo, X[200:], y, orc)
    assert g["predictions_identical"] and g["semantic_gain_accuracy"] == 0.0 and g["semantic_gain_fidelity"] == 0.0
    assert g["semantic_attempt_status"] == "SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN"
    assert g["fidelity_non_inferior"] is True and g["fidelity_strictly_superior"] is False  # F_R >= F_O vs F_R > F_O
    c = cost_report(mr, ref_total_oracle_samples=sum(so.samples.values()), ref_total_oracle_calls=sum(so.calls.values()))
    assert c["extra_oracle_samples_vs_original"] == c["semantic_oracle_samples"] - c["reused_oracle_samples"] > 0
    e = efficiency_report(g, c, mr)
    assert e["fidelity_gain_per_1k_extra_samples"] == 0.0 and e["complexity_diff_nodes"] == 0
    assert no_effect_rate([g]) == 1.0
    g0 = semantic_gain_report(mo, mo, X[200:], y, orc)
    assert g0["semantic_attempt_status"] == "NO_SEMANTIC_ATTEMPT" and no_effect_rate([g0]) is None
