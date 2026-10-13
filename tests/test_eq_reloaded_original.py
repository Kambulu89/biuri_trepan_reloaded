"""Contrato de equivalência EQ (Fase 1a): Reloaded sem enriquecimento ontológico efetivo ≡ TREPAN Original (Doc. 7, E1–E9).

Tudo sintético e dataset-agnóstico. O oráculo-espião regista o fluxo exato de entradas (método, forma, bytes).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eq_support import (BUDGET, audit_views, diff_views, ClosedFormOracle, SpyOracle, digest, fingerprint, fit_model, make_problem, portable_audit_view,
                        scenario_fit_kwargs)
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier

# Allowlist de diferenças algorítmicas autorizadas: VAZIA por contrato (CLAUDE.md §2b). Não acrescentar sem decisão explícita e versionada.
ALGORITHMIC_DIFFERENCE_ALLOWLIST: tuple = ()
# Campos de relato, fora de E1–E9, SEM efeito em decisões/consultas/RNG/previsões (D-15). Lista separada da allowlist acima.
NON_SCIENTIFIC_REPORTING_FIELDS = ("semantic_split_audit_", "error_region_audit_", "semantic_audit_summary_",
                                   "pre_mirror_semantic_audit_", "pre_mirror_semantic_split_audit_")

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
    """Com ontologia efetiva o comportamento é bit-a-bit o de main@1789c17 (âncora gerada antes da correção)."""
    kind, p, k, seed = "continuous", 8, 3, 3
    X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
    for flag, kw in FLAGS.items():
        m, spy = fit_model(TrepanReloadedClassifier, kw, X[:200], names, oracle, seed, fit_kw=scenario_fit_kwargs(scenario, p))
        key = f"{kind}-p{p}-k{k}-s{seed}|{scenario}|{flag}"
        got = fingerprint(m, spy, X[200:], SIG)
        if got != GOLDEN_PORTABLE[key]:      # diagnóstico: campos de auditoria que diferem da vista de referência (Linux)
            ref = DIAG_VIEWS.get(key)
            detail = diff_views(ref, audit_views(m)) if ref else [portable_audit_view(m)]
            assert got == GOLDEN_PORTABLE[key], (scenario, flag, detail)
        if sys.platform.startswith("linux"):                                                     # plataforma de geração: bit-a-bit
            assert fingerprint(m, spy, X[200:]) == GOLDEN[key], (scenario, flag)


def test_golden_covers_all_scenarios_and_original_baseline_unchanged():
    kind, p, k, seed = "mixed", 6, 2, 11
    X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
    m, spy = fit_model(TrepanOriginalClassifier, {}, X[:200], names, oracle, seed)
    key = f"{kind}-p{p}-k{k}-s{seed}|original"
    assert fingerprint(m, spy, X[200:], SIG) == GOLDEN_PORTABLE[key]
    if sys.platform.startswith("linux"):
        assert fingerprint(m, spy, X[200:]) == GOLDEN[key]
    assert {k_.split("|")[1] for k_ in GOLDEN if not k_.endswith("|original")} == set(NON_EFFECTIVE) | set(EFFECTIVE)


def test_v2_mirror_effective_without_measurable_effect_tree_equals_original():
    """V2 (D-16, caso 4 sem efeito mensurável): a árvore ≡ Original; a duplicação de consultas ao oráculo é registada, não corrigida na 1a."""
    (_, _, fo), (mr, _, fr) = _pair(3, 8, 3, "continuous", "projector", "default")
    assert mr.semantic_effect_mirror_applied_ is True
    assert fo["rules"] == fr["rules"] and fo["predictions"] == fr["predictions"]
    assert fo["membership_queries"] == fr["membership_queries"]


@pytest.mark.xfail(strict=True, reason="V2 conhecida (Doc. 11/12, D-16): o espelhamento reexecuta o Original no mesmo oráculo, duplicando consultas; "
                                       "alterar exigiria mexer na lógica ontológica ativa — fora do âmbito da Fase 1a")
def test_v2_mirror_oracle_stream_equals_original_known_violation():
    (_, _, fo), (_, _, fr) = _pair(3, 8, 3, "continuous", "projector", "default")
    assert fo["oracle_stream"] == fr["oracle_stream"]


def test_canonical_mofn_complement_is_same_partition():
    """m-of-n(L) e (n-m+1)-of-n(¬L) são a mesma partição; a etiqueta de auditoria pode sair em qualquer das formas (desempate por último bit)."""
    from eq_support import canonical_mofn
    a, b = "2-of-2(a0 > -0.15685; a2 <= 0.3833)", "1-of-2(a0 <= -0.15685; a2 > 0.3833)"
    assert canonical_mofn(a) == canonical_mofn(b)
    assert canonical_mofn(a) != canonical_mofn("2-of-2(a0 > -0.15685; a2 > 0.3833)")      # testes realmente diferentes continuam distintos
    assert canonical_mofn("texto livre") == "texto livre"
