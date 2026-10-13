"""B1 — calibração do contributo semântico (EXPERIMENTAL, desligada por omissão). Contrato: OFF ≡ histórico; sem ontologia efetiva ≡ Original (S0–S2);
propriedades matemáticas de cada variante; instrumentação de candidatos sem efeito nas decisões/consultas."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eq_support import fingerprint, fit_model, make_problem, scenario_fit_kwargs
from core.trepan_original import Literal, MofNTest, TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier

NOMIRROR = {"mirror_when_no_semantic_effect": False}
E_FIELDS = ("rules", "predictions", "membership_queries", "oracle_query_count", "n_nodes", "oracle_stream", "oracle_calls",
            "oracle_samples", "oracle_sequence", "split_audit", "expansion_order")
LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)


def _fit(cal=None, lam=0.2, scenario="combined", seed=3, log=False, **extra):
    X, names, oracle = make_problem(seed, p=8, k=3)
    kw = dict(NOMIRROR, semantic_calibration=cal, semantic_lambda_cap=lam, log_split_candidates=log, **extra)
    m, spy = fit_model(TrepanReloadedClassifier, kw, X[:200], names, oracle, seed, fit_kw=scenario_fit_kwargs(scenario, 8))
    return m, spy, X


def test_default_is_off_and_validation():
    assert TrepanReloadedClassifier().semantic_calibration is None and TrepanReloadedClassifier().log_split_candidates is False
    with pytest.raises(ValueError):
        TrepanReloadedClassifier(semantic_calibration="inexistente")
    with pytest.raises(ValueError):
        TrepanReloadedClassifier(semantic_calibration="bounded", semantic_lambda_cap=-0.1)


@pytest.mark.parametrize("scenario", ["weights", "groups", "combined", "depths"])
def test_off_is_bitwise_the_historical_behaviour(scenario):
    """Com ``semantic_calibration=None`` (e a instrumentação ligada ou não) a árvore e o fluxo ao oráculo são os de sempre."""
    a, sa, X = _fit(None, scenario=scenario)
    b, sb, _ = _fit(None, scenario=scenario, log=True)
    fa, fb = fingerprint(a, sa, X[200:]), fingerprint(b, sb, X[200:])
    assert all(fa[f] == fb[f] for f in E_FIELDS)
    assert len(b.split_candidate_log_) > 0 and len(a.split_candidate_log_) == 0


@pytest.mark.parametrize("cal", ["tiebreak", "bounded", "calibrated"])
@pytest.mark.parametrize("scenario", ["none", "depths_zero", "groups_unique"])
def test_calibration_has_no_effect_without_effective_ontology(cal, scenario):
    """S0–S2: a opção B1 ligada não altera NADA (≡ Original, incluindo o fluxo de consultas)."""
    X, names, oracle = make_problem(5, p=8, k=3)
    mo, so = fit_model(TrepanOriginalClassifier, {}, X[:200], names, oracle, 5)
    mr, sr = fit_model(TrepanReloadedClassifier, {"semantic_calibration": cal, "semantic_lambda_cap": 0.1}, X[:200], names, oracle, 5,
                       fit_kw=scenario_fit_kwargs(scenario, 8), forbid_proba=True)
    fo, fr = fingerprint(mo, so, X[200:]), fingerprint(mr, sr, X[200:])
    assert all(fo[f] == fr[f] for f in E_FIELDS)


def _score_pieces(model, X, seed=3, n_tests=400):
    """Pares (s, m, score_bounded, ...) para testes aleatórios, usando o estado ajustado do modelo."""
    rng = np.random.default_rng(seed)
    y = model.predict(X[:200])
    model._semantic_structure_cache_enabled_ = False
    model._current_error_profile_ = None
    out = []
    for _ in range(n_tests):
        nl = int(rng.integers(1, 4))
        fs = rng.choice(8, size=nl, replace=False)
        lits = tuple(Literal(int(f), float(rng.normal()), bool(rng.integers(0, 2))) for f in fs)
        test = MofNTest(int(rng.integers(1, nl + 1)), lits)
        mask = test.evaluate(X[:200])
        if mask.sum() < 4 or (~mask).sum() < 4:
            continue
        out.append((y, mask, test))
    return out


def test_legacy_score_equals_statistical_plus_net_semantic_term():
    """Identidade que fundamenta B1: score histórico = s + m, com m = s·(w^k−1) + α·OntoDepth + β·ErrorCoverage."""
    model, _, X = _fit(None)
    k = float(model.semantic_gain_strength)
    checked = 0
    for y, mask, test in _score_pieces(model, X):
        raw = TrepanOriginalClassifier._split_selection_score(model, y, mask, test)
        if raw <= 1e-12:
            continue
        s = model._normalized_information_gain(raw, y)
        w = model._semantic_weight_for_test(test)
        m = s * (w ** k - 1.0) + model.alpha * model._onto_depth_for_test(test) + model.beta * model._error_coverage_for_test(test)
        assert abs(model._split_selection_score(y, mask, test) - (s + m)) < 1e-12
        checked += 1
    assert checked > 50


@pytest.mark.parametrize("lam", LAMBDAS)
def test_bounded_and_calibrated_respect_the_cap_and_lambda_zero_is_purely_statistical(lam):
    model, _, X = _fit("bounded", lam)
    for cal in ("bounded", "calibrated"):
        model.semantic_calibration, model.semantic_lambda_cap = cal, lam
        for y, mask, test in _score_pieces(model, X):
            raw = TrepanOriginalClassifier._split_selection_score(model, y, mask, test)
            if raw <= 1e-12:
                assert model._split_selection_score(y, mask, test) == 0.0
                continue
            s = model._normalized_information_gain(raw, y)
            sc = model._split_selection_score(y, mask, test)
            assert abs(sc - s) <= lam * s + 1e-12                                # |bónus| <= λ·s  (desvio estatístico máx. 2λ/(1+λ))
            if lam == 0.0:
                assert sc == s                                                  # ranking puramente estatístico


def test_bounded_is_monotone_in_lambda_for_nonnegative_semantic_term():
    model, _, X = _fit("bounded", 0.0)
    prev = {}
    for lam in LAMBDAS:
        model.semantic_lambda_cap = lam
        for i, (y, mask, test) in enumerate(_score_pieces(model, X, n_tests=120)):
            sc = model._split_selection_score(y, mask, test)
            if i in prev:
                assert sc >= prev[i] - 1e-12
            prev[i] = sc


def test_tiebreak_changes_ranking_only_within_two_tau():
    model, _, X = _fit("tiebreak", 0.2)
    model._current_real_validation_y_ = np.zeros(100)
    tau = model.semantic_tie_se_multiplier / np.sqrt(100)
    for y, mask, test in _score_pieces(model, X):
        raw = TrepanOriginalClassifier._split_selection_score(model, y, mask, test)
        if raw <= 1e-12:
            continue
        s = model._normalized_information_gain(raw, y)
        assert abs(model._split_selection_score(y, mask, test) - s) <= tau + 1e-12


@pytest.mark.parametrize("cal,lam", [("bounded", 0.2), ("calibrated", 0.2), ("tiebreak", 0.2)])
def test_logging_does_not_change_decisions_queries_or_rng(cal, lam):
    a, sa, X = _fit(cal, lam, log=False)
    b, sb, _ = _fit(cal, lam, log=True)
    fa, fb = fingerprint(a, sa, X[200:]), fingerprint(b, sb, X[200:])
    assert all(fa[f] == fb[f] for f in E_FIELDS)


def test_candidate_log_records_all_candidates_not_only_winners():
    m, _, _ = _fit("bounded", 0.2, log=True)
    winners = len(m.split_audit_)
    log = m.split_candidate_log_
    assert len(log) > 20 * winners > 0
    keys = {"node_id", "m", "n", "literals", "raw_information_gain", "stat_component", "weight", "onto_depth", "error_coverage", "score",
            "variant", "lambda_cap", "phase"}
    assert all(keys <= set(r) for r in log)
    assert {r["variant"] for r in log} == {"bounded"} and {r["phase"] for r in log} == {"semantic"}


def test_variants_are_deterministic_and_reproducible_across_seeds():
    for cal in ("bounded", "calibrated", "tiebreak"):
        runs = []
        for _ in range(2):
            m, sp, X = _fit(cal, 0.2, seed=11)
            runs.append((m.export_rules(), sp.stream_digest()))
        assert runs[0] == runs[1]


def test_candidate_log_summary_reports_real_scales_and_decision_switches():
    leg, _, _ = _fit(None, log=True)
    bnd, _, _ = _fit("bounded", 0.2, log=True)
    sl, sb = leg.candidate_log_summary(), bnd.candidate_log_summary()
    assert sl["nodes_with_candidates"] > 0 and 0.0 <= sl["switch_rate"] <= 1.0
    assert sl["ratio_m_over_s_median"] > 1.0                        # score histórico: termo semântico excede o estatístico (escala real)
    assert abs(sb["ratio_m_over_s_median"] - 0.2) < 1e-9            # bounded λ=0.2: |m|/s saturado em λ
    for n in sl["nodes"]:
        assert n["n_candidates"] > 0 and 0.0 <= (n["regret"] or 0.0) <= 1.0
