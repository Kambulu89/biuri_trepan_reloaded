"""Tuning científico da estrutura do TREPAN: Repeated Stratified K-Fold no treino, estatísticas por candidato,
seleção lexicográfica (sem viés para árvores maiores), veredictos, estabilidade e fallback canónico."""
import inspect

import numpy as np
import pytest

from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.training_config import get_training_preset, resolve_trepan_structure_limits
from core.trepan_scientific_tuning import (ScientificTrepanSearchConfig, _capacity_candidates, _cv_plan,
                                           _lexicographic_select, _structure_candidates, tune_scientific_trepan)


class Oracle:
    classes_ = np.array([0, 1])

    def predict(self, X):
        X = np.asarray(X, float)
        return ((X[:, 0] + 0.6 * X[:, 1] * X[:, 2]) > 0).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.c_[1 - p, p]


def _base(**kw):
    values = dict(max_nodes=31, max_depth=31, min_samples_leaf=2, min_sample=300, max_n=2, beam_width=1,
                  max_features_per_node=4, max_queries=6000, random_state=3)
    values.update(kw)
    return ControlledTrepanConfig(**values)


def _fast(**kw):
    values = dict(cv_folds=2, cv_repeats=2, max_capacity_candidates=1, max_semantic_candidates=1,
                  purity_epsilon_grid=(0.05, 0.01), max_nodes_grid=(7, 15))
    values.update(kw)
    return ScientificTrepanSearchConfig(**values)


def _data(n=150, seed=0, d=3):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return X, Oracle().predict(X)


# ------------------------------------------------------------------------------- defaults canónicos
def test_canonical_defaults_are_the_literature_values_not_dataset_tuned():
    preset = get_training_preset("scientific")
    assert preset.trepan_purity_epsilon == 0.05
    assert resolve_trepan_structure_limits(preset)["max_nodes"] == 31
    assert ControlledTrepanConfig().purity_epsilon == 0.05 and ControlledTrepanConfig().max_nodes == 31
    for key in ("fast", "balanced", "scientific"):
        assert get_training_preset(key).trepan_purity_epsilon == 0.05


def test_structure_grid_is_configurable_and_default_has_the_requested_candidates():
    search = ScientificTrepanSearchConfig()
    assert search.purity_epsilon_grid == (0.05, 0.02, 0.01) and search.max_nodes_grid == (31, 63)
    cands = _structure_candidates(_base(), search)
    assert (cands[0].purity_epsilon, cands[0].max_nodes) == (0.05, 31)           # base canónica primeiro
    assert {(c.purity_epsilon, c.max_nodes) for c in cands} == {(e, n) for e in (0.05, 0.02, 0.01) for n in (31, 63)}
    from core.training_config import required_query_budget
    # equidade: todos os candidatos têm o orçamento necessário para os seus próprios nós (31 e 63), não só os maiores
    assert all(c.max_queries >= required_query_budget(c.max_nodes, c.min_sample, cap=250_000) for c in cands)
    assert len({c.max_queries for c in cands if c.max_nodes == 31}) == 1 and cands[0].max_queries == max(c.max_queries for c in cands if c.max_nodes == 31)
    custom = _structure_candidates(_base(), ScientificTrepanSearchConfig(purity_epsilon_grid=(0.05, 0.1), max_nodes_grid=(15, 31, 127)))
    assert {(c.purity_epsilon, c.max_nodes) for c in custom} == {(e, n) for e in (0.05, 0.1) for n in (15, 31, 127)} | {(0.05, 31)}


# ------------------------------------------------------------------------------- plano de CV e isolamento
def test_cv_plan_is_repeated_stratified_deterministic_and_train_only():
    y = np.array([0] * 40 + [1] * 20)
    search = ScientificTrepanSearchConfig(cv_folds=3, cv_repeats=3)
    plan, k, seeds = _cv_plan(y, search, base_seed=42)
    plan2, _, seeds2 = _cv_plan(y, search, base_seed=42)
    assert k == 3 and len(seeds) == 3 and len(set(seeds)) == 3 and len(plan) == 9
    assert seeds == seeds2 and all(np.array_equal(a[3], b[3]) for a, b in zip(plan, plan2))   # determinístico
    for r in range(3):
        va = np.concatenate([p[4] for p in plan if p[0] == r])
        assert sorted(va) == list(range(len(y)))                                  # cada repetição particiona o treino
    for (_r, _s, _f, tr, va) in plan:
        assert set(tr).isdisjoint(va) and max(va) < len(y)                        # só índices do treino
        assert 0.25 < y[va].mean() < 0.42                                         # estratificado (1/3 de positivos)
    assert _cv_plan(y, ScientificTrepanSearchConfig(cv_seeds=(5, 9)), 42)[2] == (5, 9)          # seeds explícitas


def test_tuning_function_cannot_receive_a_test_set():
    params = set(inspect.signature(tune_scientific_trepan).parameters)
    assert not {"X_test", "y_test", "X_val", "test_size", "test"} & params
    X, y = _data()
    r = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=_fast())
    assert r["test_used_for_selection"] is False and r["cv_plan"]["test_set_used"] is False
    assert r["selection_scope"] == "training_cv_only"


# ------------------------------------------------------------------------------- regra lexicográfica
def _meta(*sizes):
    return [{"config": {"purity_epsilon": 0.05, "max_nodes": n}, "max_nodes": n} for n in sizes]


SEARCH = ScientificTrepanSearchConfig()


def _sel(F, N, meta, valid=None):
    F = np.array(F, float); N = np.array(N, float)
    return _lexicographic_select(F, N, meta, 3, SEARCH, valid or [True] * len(F))


def test_significantly_better_fidelity_wins_even_if_the_tree_is_bigger():
    F = [[0.80, 0.81, 0.79, 0.80, 0.81, 0.79], [0.90, 0.91, 0.89, 0.90, 0.91, 0.89]]
    w, v, _ = _sel(F, [[3] * 6, [40] * 6], _meta(31, 63))
    assert w == 1 and v[1]["status"] == "WINNER" and v[0]["reason_code"] == "LOST_FIDELITY"
    assert "significativamente inferior" in v[0]["text"] and "t=" in v[0]["text"]


def test_significantly_better_fidelity_wins_even_if_the_tree_is_smaller_too():
    F = [[0.90, 0.91, 0.89, 0.90, 0.91, 0.89], [0.80, 0.81, 0.79, 0.80, 0.81, 0.79]]
    w, v, _ = _sel(F, [[3] * 6, [40] * 6], _meta(31, 63))
    assert w == 0 and v[1]["reason_code"] == "LOST_FIDELITY"


def test_indistinguishable_fidelity_prefers_stability_over_complexity_and_over_the_higher_mean():
    noisy_best = [0.95, 0.85, 0.99, 0.84, 0.97, 0.90]            # maior média, mas instável
    stable = [0.915, 0.92, 0.91, 0.915, 0.92, 0.91]
    assert np.mean(noisy_best) > np.mean(stable)
    w, v, _ = _sel([noisy_best, stable], [[3] * 6, [40] * 6], _meta(31, 63))
    assert w == 1                                                    # estável vence, mesmo sendo a árvore MAIOR
    assert v[0]["reason_code"] == "LOST_STABILITY" and "desvio-padrão" in v[0]["text"]


def test_indistinguishable_and_equally_stable_prefers_the_less_complex_never_the_bigger():
    same = [0.90, 0.91, 0.89, 0.90, 0.91, 0.89]
    w, v, _ = _sel([same, same, same], [[40] * 6, [3] * 6, [20] * 6], _meta(63, 31, 63))
    assert w == 1 and v[0]["reason_code"] == "LOST_COMPLEXITY" and v[2]["reason_code"] == "LOST_COMPLEXITY"
    w2, _v2, _ = _sel([same, same], [[3] * 6, [40] * 6], _meta(31, 63))
    assert w2 == 0


def test_real_tie_resolves_to_the_simplest_then_canonical_order():
    same = [0.90, 0.91, 0.89, 0.90, 0.91, 0.89]
    w, v, _ = _sel([same, same], [[3] * 6, [3] * 6], _meta(63, 31))
    assert w == 1 and v[0]["reason_code"] == "LOST_TIE" and "mais simples" in v[0]["text"]
    w2, _v, _ = _sel([same, same], [[3] * 6, [3] * 6], _meta(31, 31))
    assert w2 == 0                                                   # ordem canónica (base primeiro)


def test_failed_candidates_are_reported_and_never_win():
    same = [0.9] * 6
    w, v, _ = _sel([same, [0.0] * 6], [[3] * 6, [0] * 6], _meta(31, 63), valid=[True, False])
    assert w == 0 and v[1]["status"] == "FAILED"
    assert _sel([same], [[3] * 6], _meta(31), valid=[False])[0] is None


# ------------------------------------------------------------------------------- tuning completo
def test_every_candidate_records_stats_and_a_verdict_and_exactly_one_wins():
    X, y = _data()
    r = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=_fast())
    hist = r["structure_history"]
    assert len(hist) == 5                                                              # base (0.05, 31) + 4 da grelha (7, 15)
    need = {"budget_exhausted_fraction", "fidelity_mean", "fidelity_std", "fidelity_min", "nodes_mean", "nodes_std", "nodes_cv", "depth_mean",
            "between_seed_std", "within_seed_fold_std_mean", "time_mean_s", "query_budget", "queries_used_mean", "n_splits"}
    for h in hist:
        assert need <= set(h["stats"]) and h["stats"]["n_splits"] == 4               # 2 dobras x 2 repetições
        assert len(h["per_split"]) == 4 and {"repeat", "seed", "fold", "fidelity", "nodes", "depth", "time_s"} <= set(h["per_split"][0])
        assert h["verdict"]["text"] and h["verdict"]["status"] in {"WINNER", "LOST"}
        assert h["stats"]["fidelity_min"] <= h["stats"]["fidelity_mean"] <= h["stats"]["fidelity_max"]
    assert sum(h["verdict"]["status"] == "WINNER" for h in hist) == 1
    sel = r["structure_selection"]
    assert sel["selected_label"] == [h["label"] for h in hist if h["verdict"]["status"] == "WINNER"][0]
    assert len(sel["per_seed_winners"]) == 2 and len(sel["seeds"]) == 2 and 0.0 <= sel["agreement"] <= 1.0
    assert r["cv_plan"]["n_splits"] == 4 and r["cv_plan"]["repeats"] == 2


def test_tuning_is_deterministic():
    X, y = _data()
    runs = [tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=_fast())
            for _ in range(2)]
    f = [[h["stats"]["fidelity_mean"] for h in r["structure_history"]] for r in runs]
    assert f[0] == f[1] and runs[0]["structure_selected"] == runs[1]["structure_selected"]


def test_tuning_is_flagged_unstable_when_seeds_disagree_with_the_final_choice():
    X, y = _data()
    strict = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(),
                                    search=_fast(min_selection_agreement=1.01))       # exigência impossível
    lax = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(),
                                 search=_fast(min_selection_agreement=0.0))
    assert strict["tuning_stable"] is False and strict["structure_selection"]["stable"] is False
    assert lax["tuning_stable"] is True


def test_tuning_off_or_failing_keeps_the_canonical_base():
    X, y = _data(120)
    for search in (_fast(tune_structure=False), _fast(purity_epsilon_grid=()), _fast(max_nodes_grid=())):
        r = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=search)
        assert r["structure_history"] == [] and r["structure_selection"] is None
        assert (r["common_capacity"]["purity_epsilon"], r["common_capacity"]["max_nodes"]) == (0.05, 31)


def test_all_candidates_failing_falls_back_to_the_canonical_base(monkeypatch):
    import core.trepan_scientific_tuning as mod

    class Boom:
        def __init__(self, **kw):
            raise RuntimeError("falha simulada")

    monkeypatch.setattr(mod, "TrepanOriginalClassifier", Boom)
    X, y = _data(90)
    r = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=_fast())
    assert (r["common_capacity"]["purity_epsilon"], r["common_capacity"]["max_nodes"]) == (0.05, 31)
    assert all(h["failed"] and h["verdict"]["status"] == "FAILED" for h in r["structure_history"])
    assert r["structure_selection"]["fallback"] == "all_candidates_failed" and r["tuning_stable"] is False


def test_capacity_stage_does_not_grow_nodes_beyond_the_grid():
    base = _base(max_nodes=63, max_depth=63)
    assert max(c.max_nodes for c in _capacity_candidates(base, 5, 6, grow_nodes=False)) == 63
    assert max(c.max_nodes for c in _capacity_candidates(base, 5, 6, grow_nodes=True)) > 63


def test_no_dataset_specific_logic_in_the_tuning_module():
    import core.trepan_scientific_tuning as mod
    src = inspect.getsource(mod).lower()
    for name in ("breast", "iris", "wdbc", "wisconsin", "adult", "german"):
        assert name not in src
