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
                  purity_epsilon_grid=(0.05, 0.01), max_nodes_grid=(7, 15), capacity_expansion=False)
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
    assert len(sel["per_repeat_winners"]) == 2 and len(sel["seeds"]) == 2 and 0.0 <= sel["selection_probability"] <= 1.0
    assert r["cv_plan"]["n_splits"] == 4 and r["cv_plan"]["repeats"] == 2


def test_tuning_is_deterministic():
    X, y = _data()
    runs = [tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=_fast())
            for _ in range(2)]
    f = [[h["stats"]["fidelity_mean"] for h in r["structure_history"]] for r in runs]
    assert f[0] == f[1] and runs[0]["structure_selected"] == runs[1]["structure_selected"]


def test_tuning_is_flagged_uncertain_when_selection_probability_is_below_the_threshold():
    X, y = _data()
    strict = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(),
                                    search=_fast(min_selection_probability=1.01, min_equivalent_set_probability=1.01))       # exigência impossível
    lax = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(),
                                 search=_fast(min_selection_probability=0.0))
    # com exigências impossíveis nunca é "estável"; se toda a grelha for equivalente o estado é non_discriminative_grid
    assert strict["tuning_stable"] is False and strict["structure_selection"]["status"] in {"tuning_uncertain", "non_discriminative_grid"}
    st = lax["structure_selection"]["status"]
    assert st in {"stable_exact", "stable_equivalent_subset", "non_discriminative_grid"}
    assert lax["tuning_stable"] is (st in {"stable_exact", "stable_equivalent_subset"})


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


# ------------------------------------------------------------------------------- orçamento não limitante
def test_non_binding_budget_is_common_to_all_candidates_and_never_exhausted():
    from core.training_config import non_binding_query_budget
    search = _fast()
    cands = _structure_candidates(_base(), search)
    budgets = {c.max_queries for c in cands}
    assert len(budgets) == 1                                                            # condições computacionais iguais
    assert budgets.pop() >= non_binding_query_budget(15, 300) > 15 * 300               # majora o consumo do maior candidato
    X, y = _data()
    r = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=search)
    assert r["query_budget_policy"] == "non_binding"
    chk = r["budget_check"]
    assert chk["all_candidates_same_budget"] and chk["any_budget_exhausted"] is False
    for h in r["structure_history"]:
        s = h["stats"]
        assert s["budget_exhausted_count"] == 0 and s["budget_exhausted_fraction"] == 0.0
        assert s["queries_used_mean"] <= s["query_budget"] and s["query_budget"] == chk["common_budget"]
        assert all(row["queries_used"] < s["query_budget"] and row["budget_exhausted"] is False for row in h["per_split"])


def test_required_budget_policy_is_still_available_and_differs_per_candidate():
    cands = _structure_candidates(_base(), _fast(query_budget_policy="required", max_nodes_grid=(31, 63)))
    assert len({c.max_queries for c in cands}) > 1


# ------------------------------------------------------------------------------- estabilidade estrutural
def test_structural_stability_beats_complexity_among_fidelity_equivalents():
    from core.trepan_scientific_tuning import structural_instability
    same = [0.90, 0.91, 0.89, 0.90, 0.91, 0.89]
    erratic = [3, 3, 3, 3, 27, 3]                    # 3,3,3,3,27: pequeno na média, mas dependente da amostragem
    steady = [20, 22, 21, 20, 23, 21]
    assert structural_instability(erratic) > 3 * structural_instability(steady)
    w, v, _ = _sel([same, same], [erratic, steady], _meta(31, 31))
    assert w == 1 and v[0]["reason_code"] == "LOST_STRUCTURAL_STABILITY"
    assert "estabilidade estrutural" in v[0]["text"] and "3–27" in v[0]["text"]


def test_structural_stability_does_not_favour_bigger_or_smaller_trees():
    same = [0.90, 0.91, 0.89, 0.90, 0.91, 0.89]
    small = [10, 10, 11, 10, 10, 11]
    big = [50, 50, 51, 50, 50, 51]
    w, _v, _ = _sel([same, same], [small, big], _meta(31, 63))
    assert w == 0                                    # igualmente estáveis: decide a complexidade (a mais simples)
    w2, _v2, _ = _sel([same, same], [big, small], _meta(63, 31))
    assert w2 == 1


def test_structural_instability_uses_depth_and_leaves_too():
    from core.trepan_scientific_tuning import structural_instability
    nodes = [21] * 6
    assert structural_instability(nodes) == 0.0
    assert structural_instability(nodes, depth=[2, 2, 2, 12, 2, 2]) > 0.5
    assert structural_instability(nodes, leaves=[2, 2, 2, 12, 2, 2]) > 0.5


def test_fidelity_still_comes_before_structure():
    good_but_erratic = [0.95, 0.96, 0.94, 0.95, 0.96, 0.94]
    steady_but_worse = [0.80, 0.81, 0.79, 0.80, 0.81, 0.79]
    w, v, _ = _sel([good_but_erratic, steady_but_worse], [[3, 3, 3, 3, 27, 3], [20] * 6], _meta(31, 31))
    assert w == 0 and v[1]["reason_code"] == "LOST_FIDELITY"


# ------------------------------------------------------------------------------- diagnóstico semântico/estrutural
def test_structure_diagnostics_are_recorded_and_not_used_for_selection():
    X, y = _data()
    r = tune_scientific_trepan(X, y, oracle=Oracle(), feature_names=list("abc"), base_config=_base(), search=_fast())
    for h in r["structure_history"]:
        g = h["stats"]["structure_diagnostics"]
        assert {"root_feature_freq", "root_feature_modal_share", "feature_usage_freq", "feature_set_jaccard_mean",
                "split_signature_jaccard_mean", "same_size_pairs", "same_size_feature_jaccard_mean", "stump_fraction"} <= set(g)
        assert 0.0 <= g["feature_set_jaccard_mean"] <= 1.0 and 0.0 <= g["root_feature_modal_share"] <= 1.0
        assert set(g["root_feature_freq"]) <= {"a", "b", "c"}
        assert {"root_features", "features", "splits"} <= set(h["per_split"][0])
    import core.trepan_scientific_tuning as mod
    src = inspect.getsource(mod._lexicographic_select)
    assert "structure_diagnostics" not in src and "jaccard" not in src.lower()


def test_jaccard_and_diagnostics_distinguish_same_size_but_different_explanations():
    from core.trepan_scientific_tuning import _structure_diagnostics
    rows = [{"nodes": 5, "root_features": [0], "features": [0, 1], "splits": [[0, 1], [1, 2]]},
            {"nodes": 5, "root_features": [2], "features": [2, 3], "splits": [[2, 4], [3, 5]]},
            {"nodes": 5, "root_features": [0], "features": [0, 1], "splits": [[0, 1], [1, 2]]}]
    g = _structure_diagnostics(rows, list("abcd"))
    assert g["root_feature_freq"] == {"a": 2 / 3, "c": 1 / 3} and g["same_size_pairs"] == 3
    assert g["same_size_feature_jaccard_mean"] == pytest.approx((0 + 1 + 0) / 3)    # mesmo nº de nós, explicações diferentes
