"""Expansão adaptativa da capacidade (genérica) e semântica exata do block bootstrap da seleção."""
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.trepan_scientific_tuning as tm
from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.trepan_scientific_tuning import (ScientificTrepanSearchConfig, _block_bootstrap, _lexicographic_select,
                                           _saturation_state, tune_scientific_trepan)

ROOT = Path(__file__).resolve().parents[1]


class Oracle:
    """Fronteira de decisão complexa (muitos nós úteis) ou simples, conforme ``rule``."""
    classes_ = np.array([0, 1])

    def __init__(self, rule="complex"):
        self.rule = rule

    def predict(self, X):
        X = np.asarray(X, float)
        if self.rule == "simple":
            return (X[:, 0] > 0).astype(int)
        return (np.sin(3 * X[:, 0]) + np.cos(2.5 * X[:, 1]) + 0.5 * np.sin(4 * X[:, 2]) > 0).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.c_[1 - p, p]


def _base(**kw):
    values = dict(max_nodes=7, max_depth=31, min_samples_leaf=2, min_sample=200, max_n=1, beam_width=1,
                  max_features_per_node=3, max_queries=2000, random_state=1)
    values.update(kw)
    return ControlledTrepanConfig(**values)


def _search(**kw):
    values = dict(cv_folds=2, cv_repeats=3, max_capacity_candidates=1, max_semantic_candidates=1, purity_epsilon_grid=(0.05,),
                  max_nodes_grid=(7, 15), bootstrap_resamples=30, max_nodes_safety_limit=63, capacity_expansion=True,
                  capacity_expansion_saturation_threshold=0.5, max_capacity_expansion_rounds=3)
    values.update(kw)
    return ScientificTrepanSearchConfig(**values)


def _data(n=220, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 3))


def _run(rule, seed=0, names=("a", "b", "c"), **skw):
    X = _data(seed=seed)
    return tune_scientific_trepan(X, Oracle(rule).predict(X), oracle=Oracle(rule), feature_names=list(names), base_config=_base(),
                                  search=_search(**skw))


# --------------------------------------------------------------------------------- expansão adaptativa
def test_expansion_is_triggered_by_saturation_and_is_fully_recorded():
    r = _run("complex")
    ex = r["capacity_expansion"]
    assert ex["initial_node_grid"] == [7, 15] and ex["expansion_triggered"] is True
    assert ex["capacity_expansion_rounds"] >= 1 and ex["final_node_grid"][:2] == [7, 15] and max(ex["final_node_grid"]) > 15
    for k in ("initial_node_grid", "final_node_grid", "capacity_expansion_rounds", "expansion_triggered", "expansion_stop_reason",
              "fraction_at_node_cap"):
        assert r[k] == ex[k]                                           # também no topo do relatório
    assert ex["expansion_stop_reason"] in {"saturation_resolved", "safety_limit_reached", "max_rounds_reached"}
    # geométrico: next = 2*current + 1
    grid = ex["final_node_grid"]
    assert all(b == 2 * a + 1 for a, b in zip(grid[1:], grid[2:]))
    assert all(rd["to_max_nodes"] == 2 * rd["from_max_nodes"] + 1 for rd in ex["rounds"])


def test_expansion_is_evaluated_on_exactly_the_same_folds_and_never_changes_purity_epsilon():
    r = _run("complex", purity_epsilon_grid=(0.05, 0.01))
    hist = r["structure_history"]
    first = [(row["seed"], row["fold"]) for row in hist[0]["per_split"]]
    assert all([(row["seed"], row["fold"]) for row in h["per_split"]] == first for h in hist)       # mesmas dobras
    assert {h["config"]["purity_epsilon"] for h in hist} == {0.05, 0.01}                          # só a grelha original
    for h in hist:                                                                                   # mesmo orçamento comum, não limitante
        assert h["stats"]["query_budget"] == hist[0]["stats"]["query_budget"] and h["stats"]["budget_exhausted_count"] == 0
    assert r["budget_check"]["any_budget_exhausted"] is False


def test_simple_data_stops_without_expansion_small_datasets_may_stay_at_the_initial_grid():
    r = _run("simple", max_nodes_grid=(7, 15))
    ex = r["capacity_expansion"]
    # um alvo simples não satura o teto: nenhum candidato competitivo precisa de mais nós
    assert ex["capacity_expansion_rounds"] == 0 or ex["expansion_stop_reason"] in {"saturation_resolved", "not_triggered",
                                                                                   "no_competitive_at_max_capacity"}
    assert max(ex["final_node_grid"]) <= 31


def test_expansion_depends_on_saturation_and_reaches_larger_capacity_only_when_data_justify_it():
    simple = _run("simple")["capacity_expansion"]
    complex_ = _run("complex")["capacity_expansion"]
    assert complex_["expansion_triggered"] is True
    assert max(complex_["final_node_grid"]) > max(simple["final_node_grid"])
    assert complex_["capacity_expansion_rounds"] > simple["capacity_expansion_rounds"]


def test_expansion_can_reach_127_or_beyond_when_allowed_and_the_safety_limit_stops_it():
    deep = _run("complex", max_nodes_safety_limit=255, max_capacity_expansion_rounds=4, max_nodes_grid=(7, 15))
    assert max(deep["capacity_expansion"]["final_node_grid"]) >= 63
    capped = _run("complex", max_nodes_safety_limit=15, max_nodes_grid=(7, 15))["capacity_expansion"]
    assert capped["capacity_expansion_rounds"] == 0 and capped["expansion_stop_reason"] == "safety_limit_reached"
    assert max(capped["final_node_grid"]) == 15
    rounds = _run("complex", max_capacity_expansion_rounds=1, max_nodes_safety_limit=255)["capacity_expansion"]
    assert rounds["capacity_expansion_rounds"] <= 1


def test_expansion_can_be_disabled_and_is_deterministic():
    off = _run("complex", capacity_expansion=False)["capacity_expansion"]
    assert off["expansion_stop_reason"] == "expansion_disabled" and off["capacity_expansion_rounds"] == 0
    a, b = _run("complex"), _run("complex")
    assert a["capacity_expansion"] == b["capacity_expansion"] and a["structure_selected"] == b["structure_selected"]


def test_renaming_the_dataset_gives_the_same_expansion_decision():
    X = _data(seed=3)
    y = Oracle("complex").predict(X)
    runs = [tune_scientific_trepan(X, y, oracle=Oracle("complex"), feature_names=names, base_config=_base(), search=_search())
            for names in (["a", "b", "c"], ["zz_measure_1", "outro", "terceira_coluna"])]
    assert runs[0]["capacity_expansion"] == runs[1]["capacity_expansion"]
    assert runs[0]["structure_selected"] == runs[1]["structure_selected"]
    labels = [{h["label"] for h in r["structure_history"]} for r in runs]
    assert labels[0] == labels[1]


def test_a_new_dataset_can_trigger_expansion_without_any_code_change(tmp_path):
    import core.scientific_benchmark_service as svc
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(260, 3)), columns=["novo_a", "novo_b", "novo_c"])
    df["classe_nova"] = np.where(np.sin(3 * df.novo_a) + np.cos(2.5 * df.novo_b) + 0.5 * np.sin(4 * df.novo_c) > 0, "sim", "nao")
    search = svc.scientific_search_config(cv_folds=2, cv_repeats=3, max_capacity_candidates=1, max_semantic_candidates=1,
                                          purity_epsilon_grid=(0.05,), max_nodes_grid=(7, 15), bootstrap_resamples=20,
                                          max_nodes_safety_limit=63)
    before = [p.read_bytes() for p in sorted((ROOT / "core").glob("*.py"))]
    out = svc.run_scientific_benchmark(df, target="classe_nova", seed=2, search=search)
    ex = out.report["evaluation"]["trepan_scientific_tuning"]["capacity_expansion"]
    assert {7, 15} <= set(ex["initial_node_grid"]) and ex["final_node_grid"][:len(ex["initial_node_grid"])] == ex["initial_node_grid"]
    assert before == [p.read_bytes() for p in sorted((ROOT / "core").glob("*.py"))]
    # o nó selecionado pode ser maior do que a grelha inicial: capacidade emerge dos dados
    assert out.report["evaluation"]["trepan_scientific_tuning"]["common_capacity"]["max_nodes"] in ex["final_node_grid"]


def test_expansion_never_uses_the_test_set(monkeypatch):
    X, y = _data(seed=1), None
    y = Oracle("complex").predict(X)
    seen = []
    real = tm.TrepanOriginalClassifier.fit

    def spy(self, Xfit, *a, **k):
        seen.append(np.asarray(Xfit))
        return real(self, Xfit, *a, **k)

    monkeypatch.setattr(tm.TrepanOriginalClassifier, "fit", spy)
    import inspect
    assert not {"X_test", "y_test"} & set(inspect.signature(tune_scientific_trepan).parameters)
    holdout = _data(60, seed=99)
    r = tune_scientific_trepan(X, y, oracle=Oracle("complex"), feature_names=list("abc"), base_config=_base(), search=_search())
    assert r["capacity_expansion"]["expansion_triggered"] and r["test_used_for_selection"] is False
    train_rows = {tuple(row) for row in X}
    assert seen and all(tuple(row) in train_rows for arr in seen for row in arr)       # só linhas de treino
    assert not any(tuple(row) in {tuple(h) for h in holdout} for arr in seen for row in arr)


def test_saturation_state_is_a_pure_function_of_cv_statistics():
    def h(label, nodes, frac, code):
        return {"label": label, "stats": {"max_nodes": nodes, "fraction_at_node_cap": frac}, "verdict": {"reason_code": code}}
    s = ScientificTrepanSearchConfig(capacity_expansion_saturation_threshold=0.5)
    assert _saturation_state([h("a", 31, 0.9, "WINNER"), h("b", 63, 0.6, "LOST_STABILITY")], s)["triggered"] is True
    assert _saturation_state([h("a", 31, 0.9, "WINNER"), h("b", 63, 0.2, "LOST_COMPLEXITY")], s)["triggered"] is False
    state = _saturation_state([h("a", 31, 0.9, "WINNER"), h("b", 63, 0.9, "LOST_FIDELITY")], s)
    assert state["triggered"] is False and state["reason"] == "no_competitive_at_max_capacity"
    assert _saturation_state([], s)["triggered"] is False


def test_expansion_code_contains_no_dataset_identity():
    text = (ROOT / "core" / "trepan_scientific_tuning.py").read_text()
    assert not re.search(r"iris|breast|cancer|wine|adult|diabetes|dataset_name", text, re.I)


# --------------------------------------------------------------------------------- bootstrap exato e semântica
def _arrays(means, stds, nodes, repeats=5, folds=3, seed=0):
    rng = np.random.default_rng(seed)
    J = repeats * folds
    F = np.array([m + s * rng.standard_normal(J) for m, s in zip(means, stds)])
    N = np.array([np.full(J, float(n)) for n in nodes])
    plan = [(r, 100 + r, f, None, None) for r in range(repeats) for f in range(folds)]
    meta = [{"config": {"purity_epsilon": 0.05, "max_nodes": int(n)}, "max_nodes": int(n)} for n in nodes]
    return F, N, plan, meta


def _bs(F, N, plan, meta, final, repeats=5, **skw):
    labels = [f"c{i}" for i in range(len(F))]
    search = ScientificTrepanSearchConfig(**skw)
    return _block_bootstrap(F, N, N, N, meta, 3, search, [True] * len(F), plan, repeats, labels, labels[final], None), labels


def test_five_blocks_use_exact_enumeration_of_all_3125_resamples_with_126_evaluations():
    F, N, plan, meta = _arrays([0.90, 0.895, 0.80], [0.02, 0.02, 0.02], [10, 12, 30])
    out, _ = _bs(F, N, plan, meta, final=0)
    assert out["method"] == "exact" and out["bootstrap_samples"] == 5 ** 5 == 3125
    assert out["bootstrap_evaluations"] == math.comb(9, 5) == 126
    assert abs(sum(out["distribution"].values()) - 1.0) < 1e-12 and out["selection_probability_mc_interval"] is None


def test_exact_enumeration_matches_brute_force_over_all_ordered_resamples():
    F, N, plan, meta = _arrays([0.90, 0.895, 0.85], [0.03, 0.03, 0.03], [10, 12, 30], repeats=4)
    out, labels = _bs(F, N, plan, meta, final=0, repeats=4)
    blocks = [[j for j, (rr, *_x) in enumerate(plan) if rr == r] for r in range(4)]
    counts = {}
    search = ScientificTrepanSearchConfig()
    for picks in itertools.product(range(4), repeat=4):                  # as 4^4 = 256 reamostragens ordenadas
        cols = [j for p in picks for j in blocks[p]]
        w = _lexicographic_select(F[:, cols], N[:, cols], meta, 3, search, [True] * 3, N[:, cols], N[:, cols])[0]
        counts[labels[w]] = counts.get(labels[w], 0) + 1
    brute = {k: v / 256 for k, v in counts.items()}
    assert out["method"] == "exact" and out["bootstrap_samples"] == 256
    assert {k: pytest.approx(v) for k, v in brute.items()} == {k: pytest.approx(v) for k, v in out["distribution"].items()}


def test_many_blocks_fall_back_to_deterministic_monte_carlo():
    F, N, plan, meta = _arrays([0.90, 0.895], [0.02, 0.02], [10, 12], repeats=10)
    kw = dict(bootstrap_exact_limit=100, bootstrap_resamples=80, bootstrap_seed=7)
    a, _ = _bs(F, N, plan, meta, final=0, repeats=10, **kw)
    b, _ = _bs(F, N, plan, meta, final=0, repeats=10, **kw)
    assert a == b and a["method"] == "monte_carlo" and a["bootstrap_samples"] == 80 and a["seed"] == 7
    assert a["selection_probability_mc_interval"] is not None and a["bootstrap_evaluations"] <= 80


def test_semantics_full_cv_winner_vs_bootstrap_mode_are_distinguished_never_a_negative_margin():
    # candidato 0 vence na CV completa por pouco; nos reamostrados o candidato 1 vence mais vezes
    F, N, plan, meta = _arrays([0.0, 0.0], [0.0, 0.0], [10, 10], repeats=5)
    good = np.array([0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90])
    F = np.array([good, good + np.array([-0.02, -0.02, -0.02] * 5)])
    N = np.array([np.full(15, 10.0), np.full(15, 10.0)])
    labels = ["c0", "c1"]
    search = ScientificTrepanSearchConfig()
    out = _block_bootstrap(F, N, N, N, meta, 3, search, [True, True], plan, 5, labels, "c1", None)   # "escolhida" menos provável
    assert out["selected_config_full_cv"] == "c1" and out["bootstrap_modal_config"] != "c1"
    assert out["selected_is_modal"] is False and out["full_cv_selection_fragile"] is True
    assert out["bootstrap_modal_probability"] > out["selected_config_probability"]
    ru = out["bootstrap_runner_up"]
    assert ru is None or ru["probability"] <= out["bootstrap_modal_probability"]
    assert out["top1_top2_margin"] >= 0 and out["full_cv_vs_modal_gap"] > 0
    assert "runner_up" not in out and "margin" not in out                  # nomes ambíguos eliminados


def test_tuning_is_uncertain_when_the_full_cv_winner_is_not_the_bootstrap_mode(monkeypatch):
    real = tm._block_bootstrap

    def fake(*a, **k):
        out = real(*a, **k)
        labels = a[10]
        other = [l for l in labels if l != a[11]][0]
        out.update({"bootstrap_modal_config": other, "bootstrap_modal_probability": 0.65, "selection_probability": 0.12,
                    "selected_config_probability": 0.12, "selected_is_modal": False, "full_cv_selection_fragile": True,
                    "bootstrap_runner_up": {"label": a[11], "probability": 0.12}, "top1_top2_margin": 0.53, "full_cv_vs_modal_gap": 0.53})
        return out
    monkeypatch.setattr(tm, "_block_bootstrap", fake)
    r = _run("complex", capacity_expansion=False, min_selection_probability=0.0)
    sel = r["structure_selection"]
    assert sel["status"] == "tuning_uncertain" and sel["status_reason"] == "full_cv_selection_not_bootstrap_modal"
    assert sel["selected_config_full_cv"] == sel["selected_label"] and sel["bootstrap_modal_config"] != sel["selected_label"]
    assert r["tuning_stable"] is False and sel["full_cv_selection_fragile"] is True


def test_stable_requires_modal_and_enough_probability():
    r = _run("simple", capacity_expansion=False, max_nodes_grid=(7,))
    sel = r["structure_selection"]
    if sel["selected_config_full_cv"] == sel["bootstrap_modal_config"] and sel["selection_probability"] >= sel["threshold"]:
        assert sel["status"] == "tuning_stable" and sel["status_reason"] == "ok"
    for key in ("bootstrap_modal_probability", "bootstrap_runner_up", "top1_top2_margin", "selected_config_probability"):
        assert key in sel
    assert sel["bootstrap"]["method"] == "exact" and sel["bootstrap"]["bootstrap_samples"] == 3 ** 3


def test_censored_structural_stability_is_labelled_as_weak_evidence():
    r = _run("complex", capacity_expansion=False, max_nodes_grid=(7,))
    for h in r["structure_history"]:
        s = h["stats"]
        if s["structural_stability_censored"]:
            assert s["structural_stability_evidence"] == "censored_by_node_cap"
            assert "não é evidência forte" in h["verdict"]["text"]
        else:
            assert s["structural_stability_evidence"] == "observed"
