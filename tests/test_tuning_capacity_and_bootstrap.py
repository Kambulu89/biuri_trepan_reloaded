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
                                           tune_scientific_trepan)

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


# --------------------------------------------------------------------------------- expansão: saturação + ganho
def _entry(eps, nodes, fid, frac, J=15):
    return {"config": {"purity_epsilon": eps, "max_nodes": nodes}, "label": f"purity_epsilon={eps:g}, max_nodes={nodes}",
            "stats": {"fraction_at_node_cap": frac, "max_nodes": nodes},
            "per_split": [{"fidelity": float(f)} for f in fid]}


def _level(nodes, base, frac, eps=(0.05, 0.01), J=15, seed=0):
    """Fidelity por partição = base + ruído PARTILHADO entre capacidades (emparelhado) + pequena variação por epsilon."""
    shared = np.random.default_rng(seed).normal(0, 0.004, J)                 # ruído das partições (partilhado entre capacidades)
    own = np.random.default_rng(1000 + nodes).normal(0, 0.003, J)            # variação própria de cada capacidade
    return [_entry(e, nodes, base + shared + own + 0.0005 * i, frac, J) for i, e in enumerate(eps)]


class FakeEval:
    """Substitui o ajuste das árvores: devolve fidelity sintética por nível e regista que níveis foram avaliados."""
    def __init__(self, base_by_level, frac=1.0, eps=(0.05, 0.01)):
        self.base, self.frac, self.eps, self.calls = base_by_level, frac, eps, []

    def __call__(self, nodes):
        self.calls.append(nodes)
        return _level(nodes, self.base[nodes], self.frac, self.eps)


def _expand(base_by_level, initial, frac=1.0, **skw):
    hist = [e for n in initial for e in _level(n, base_by_level[n], frac)]
    fe = FakeEval(base_by_level, frac)
    search = ScientificTrepanSearchConfig(**{"max_nodes_safety_limit": 255, "max_capacity_expansion_rounds": 5, **skw})
    out = tm.run_capacity_expansion(hist, fe, search, 3)
    return out, fe, hist


def test_saturated_with_supported_gain_expands_and_the_new_level_is_evaluated_on_the_same_partitions():
    out, fe, hist = _expand({31: 0.80, 63: 0.85, 127: 0.85}, [31, 63])
    assert fe.calls == [127] and out["capacity_expansion_rounds"] == 1 and out["expansion_triggered"] is True
    assert out["steps"][0]["capacity_gain_supported"] is True and out["steps"][0]["decision"] == "expand"
    assert out["final_node_grid"] == [31, 63, 127]
    assert all(len(h["per_split"]) == 15 for h in hist)                         # mesmas 15 partições


def test_saturated_but_plateau_does_not_expand_and_reports_validation_plateau():
    out, fe, _ = _expand({31: 0.850, 63: 0.8502}, [31, 63])
    assert fe.calls == [] and out["capacity_expansion_rounds"] == 0 and out["expansion_triggered"] is False
    assert out["expansion_stop_reason"] == "validation_plateau" and out["final_node_grid"] == [31, 63]
    assert "não justificam mais capacidade" in out["interpretation"] and "crescer estruturalmente" in out["interpretation"]
    st = out["steps"][0]
    assert st["saturated"] is True and st["capacity_gain_supported"] is False and st["decision"] == "stop"


def test_not_saturated_never_expands_even_with_a_large_gain():
    out, fe, _ = _expand({31: 0.70, 63: 0.90}, [31, 63], frac=0.1)
    assert fe.calls == [] and out["expansion_stop_reason"] == "not_saturated" and out["capacity_expansion_rounds"] == 0
    assert out["steps"][0]["saturated"] is False and out["steps"][0]["capacity_gain_supported"] is True


def test_plateau_at_63_prevents_127_and_255_from_being_evaluated():
    out, fe, hist = _expand({31: 0.80, 63: 0.8001, 127: 0.9, 255: 0.95}, [31, 63])
    assert fe.calls == [] and out["final_node_grid"] == [31, 63]
    assert {h["config"]["max_nodes"] for h in hist} == {31, 63}


def test_gain_up_to_63_then_plateau_at_127_stops_at_127_without_evaluating_255():
    out, fe, hist = _expand({31: 0.80, 63: 0.86, 127: 0.8601, 255: 0.99}, [31, 63])
    assert fe.calls == [127] and out["final_node_grid"] == [31, 63, 127]
    assert out["expansion_stop_reason"] == "validation_plateau" and out["last_supported_max_nodes"] == 63
    assert {h["config"]["max_nodes"] for h in hist} == {31, 63, 127}                # 255 nunca avaliado


def test_continued_gains_expand_geometrically_until_the_safety_limit():
    base = {31: 0.70, 63: 0.76, 127: 0.82, 255: 0.88, 511: 0.94}
    out, fe, _ = _expand(base, [31, 63], max_nodes_safety_limit=255)
    assert fe.calls == [127, 255] and out["expansion_stop_reason"] == "safety_limit_reached" and out["final_node_grid"] == [31, 63, 127, 255]
    assert all(b == 2 * a + 1 for a, b in zip(out["final_node_grid"][1:], out["final_node_grid"][2:]))
    capped, fe2, _ = _expand(base, [31, 63], max_capacity_expansion_rounds=1)
    assert fe2.calls == [127] and capped["expansion_stop_reason"] == "max_rounds_reached"


def test_every_step_records_the_capacity_comparison_evidence():
    out, _fe, _ = _expand({31: 0.80, 63: 0.86, 127: 0.8601}, [31, 63])
    for st in out["steps"]:
        for k in ("previous_max_nodes", "candidate_max_nodes", "previous_fidelity", "candidate_fidelity", "fidelity_delta",
                  "statistical_test", "capacity_gain_supported", "fraction_at_node_cap", "paired_purity_epsilon", "per_epsilon_delta"):
            assert k in st
        t = st["statistical_test"]
        assert t["available"] and {"name", "t", "t_critical", "p_value", "ci95_delta", "alpha", "n_splits"} <= set(t) and 0.0 <= t["p_value"] <= 1.0
        assert st["fidelity_delta"] == pytest.approx(st["candidate_fidelity"] - st["previous_fidelity"])
    first = out["steps"][0]
    assert first["capacity_gain_supported"] is True and first["statistical_test"]["p_value"] < 0.05
    assert first["statistical_test"]["ci95_delta"][0] > 0                      # IC do ganho exclui zero
    assert out["steps"][1]["capacity_gain_supported"] is False and out["steps"][1]["statistical_test"]["p_value"] > 0.05


def test_capacity_is_compared_at_the_same_purity_epsilon_only():
    hist = _level(31, 0.80, 1.0, eps=(0.05, 0.01)) + _level(63, 0.85, 1.0, eps=(0.05, 0.02))
    step = tm.capacity_gain_step(hist, 31, 63, ScientificTrepanSearchConfig(), 3)
    assert step["paired_purity_epsilon"] == [0.05] and list(step["per_epsilon_delta"]) == [0.05]
    none = tm.capacity_gain_step(_level(31, 0.8, 1.0, eps=(0.05,)) + _level(63, 0.85, 1.0, eps=(0.01,)), 31, 63,
                                 ScientificTrepanSearchConfig(), 3)
    assert none["capacity_gain_supported"] is False and none["statistical_test"]["available"] is False


def test_expansion_decisions_are_deterministic_and_independent_of_labels():
    a, _, _ = _expand({31: 0.80, 63: 0.86, 127: 0.8601}, [31, 63])
    b, _, _ = _expand({31: 0.80, 63: 0.86, 127: 0.8601}, [31, 63])
    assert a == b
    hist = _level(31, 0.80, 1.0) + _level(63, 0.86, 1.0)
    for h in hist:                                       # nomes de dataset/feature/classe não existem aqui; trocar rótulos não muda nada
        h["label"] = "x_" + h["label"][::-1]
    assert tm.capacity_gain_step(hist, 31, 63, ScientificTrepanSearchConfig(), 3)["capacity_gain_supported"] is True


def test_expansion_can_be_disabled_or_lack_levels():
    off, fe, _ = _expand({31: 0.8, 63: 0.9}, [31, 63], capacity_expansion=False)
    assert off["expansion_stop_reason"] == "expansion_disabled" and fe.calls == []
    one, fe1, _ = _expand({31: 0.8}, [31])
    assert one["expansion_stop_reason"] == "insufficient_capacity_levels" and fe1.calls == []


def test_saturation_threshold_is_configurable():
    low, fe, _ = _expand({31: 0.80, 63: 0.86, 127: 0.8601}, [31, 63], frac=0.4, capacity_expansion_saturation_threshold=0.3)
    assert fe.calls == [127]
    high, fe2, _ = _expand({31: 0.80, 63: 0.86}, [31, 63], frac=0.4, capacity_expansion_saturation_threshold=0.9)
    assert fe2.calls == [] and high["expansion_stop_reason"] == "not_saturated"


# --------------------------------------------------------------------------------- expansão: ponta a ponta com dados reais
def _check_expansion_invariants(r):
    ex = r["capacity_expansion"]
    for k in ("initial_node_grid", "final_node_grid", "capacity_expansion_rounds", "expansion_triggered", "expansion_stop_reason",
              "fraction_at_node_cap"):
        assert r[k] == ex[k]
    grid = ex["final_node_grid"]
    assert grid[:len(ex["initial_node_grid"])] == ex["initial_node_grid"] and len(grid) == len(ex["initial_node_grid"]) + ex["capacity_expansion_rounds"]
    assert all(b == 2 * a + 1 for a, b in zip(grid[len(ex["initial_node_grid"]) - 1:], grid[len(ex["initial_node_grid"]):]))
    assert ex["expansion_stop_reason"] in {"validation_plateau", "not_saturated", "safety_limit_reached", "max_rounds_reached",
                                           "insufficient_capacity_levels", "expansion_disabled", "no_new_candidates"}
    for i, st in enumerate(ex["steps"]):
        if st["decision"] == "expand":
            assert st["saturated"] and st["capacity_gain_supported"] and i < len(ex["steps"]) - 1 or i == len(ex["steps"]) - 1
    if ex["expansion_stop_reason"] == "validation_plateau":
        assert ex["steps"][-1]["capacity_gain_supported"] is False and ex["steps"][-1]["saturated"] is True
        assert max(grid) == ex["steps"][-1]["candidate_max_nodes"]               # nenhum nível acima do plateau
    hist = r["structure_history"]
    first = [(row["seed"], row["fold"]) for row in hist[0]["per_split"]]
    assert all([(row["seed"], row["fold"]) for row in h["per_split"]] == first for h in hist)           # mesmas dobras


def test_real_runs_satisfy_the_expansion_invariants_and_never_change_purity_epsilon():
    for rule in ("complex", "simple"):
        r = _run(rule, purity_epsilon_grid=(0.05, 0.01))
        _check_expansion_invariants(r)
        assert {h["config"]["purity_epsilon"] for h in r["structure_history"]} == {0.05, 0.01}
        assert r["budget_check"]["any_budget_exhausted"] is False and r["test_used_for_selection"] is False


def test_renaming_the_dataset_gives_the_same_expansion_decision():
    X = _data(seed=3)
    y = Oracle("complex").predict(X)
    runs = [tune_scientific_trepan(X, y, oracle=Oracle("complex"), feature_names=names, base_config=_base(), search=_search())
            for names in (["a", "b", "c"], ["zz_measure_1", "outro", "terceira_coluna"])]
    assert runs[0]["capacity_expansion"] == runs[1]["capacity_expansion"]
    assert runs[0]["structure_selected"] == runs[1]["structure_selected"]


def test_a_new_dataset_runs_the_expansion_policy_without_any_code_change():
    import core.scientific_benchmark_service as svc
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(260, 3)), columns=["novo_a", "novo_b", "novo_c"])
    df["classe_nova"] = np.where(np.sin(3 * df.novo_a) + np.cos(2.5 * df.novo_b) + 0.5 * np.sin(4 * df.novo_c) > 0, "sim", "nao")
    search = svc.scientific_search_config(cv_folds=2, cv_repeats=3, max_capacity_candidates=1, max_semantic_candidates=1,
                                          purity_epsilon_grid=(0.05,), max_nodes_grid=(7, 15), bootstrap_resamples=20,
                                          max_nodes_safety_limit=63)
    before = [p.read_bytes() for p in sorted((ROOT / "core").glob("*.py"))]
    out = svc.run_scientific_benchmark(df, target="classe_nova", seed=2, search=search)
    t = out.report["evaluation"]["trepan_scientific_tuning"]
    ex = t["capacity_expansion"]
    assert {7, 15} <= set(ex["initial_node_grid"]) and ex["expansion_stop_reason"] and ex["steps"]
    assert before == [p.read_bytes() for p in sorted((ROOT / "core").glob("*.py"))]
    assert t["common_capacity"]["max_nodes"] in ex["final_node_grid"]


def test_expansion_never_uses_the_test_set(monkeypatch):
    X = _data(seed=1)
    y = Oracle("complex").predict(X)
    seen = []
    real = tm.TrepanOriginalClassifier.fit

    def spy(self, Xfit, *a, **k):
        seen.append(np.asarray(Xfit))
        return real(self, Xfit, *a, **k)

    monkeypatch.setattr(tm.TrepanOriginalClassifier, "fit", spy)
    import inspect
    assert not {"X_test", "y_test"} & set(inspect.signature(tune_scientific_trepan).parameters)
    assert not {"X_test", "y_test"} & set(inspect.signature(tm.run_capacity_expansion).parameters)
    holdout = {tuple(h) for h in _data(60, seed=99)}
    r = tune_scientific_trepan(X, y, oracle=Oracle("complex"), feature_names=list("abc"), base_config=_base(), search=_search())
    assert r["test_used_for_selection"] is False and r["capacity_expansion"]["steps"]
    train_rows = {tuple(row) for row in X}
    assert seen and all(tuple(row) in train_rows for arr in seen for row in arr)
    assert not any(tuple(row) in holdout for arr in seen for row in arr)


def test_expansion_code_contains_no_dataset_identity():
    text = (ROOT / "core" / "trepan_scientific_tuning.py").read_text()
    assert not re.search(r"iris|breast|cancer|wine|adult|diabetes|german|dataset_name", text, re.I)


# --------------------------------------------------------------------------------- equivalência e estados científicos
def test_equivalent_candidate_set_and_probabilities_are_reported():
    r = _run("complex", capacity_expansion=False, purity_epsilon_grid=(0.05, 0.01, 0.02))
    sel = r["structure_selection"]
    ids = sel["equivalent_candidate_ids"]
    assert sel["selected_config_full_cv"] in ids and sel["equivalent_candidate_count"] == len(ids) == sel["equivalent_candidate_set"]["count"]
    labels = {h["label"] for h in r["structure_history"]}
    assert set(ids) <= labels
    assert sel["exact_selection_probability"] == sel["selected_config_probability"] == sel["bootstrap"]["selected_config_probability"]
    assert sel["selected_config_probability"] - 1e-9 <= sel["equivalent_set_probability"] <= 1.0 + 1e-9    # a família contém a escolhida
    assert sel["bootstrap_modal_probability"] >= sel["selected_config_probability"] - 1e-12
    assert "tree_behavior" in sel and "behavior_unstable" in sel["tree_behavior"]


def test_statuses_distinguish_exact_family_and_uncertain_without_changing_the_selected_config():
    kw = dict(capacity_expansion=False, purity_epsilon_grid=(0.05, 0.01), behavior_instability_threshold=1e9)
    exact = _run("simple", min_selection_probability=0.0, **kw)["structure_selection"]
    assert exact["status"] == "stable_exact" and exact["stable_exact"] is True
    base = _run("simple", min_selection_probability=1.01, min_equivalent_set_probability=0.0, **kw)["structure_selection"]
    assert base["status"] == "stable_equivalent_set" and base["family_stable"] is True and base["stable_exact"] is False
    assert base["status_reason"] == "exact_hyperparameter_unstable_but_equivalent_family_stable"
    bad = _run("simple", min_selection_probability=1.01, min_equivalent_set_probability=1.01, **kw)["structure_selection"]
    assert bad["status"] == "tuning_uncertain"
    assert exact["selected_config_full_cv"] == base["selected_config_full_cv"] == bad["selected_config_full_cv"]   # bootstrap só diagnostica


def test_unstable_tree_behaviour_makes_the_tuning_uncertain_even_if_the_family_is_stable():
    r = _run("simple", capacity_expansion=False, purity_epsilon_grid=(0.05, 0.01), min_selection_probability=0.0,
             behavior_instability_threshold=-1.0)
    sel = r["structure_selection"]
    assert sel["tree_behavior"]["behavior_unstable"] is True and sel["status"] == "tuning_uncertain"
    assert sel["status_reason"] == "tree_behavior_unstable"


def test_overall_status_aggregates_stage_statuses():
    assert tm._overall_status([]) == "not_assessed"
    assert tm._overall_status([{"status": "stable_exact"}, {"status": "stable_exact"}]) == "stable_exact"
    assert tm._overall_status([{"status": "stable_exact"}, {"status": "stable_equivalent_set"}]) == "stable_equivalent_set"
    assert tm._overall_status([{"status": "stable_exact"}, {"status": "tuning_uncertain"}]) == "tuning_uncertain"


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
    r = _run("complex", capacity_expansion=False, min_selection_probability=0.0, min_equivalent_set_probability=1.01,
             behavior_instability_threshold=1e9)
    sel = r["structure_selection"]
    assert sel["status"] == "tuning_uncertain" and sel["status_reason"] == "full_cv_selection_not_bootstrap_modal"
    fam = _run("complex", capacity_expansion=False, min_selection_probability=0.0, min_equivalent_set_probability=0.0,
               behavior_instability_threshold=1e9)["structure_selection"]
    # uma família equivalente estável não esconde a fragilidade do hiperparâmetro exato
    assert fam["status"] == "stable_equivalent_set" and fam["stable_exact"] is False and fam["full_cv_selection_fragile"] is True
    assert sel["selected_config_full_cv"] == sel["selected_label"] and sel["bootstrap_modal_config"] != sel["selected_label"]
    assert r["tuning_stable"] is False and sel["full_cv_selection_fragile"] is True


def test_stable_requires_modal_and_enough_probability():
    r = _run("simple", capacity_expansion=False, max_nodes_grid=(7,))
    sel = r["structure_selection"]
    if sel["selected_config_full_cv"] == sel["bootstrap_modal_config"] and sel["selection_probability"] >= sel["threshold"]:
        assert sel["status"] in {"stable_exact", "stable_equivalent_set"}
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


def test_equivalent_set_flags_when_it_covers_the_whole_grid():
    r = _run("simple", capacity_expansion=False, purity_epsilon_grid=(0.05, 0.01))
    sel = r["structure_selection"]
    n_valid = sum(1 for h in r["structure_history"] if h["stats"])
    assert sel["equivalent_set_covers_all_candidates"] == (sel["equivalent_candidate_count"] == n_valid)
