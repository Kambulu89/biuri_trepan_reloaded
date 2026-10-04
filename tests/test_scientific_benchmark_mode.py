"""Modo SCIENTIFIC / BENCHMARK da GUI, block bootstrap da seleção, saturação em max_nodes e isolamento do teste."""
import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import core.scientific_benchmark_service as svc
import core.trepan_scientific_tuning as tuning_mod
from core.execution_mode import DEFAULT_MODE, ExecutionMode, parse_mode
from core.experiment_builders import scientific_diagnostics_from_tuning
from core.trepan_scientific_tuning import (ScientificTrepanSearchConfig, _block_bootstrap, _lexicographic_select,
                                           _split_stats, tune_scientific_trepan)
from core.controlled_trepan_experiment import ControlledTrepanConfig

ROOT = Path(__file__).resolve().parents[1]
FAST = dict(cv_folds=2, cv_repeats=3, max_capacity_candidates=1, max_semantic_candidates=1,
            purity_epsilon_grid=(0.05, 0.01), max_nodes_grid=(7, 15), bootstrap_resamples=40, capacity_expansion=False)


def _fast_search(**kw):
    return svc.scientific_search_config(**{**FAST, **kw})


def _frame(n=160, seed=3):
    import pandas as pd
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["y"] = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, "pos", "neg")
    return df


# ------------------------------------------------------------------------------- modos distintos
def test_interactive_and_benchmark_modes_are_explicitly_distinct():
    assert ExecutionMode.INTERACTIVE_EXPLORATORY != ExecutionMode.SCIENTIFIC_BENCHMARK
    assert DEFAULT_MODE is ExecutionMode.INTERACTIVE_EXPLORATORY
    assert ExecutionMode.SCIENTIFIC_BENCHMARK.usable_as_benchmark and not ExecutionMode.INTERACTIVE_EXPLORATORY.usable_as_benchmark
    assert ExecutionMode.SCIENTIFIC_BENCHMARK.label == "SCIENTIFIC / BENCHMARK"
    assert ExecutionMode.INTERACTIVE_EXPLORATORY.label == "INTERACTIVE / EXPLORATORY"
    assert parse_mode("SCIENTIFIC_BENCHMARK") is ExecutionMode.SCIENTIFIC_BENCHMARK
    with pytest.raises(ValueError):
        parse_mode("qualquer")


def test_main_scientific_configuration_is_cv_5x3():
    s = svc.scientific_search_config()
    assert (s.cv_repeats, s.cv_folds) == (5, 3) == (svc.SCIENTIFIC_CV_REPEATS, svc.SCIENTIFIC_CV_FOLDS)
    assert ScientificTrepanSearchConfig().cv_repeats == 5


# ------------------------------------------------------------------------------- serviço científico + contrato
@pytest.fixture(scope="module")
def outcome():
    return svc.run_scientific_benchmark(_frame(), target="y", seed=3, search=_fast_search())


def test_benchmark_outcome_proves_one_oracle_for_original_and_reloaded(outcome):
    ev = outcome.report["evaluation"]
    oc = ev["oracle_contract"]
    ids = oc["tree_oracle_ids"]
    assert ids["trepan_original"] == ids["trepan_reloaded"] == outcome.oracle_id == oc["oracle"]["oracle_id"]
    assert oc["unchanged_after"] and oc["single_oracle_for_all_trees"] and {"tuning", "trepan_pair"} <= set(oc["trees"])
    sci = outcome.result.scientific
    assert sci.benchmark_eligible and sci.same_oracle_original_reloaded is True and sci.oracle_id == outcome.oracle_id
    assert outcome.usable_as_benchmark and outcome.result.provenance.execution_mode == "SCIENTIFIC_BENCHMARK"


def test_the_contract_check_rejects_diverging_oracles():
    from core.scientific_experiment_contract import OracleContractViolation
    bad = {"evaluation": {"oracle_contract": {"oracle": {"oracle_id": "aaa"}, "unchanged_after": True,
                                              "single_oracle_for_all_trees": True,
                                              "tree_oracle_ids": {"trepan_original": "aaa", "trepan_reloaded": "bbb"}}}}
    with pytest.raises(OracleContractViolation):
        svc.verify_oracle_contract(bad)
    with pytest.raises(OracleContractViolation):
        svc.verify_oracle_contract({"evaluation": {}})


def test_benchmark_service_delegates_to_the_production_pipeline_and_the_test_never_enters_tuning(monkeypatch):
    import core.production_training as pt
    seen = {}
    real_tune = pt.tune_scientific_trepan
    real_train = svc.train_production_dataframe

    def spy_tune(X, y, **kw):
        seen["tune_rows"] = len(X)
        seen["tune_params"] = set(kw)
        return real_tune(X, y, **kw)

    def spy_train(df, **kw):
        seen["train_called"] = True
        return real_train(df, **kw)

    monkeypatch.setattr(pt, "tune_scientific_trepan", spy_tune)
    monkeypatch.setattr(svc, "train_production_dataframe", spy_train)
    out = svc.run_scientific_benchmark(_frame(), target="y", seed=3, search=_fast_search())
    man = out.report["manifest"]
    assert seen["train_called"] is True                                       # o MESMO pipeline da produção
    assert seen["tune_rows"] == man["train_rows"] < man["train_rows"] + man["test_rows"]   # o tuning só viu o treino
    assert not ({"X_test", "y_test", "test"} & seen["tune_params"])
    tuning = out.report["evaluation"]["trepan_scientific_tuning"]
    assert tuning["test_used_for_selection"] is False and tuning["cv_plan"]["test_set_used"] is False
    assert out.result.scientific.test_used_for_selection is False
    train_idx = set(out.report["split"]["train_indices"]); test_idx = set(out.report["split"]["test_indices"])
    assert not (train_idx & test_idx)


# ------------------------------------------------------------------------------- GUI
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def arff(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "small.arff"
    lines = ["@relation small", "@attribute a numeric", "@attribute b numeric", "@attribute c numeric",
             "@attribute class {neg,pos}", "@data"]
    for _ in range(120):
        a, b, c = rng.normal(size=3)
        lines.append(f"{a:.3f},{b:.3f},{c:.3f},{'pos' if a + 0.5 * b > 0 else 'neg'}")
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def window(qapp):
    from gui.biuri_app_complete import BiuriApp
    w = BiuriApp()
    yield w
    w.close()


def test_gui_has_two_explicit_modes_and_defaults_to_interactive(window):
    combo = window.execution_mode_combo
    data = [combo.itemData(i) for i in range(combo.count())]
    assert data == ["INTERACTIVE_EXPLORATORY", "SCIENTIFIC_BENCHMARK"]
    assert [combo.itemText(i) for i in range(combo.count())] == ["INTERACTIVE / EXPLORATORY", "SCIENTIFIC / BENCHMARK"]
    assert window._get_execution_mode() is ExecutionMode.INTERACTIVE_EXPLORATORY and combo.toolTip()


def test_gui_benchmark_mode_calls_the_shared_service_and_publishes_the_diagnostics(window, arff, monkeypatch):
    calls = {}
    real = svc.run_scientific_benchmark

    def spy(df, **kw):
        calls["rows"] = len(df)
        calls["kw"] = kw
        return real(df, **{**kw, "search": _fast_search()})

    monkeypatch.setattr(svc, "run_scientific_benchmark", spy)
    window.load_data_from_file(arff)
    window.execution_mode_combo.setCurrentIndex(window.execution_mode_combo.findData("SCIENTIFIC_BENCHMARK"))
    assert window._get_execution_mode() is ExecutionMode.SCIENTIFIC_BENCHMARK
    outcome = window._execute_benchmark_pipeline()
    assert calls["rows"] == 120 and outcome.usable_as_benchmark                 # reutiliza o serviço partilhado
    class _Dlg:
        def close(self):
            pass
    window._on_benchmark_finished({"outcome": outcome}, _Dlg())
    res = window.audit.result
    assert res is outcome.result and res.provenance.execution_mode == "SCIENTIFIC_BENCHMARK"
    from gui import result_presenter as rp
    rows = dict(rp.scientific_rows(res))
    from gui.strings import tr
    for key in ("sci.oracle_id", "sci.seed", "sci.cv_plan", "sci.selected", "sci.fidelity", "sci.predictive_stability",
                "sci.structural_stability", "sci.selection_probability", "sci.node_cap", "sci.queries", "sci.budget_exhausted",
                "sci.test_used", "sci.tuning_status"):
        assert tr(key) in rows, key
    assert rows[tr("sci.oracle_id")] == outcome.oracle_id[:8]
    assert rows[tr("sci.tuning_status")].split(" ")[0] in {"stable_exact", "stable_equivalent_set", "tuning_uncertain"}
    assert not [m for m in res.messages if m.code == "exploratory_mode"]
    # os visualizadores recebem as MESMAS árvores avaliadas (identidade de objeto), sem retreino
    assert window.trepan_original_tree is outcome.artifacts["trepan_original"]
    assert window.trepan_reloaded_tree is outcome.artifacts["trepan_reloaded"]
    assert window.benchmark_view["oracle_id"] == outcome.oracle_id == outcome.artifacts["oracle_id"]
    window.visualize_tree()
    w = window.tree_widget
    assert w.trepan_original_tree is outcome.artifacts["trepan_original"] and w.trepan_reloaded_tree is outcome.artifacts["trepan_reloaded"]
    assert w.feature_names == outcome.artifacts["feature_names_original"] and outcome.oracle_id[:8] in w.dataset_name
    assert outcome.artifacts["trepan_original"].node_count_ == outcome.result.trees["trepan_original"].logical_nodes.value
    cfg = outcome.artifacts["selected_config"]
    assert outcome.report["evaluation"]["trepan_scientific_tuning"]["selected_config"]["max_nodes"] == cfg.max_nodes
    assert "max_nodes" in w.dataset_name and str(cfg.max_nodes) in w.dataset_name
    # o modo interativo nunca herda as árvores do benchmark
    window._clear_benchmark_view()
    assert window.benchmark_view is None and window.benchmark_outcome is None


def test_gui_interactive_result_is_marked_exploratory_and_not_a_benchmark(window, arff):
    from gui.messages import derive_messages
    from gui.result_builder import build_experiment_result
    from gui import result_presenter as rp
    window.load_data_from_file(arff)
    r = build_experiment_result(window)
    assert r.provenance.execution_mode == "INTERACTIVE_EXPLORATORY"
    assert r.scientific is not None and r.scientific.benchmark_eligible is False and r.scientific.oracle_id is None
    assert [m for m in derive_messages(r) if m.code == "exploratory_mode"]
    assert "INTERACTIVE" in dict(rp.experiment_rows(r))["Modo de execução"]
    assert "não utilizar como benchmark" in dict(rp.experiment_rows(r))["Modo de execução"]


def test_gui_benchmark_branch_does_not_reimplement_the_pipeline():
    src = (ROOT / "gui" / "biuri_app_complete.py").read_text()
    body = src[src.index("def _execute_benchmark_pipeline"):src.index("def _launch_benchmark_worker")]
    assert "run_scientific_benchmark" in body and "frame_from_arrays" in body
    for forbidden in ("train_production_dataframe", "tune_scientific_trepan", "MLPTrainer", "train_robust_mlp", "fit_controlled_trepan_pair"):
        assert forbidden not in body


# ------------------------------------------------------------------------------- candidatos diferentes por dataset
def test_different_candidates_can_win_on_different_datasets():
    search = ScientificTrepanSearchConfig()
    rng = np.random.default_rng(0)
    J = 15
    meta = [{"config": {"purity_epsilon": 0.05, "max_nodes": 31}, "max_nodes": 31},
            {"config": {"purity_epsilon": 0.01, "max_nodes": 63}, "max_nodes": 63}]
    noise = 0.004 * rng.standard_normal(J)
    # dataset A: o candidato 0 é significativamente melhor; dataset B: o candidato 1
    FA = np.array([0.95 + noise, 0.80 + noise]); FB = FA[::-1].copy()
    N = np.array([np.full(J, 20.0), np.full(J, 50.0)])
    wa = _lexicographic_select(FA, N, meta, 3, search, [True, True], N, N)[0]
    wb = _lexicographic_select(FB, N, meta, 3, search, [True, True], N, N)[0]
    assert (wa, wb) == (0, 1)                                                # sem preferência fixa por tamanho


def test_no_dataset_specific_checks_in_the_scientific_modules():
    banned = re.compile(r"iris|breast|cancer|diagnosis|wine|titanic|mnist|heart|diabetes", re.I)
    for rel in ("core/trepan_scientific_tuning.py", "core/scientific_experiment_contract.py", "core/scientific_benchmark_service.py",
                "core/execution_mode.py"):
        text = (ROOT / rel).read_text()
        assert not banned.search(text), rel
    body = (ROOT / "core" / "experiment_builders.py").read_text()
    seg = body[body.index("def scientific_diagnostics_from_tuning"):body.index("def from_production_report") if "def from_production_report" in body else None]
    assert not banned.search(seg)


# ------------------------------------------------------------------------------- saturação em max_nodes
def _rows(n, at_cap, nodes):
    return [{"repeat": i // 3, "fidelity": 0.9, "nodes": nodes, "depth": 5, "leaves": nodes // 2 + 1, "time_s": 0.1,
             "queries_used": 100, "budget_exhausted": False, "balanced_accuracy": 0.9, "macro_f1": 0.9,
             "max_nodes_reached": i < at_cap, "root_features": [0], "features": [0], "splits": [[0, 1]]} for i in range(n)]


def test_node_cap_saturation_is_diagnosed_and_does_not_change_the_selection():
    search = ScientificTrepanSearchConfig()
    cfg = ControlledTrepanConfig(max_nodes=31)
    capped = _split_stats(_rows(15, 14, 31), cfg, 5, ["a"], search)
    free = _split_stats(_rows(15, 1, 25), cfg, 5, ["a"], search)
    assert capped["fraction_at_node_cap"] == pytest.approx(14 / 15) and capped["node_cap_reached_count"] == 14
    assert capped["max_nodes"] == 31 and capped["structural_stability_censored"] is True
    assert free["structural_stability_censored"] is False
    F = np.array([[0.9, 0.91, 0.89] * 5] * 2); N = np.array([np.full(15, 31.0), np.full(15, 25.0)])
    meta_off = [{"config": {"purity_epsilon": 0.05, "max_nodes": 31}, "max_nodes": 31, "censored": False}] * 2
    meta_on = [{**meta_off[0], "censored": True}, meta_off[1]]
    w0, v0, _ = _lexicographic_select(F, N, meta_off, 3, search, [True, True], N, N)
    w1, v1, _ = _lexicographic_select(F, N, meta_on, 3, search, [True, True], N, N)
    assert w0 == w1                                                         # o rótulo de censura não altera a escolha
    assert v1[0]["structural_stability_censored"] is True and "CENSURADA" in v1[0]["text"]
    assert "CENSURADA" not in v0[0]["text"] and v1[1]["structural_stability_censored"] is False


def test_tuning_history_records_node_cap_fields():
    X = np.random.default_rng(4).normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)

    class O:
        classes_ = np.array([0, 1])

        def predict(self, A):
            return (np.asarray(A)[:, 0] > 0).astype(int)
    base = ControlledTrepanConfig(max_nodes=15, max_depth=15, min_samples_leaf=2, min_sample=100, max_n=2, beam_width=1,
                                  max_features_per_node=3, max_queries=3000, random_state=1)
    r = tune_scientific_trepan(X, y, oracle=O(), feature_names=list("abc"), base_config=base,
                               search=ScientificTrepanSearchConfig(**FAST))
    for h in r["structure_history"]:
        s = h["stats"]
        assert {"fraction_at_node_cap", "node_cap_reached_count", "max_nodes", "structural_stability_censored"} <= set(s)
        assert "structural_stability_censored" in h["verdict"] and 0.0 <= s["fraction_at_node_cap"] <= 1.0
    assert {"max_nodes", "fraction_at_node_cap", "structural_stability_censored"} <= set(r["structure_selection"]["node_cap"])


def test_scientific_diagnostics_map_uncertainty_and_missing_tuning():
    d = scientific_diagnostics_from_tuning(None, None, seed=7, execution_mode="INTERACTIVE_EXPLORATORY")
    assert d.tuning_status == "not_run" and d.benchmark_eligible is False and not d.fidelity_mean.available
    f = scientific_diagnostics_from_tuning({"failed": "boom"}, None, seed=7, execution_mode="SCIENTIFIC_BENCHMARK")
    assert f.tuning_status == "failed" and f.benchmark_eligible is False
