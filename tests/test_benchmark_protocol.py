"""Contratos do benchmark científico: métricas correctas, comparáveis, pareadas e reproduzíveis."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.benchmark import analysis as an
from core.benchmark import manifest as mf
from core.benchmark import report as rp
from core.benchmark import runner as rn
from core.benchmark import stats as st
from core.benchmark.io import analyze, save_result, verify_metrics_from_predictions
from core.benchmark.metrics import (
    OracleInfo, accuracy_real_labels, classification_bundle, evaluate_model, fidelity_to_oracle, format_fidelity, minority_class,
)
from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, Dataset, TreeBudget, default_arms
from core.benchmark.semantic import GroupSemanticProvider, NoSemanticProvider, OwlSemanticProvider, ShuffledSemanticProvider
from core.benchmark.splits import assert_disjoint, inner_folds, make_splits
from core.benchmark.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
FAST = BenchmarkConfig(seeds=(11, 22), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7, max_depth=4), extra_ablations=False,
                       n_boot=300, inner_cv_splits=2)


@pytest.fixture(scope="module")
def synth():
    ds, groups = make_synthetic("syn_bin", n_samples=140, n_features=6, n_classes=2, n_groups=2, seed=3)
    return ds, groups


@pytest.fixture(scope="module")
def result(synth):
    ds, groups = synth
    return BenchmarkRunner(FAST).run(ds, GroupSemanticProvider(groups))


# ------------------------------------------------------------------ métricas
def test_accuracy_and_fidelity_are_different_quantities():
    y = np.array([0, 0, 1, 1, 1, 0])
    oracle = np.array([0, 1, 1, 1, 0, 0])
    tree = np.array([0, 1, 1, 1, 0, 1])
    info = OracleInfo("MLP Original", "MLPClassifier", 4, "v1", 0.5)
    assert accuracy_real_labels(y, tree) == pytest.approx(np.mean(y == tree))
    assert fidelity_to_oracle(oracle, tree, info) == pytest.approx(np.mean(oracle == tree))
    assert accuracy_real_labels(y, tree) != fidelity_to_oracle(oracle, tree, info)


def test_fidelity_requires_explicit_oracle():
    with pytest.raises(TypeError):
        fidelity_to_oracle([0, 1], [0, 1], "MLP Original")
    with pytest.raises(ValueError):
        OracleInfo("  ", "x", 1)
    with pytest.raises(ValueError):
        evaluate_model(y_real=[0, 1], prediction=[0, 1], labels=[0, 1], oracle=OracleInfo("o", "t", 1))  # sem predições do oráculo
    s = format_fidelity(0.94, "TREPAN Reloaded", OracleInfo("MLP Ontológico", "t", 9))
    assert "MLP Ontológico" in s and "TREPAN Reloaded" in s


def test_evaluate_model_naming_separates_families():
    out = evaluate_model(y_real=[0, 1, 1, 0], prediction=[0, 1, 0, 0], labels=[0, 1], oracle=OracleInfo("O", "t", 2),
                         oracle_prediction=[0, 1, 1, 0], minority_label=1)
    assert out["accuracy_real_labels"] == 0.75 and out["fidelity_to_oracle"] == 0.75 and out["fidelity_oracle_name"] == "O"
    assert "minority_recall_real_labels" in out and not any(k == "fidelity" for k in out)
    no_oracle = evaluate_model(y_real=[0, 1], prediction=[0, 1], labels=[0, 1])
    assert no_oracle["fidelity_to_oracle"] is None and no_oracle["fidelity_oracle_name"] is None


def test_multiclass_metrics_and_label_order():
    y = np.array([0, 1, 2, 2, 1, 0, 2, 2])
    p = np.array([0, 1, 2, 1, 1, 0, 2, 0])
    b = classification_bundle(y, p, labels=[2, 1, 0])
    for key in ("precision_macro_real_labels", "recall_macro_real_labels", "macro_f1_real_labels", "balanced_accuracy_real_labels", "weighted_f1_real_labels"):
        assert 0 <= b[key] <= 1
    assert classification_bundle(y, p, labels=[0, 1, 2]) == b      # macro não depende da ordem; ordem é preservada para a matriz
    assert b["macro_f1_real_labels"] != b["accuracy_real_labels"]


def test_minority_class_from_training_not_assumed_class_one():
    y_train = np.array(["a"] * 7 + ["b"] * 2 + ["c"] * 5)
    assert minority_class(y_train, ["a", "b", "c"]) == "b"
    y2 = np.array([0] * 3 + [1] * 30)
    assert minority_class(y2, [0, 1]) == 0                           # a minoritária é a 0
    assert minority_class(np.array([0, 0, 1, 1]), [1, 0]) == 1       # empate: ordem das classes


# ------------------------------------------------------------------ estatística
def test_describe_ci_and_determinism():
    v = [0.8, 0.82, 0.85, 0.9, 0.88]
    d = st.describe(v)
    assert d["mean"] == pytest.approx(np.mean(v)) and d["std"] == pytest.approx(np.std(v, ddof=1)) and d["min"] == 0.8 and d["max"] == 0.9
    lo, hi = st.bootstrap_ci(v, n_boot=2000, seed=1)
    assert lo < d["mean"] < hi and (lo, hi) == st.bootstrap_ci(v, n_boot=2000, seed=1)
    assert np.isnan(st.bootstrap_ci([0.5])[0])


def test_paired_comparison_known_case_and_effect_sizes():
    a = np.array([0.90, 0.92, 0.94, 0.93, 0.95, 0.91, 0.96, 0.92, 0.94, 0.93])
    b = a - np.array([0.02, 0.03, 0.01, 0.02, 0.04, 0.02, 0.03, 0.01, 0.02, 0.03])
    c = st.paired_comparison(a, b, n_boot=2000)
    assert c["mean_diff"] > 0 and c["ci_low"] > 0 and c["wilcoxon_p"] < 0.01 and c["wins_a"] == 10 and c["rank_biserial"] == pytest.approx(1.0)
    assert c["cliffs_delta"] > 0.3 and c["cohens_dz"] > 1 and c["ttest_valid"] in (True, False)
    same = st.paired_comparison(a, a)
    assert same["wilcoxon_p"] == 1.0 and same["mean_diff"] == 0 and same["ties"] == 10
    with pytest.raises(ValueError):
        st.paired_comparison([1, 2], [1])


def test_multiple_comparison_corrections():
    p = [0.01, 0.04, 0.03, 0.2]
    assert st.holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    bh = st.benjamini_hochberg(p)
    assert bh[0] == pytest.approx(0.04) and all(x >= y for x, y in zip(bh, p))
    assert np.isnan(st.holm([np.nan, 0.01])[0])


def test_friedman_posthoc_and_evidence_levels():
    rng = np.random.default_rng(0)
    M = np.column_stack([rng.normal(0.9, 0.01, 8), rng.normal(0.85, 0.01, 8), rng.normal(0.8, 0.01, 8)])
    fr = st.friedman_posthoc(M, ["a", "b", "c"])
    assert fr["valid"] and fr["friedman_p"] < 0.05 and len(fr["posthoc"]) == 3
    assert not st.friedman_posthoc(M[:2], ["a", "b", "c"])["valid"]
    a = np.linspace(0.9, 0.95, 12); b = a - 0.03 - np.linspace(0, 0.01, 12)
    c = st.paired_comparison(a, b, n_boot=1000)
    assert st.evidence_level(c, p_adjusted=0.001, min_test_samples=300)[0] == st.SUPPORTED
    assert st.evidence_level(c, p_adjusted=0.001, min_test_samples=100)[0] == st.INDICATIVE         # teste pequeno
    assert st.evidence_level(c, p_adjusted=0.001, min_test_samples=300, needs_negative_control=True, negative_control_passed=False)[0] == st.INDICATIVE
    small = st.paired_comparison(a[:2], b[:2]); assert st.evidence_level(small, p_adjusted=0.01, min_test_samples=999)[0] == st.MECHANISM_ONLY


# ------------------------------------------------------------------ splits pareados
def test_splits_deterministic_stratified_disjoint():
    y = np.array([0] * 70 + [1] * 30)
    s1, s2 = make_splits(y, seeds=[1, 2, 3]), make_splits(y, seeds=[1, 2, 3])
    assert [s.split_hash for s in s1] == [s.split_hash for s in s2] and len({s.split_hash for s in s1}) == 3
    for s in s1:
        assert_disjoint(s)
        assert abs(np.mean(y[s.test_idx]) - 0.3) < 0.05
    rep = make_splits(y, scheme="repeated_cv", n_splits=5, n_repeats=3)
    assert len(rep) == 15 and all(abs(np.mean(y[s.test_idx]) - 0.3) < 0.08 for s in rep)
    with pytest.raises(ValueError):
        make_splits(y, scheme="bogus")


def test_inner_folds_only_use_training_indices():
    y_train = np.array([0, 1] * 20)
    for tr, va in inner_folds(y_train, n_splits=3, seed=0):
        assert max(tr.max(), va.max()) < len(y_train) and not set(tr) & set(va)


def test_all_arms_share_identical_splits(result):
    df = result.frame()
    for split_id, g in df.groupby("split_id"):
        assert g["split_hash"].nunique() == 1 and g["n_train"].nunique() == 1 and g["n_test"].nunique() == 1
    assert len(result.splits) == 2 and {s["split_hash"] for s in result.splits} == set(df["split_hash"])


# ------------------------------------------------------------------ runner / protocolo
def test_oracle_identification_and_label_usage(result):
    df = result.frame()
    trees = df[df["family"].isin(["trepan_original", "trepan_reloaded"])]
    assert trees["oracle_name"].notna().all() and trees["oracle_type"].notna().all() and trees["oracle_feature_space"].notna().all()
    assert trees["fidelity_to_oracle"].notna().all() and not trees["uses_real_labels_for_training"].any()
    non_surrogates = df[df["family"].isin(["mlp", "c45"])]
    assert non_surrogates["fidelity_to_oracle"].isna().all() and non_surrogates["oracle_name"].isna().all()
    c45 = df[df["family"] == "c45"]
    assert c45["uses_real_labels_for_training"].all() and not c45["uses_oracle"].any()
    assert set(trees[trees["arm"] != "reloaded_e2e"]["oracle_name"]) == {rn.ORACLE_ORIGINAL}


def test_original_vs_reloaded_same_oracle_budget_split_seed(result):
    df = result.frame()
    for _, g in df[df["family"].isin(["trepan_original", "trepan_reloaded"])].groupby("split_id"):
        main = g[g["arm"].isin(["trepan_original", "reloaded_lambda0", "reloaded_owl_full", "reloaded_owl_shuffled"])]
        assert main["query_budget"].nunique() == 1 and main["oracle_name"].nunique() == 1 and main["seed"].nunique() == 1
        assert main["split_hash"].nunique() == 1


def test_reloaded_lambda0_matches_original_exactly(result):
    r = result.frame()
    l0 = r[r["arm"] == "reloaded_lambda0"]
    assert l0["identical_predictions_to_trepan_original"].all() and (l0["semantic_split_count"] == 0).all()
    for sid, g in r.groupby("split_id"):
        o, z = g[g["arm"] == "trepan_original"].iloc[0], g[g["arm"] == "reloaded_lambda0"].iloc[0]
        assert o["fidelity_to_oracle"] == z["fidelity_to_oracle"] and o["node_count"] == z["node_count"]


def test_predictions_are_stored_and_metrics_recomputable(result, tmp_path):
    run = save_result(result, analyze(result), tmp_path)
    ver = verify_metrics_from_predictions(run)
    assert ver["ok"] and ver["checked_values"] > 50 and ver["max_abs_diff"] < 1e-9
    pred = pd.read_csv(run / "predictions.csv")
    assert {"y_real", "pred__mlp_original", "pred__c45", "pred__trepan_original", "pred__reloaded_owl_full"} <= set(pred.columns)
    assert any(c.startswith("oraclepred__") for c in pred.columns)
    raw = pd.read_csv(run / "raw_results.csv")
    raw.loc[raw["arm"] == "trepan_original", "fidelity_to_oracle"] += 0.05           # adulteração tem de ser detectada
    raw.to_csv(run / "raw_results.csv", index=False)
    assert not verify_metrics_from_predictions(run)["ok"]


def test_deterministic_seeds_reproduce_metrics(synth):
    ds, groups = synth
    cfg = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7), extra_ablations=False, n_boot=100, inner_cv_splits=2)
    arms = [a for a in default_arms(cfg) if a.arm_id in ("mlp_original", "c45", "trepan_original", "reloaded_owl_full")]
    r1 = BenchmarkRunner(cfg, arms).run(ds, GroupSemanticProvider(groups)).frame()
    r2 = BenchmarkRunner(cfg, arms).run(ds, GroupSemanticProvider(groups)).frame()
    cols = ["arm", "accuracy_real_labels", "fidelity_to_oracle", "node_count", "membership_queries"]
    pd.testing.assert_frame_equal(r1[cols].reset_index(drop=True), r2[cols].reset_index(drop=True))


def test_no_ontology_runs_without_crash(synth):
    ds, _ = synth
    cfg = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7), extra_ablations=False, n_boot=100, inner_cv_splits=2)
    r = BenchmarkRunner(cfg).run(ds, None)
    arms = set(r.frame()["arm"])
    assert {"mlp_original", "c45", "trepan_original", "reloaded_lambda0"} <= arms
    assert not arms & {"mlp_ontological", "reloaded_owl_full", "reloaded_owl_shuffled", "reloaded_e2e"}
    assert all(s["semantic_available"] is False for s in r.semantic_report) and any("semantic_available=false" in s["reason"] for s in r.skipped)
    a = analyze(r)
    assert len(a["aggregate"]) and "NOT_RUN" in {x["negative_control"] for x in a["attribution"]} or a["attribution"] == []


def test_invalid_owl_does_not_discard_experiment(synth, tmp_path):
    ds, _ = synth
    ctx = OwlSemanticProvider(str(tmp_path / "missing.owl")).build(ds.X, ds.feature_names, 0)
    assert ctx.available is False and ctx.ontology_valid is False
    cfg = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7), extra_ablations=False, n_boot=100, inner_cv_splits=2)
    r = BenchmarkRunner(cfg).run(ds, str(tmp_path / "missing.owl"))
    assert "trepan_original" in set(r.frame()["arm"]) and r.semantic_report[0]["ontology_valid"] is False
    assert r.semantic_report[0]["mlp_enrichment_accepted"] is False and r.semantic_report[0]["trepan_semantics_available"] is False


def test_multiclass_dataset_runs():
    ds, groups = make_synthetic("syn_3c", n_samples=150, n_features=6, n_classes=3, n_groups=2, seed=5)
    cfg = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7), extra_ablations=False, n_boot=100, inner_cv_splits=2)
    arms = [a for a in default_arms(cfg) if a.arm_id in ("mlp_original", "c45", "trepan_original", "reloaded_owl_full")]
    df = BenchmarkRunner(cfg, arms).run(ds, GroupSemanticProvider(groups)).frame()
    assert df["macro_f1_real_labels"].between(0, 1).all() and df["balanced_accuracy_real_labels"].between(0, 1).all()
    assert (df["accuracy_real_labels"] != df["macro_f1_real_labels"]).any()


# ------------------------------------------------------------------ anti-leakage
def test_selection_functions_cannot_receive_test_data():
    assert list(inspect.signature(rn.select_oracle).parameters) == ["X_train", "X_train_enriched", "y_train", "seed", "margin", "inner_splits"]
    assert "X_test" not in inspect.signature(rn.fit_mlp).parameters


def test_runner_only_feeds_training_rows_to_selection(synth, monkeypatch):
    ds, groups = synth
    seen = []
    orig = rn.select_oracle

    def spy(X_train, X_train_enriched, y_train, **kw):
        seen.append(len(X_train))
        return orig(X_train, X_train_enriched, y_train, **kw)

    monkeypatch.setattr(rn, "select_oracle", spy)
    cfg = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7), extra_ablations=False, n_boot=100, inner_cv_splits=2)
    r = BenchmarkRunner(cfg, [a for a in default_arms(cfg) if a.arm_id in ("mlp_original", "trepan_original")]).run(ds, GroupSemanticProvider(groups))
    assert seen == [r.splits[0]["n_train"]]
    audit = r.semantic_report[0]["protocol_audit"]
    assert audit["valid"] and not audit["test_used_for_selection"] and audit["final_test_evaluated"]


def test_config_is_frozen_and_seeds_declared_before_results():
    with pytest.raises(Exception):
        BenchmarkConfig().seeds = (1,)
    assert tuple(BenchmarkConfig().seeds) == (11, 22, 33, 44, 55, 66, 77, 88, 99, 111)


def test_mlp_original_and_ontological_get_same_tuning_budget(synth):
    ds, groups = synth
    cfg = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=5), extra_ablations=False, n_boot=100, inner_cv_splits=2, mlp_trials=2)
    arms = [a for a in default_arms(cfg) if a.arm_id in ("mlp_original", "mlp_ontological")]
    df = BenchmarkRunner(cfg, arms).run(ds, GroupSemanticProvider(groups)).frame()
    assert set(df["mlp_trials"]) == {2} and set(df["mlp_selection"]) == {"random_search_inner_cv"}


# ------------------------------------------------------------------ controlo negativo
def test_shuffled_control_preserves_shape_but_destroys_correspondence(synth):
    ds, groups = synth
    real = GroupSemanticProvider(groups).build(ds.X, ds.feature_names, 3)
    shuf = ShuffledSemanticProvider(GroupSemanticProvider(groups)).build(ds.X, ds.feature_names, 3)
    assert shuf.source == "shuffled" and real.n_features == shuf.n_features and len(real.onto_idx) == len(shuf.onto_idx)
    assert sorted(real.weights) == sorted(shuf.weights) and sorted(map(str, real.groups)) == sorted(map(str, shuf.groups))
    assert np.allclose(np.sort(np.linalg.eigvalsh(real.relatedness)), np.sort(np.linalg.eigvalsh(shuf.relatedness)))
    assert real.structure_signature() != shuf.structure_signature()
    a, b = real.enrich(ds.X), shuf.enrich(ds.X)
    assert a.shape == b.shape and np.allclose(a[:, :6], b[:, :6]) and not np.allclose(a[:, 6:], b[:, 6:])
    shuf2 = ShuffledSemanticProvider(GroupSemanticProvider(groups)).build(ds.X, ds.feature_names, 3)
    assert shuf.structure_signature() == shuf2.structure_signature()                  # determinístico por seed
    assert NoSemanticProvider().build(ds.X, ds.feature_names).available is False


def _fake_frame(delta, n=10, seed=0, arm_b="reloaded_owl_shuffled"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = 0.80 + rng.normal(0, 0.02)
        for arm, val in (("reloaded_owl_full", base + delta + rng.normal(0, 0.003)), (arm_b, base)):
            rows.append(dict(dataset="d", split_id=f"s{i}", arm=arm, oracle_name="MLP Original", n_train=100, n_test=300,
                             fidelity_to_oracle=val, accuracy_real_labels=val, node_count=9.0))
    return pd.DataFrame(rows)


def test_negative_control_verdicts():
    res = an.run_contrasts(_fake_frame(0.05), n_boot=500)
    v = an.negative_control_verdicts(res)[("d", "fidelity_to_oracle")]
    assert v["verdict"] == "REAL_GT_SHUFFLED" and v["passed"]
    res0 = an.run_contrasts(_fake_frame(0.0, seed=2), n_boot=500)
    v0 = an.negative_control_verdicts(res0)[("d", "fidelity_to_oracle")]
    assert v0["verdict"] in ("REAL_APPROX_SHUFFLED", "SHUFFLED_GT_REAL", "REAL_GT_SHUFFLED") and v0["verdict"] != "INSUFFICIENT_DATA"
    few = an.run_contrasts(_fake_frame(0.05, n=3), n_boot=200)
    assert an.negative_control_verdicts(few)[("d", "fidelity_to_oracle")]["verdict"] == "INSUFFICIENT_DATA"
    att = an.semantic_attribution(res0)[0]
    assert "NÃO atribuível" in att["attribution"] or "OWL permutada" in att["attribution"] or "atribuível" in att["attribution"]


def test_fidelity_not_compared_across_different_oracles():
    rows = []
    for i in range(6):
        rows.append(dict(dataset="d", split_id=f"s{i}", arm="trepan_original", oracle_name=rn.ORACLE_ORIGINAL, n_train=90, n_test=40,
                         fidelity_to_oracle=0.8, accuracy_real_labels=0.8, node_count=5.0))
        rows.append(dict(dataset="d", split_id=f"s{i}", arm="reloaded_e2e", oracle_name=rn.ORACLE_ONTOLOGICAL, n_train=90, n_test=40,
                         fidelity_to_oracle=0.9, accuracy_real_labels=0.85 + 0.01 * i, node_count=5.0))
    res = an.run_contrasts(pd.DataFrame(rows), n_boot=200)
    fid = res[(res["contrast"] == "end_to_end_vs_original") & (res["metric"] == "fidelity_to_oracle")].iloc[0]
    acc = res[(res["contrast"] == "end_to_end_vs_original") & (res["metric"] == "accuracy_real_labels")].iloc[0]
    assert fid["status"] == "not_comparable" and "oráculos diferentes" in fid["reason"] and acc["status"] == "ok"


def test_evidence_capped_for_small_tests_and_repeated_cv():
    df = _fake_frame(0.05, n=12)
    df["n_test"] = 40
    res = an.run_contrasts(df, n_boot=300)
    ok = res[(res["status"] == "ok") & (res["metric"] == "fidelity_to_oracle")]
    assert set(ok["evidence_level"]) <= {st.INDICATIVE, st.MECHANISM_ONLY}
    big = _fake_frame(0.05, n=12); big["n_test"] = 400
    rc = an.run_contrasts(big, scheme="repeated_cv(5x2)", n_boot=300)
    prim = rc[(rc["status"] == "ok") & rc["metric"].isin(an.PRIMARY_METRICS)]
    assert "nadeau_bengio_p" in rc.columns and set(prim["evidence_level"].dropna()) <= {st.INDICATIVE}
    ident = rc[(rc["status"] == "ok") & (rc["metric"] == "node_count")]
    assert set(ident["evidence_level"]) == {"IDENTICAL_RESULTS"}                       # nó constante: diferenças todas 0


def test_aggregation_keeps_raw_and_reports_mean_std_ci(result):
    df = result.frame()
    agg = an.aggregate(df, n_boot=200)
    assert {"mean", "std", "median", "min", "max", "ci_low", "ci_high", "n"} <= set(agg.columns)
    row = agg[(agg["arm"] == "trepan_original") & (agg["metric"] == "fidelity_to_oracle")].iloc[0]
    vals = df[df["arm"] == "trepan_original"]["fidelity_to_oracle"]
    assert row["mean"] == pytest.approx(vals.mean()) and row["n"] == len(vals)
    # invariante robusta ao ambiente: a agregação nunca inventa linhas (no máx. uma por linha bruta x métrica agregada) e a tabela bruta é
    # preservada (antes comparava-se o nº de linhas brutas com o nº de métricas disponíveis, que depende do ambiente: ex. tempos do reasoner)
    assert len(agg) <= len(df) * len(an.AGG_METRICS) and len(df) == df.groupby(["arm", "oracle_name"], dropna=False).size().sum()


# ------------------------------------------------------------------ manifest / persistência / relatórios
def test_manifest_and_no_silent_overwrite(result, tmp_path):
    a = analyze(result)
    d1 = save_result(result, a, tmp_path); d2 = save_result(result, a, tmp_path)
    assert d1 != d2 and d1.exists() and d2.exists()
    for f in ("manifest.json", "config.json", "metrics.json", "tree_metrics.json", "semantic_report.json", "predictions.csv", "summary.md", "raw_results.csv", "aggregate.csv", "contrasts.csv"):
        assert (d1 / f).exists(), f
    m = json.loads((d1 / "manifest.json").read_text(encoding="utf-8"))
    for k in ("experiment_id", "timestamp", "build_version", "git_commit", "dataset_hash", "ontology_hash", "config_hash", "seeds", "split_hash",
              "library_versions", "python_version", "platform"):
        assert k in m, k
    assert m["dataset_hash"] == result.dataset_hash and m["seeds"] == [11, 22]
    assert set(m["library_versions"]) >= {"numpy", "scipy", "pandas", "sklearn"}


def test_hashes_change_with_inputs(synth):
    ds, _ = synth
    h = mf.hash_dataset(ds.X, ds.y, ds.feature_names)
    assert h == mf.hash_dataset(ds.X.copy(), ds.y.copy(), ds.feature_names) and h != mf.hash_dataset(ds.X + 1e-9, ds.y, ds.feature_names)
    assert mf.stable_hash({"a": 1}) != mf.stable_hash({"a": 2})


def test_reports_contain_required_sections(result):
    a = analyze(result)
    entries = [dict(result=result, analysis=a, run_dir="x")]
    sci = rp.scientific_validation_markdown(entries, verification={"checked_values": 10, "max_abs_diff": 0.0, "ok": True})
    for needle in ("## 1. Protocolo", "Negative / Null Results", "Limitações", "STATISTICALLY_SUPPORTED", "Wilcoxon", "Holm", "bootstrap", "fidelity_to_oracle", "MLP Original"):
        assert needle in sci, needle
    assert "Fidelity" in sci and "rótulos reais" in sci
    nc = rp.negative_control_markdown(entries); ab = rp.ablation_markdown(entries)
    assert "NEGATIVE CONTROL" in nc and "ABLATION REPORT" in ab and "sem_reasoner" in ab and "reloaded_owl_shuffled" in ab


def test_negative_results_are_reported_when_semantics_missing(synth):
    ds, _ = synth
    cfg = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=5), extra_ablations=False, n_boot=100, inner_cv_splits=2)
    r = BenchmarkRunner(cfg, [a for a in default_arms(cfg) if a.arm_id in ("mlp_original", "trepan_original")]).run(ds, None)
    items = rp.null_results([dict(result=r, analysis=analyze(r))])
    assert any("semântica indisponível" in i for i in items)


# ------------------------------------------------------------------ agnosticismo
def test_benchmark_core_is_dataset_agnostic():
    pattern = re.compile(r"\b(iris|wine|wdbc|sonar|german|hepatitis|adult|breast|cancer|digits|diabetes|titanic|mnist)\b", re.I)
    bad = []
    for p in (ROOT / "core" / "benchmark").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        text = re.sub(r"#.*", "", text)
        if pattern.search(text):
            bad.append(p.name)
    assert not bad, bad


def test_arff_class_order_is_preserved(tmp_path):
    p = tmp_path / "t.arff"
    p.write_text("@relation t\n@attribute a numeric\n@attribute b numeric\n@attribute cls {zeta,alpha}\n@data\n1,2,alpha\n3,4,zeta\n5,6,alpha\n")
    ds = Dataset.from_file(str(p))
    assert ds.y.tolist() == [1, 0, 1] and ds.feature_names == ["a", "b"]          # zeta=0, alpha=1 (ordem declarada)


def test_semantic_split_count_refers_to_final_tree_only(result):
    df = result.frame()
    trees = df[df["family"] == "trepan_reloaded"]
    assert (trees["semantic_split_count"] <= trees["internal_nodes"]).all()
    assert (trees["semantic_decision_changed_count"] <= trees["semantic_split_count"]).all()
    assert (df[df["arm"].isin(["trepan_original", "reloaded_lambda0"])]["semantic_split_count"] == 0).all()
