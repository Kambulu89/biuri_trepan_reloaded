"""Protocolo comum do benchmark, passo 1: um oráculo CONGELADO por split, com oracle_id, partilhado por todas as árvores."""
import numpy as np
import pandas as pd
import pytest

from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, TreeBudget
from core.benchmark.semantic import GroupSemanticProvider
from core.benchmark.synthetic import make_synthetic
from core.scientific_experiment_contract import OracleContractViolation, freeze_oracle

FAST = BenchmarkConfig(seeds=(11, 22), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7, max_depth=4), extra_ablations=False,
                       n_boot=100, inner_cv_splits=2)


@pytest.fixture(scope="module")
def result():
    ds, groups = make_synthetic("syn_cc", n_samples=140, n_features=6, n_classes=2, n_groups=2, seed=3)
    return BenchmarkRunner(FAST).run(ds, GroupSemanticProvider(groups))


def test_every_tree_row_records_the_frozen_oracle_it_consulted(result):
    df = result.frame()
    trees = df[df["uses_oracle"] == True]       # noqa: E712
    assert len(trees) > 0 and trees["oracle_id"].notna().all() and (trees["oracle_queries"] > 0).all()
    assert set(trees["oracle_scope"]) <= {"shared_frozen_oracle", "ontological_oracle_separate"}
    c45 = df[df["family"] == "c45"]
    assert (c45["oracle_scope"] == "no_oracle_real_labels").all() and c45["oracle_id"].isna().all()


def test_all_shared_oracle_trees_of_a_split_use_exactly_the_same_oracle_id_and_the_mlp_itself(result):
    df = result.frame()
    for split_id, g in df.groupby("split_id"):
        shared = g[g["oracle_scope"] == "shared_frozen_oracle"]
        mlp = g[g["arm"] == "mlp_original"].iloc[0]
        assert shared["oracle_id"].nunique() == 1 and shared["oracle_id"].iloc[0] == mlp["oracle_id"]
        assert f"oracle_id={mlp['oracle_id']}" in shared["oracle_version"].iloc[0]
    ids = df[df["arm"] == "mlp_original"]["oracle_id"]
    assert ids.nunique() == df["split_id"].nunique()                  # splits/seeds diferentes -> oráculos diferentes


def test_the_ontological_oracle_is_a_separate_object_and_clearly_identified(result):
    df = result.frame()
    e2e = df[df["arm"] == "reloaded_e2e"]
    assert len(e2e) > 0
    for _, r in e2e.iterrows():
        shared = df[(df["split_id"] == r["split_id"]) & (df["arm"] == "trepan_original")].iloc[0]
        if r["oracle_name"] == shared["oracle_name"]:          # o gate rejeitou o MLP ontológico: usa o MESMO oráculo congelado
            assert r["oracle_scope"] == "shared_frozen_oracle" and r["oracle_id"] == shared["oracle_id"]
        else:                                                  # gate aceitou: outro modelo, identificado e com outro oracle_id
            assert r["oracle_scope"] == "ontological_oracle_separate" and r["oracle_id"] != shared["oracle_id"]


def test_semantic_report_carries_the_contract_proof_per_split(result):
    assert len(result.semantic_report) == len(FAST.seeds)
    for rep in result.semantic_report:
        oc = rep["oracle_contract"]
        assert oc["single_oracle_for_shared_arms"] and oc["unchanged_after"] and oc["oracle_id_original"]
        assert "trepan_original" in oc["shared_oracle_arms"] and all(v > 0 for k, v in oc["queries_per_scope"].items())


def test_a_different_oracle_among_shared_arms_is_rejected():
    rows = [dict(arm="trepan_original", oracle_scope="shared_frozen_oracle", oracle_id="aaa"),
            dict(arm="reloaded_lambda0", oracle_scope="shared_frozen_oracle", oracle_id="bbb")]
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    from sklearn.linear_model import LogisticRegression
    fo = freeze_oracle(LogisticRegression().fit(X, y), X, builder="t")
    with pytest.raises(OracleContractViolation):
        BenchmarkRunner._verify_oracle_contract(rows, fo, None)


def test_the_frozen_oracle_cannot_be_refit_during_the_benchmark():
    X = np.random.default_rng(1).normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    from sklearn.linear_model import LogisticRegression
    fo = freeze_oracle(LogisticRegression().fit(X, y), X, builder="t")
    with pytest.raises(OracleContractViolation):
        fo.fit(X, y)
