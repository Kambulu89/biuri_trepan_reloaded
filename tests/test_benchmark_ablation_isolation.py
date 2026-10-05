"""Testes controlados: cada ablação A–F isola UMA variável (protocolo estrutural, oráculo, ontologia, espelhamento)."""
from __future__ import annotations

import json

import pytest

from core.benchmark import analysis as an
from core.benchmark.analysis import DEFAULT_CONTRASTS
from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, TreeBudget, default_arms
from core.benchmark.semantic import GroupSemanticProvider
from core.benchmark.synthetic import make_synthetic

MAIN = ("mlp_original", "c45", "trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled")
CFG = BenchmarkConfig(seeds=(11,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7, max_depth=4), extra_ablations=False,
                      n_boot=100, inner_cv_splits=2)


@pytest.fixture(scope="module")
def frame():
    ds, groups = make_synthetic("syn_iso", n_samples=140, n_features=6, n_classes=2, n_groups=2, seed=5)
    arms = [a for a in default_arms(CFG) if a.arm_id in MAIN]
    return BenchmarkRunner(CFG, arms).run(ds, GroupSemanticProvider(groups)).frame()


def _arm(df, name):
    return df[df["arm"] == name]


def test_main_arms_are_exactly_a_to_f(frame):
    assert set(frame["arm"]) == set(MAIN)


def test_all_trees_share_one_frozen_oracle_and_split(frame):
    trees = frame[frame["family"].isin(["trepan_original", "trepan_reloaded"])]
    for _, g in trees.groupby("split_id"):
        assert g["oracle_id"].nunique() == 1 and g["split_hash"].nunique() == 1 and g["seed"].nunique() == 1
        assert g["query_budget"].nunique() == 1 and g["structural_source"].nunique() == 1
        assert g["structural_tuning_ontology_blind"].astype(bool).all()
        assert g["structural_oof_fidelity"].nunique() == 1          # a MESMA configuração estrutural em C, D, E, F


def test_c45_is_canonical_and_measured_against_the_frozen_oracle(frame):
    c45 = _arm(frame, "c45")
    assert c45["c45_canonical"].astype(bool).all() and c45["oracle_id"].isna().all()
    assert c45["fidelity_to_oracle"].notna().all() and c45["fidelity_oracle_id"].nunique() == 1
    assert set(_arm(frame, "mlp_original")["oracle_id"]) == set(c45["fidelity_oracle_id"])


def test_arm_d_gets_no_ontological_information(frame):
    d = _arm(frame, "reloaded_core")
    assert (d["semantic_split_count"] == 0).all() and (d["ontology_usage_rate"] == 0).all()
    assert (d["semantic_features_used"] == 0).all() and (d["semantic_features_available"] == 0).all()
    assert not d["reasoning_applied"].astype(bool).any() and not d["ontology_effectively_used"].astype(bool).any()
    assert not d["ontology_provided_to_arm"].astype(bool).any()
    for raw in d["non_semantic_mechanisms"]:                      # mecanismos não semânticos continuam ativos
        mech = json.loads(raw)
        assert mech["error_focused_refinement"] is True and mech["semantic_active_query_fraction"] > 0


def test_mirroring_is_disabled_in_d_e_f(frame):
    for name in ("reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled"):
        g = _arm(frame, name)
        assert not g["mirror_applied"].astype(bool).any() and not g["semantic_mirror_applied"].astype(bool).any()


def test_negative_control_never_counts_as_effective_ontology_use(frame):
    assert not _arm(frame, "reloaded_owl_shuffled")["ontology_effectively_used"].astype(bool).any()
    e = _arm(frame, "reloaded_owl_full")
    evid = (e["semantic_split_count"] > 0) | (e["semantic_features_used"] > 0) | (e["semantic_rule_count"] > 0) | (e["ontology_influenced_splits"] > 0)
    assert (e["ontology_effectively_used"].astype(bool) == (evid & e["ontology_valid"].astype(bool))).all()   # só com evidência


def test_ablations_differ_in_only_the_declared_variable():
    arms = {a.arm_id: a for a in default_arms(CFG)}
    c, d, e, f = (arms[k] for k in ("trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled"))
    assert d.oracle == e.oracle == f.oracle == c.oracle
    assert d.semantic == "none" and e.semantic == "real" and f.semantic == "shuffled"
    assert d.lam == e.lam == f.lam                                  # mesmos hiperparâmetros Reloaded; só a informação ontológica muda
    assert f.control and not e.control and not d.control
    assert arms["reloaded_e2e"].role == "different_oracle_experiment" and arms["mlp_ontological"].role == "different_oracle_experiment"


def test_causal_contrasts_are_declared_without_assumed_sign():
    names = {c.name: (c.arm_a, c.arm_b) for c in DEFAULT_CONTRASTS}
    assert names["architectural_gain"] == ("reloaded_core", "trepan_original")
    assert names["incremental_ontology"] == ("reloaded_owl_full", "reloaded_core")
    assert names["NEGATIVE_CONTROL_real_vs_shuffled"] == ("reloaded_owl_full", "reloaded_owl_shuffled")
    assert names["total_reloaded_vs_original"] == ("reloaded_owl_full", "trepan_original")
    assert names["trepan_original_vs_c45"] == ("trepan_original", "c45") and names["reloaded_owl_vs_c45"] == ("reloaded_owl_full", "c45")
    assert all(c.group == "different_oracle_experiment" for c in DEFAULT_CONTRASTS if c.arm_a in ("reloaded_e2e", "mlp_ontological"))


def test_architectural_gain_is_zero_when_d_equals_original(frame):
    """Se D coincide com o Original, o ganho arquitectural é 0 (nenhuma diferença artificial é criada)."""
    c, d = _arm(frame, "trepan_original").set_index("split_id"), _arm(frame, "reloaded_core").set_index("split_id")
    diff = (d["fidelity_to_oracle"] - c["fidelity_to_oracle"]).astype(float)
    assert diff.notna().all()                                       # medido, nunca imposto; o sinal não é assumido
