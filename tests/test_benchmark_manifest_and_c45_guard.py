"""Manifesto imutável, auditoria do C4.5, prevenção de leakage, controlo negativo F e resultados brutos imutáveis."""
from __future__ import annotations

import copy
import inspect
import json
import re
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from core.benchmark import ontology_gate as og
from core.benchmark import runner as rn
from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, Dataset, TreeBudget, default_arms
from core.benchmark.semantic import GroupSemanticProvider, ShuffledSemanticProvider, structure_counts
from core.benchmark.splits import make_splits
from core.benchmark.synthetic import make_synthetic
from core.c45_j48_tree import C45Classifier
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig
from validation.benchmark import datasets as ds_mod
from validation.benchmark import invariants as inv
from validation.benchmark import manifest as mf
from validation.benchmark import raw_store as rs
from validation.benchmark.c45_audit import audit_c45_implementation, audit_native_c45

ARMS = ("mlp_original", "c45", "trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled")
CHEAP = ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1, purity_epsilon_grid=(0.05,), max_nodes_grid=(7,))
CFG = BenchmarkConfig(seeds=(42,), tree=TreeBudget(min_sample=150, max_queries=1500, max_nodes=7, max_depth=4), extra_ablations=False,
                      different_oracle_experiment=False, structure_tuning=True, structure_search=CHEAP, n_boot=100, inner_cv_splits=2)


def _arms(cfg=CFG):
    return [a for a in default_arms(cfg) if a.arm_id in ARMS]


# ------------------------------------------------------------------ 2. C4.5 é C4.5
def test_native_c45_passes_the_audit_and_cart_does_not():
    native = audit_native_c45()
    assert native["is_c45"], {k: v for k, v in native["checks"].items() if not v["passed"]}
    assert native["algorithm_name"] == "C4.5-Nativo" and native["known_deviations"]
    cart = audit_c45_implementation(lambda **kw: DecisionTreeClassifier(
        criterion="entropy", min_samples_split=kw.get("min_samples_split", 2), min_samples_leaf=kw.get("min_samples_leaf", 1)))
    failed = {k for k, v in cart["checks"].items() if not v["passed"]}
    assert not cart["is_c45"] and {"gain_ratio_selection", "categorical_multiway_split", "not_a_sklearn_cart_tree"} <= failed


def test_runner_cannot_silently_present_cart_as_c45():
    src = inspect.getsource(rn)
    assert "DecisionTreeClassifier" not in src and "sklearn.tree" not in src
    assert not any(c.__module__.startswith("sklearn.tree") for c in C45Classifier.__mro__)
    ds, groups = make_synthetic("syn_c45", n_samples=120, n_features=6, n_classes=2, n_groups=2, seed=2)
    cfg = BenchmarkConfig(seeds=(42,), tree=TreeBudget(min_sample=100, max_queries=500, max_nodes=5), extra_ablations=False,
                          different_oracle_experiment=False, structure_tuning=False, n_boot=50, inner_cv_splits=2)
    arms = [a for a in default_arms(cfg) if a.arm_id in ("mlp_original", "c45")]
    row = BenchmarkRunner(cfg, arms).run(ds, None).frame().set_index("arm").loc["c45"]
    assert row["c45_algorithm"] == "C4.5-Nativo" and bool(row["c45_is_native"]) and bool(row["c45_canonical"])
    assert row["c45_pruning"] == "pessimistic_error_subtree_replacement" and pd.isna(row["c45_node_cap"])


# ------------------------------------------------------------------ 1. manifesto
def test_manifest_has_all_required_fields_and_no_hyperparameters():
    m = mf.build_manifest()
    required = {"dataset_id", "origin", "version", "data_hash", "n_samples", "n_features", "feature_types", "target_column", "n_classes",
                "classes", "has_missing_values", "ontology_id", "ontology_hash", "expected_mapping_rate", "master_seeds"}
    for e in m["datasets"]:
        assert required <= set(e)
    assert m["master_seeds"] == [42, 7, 123, 2024, 11]
    assert [a["group"] for a in m["arms"]] == list("ABCDEF")
    assert {"code_commit", "library_versions", "frozen_code_sha256", "manifest_sha256"} <= set(m)
    assert m["main_ranking_datasets"] == ["iris", "wine", "breast_cancer", "digits"]
    assert set(m["controlled_validation_datasets"]) == set(ds_mod.CONTROLLED_DATASETS)            # sintéticos fora do ranking
    mf.assert_no_hyperparameters(m)
    bad = copy.deepcopy(m)
    bad["datasets"][0]["max_nodes"] = 31
    with pytest.raises(ValueError):
        mf.assert_no_hyperparameters(bad)
    bad = copy.deepcopy(m)
    bad["datasets"][1]["config"] = {"purity_epsilon": 0.01}
    with pytest.raises(ValueError):
        mf.assert_no_hyperparameters(bad)


def test_manifest_is_immutable_and_tamper_evident(tmp_path):
    path = tmp_path / "BENCHMARK_MANIFEST_test.json"
    mf.write_manifest(path, dataset_ids=["iris", "wine"])
    assert not (path.stat().st_mode & stat.S_IWUSR)
    with pytest.raises(FileExistsError):
        mf.write_manifest(path, dataset_ids=["iris"])
    assert mf.verify_manifest(path)["ok"]
    data = json.loads(path.read_text()); data["datasets"][0]["n_samples"] += 1
    forged = tmp_path / "forged.json"; forged.write_text(json.dumps(data))
    rep = mf.verify_manifest(forged)
    assert not rep["ok"] and any("manifesto" in p or "n_samples" in p for p in rep["problems"])


def test_frozen_ontology_files_match_registry_hashes():
    from core.benchmark.manifest import hash_file
    for spec in ds_mod.REGISTRY.values():
        assert spec.ontology_path.exists(), spec.ontology_path
        assert hash_file(str(spec.ontology_path)) == mf.dataset_entry(spec)["ontology_hash"]
    assert ds_mod.freeze_ontologies() == {s.ontology_id: str(s.ontology_path) for s in ds_mod.REGISTRY.values() if s.ontology_id}   # não regera


def test_validation_benchmark_has_no_dataset_specific_branching():
    for f in (Path(inv.__file__), Path(rs.__file__), Path(mf.__file__)):
        src = f.read_text(encoding="utf-8")
        assert not re.search(r"if\s+[\w.\[\]\"']*dataset[\w_]*\s*==", src), f
        assert not re.search(r"dataset_id\s*==", src), f


# ------------------------------------------------------------------ 4/5. controlo negativo F
def test_shuffled_control_is_deterministic_and_preserves_structure():
    ds, groups = make_synthetic("syn_shuf", n_samples=120, n_features=8, n_classes=2, n_groups=3, seed=4)
    base = GroupSemanticProvider(groups)
    real = base.build(ds.X, ds.feature_names, 7)
    a = ShuffledSemanticProvider(base).build(ds.X, ds.feature_names, 7)
    b = ShuffledSemanticProvider(base).build(ds.X, ds.feature_names, 7)
    c = ShuffledSemanticProvider(base).build(ds.X, ds.feature_names, 8)
    assert a.info["shuffled_ontology_hash"] == b.info["shuffled_ontology_hash"] and a.info["shuffle_seed"] == b.info["shuffle_seed"]
    assert a.info["shuffled_ontology_hash"] != c.info["shuffled_ontology_hash"]
    assert a.info["original_ontology_hash"] == real.info["ontology_hash"] and a.info["shuffled_ontology_hash"] != a.info["original_ontology_hash"]
    assert a.info["counts_before"] == a.info["counts_after"] == structure_counts(real)           # o controlo não é «uma ontologia menor»
    assert sorted(np.asarray(a.weights).tolist()) == sorted(np.asarray(real.weights).tolist())
    assert sorted(a.relatedness.ravel().tolist()) == sorted(real.relatedness.ravel().tolist())
    assert a.structure_signature() != real.structure_signature()                                    # mas a correspondência semântica mudou


# ------------------------------------------------------------------ 9. categorias do quality gate
def test_ontology_category_distinguishes_no_evidence_from_no_benefit():
    used = dict(ontology_effectively_used=True, ontology_valid=True, mapped_feature_count=4, semantic_features_generated=2)
    loaded_not_effective = dict(ontology_effectively_used=False, ontology_valid=True, mapped_feature_count=4, semantic_features_generated=2)
    unmapped = dict(ontology_effectively_used=False, ontology_valid=True, mapped_feature_count=0, semantic_features_generated=0)
    invalid = dict(ontology_effectively_used=False, ontology_valid=False, mapped_feature_count=3, semantic_features_generated=1)
    assert og.ontology_category(used) == "ontology_effectively_used"
    assert og.ontology_category(loaded_not_effective) == "ontology_available_but_not_effective"
    assert og.ontology_category(unmapped) == og.ontology_category(invalid) == "ontology_invalid_or_unmapped"
    assert set(og.ONTOLOGY_CATEGORIES) == {"ontology_effectively_used", "ontology_available_but_not_effective", "ontology_invalid_or_unmapped"}


# ------------------------------------------------------------------ 3. leakage
@pytest.fixture(scope="module")
def synth():
    return make_synthetic("syn_unit", n_samples=140, n_features=6, n_classes=2, n_groups=2, seed=3)


def test_test_partition_never_influences_anything_learned(synth):
    """Substituir APENAS as linhas de teste por ruído não pode alterar oráculo, pré-processamento, tuning nem a árvore."""
    ds, groups = synth
    sp = make_splits(ds.y, scheme="holdout", seeds=(42,), test_size=0.25)[0]
    X2 = ds.X.copy()
    X2[sp.test_idx] = np.random.default_rng(0).normal(5.0, 9.0, size=X2[sp.test_idx].shape)
    ds2 = Dataset("syn_unit", X2, ds.y, ds.feature_names, ds.class_labels)
    arms = [a for a in _arms() if a.arm_id in ("mlp_original", "trepan_original", "reloaded_owl_full")]
    r1 = BenchmarkRunner(CFG, arms).run(ds, GroupSemanticProvider(groups))
    r2 = BenchmarkRunner(CFG, arms).run(ds2, GroupSemanticProvider(groups))
    f1, f2 = r1.frame().set_index("arm"), r2.frame().set_index("arm")
    for arm in ("trepan_original", "reloaded_owl_full"):
        for col in ("oracle_id", "preprocessing_id", "node_count", "features_used_names", "split_signatures", "membership_queries", "fidelity_train"):
            assert f1.loc[arm, col] == f2.loc[arm, col], (arm, col)
    assert r1.semantic_report[0]["structural_protocol"]["selected"] == r2.semantic_report[0]["structural_protocol"]["selected"]


# ------------------------------------------------------------------ 6/10. invariantes + brutos imutáveis
@pytest.fixture(scope="module")
def unit(synth, tmp_path_factory):
    ds, groups = synth
    result = BenchmarkRunner(CFG, _arms()).run(ds, GroupSemanticProvider(groups))
    root = tmp_path_factory.mktemp("bench")
    rs.write_unit(result, root, manifest_sha256="test", labels=[0, 1])
    return result, root


def test_protocol_invariants_hold_on_raw_results(unit):
    _, root = unit
    rows, records, splits = rs.load_raw(root)
    report = inv.check_all(records, splits, expected_arms=list(ARMS))
    failed = {u: [c for c in cs if not c["passed"]] for u, cs in report["units"].items()}
    assert report["all_passed"], failed
    assert report["metrics_recomputable_from_raw"]["ok"] and report["metrics_recomputable_from_raw"]["checked_values"] > 30
    assert set(rows["arm"]) == set(ARMS) and rows["preprocessing_id"].nunique() == 1


def test_raw_results_are_immutable_and_complete(unit):
    result, root = unit
    udir = rs.unit_dir(root, "syn_unit", 42)
    files = {p.name for p in udir.glob("*.json")}
    assert {f"{a}.json" for a in ARMS} | {"split.json", "UNIT_SHA256.json"} == files
    arm = json.loads((udir / "reloaded_owl_full.json").read_text())
    assert {"y_true", "oracle_predictions", "surrogate_predictions", "test_row_index"} <= set(arm["predictions"])
    assert {"oracle_id", "preprocessing_id", "ontology_hash", "split_hash", "config", "row"} <= set(arm)
    assert {"fidelity_to_oracle", "node_count", "membership_queries", "tree_training_time", "ontology_effectively_used"} <= set(arm["row"])
    assert all(not (p.stat().st_mode & stat.S_IWUSR) for p in udir.glob("*.json"))
    with pytest.raises(FileExistsError):
        rs.write_unit(result, root, manifest_sha256="again", labels=[0, 1])
    assert rs.verify_raw_integrity(root)["ok"]
    target = udir / "c45.json"
    target.chmod(0o644); target.write_text(target.read_text().replace('"fidelity_to_oracle"', '"fidelity_to_oracle_x"', 1))
    assert not rs.verify_raw_integrity(root)["ok"]
    with pytest.raises(RuntimeError):
        rs.load_raw(root)
