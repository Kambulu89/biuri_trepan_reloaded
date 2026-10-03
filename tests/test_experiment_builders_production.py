"""O construtor reproduz, sem recalcular, o que o pipeline de produção produziu."""
import json

import numpy as np
import pytest

owlready2 = pytest.importorskip("owlready2")

from core.experiment_builders import from_production_report
from core.experiment_result import MODEL_ORDER, Reason
from core.production_training import train_production_dataframe
from core.semantic_enrichment import EnrichmentConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from test_production_attribution_gate import _family_onto, _hidden_relation  # noqa: E402

ENRICH = EnrichmentConfig(cv_folds=3, tuning_candidates=2, tuning_inner_folds=2, max_iter=150, n_bootstrap=200, random_state=5)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return train_production_dataframe(_hidden_relation(n=300, seed=4), target="target", out_dir=tmp_path_factory.mktemp("p"),
                                      seed=5, ontology=_family_onto(20, "bld"), require_reasoner=True,
                                      scientific_tuning=False, semantic_enrichment=ENRICH)


@pytest.fixture(scope="module")
def result(report):
    return from_production_report(report)


def test_result_is_complete_and_json_serializable(result):
    assert result.state == "RESULTS_READY"
    assert set(result.models) == set(MODEL_ORDER) and set(result.trees) == {"trepan_original", "trepan_reloaded"}
    json.dumps(result.to_dict())


def test_dataset_and_provenance_come_from_the_manifest(result, report):
    m = report["manifest"]
    assert result.dataset.train_rows == m["train_rows"] and result.dataset.test_rows == m["test_rows"]
    assert result.dataset.rows == m["train_rows"] + m["test_rows"] and result.dataset.classes == 2
    assert result.dataset.features == 23 and result.dataset.hash == m["dataset_sha256"]
    p = result.provenance
    assert p.seed == 5 and p.source == "headless" and p.build["commit"] and p.build["semantic_pipeline_version"]
    assert p.experiment_id and p.config_hash


def test_metrics_are_copied_not_recomputed(result, report):
    ev = report["evaluation"]["models"]
    assert result.models["mlp_original"].metrics["accuracy"].value == ev["mlp_original"]["accuracy"]
    assert result.models["trepan_reloaded"].metrics["accuracy"].value == ev["reloaded"]["accuracy"]
    assert result.models["trepan_original"].fidelity.value == ev["original"]["oracle_fidelity"]
    assert result.models["trepan_reloaded"].complexity["nodes"].value == ev["reloaded"]["nodes"]
    assert result.models["c45"].metrics["balanced_accuracy"].value == ev["c45_native"]["balanced_accuracy"]


def test_oracle_is_always_identified_for_the_trees_and_c45_has_none(result):
    assert result.models["c45"].oracle is None and result.models["c45"].fidelity.reason == Reason.NO_ORACLE
    assert result.models["mlp_original"].fidelity.reason == Reason.NO_ORACLE
    for key in ("trepan_original", "trepan_reloaded"):
        assert result.models[key].oracle in {"mlp_original", "mlp_ontological"}
    assert result.models["trepan_original"].oracle == result.models["trepan_reloaded"].oracle   # mesmo professor


def test_original_and_ontological_mlp_are_separate_cards(result, report):
    ev = report["evaluation"]["models"]
    assert "mlp_semantic" in ev                      # neste cenário o professor semântico foi usado
    assert result.models["mlp_ontological"].status == "ACCEPTED"
    assert result.models["mlp_ontological"].metrics["accuracy"].value == ev["mlp_semantic"]["accuracy"]
    assert result.models["mlp_original"].metrics["accuracy"].value == ev["mlp_original"]["accuracy"]
    assert result.models["mlp_ontological"].metrics["accuracy"].value != result.models["mlp_original"].metrics["accuracy"].value


def test_ontology_and_enrichment_axes(result):
    o, e = result.ontology, result.enrichment
    assert (o.structural_status, o.reasoner_status, o.abox_status) == ("VALID", "CONSISTENT", "SAFE")
    assert o.mapped == o.total == 23 and o.coverage == 1.0
    assert e.mlp_status == "ACCEPTED" and e.teacher == "mlp_semantic" and e.trepan_semantics_available is True
    assert e.reloaded_mode in {"augmented", "original_space", "neutral", "none"}
    assert e.base_utility.available and e.owl_utility.available and e.delta_utility.available


def test_tree_diagnostics_explain_why_trees_stopped(result):
    for key in ("trepan_original", "trepan_reloaded"):
        d = result.trees[key]
        assert d.available and d.loop_end_reason in {"node_budget_exhausted", "no_expandable_nodes_left"}
        assert d.nodes_before_pruning.value >= d.nodes_after_pruning.value == d.logical_nodes.value
        assert d.query_budget.available and d.queries_used.available and d.stop_reasons


def test_semantic_tables_are_populated_from_backend_audits(result, report):
    assert len(result.semantic_features) == len(report["evaluation"]["semantic_enrichment"]["feature_audit"])
    assert any(f.selected for f in result.semantic_features)
    assert len(result.semantic_splits) == len(report["evaluation"]["semantic_split_audit"])
    assert all(r.final_score.available for r in result.semantic_splits)
