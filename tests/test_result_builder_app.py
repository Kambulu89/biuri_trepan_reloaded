"""Adaptador BiuriApp -> ExperimentResult sem Qt e sem recálculo."""
import json
from types import SimpleNamespace

from core.experiment_result import Reason
from gui.result_builder import build_experiment_result, config_of, current_fingerprint


def _app(**kw):
    app = SimpleNamespace(
        metrics_comparator=SimpleNamespace(comparison_results={}), ontology_acceptance=None, ontology_quality_report=None,
        ontology_reasoner_report=None, loaded_ontology=None, loaded_ontology_path=None, mlp_model=None,
        trepan_original_tree=None, trepan_reloaded_tree=None, trepan_original_audit=None, trepan_reloaded_audit=None,
        onto_feature_bias_weight=0.5, ontology_match_threshold=0.72, dataset_fingerprint="dh", current_seed=42,
        dataset_info={"name": "iris", "rows": 150, "features": 4, "classes": 3, "train_rows": 120, "test_rows": 30})
    for k, v in kw.items():
        setattr(app, k, v)
    return app


def test_untrained_app_has_na_with_reasons_not_zeros():
    r = build_experiment_result(_app(dataset_info={}))
    assert r.state == "NO_DATA" or r.state == "DATA_LOADED"
    assert r.models["mlp_original"].status == "NOT_AVAILABLE"
    assert r.models["mlp_original"].metrics["accuracy"].reason == Reason.NOT_TRAINED
    assert r.models["trepan_original"].fidelity.reason == Reason.TREE_NOT_BUILT
    assert r.models["mlp_ontological"].status_reason == Reason.NO_ONTOLOGY
    assert r.trees["trepan_original"].available is False
    json.dumps(r.to_dict())


def test_trained_app_copies_values_without_recomputation():
    comparison = {"precision": {"mlp": {"accuracy": .9, "f1_macro": .88, "precision_macro": .87, "recall_macro": .86, "balanced_accuracy": .85},
                                "trepan_original": {"accuracy": .8, "f1_macro": .7},
                                "trepan_reloaded": {"accuracy": .82, "f1_macro": .72},
                                "c45_j48": {"accuracy": .75}},
                  "fidelity": {"trepan_original": {"overall_fidelity": .93}, "trepan_reloaded": {"overall_fidelity": .95}},
                  "model_info": {"n_test_samples": 30}}
    audit_o = {"node_count": 9, "depth": 3, "leaves": 5, "membership_queries": 200,
               "stop_summary": {"loop_end_reason": "node_budget_exhausted", "stop_reasons": {"pure_node": 2}, "queries_used": 200,
                                "nodes_before_pruning": 11, "nodes_after_pruning": 9}}
    app = _app(mlp_model=object(), metrics_comparator=SimpleNamespace(comparison_results=comparison),
               trepan_original_tree=object(), trepan_reloaded_tree=object(), trepan_original_audit=audit_o,
               trepan_reloaded_audit={"node_count": 13, "depth": 4, "semantic_split_audit": [{"node_id": 1, "selected_feature": "f", "semantic_bonus": .1}]})
    r = build_experiment_result(app)
    assert r.state == "RESULTS_READY"
    assert r.models["mlp_original"].metrics["accuracy"].value == .9 and r.models["mlp_original"].metrics["macro_f1"].value == .88
    assert r.models["trepan_original"].fidelity.value == .93 and r.models["trepan_original"].oracle == "mlp_original"
    assert r.models["trepan_reloaded"].oracle == "mlp_original"  # sem aceitação ontológica
    assert r.models["c45"].fidelity.reason == Reason.NO_ORACLE and r.models["c45"].oracle is None
    assert r.trees["trepan_original"].loop_end_reason == "node_budget_exhausted"
    assert r.trees["trepan_original"].nodes_before_pruning.value == 11
    assert len(r.semantic_splits) == 1 and r.models["mlp_original"].evaluation_samples == 30


def test_accepted_ontology_changes_reloaded_oracle_only():
    app = _app(mlp_model=object(), loaded_ontology=object(), ontology_acceptance={"accepted": True, "selected_feature_names": ["a"],
               "balanced_accuracy_original": .8, "balanced_accuracy_onto": .85, "balanced_accuracy_gain": .05},
               trepan_original_audit={"node_count": 5}, trepan_reloaded_audit={"node_count": 5},
               trepan_original_tree=object(), trepan_reloaded_tree=object())
    r = build_experiment_result(app)
    assert r.models["trepan_reloaded"].oracle == "mlp_ontological" and r.models["trepan_original"].oracle == "mlp_original"
    assert r.enrichment.mlp_status == "ACCEPTED" and r.enrichment.delta_utility.value == .05


def test_rejected_ontology_is_explicit():
    app = _app(mlp_model=object(), loaded_ontology=object(),
               ontology_acceptance={"accepted": False, "reason": "ONTOLOGY_VALID_BUT_NO_PREDICTIVE_UTILITY"})
    r = build_experiment_result(app)
    assert r.enrichment.mlp_status == "REJECTED" and r.models["mlp_ontological"].status == "REJECTED"
    assert r.models["mlp_ontological"].metrics["accuracy"].reason == Reason.TEACHER_REJECTED
    assert r.enrichment.trepan_semantics_available is False


def test_cache_info_and_fingerprint_follow_config():
    app = _app(mlp_model=object(), _last_cache_info={"used": True, "key": "abc123"})
    r = build_experiment_result(app)
    assert r.provenance.cache_used and r.provenance.cache_key == "abc123" and r.models["mlp_original"].cached
    fp1 = current_fingerprint(app)
    app.onto_feature_bias_weight = 0.9
    fp2 = current_fingerprint(app)
    assert fp1.config_hash != fp2.config_hash and fp1.dataset_hash == fp2.dataset_hash
    assert config_of(app)["onto_feature_bias_weight"] == 0.9


def test_builder_does_not_mutate_app_state():
    comparison = {"precision": {"mlp": {"accuracy": .9}}, "fidelity": {}}
    app = _app(mlp_model=object(), metrics_comparator=SimpleNamespace(comparison_results=comparison))
    before = json.dumps(comparison, sort_keys=True)
    build_experiment_result(app)
    assert json.dumps(comparison, sort_keys=True) == before


def test_counterfactual_summary_and_mlp_tuning_are_read_only_views():
    trainer = SimpleNamespace(arff_meta={"mlp_optimization": {"method": "optuna", "best_params": {"hidden": 32}}})
    app = _app(mlp_model=object(), trepan=SimpleNamespace(mlp_trainer=trainer), cf_result=[1, 2, 3])
    r = build_experiment_result(app)
    assert r.counterfactual == {"cf_result": {"available": True, "type": "list", "n": 3}}
    assert r.models["mlp_original"].hyperparameters == {"method": "optuna", "hidden": 32}
    assert build_experiment_result(_app()).counterfactual is None
