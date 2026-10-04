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


def test_node_count_falls_back_to_the_tree_attribute_when_audit_has_none():
    tree = SimpleNamespace(node_count_=3)
    app = _app(mlp_model=object(), trepan_original_tree=tree, trepan_original_audit={"node_count": 0})
    assert build_experiment_result(app).models["trepan_original"].complexity["nodes"].value == 3


def test_pipeline_audit_counts_trepan_nodes():
    from core.pipeline_audit import _tree_stats
    t = SimpleNamespace(node_count_=3, get_n_leaves=lambda: 2)
    assert _tree_stats(t) == {"n_nodes": 3, "n_leaves": 2, "is_trivial": False}


def test_legacy_audit_without_stop_summary_queries_or_leaves_reads_them_from_the_tree():
    """O last_audit do TREPAN Original/Reloaded não traz stop_summary nem (Reloaded) queries: ler da árvore."""
    import numpy as np
    from core.trepan_original import TrepanOriginalClassifier

    class Oracle:
        classes_ = np.array([0, 1])
        def predict(self, X):
            return (np.asarray(X, float)[:, 0] > 0).astype(int)

    X = np.random.default_rng(0).normal(size=(120, 3))
    tree = TrepanOriginalClassifier(max_nodes=11, max_depth=4, min_sample=100, max_queries=3000, max_n=2, beam_width=1,
                                    random_state=0).fit(X, oracle=Oracle(), feature_names=list("abc"))
    app = _app(mlp_model=object(), trepan_original_tree=tree, trepan_reloaded_tree=tree,
               trepan_original_audit={"trepan_accuracy": .9}, trepan_reloaded_audit={"trepan_accuracy": .9})
    r = build_experiment_result(app)
    for key in ("trepan_original", "trepan_reloaded"):
        diag = r.trees[key]
        assert diag.loop_end_reason in {"node_budget_exhausted", "no_expandable_nodes_left"}
        assert diag.stop_reasons and diag.queries_used.value == tree.membership_queries_
        assert r.models[key].complexity["leaves"].value == tree.get_n_leaves()
        assert r.models[key].complexity["queries"].value == tree.membership_queries_


def test_builder_exposes_the_tuning_selection_and_stability_without_changing_the_config_hash():
    base = _app(mlp_model=object())
    h0 = build_experiment_result(base).provenance.config_hash
    app = _app(mlp_model=object(), _trepan_scientific_tuning={
        "structure_selected": {"purity_epsilon": 0.02, "max_nodes": 31}, "tuning_stable": False,
        "structure_selection": {"selection_probability": 0.33, "threshold": 0.6, "per_repeat_winners": ["a", "b", "a"]},
        "cv_plan": {"n_splits": 9}})
    r = build_experiment_result(app)
    t = r.config["trepan_tuning"]
    assert t["selected"] == {"purity_epsilon": 0.02, "max_nodes": 31} and t["stable"] is False and t["n_splits"] == 9
    assert r.provenance.config_hash == h0                       # o resultado do tuning não entra no hash da configuração
