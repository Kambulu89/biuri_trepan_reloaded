"""Adaptador BiuriApp -> ``ExperimentResult`` (Parte 2). Qt-free e **sem recálculo**.

Lê apenas o que o pipeline e o ``MetricsComparator`` já guardaram nos atributos da aplicação
(duck-typing: aceita qualquer objeto com esses atributos, o que permite testar sem Qt). Valores
em falta tornam-se ``Measure`` com razão — nunca ``0``. A única derivação feita aqui é de
identidade/proveniência (hash da configuração, estado), nunca de métricas científicas.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from core.build_info import get_build_info
from core.experiment_builders import (enrichment_from_reports, ontology_from_quality, predictive_metrics,
                                      semantic_features_from_audit, semantic_splits_from_audit, tree_diagnostics)
from core.experiment_result import (DatasetInfo, ExperimentResult, ExperimentState, Measure, ModelCard, Provenance,
                                    Reason)
from core.model_cache import compute_semantic_config_hash
from gui.experiment_state import Fingerprint


def _file_hash(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def config_of(app) -> Dict[str, Any]:
    """Configuração que influencia os resultados; usada no hash e na deteção de stale."""
    return {
        "onto_feature_bias_weight": getattr(app, "onto_feature_bias_weight", None),
        "ontology_match_threshold": getattr(app, "ontology_match_threshold", None),
        "training_preset": getattr(app, "_last_training_preset", None),
        "seed": getattr(app, "current_seed", None),
    }


def current_fingerprint(app) -> Fingerprint:
    """Identidade da configuração *atual* da aplicação (comparada com a do último resultado)."""
    return Fingerprint(dataset_hash=getattr(app, "dataset_fingerprint", None),
                       owl_hash=_file_hash(getattr(app, "loaded_ontology_path", None)),
                       config_hash=compute_semantic_config_hash(config_of(app)),
                       seed=getattr(app, "current_seed", None))


def _g(d: Optional[Mapping[str, Any]], *keys, default=None):
    for k in keys:
        if isinstance(d, Mapping) and d.get(k) is not None:
            return d[k]
    return default


def _block_metrics(block: Optional[Mapping[str, Any]], missing: str) -> Dict[str, Measure]:
    return predictive_metrics(block, missing=missing)


def _legacy_enrichment(acceptance: Optional[Mapping[str, Any]]):
    """Relatório de enriquecimento do pipeline legado no formato de ``enrichment_from_reports``."""
    if not acceptance:
        return None
    fs = dict(acceptance.get("feature_selection") or {})
    if "decision" in fs and "semantic_mlp_accepted" in fs:
        return fs  # já é um relatório completo de enriquecimento
    accepted = bool(acceptance.get("accepted"))
    out = {
        "semantic_mlp_accepted": accepted,
        "decision": acceptance.get("reason") or fs.get("status") or acceptance.get("semantic_utility_status")
        or ("ACCEPT" if accepted else "NOT_EVALUATED"),
        "evidence_strength": fs.get("evidence_strength"),
        "selected_semantic_features": list(acceptance.get("selected_feature_names") or []),
        "feature_audit": fs.get("feature_audit") or [],
        "stages": {"D_mlp_comparison": {
            "base": {"utility": acceptance.get("balanced_accuracy_original")},
            "with_owl": {"utility": acceptance.get("balanced_accuracy_onto")},
            "utility_gain": acceptance.get("balanced_accuracy_gain")}},
    }
    if fs.get("stages"):
        out["stages"].update(fs["stages"])
    return out


def _counterfactual_summary(app) -> Optional[Dict[str, Any]]:
    """Presença/tamanho dos resultados contrafactuais já gerados (sem os recalcular)."""
    out: Dict[str, Any] = {}
    for name in ("cf_result", "cf_interactive_result", "cf_global_result", "cf_tree_result", "cf_transfer_result",
                 "cf_improve_result"):
        value = getattr(app, name, None)
        if value is None:
            continue
        entry: Dict[str, Any] = {"available": True, "type": type(value).__name__}
        try:
            entry["n"] = len(value)
        except TypeError:
            pass
        out[name] = entry
    return out or None


def _mlp_hyperparameters(app) -> Dict[str, Any]:
    """Tuning reportado pelo treino do MLP (apenas leitura)."""
    meta = getattr(getattr(getattr(app, "trepan", None), "mlp_trainer", None), "arff_meta", None) or {}
    opt = meta.get("mlp_optimization") or {}
    out: Dict[str, Any] = {}
    if opt.get("method"):
        out["method"] = opt["method"]
    for k, v in (opt.get("best_params") or {}).items():
        out[k] = v
    return out


def build_experiment_result(app, *, experiment_id: Optional[str] = None, state: Optional[str] = None,
                            previous: Optional[ExperimentResult] = None) -> ExperimentResult:
    """Constrói o ``ExperimentResult`` a partir do estado atual da aplicação."""
    comparison = dict(getattr(getattr(app, "metrics_comparator", None), "comparison_results", None) or {})
    precision = dict(comparison.get("precision") or {})
    fidelity = dict(comparison.get("fidelity") or {})
    acceptance = getattr(app, "ontology_acceptance", None) or None
    quality = getattr(app, "ontology_quality_report", None)
    reasoner = getattr(app, "ontology_reasoner_report", None)
    path = getattr(app, "loaded_ontology_path", None)
    owl_hash = _file_hash(path)
    n_test = _g(comparison.get("model_info"), "n_test_samples")
    cache = dict(getattr(app, "_last_cache_info", None) or {})
    cfg = config_of(app)

    prov = Provenance(
        seed=getattr(app, "current_seed", None), dataset_hash=getattr(app, "dataset_fingerprint", None), owl_hash=owl_hash,
        config_hash=compute_semantic_config_hash(cfg), build=get_build_info(), source="gui",
        split="train/test (holdout bloqueado)", training_mode=getattr(app, "_last_training_preset", None),
        cache_used=bool(cache.get("used")), cache_key=cache.get("key"))
    if experiment_id:
        prov.experiment_id = experiment_id

    d = getattr(app, "dataset_info", None) or {}
    dataset = DatasetInfo(name=d.get("name") or "", path=d.get("path"), rows=d.get("rows"), features=d.get("features"),
                          classes=d.get("classes"), class_names=list(d.get("class_names") or []),
                          train_rows=d.get("train_rows"), test_rows=d.get("test_rows"),
                          seed=prov.seed, hash=prov.dataset_hash, target=d.get("target")) if d else None

    ontology_loaded = getattr(app, "loaded_ontology", None) is not None
    ontology = ontology_from_quality(quality, reasoner, path=path, owl_hash=owl_hash) if ontology_loaded else None
    enr_report = _legacy_enrichment(acceptance)
    teacher_has_onto = bool((acceptance or {}).get("accepted")) if acceptance else False
    oracle_reloaded = "mlp_ontological" if teacher_has_onto else "mlp_original"
    enrichment = enrichment_from_reports(
        enr_report, {"semantic_trepan_available": teacher_has_onto if acceptance else None,
                     "semantic_trepan_reason": None if teacher_has_onto else
                     (None if acceptance is None else "professor semântico rejeitado")},
        {"teacher": "mlp_semantic" if teacher_has_onto else "mlp_original"}, None, ontology_loaded=ontology_loaded)
    if enrichment.trepan_semantics_available is None and acceptance is None and ontology_loaded:
        enrichment.mlp_status = "NOT_EVALUATED"

    trained = getattr(app, "mlp_model", None) is not None
    models: Dict[str, ModelCard] = {}
    mlp_block = precision.get("mlp")
    if not mlp_block:
        opt = getattr(getattr(getattr(app, "trepan", None), "mlp_trainer", None), "last_optimization_summary", None) or {}
        mlp_block = opt.get("test_metrics")
    models["mlp_original"] = ModelCard(
        key="mlp_original", status="AVAILABLE" if trained else "NOT_AVAILABLE",
        status_reason=None if trained else Reason.NOT_TRAINED,
        metrics=_block_metrics(mlp_block, Reason.NOT_TRAINED if not trained else Reason.COMPARISON_NOT_RUN),
        evaluation_samples=n_test, cached=prov.cache_used, cache_key=prov.cache_key, fidelity=Measure.na(Reason.NO_ORACLE),
        hyperparameters=_mlp_hyperparameters(app))
    onto_block = precision.get("mlp_ontological")
    if onto_block:
        models["mlp_ontological"] = ModelCard(key="mlp_ontological", status="ACCEPTED", metrics=_block_metrics(onto_block, Reason.NOT_REPORTED),
                                              evaluation_samples=n_test, fidelity=Measure.na(Reason.NO_ORACLE))
    else:
        status = "NOT_AVAILABLE" if not ontology_loaded else ("REJECTED" if acceptance and not teacher_has_onto else "NOT_AVAILABLE")
        reason = Reason.NO_ONTOLOGY if not ontology_loaded else Reason.TEACHER_REJECTED
        models["mlp_ontological"] = ModelCard(
            key="mlp_ontological", status=status, status_reason=(enr_report or {}).get("decision") if status == "REJECTED" else reason,
            metrics=predictive_metrics(None, missing=reason), fidelity=Measure.na(Reason.NO_ORACLE))
    c45 = precision.get("c45_j48")
    c45_fid = fidelity.get("c45_j48")
    models["c45"] = ModelCard(
        key="c45", status="AVAILABLE" if c45 else "NOT_AVAILABLE", status_reason=None if c45 else Reason.COMPARISON_NOT_RUN,
        oracle=None, metrics=_block_metrics(c45, Reason.COMPARISON_NOT_RUN), fidelity=Measure.na(Reason.NO_ORACLE),
        agreement_with_mlp=Measure.of(_g(c45_fid, "overall_fidelity", "agreement_rate"), Reason.COMPARISON_NOT_RUN),
        evaluation_samples=n_test, complexity={"nodes": Measure.of(_g(c45, "n_nodes", "node_count"), Reason.NOT_REPORTED)})

    trees = {}
    for key, tree_attr, audit_attr, oracle in (
            ("trepan_original", "trepan_original_tree", "trepan_original_audit", "mlp_original"),
            ("trepan_reloaded", "trepan_reloaded_tree", "trepan_reloaded_audit", oracle_reloaded)):
        audit = dict(getattr(app, audit_attr, None) or {})
        built = getattr(app, tree_attr, None) is not None or bool(audit)
        if not built:
            models[key] = ModelCard(key=key, status="NOT_AVAILABLE", status_reason=Reason.TREE_NOT_BUILT,
                                    metrics=predictive_metrics(None, missing=Reason.TREE_NOT_BUILT),
                                    fidelity=Measure.na(Reason.TREE_NOT_BUILT))
            trees[key] = tree_diagnostics(key, None, built=False)
            continue
        block = precision.get(key)
        if not block and audit:
            block = {"accuracy": audit.get("trepan_accuracy"), "balanced_accuracy": audit.get("trepan_balanced_accuracy"),
                     "f1_macro": audit.get("macro_f1"), "precision_macro": audit.get("precision_macro", audit.get("precision")),
                     "recall_macro": audit.get("recall_macro", audit.get("recall"))}
        fid = _g(fidelity.get(key), "overall_fidelity") if fidelity.get(key) else audit.get("trepan_fidelity")
        nodes = _g(audit, "node_count", "nodes", "tree_n_nodes")
        if not nodes:  # auditorias antigas escreviam 0/ausente para árvores TREPAN: ler o atributo da própria árvore
            nodes = getattr(getattr(app, tree_attr, None), "node_count_", None) or nodes
        depth = audit.get("depth")
        queries = _g(audit, "membership_queries", "oracle_query_count")
        models[key] = ModelCard(
            key=key, status="AVAILABLE", oracle=oracle, metrics=_block_metrics(block, Reason.COMPARISON_NOT_RUN),
            fidelity=Measure.of(fid, Reason.COMPARISON_NOT_RUN), evaluation_samples=n_test,
            complexity={"nodes": Measure.of(nodes), "depth": Measure.of(depth), "leaves": Measure.of(audit.get("leaves")),
                        "queries": Measure.of(queries)})
        trees[key] = tree_diagnostics(key, audit.get("stop_summary"), nodes=nodes, depth=depth, leaves=audit.get("leaves"),
                                      queries=queries, split_audit=audit.get("semantic_split_audit") if key == "trepan_reloaded" else None)

    reloaded_audit = dict(getattr(app, "trepan_reloaded_audit", None) or {})
    split_rows = reloaded_audit.get("semantic_split_audit") or []
    if state is None:
        state = (ExperimentState.RESULTS_READY if comparison else
                 ExperimentState.TREES_BUILT if any(t.available for t in trees.values()) else
                 ExperimentState.MODEL_TRAINED if trained else ExperimentState.DATA_LOADED).value
    result = ExperimentResult(
        state=state, provenance=prov, dataset=dataset, ontology=ontology, enrichment=enrichment, models=models, trees=trees,
        semantic_features=semantic_features_from_audit(enr_report), semantic_splits=semantic_splits_from_audit(split_rows),
        config=cfg, counterfactual=_counterfactual_summary(app))
    if previous is not None:
        result.messages = list(previous.messages)
    return result
