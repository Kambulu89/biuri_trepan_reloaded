"""Construção de ``ExperimentResult`` a partir do que o backend JÁ calculou.

Nada aqui recalcula métricas: cada função só **mapeia** campos de relatórios existentes
para o modelo de dados, marcando com ``Reason`` o que o backend não forneceu. Os auxiliares
são partilhados pelo caminho headless (``from_production_report``) e pelo adaptador da GUI
(``gui/result_builder.py``), para que ambos digam exatamente o mesmo.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.build_info import get_build_info
from core.model_cache import compute_semantic_config_hash
from core.experiment_result import (
    ControlRow, DatasetInfo, EnrichmentInfo, ExperimentResult, ExperimentState, Measure, Message, ModelCard,
    OntologyInfo, Provenance, Reason, ScientificDiagnostics, SemanticFeatureRow, SemanticSplitRow, TreeDiagnostics,
)

PREDICTIVE_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "precision_macro", "recall_macro")


def _get(d: Optional[Mapping[str, Any]], *path, default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


# --------------------------------------------------------------------------- ontologia ---
def ontology_from_quality(
    quality: Optional[Mapping[str, Any]],
    reasoner: Optional[Mapping[str, Any]] = None,
    *,
    path: Optional[str] = None,
    owl_hash: Optional[str] = None,
) -> OntologyInfo:
    """Estado da ontologia em eixos independentes, a partir do relatório do quality gate."""
    if not quality:
        return OntologyInfo(loaded=bool(path), path=path, hash=owl_hash)
    metrics = dict(quality.get("metrics") or {})
    reasoner = dict(reasoner or quality.get("reasoner") or {})
    split = metrics.get("knowledge_split") or {}
    tbox = split.get("tbox") or {}
    tbox_entities = sum(int(v or 0) for v in tbox.values()) if tbox else None
    issues = [str(i) for i in (quality.get("issues") or [])]
    if reasoner.get("executed") or reasoner.get("reasoner_used"):
        reasoner_status = "CONSISTENT" if reasoner.get("consistent") else "INCONSISTENT"
    else:
        reasoner_status = "NOT_EXECUTED"
    abox = quality.get("abox") or {}
    coverage = metrics.get("feature_coverage")
    mapped, total = metrics.get("mapped_features"), metrics.get("total_features")
    return OntologyInfo(
        loaded=True, path=path, hash=owl_hash,
        structural_status="VALID" if quality.get("accepted") else str(quality.get("status") or "INVALID"),
        reasoner_status=reasoner_status, reasoner_engine=reasoner.get("engine"),
        reasoner_seconds=reasoner.get("duration_seconds"),
        reasoner_inferred_axioms=reasoner.get("inferred_axioms_count"),
        reasoner_fallback=reasoner.get("fallback"),
        mapped=None if mapped is None else int(mapped), total=None if total is None else int(total),
        coverage=None if coverage is None else float(coverage),
        ambiguous=metrics.get("ambiguous_matches"), collisions=metrics.get("entity_collisions"),
        tbox_status=("NOT_EVALUATED" if tbox_entities is None else
                     "INVALID" if tbox_entities == 0 or any("TBox" in i for i in issues) else "VALID"),
        abox_status=str(abox.get("status") or ("SAFE" if quality.get("accepted") else "NOT_EVALUATED")),
        richness=(metrics.get("semantic_richness") or {}).get("level"),
        issues=issues, warnings=[str(w) for w in (quality.get("warnings") or [])],
    )


# ------------------------------------------------------------------------- enriquecimento ---
def enrichment_from_reports(
    enrichment: Optional[Mapping[str, Any]],
    stage_status: Optional[Mapping[str, Any]] = None,
    teacher: Optional[Mapping[str, Any]] = None,
    attribution: Optional[Mapping[str, Any]] = None,
    *,
    ontology_loaded: bool = True,
) -> EnrichmentInfo:
    stage_status = dict(stage_status or {})
    if not ontology_loaded:
        return EnrichmentInfo(mlp_status="NOT_AVAILABLE",
                              trepan_semantics_available=False, trepan_semantics_reason="no_ontology")
    stages = dict((enrichment or {}).get("stages") or {})
    novelty = dict(stages.get("B_novelty") or {})
    screening = dict(stages.get("C_screening") or {})
    compare = dict(stages.get("D_mlp_comparison") or {})
    if enrichment:
        accepted = bool(enrichment.get("semantic_mlp_accepted"))
        status = "ACCEPTED" if accepted else ("REJECTED" if enrichment.get("decision") != "NOT_EVALUATED" else "NOT_EVALUATED")
    else:
        status = "NOT_EVALUATED"
    base_u = _get(compare, "base", "utility")
    owl_u = _get(compare, "with_owl", "utility")
    return EnrichmentInfo(
        mlp_status=status,
        decision=(enrichment or {}).get("decision"),
        evidence_strength=(enrichment or {}).get("evidence_strength"),
        base_utility=Measure.of(base_u, Reason.ENRICHMENT_NOT_EVALUATED),
        owl_utility=Measure.of(owl_u, Reason.ENRICHMENT_NOT_EVALUATED),
        delta_utility=Measure.of(compare.get("utility_gain"), Reason.ENRICHMENT_NOT_EVALUATED),
        utility_ci=list(compare["utility_gain_ci"]) if compare.get("utility_gain_ci") else None,
        generated=novelty.get("generated"),
        stable=len(screening.get("stable_features") or []) if screening else None,
        selected=len((enrichment or {}).get("selected_semantic_features") or []) if enrichment else None,
        trepan_semantics_available=(stage_status.get("semantic_trepan_available")
                                    if "semantic_trepan_available" in stage_status
                                    else (enrichment or {}).get("semantic_trepan_available")),
        trepan_semantics_reason=stage_status.get("semantic_trepan_reason"),
        teacher=(teacher or {}).get("teacher"),
        reloaded_mode=(attribution or {}).get("selected_mode") or stage_status.get("reloaded_mode"),
        attribution_status=(attribution or {}).get("status"),
    )


def semantic_features_from_audit(enrichment: Optional[Mapping[str, Any]]) -> List[SemanticFeatureRow]:
    """Tabela de features semânticas a partir do relatório por feature do enriquecimento."""
    rows = []
    for item in (enrichment or {}).get("feature_audit") or []:
        sources = item.get("source_features") or []
        rows.append(SemanticFeatureRow(
            name=str(item.get("feature")), type=str(item.get("origin") or ""),
            source=", ".join(map(str, sources)),
            stability=Measure.of(item.get("selection_frequency"), Reason.NOT_REPORTED),
            selected=item.get("decision") == "SELECTED", reason=str(item.get("reason") or ""),
        ))
    return rows


def semantic_splits_from_audit(split_rows: Optional[Sequence[Mapping[str, Any]]]) -> List[SemanticSplitRow]:
    """Tabela de splits do TREPAN Reloaded; vem diretamente da auditoria por nó do backend."""
    out = []
    for r in split_rows or []:
        feats = r.get("selected_feature")
        feature = ", ".join(feats) if isinstance(feats, (list, tuple)) else (str(feats) if feats else str(r.get("test") or ""))
        out.append(SemanticSplitRow(
            node=r.get("node_id"), feature=feature,
            base_score=Measure.of(r.get("base_score", r.get("information_gain")), Reason.NOT_REPORTED),
            semantic_bonus=Measure.of(r.get("semantic_bonus"), Reason.NOT_REPORTED),
            final_score=Measure.of(r.get("final_score", r.get("selection_score")), Reason.NOT_REPORTED),
            reason=str(r.get("semantic_reason") or ""), decision_changed=r.get("decision_changed"),
        ))
    return out


# ------------------------------------------------------------------------------- árvores ---
def tree_diagnostics(
    tree: str,
    stop_summary: Optional[Mapping[str, Any]],
    *,
    nodes: Any = None, depth: Any = None, leaves: Any = None, queries: Any = None,
    split_audit: Optional[Sequence[Mapping[str, Any]]] = None, built: bool = True,
) -> TreeDiagnostics:
    """Diagnóstico estrutural de uma árvore. Sem árvore construída, tudo fica 'não executado'."""
    if not built:
        na = Measure.na(Reason.TREE_NOT_BUILT)
        return TreeDiagnostics(tree=tree, logical_nodes=na, rendered_nodes=na, depth=na, leaves=na, queries_used=na,
                               query_budget=na, node_budget=na, nodes_before_pruning=na, nodes_after_pruning=na,
                               available=False)
    s = dict(stop_summary or {})
    unreported = Reason.NOT_REPORTED
    return TreeDiagnostics(
        tree=tree, available=True,
        logical_nodes=Measure.of(nodes if nodes is not None else s.get("nodes_after_pruning"), unreported),
        rendered_nodes=Measure.na(Reason.NOT_RENDERED),
        depth=Measure.of(depth, unreported), leaves=Measure.of(leaves, unreported),
        queries_used=Measure.of(queries if queries is not None else s.get("queries_used"), unreported),
        query_budget=Measure.of(s.get("query_budget"), unreported),
        query_budget_exhausted=s.get("query_budget_exhausted"),
        node_budget=Measure.of(s.get("node_budget"), unreported),
        nodes_before_pruning=Measure.of(s.get("nodes_before_pruning"), unreported),
        nodes_after_pruning=Measure.of(s.get("nodes_after_pruning"), unreported),
        loop_end_reason=s.get("loop_end_reason"), stop_reasons=dict(s.get("stop_reasons") or {}),
        m_of_n_splits=(sum(1 for r in split_audit if (r.get("n") or 1) > 1) if split_audit is not None else None),
    )


def predictive_metrics(metrics: Optional[Mapping[str, Any]], *, missing: str = Reason.NOT_REPORTED,
                       aliases: Optional[Mapping[str, str]] = None) -> Dict[str, Measure]:
    """Métricas preditivas (vs rótulos reais). Valor ausente -> sem valor com razão, nunca 0."""
    aliases = dict(aliases or {})
    out: Dict[str, Measure] = {}
    for key in PREDICTIVE_KEYS:
        raw = None
        if metrics:
            raw = metrics.get(aliases.get(key, key))
            if raw is None and key == "macro_f1":
                raw = metrics.get("f1_macro")
        out[key] = Measure.of(raw, missing)
    return out


# --------------------------------------------------------------------- diagnóstico científico ---
def _winner_stats(tuning: Mapping[str, Any]):
    """(rótulo, estatísticas, seleção) do estágio de tuning que decidiu a configuração; None se não houve tuning."""
    for hist_key, sel_key in (("capacity_history", "capacity_selection"), ("structure_history", "structure_selection")):
        sel = tuning.get(sel_key)
        if sel:
            for h in tuning.get(hist_key) or []:
                if h.get("label") == sel.get("selected_label") and h.get("stats"):
                    return sel["selected_label"], h["stats"], sel
    return None, None, None


def scientific_diagnostics_from_tuning(
    tuning: Optional[Mapping[str, Any]], oracle_contract: Optional[Mapping[str, Any]] = None, *,
    seed: Optional[int] = None, execution_mode: Optional[str] = None,
) -> ScientificDiagnostics:
    """Mapeia o relatório do tuning e do contrato do oráculo para o diagnóstico científico mostrado na GUI."""
    from core.execution_mode import ExecutionMode
    contract = dict(oracle_contract or {})
    oracle = dict(contract.get("oracle") or {})
    ids = dict(contract.get("tree_oracle_ids") or {})
    same = (len(set(ids.values())) == 1 and None not in ids.values()) if ids else None
    benchmark = bool(execution_mode == ExecutionMode.SCIENTIFIC_BENCHMARK.value and oracle.get("oracle_id")
                     and contract.get("single_oracle_for_all_trees") and same)
    diag = ScientificDiagnostics(
        execution_mode=execution_mode, benchmark_eligible=benchmark, oracle_id=oracle.get("oracle_id"),
        oracle_builder=oracle.get("builder"), same_oracle_original_reloaded=same, seed=seed,
        test_used_for_selection=None if not tuning else bool(tuning.get("test_used_for_selection", False)),
    )
    if not oracle:
        diag.fidelity_mean = diag.fidelity_std = Measure.na(Reason.NO_ORACLE_CONTRACT)
    if not tuning:
        diag.tuning_status = "not_run"
        for name in ("fidelity_mean", "fidelity_std", "predictive_stability", "structural_stability", "selection_probability",
                     "selection_margin", "fraction_at_node_cap", "queries_used", "query_budget"):
            setattr(diag, name, Measure.na(Reason.TUNING_NOT_RUN))
        return diag
    if tuning.get("failed"):
        diag.tuning_status = "failed"
        return diag
    label, stats, sel = _winner_stats(tuning)
    plan = dict(tuning.get("cv_plan") or {})
    if plan:
        diag.cv_plan = (f"{plan.get('scheme', 'RepeatedStratifiedKFold')}: {plan.get('repeats')}×{plan.get('folds')} "
                        f"(seeds {plan.get('seeds')}; {plan.get('n_splits')} partições)")
    diag.selected_config = label or "; ".join(
        f"{k}={v}" for k, v in (tuning.get("structure_selected") or {}).items()) or None
    if not stats:
        diag.tuning_status = "not_run"
        return diag
    bs = dict(sel.get("bootstrap") or {})
    ru = sel.get("bootstrap_runner_up")
    assessable = bool(bs.get("assessable"))
    diag.fidelity_mean = Measure.of(stats.get("fidelity_mean"))
    diag.fidelity_std = Measure.of(stats.get("fidelity_std"))
    diag.predictive_stability = Measure.of(stats.get("fidelity_std"))
    diag.structural_stability = Measure.of(stats.get("structural_instability"))
    diag.structural_stability_evidence = stats.get("structural_stability_evidence")
    diag.selection_probability = Measure.of(bs.get("selected_config_probability") if assessable else None)
    diag.selected_config_full_cv = sel.get("selected_config_full_cv")
    diag.bootstrap_modal_config = sel.get("bootstrap_modal_config")
    diag.bootstrap_modal_probability = Measure.of(sel.get("bootstrap_modal_probability") if assessable else None)
    diag.selection_runner_up = f"{ru['label']} ({ru['probability']:.0%})" if ru else None
    diag.selection_margin = Measure.of(sel.get("top1_top2_margin") if assessable else None)
    diag.full_cv_selection_fragile = sel.get("full_cv_selection_fragile")
    diag.bootstrap_method = bs.get("method")
    diag.selection_resamples = bs.get("bootstrap_samples")
    ex = dict(tuning.get("capacity_expansion") or {})
    diag.expansion_triggered = ex.get("expansion_triggered")
    diag.capacity_expansion_rounds = ex.get("capacity_expansion_rounds")
    diag.initial_node_grid = list(ex.get("initial_node_grid") or [])
    diag.final_node_grid = list(ex.get("final_node_grid") or [])
    diag.expansion_stop_reason = ex.get("expansion_stop_reason")
    diag.fraction_at_node_cap = Measure.of(stats.get("fraction_at_node_cap"))
    diag.node_cap_censored = bool(stats.get("structural_stability_censored"))
    diag.queries_used = Measure.of(stats.get("queries_used_mean"))
    diag.query_budget = Measure.of(stats.get("query_budget"))
    diag.budget_exhausted = bool(stats.get("budget_exhausted_count", 0) > 0)
    diag.tuning_status = str(sel.get("status") or ("tuning_stable" if sel.get("stable") else "tuning_uncertain"))
    return diag


# ------------------------------------------------------------------- relatório headless ---
def from_production_report(report: Mapping[str, Any]) -> ExperimentResult:
    """Monta o resultado de uma experiência a partir de ``train_production_dataframe``."""
    manifest = dict(report.get("manifest") or {})
    ev = dict(report.get("evaluation") or {})
    models_ev = dict(ev.get("models") or {})
    quality = ev.get("ontology_quality")
    enrichment = ev.get("semantic_enrichment")
    stage = ev.get("ontology_stage_status")
    teacher = ev.get("semantic_teacher") or {}
    attribution = ev.get("semantic_attribution")
    contract = dict(report.get("contract") or {})
    split = dict(report.get("split") or {})
    audit = dict(ev.get("experiment_audit") or {})
    cfg = dict(audit.get("config") or {})

    provenance = Provenance(
        seed=manifest.get("seed"), dataset_hash=manifest.get("dataset_sha256"), owl_hash=manifest.get("owl_sha256"),
        build=get_build_info(), source="headless", split="train/test estratificado",
        config_hash=compute_semantic_config_hash(manifest.get("trepan_selected_config"), manifest.get("scientific_tuning")),
        training_mode="headless", cache_used=False,
    )
    classes = [str(c) for c in (contract.get("class_counts") or {})]
    dataset = DatasetInfo(
        name=str(manifest.get("target") or "dataset"),
        rows=((manifest.get("train_rows") or 0) + (manifest.get("test_rows") or 0)) or contract.get("n_rows"),
        features=len([c for c in contract.get("columns", []) if c.get("name") != manifest.get("target")]) or None,
        classes=len(classes) or None, class_names=classes, train_rows=manifest.get("train_rows"),
        test_rows=manifest.get("test_rows"), seed=manifest.get("seed"), hash=manifest.get("dataset_sha256"),
        target=manifest.get("target"),
    )
    ontology = ontology_from_quality(quality, (quality or {}).get("reasoner"), owl_hash=manifest.get("owl_sha256")) if quality else None
    enrich = enrichment_from_reports(enrichment, stage, teacher, attribution, ontology_loaded=bool(quality))
    if enrich.reloaded_mode is None:
        enrich.reloaded_mode = reloaded_space_mode(ev)
    n_test = manifest.get("test_rows")
    teacher_key = "mlp_ontological" if teacher.get("teacher") == "mlp_semantic" else "mlp_original"

    models: Dict[str, ModelCard] = {}
    mo = models_ev.get("mlp_original")
    models["mlp_original"] = ModelCard(
        key="mlp_original", status="AVAILABLE" if mo else "NOT_AVAILABLE",
        status_reason=None if mo else Reason.NOT_TRAINED, metrics=predictive_metrics(mo, missing=Reason.NOT_TRAINED),
        evaluation_samples=n_test, notes=[], fidelity=Measure.na(Reason.NO_ORACLE))
    ms = models_ev.get("mlp_semantic")
    if ms:
        models["mlp_ontological"] = ModelCard(key="mlp_ontological", status="ACCEPTED", metrics=predictive_metrics(ms),
                                              evaluation_samples=n_test, fidelity=Measure.na(Reason.NO_ORACLE))
    else:
        reason = Reason.NO_ONTOLOGY if not quality else Reason.TEACHER_REJECTED
        models["mlp_ontological"] = ModelCard(key="mlp_ontological",
                                              status="NOT_AVAILABLE" if not quality else "REJECTED", status_reason=reason,
                                              metrics=predictive_metrics(None, missing=reason), fidelity=Measure.na(Reason.NO_ORACLE))
    c45 = models_ev.get("c45_native")
    models["c45"] = ModelCard(
        key="c45", status="AVAILABLE" if c45 else "NOT_AVAILABLE", oracle=None,
        metrics=predictive_metrics(c45, missing=Reason.NOT_TRAINED), fidelity=Measure.na(Reason.NO_ORACLE),
        evaluation_samples=n_test, status_reason=None if c45 else Reason.NOT_TRAINED)
    diag = dict(ev.get("tree_diagnostics") or {})
    trees: Dict[str, TreeDiagnostics] = {}
    for key, arm, label in (("trepan_original", "original", "trepan_original"), ("trepan_reloaded", "reloaded", "trepan_reloaded")):
        m = models_ev.get(arm)
        if not m:
            models[key] = ModelCard(key=key, status="NOT_AVAILABLE", status_reason=Reason.TREE_NOT_BUILT,
                                    metrics=predictive_metrics(None, missing=Reason.TREE_NOT_BUILT), fidelity=Measure.na(Reason.TREE_NOT_BUILT))
            trees[key] = tree_diagnostics(key, None, built=False)
            continue
        models[key] = ModelCard(
            key=key, status="AVAILABLE", oracle=teacher_key, metrics=predictive_metrics(m),
            fidelity=Measure.of(m.get("oracle_fidelity"), Reason.NOT_REPORTED), evaluation_samples=n_test,
            complexity={"nodes": Measure.of(m.get("nodes")), "depth": Measure.of(m.get("depth")),
                        "leaves": Measure.of(m.get("leaves")), "queries": Measure.of(m.get("membership_queries"))})
        trees[key] = tree_diagnostics(key, diag.get(arm), nodes=m.get("nodes"), depth=m.get("depth"), leaves=m.get("leaves"),
                                      queries=m.get("membership_queries"),
                                      split_audit=ev.get("semantic_split_audit") if arm == "reloaded" else None)
    if "trepan_reloaded" in models and models["trepan_reloaded"].status == "AVAILABLE":
        models["trepan_reloaded"].notes.append(f"modo do Reloaded: {enrich.reloaded_mode}")
    result = ExperimentResult(
        state=ExperimentState.RESULTS_READY.value, provenance=provenance, dataset=dataset, ontology=ontology,
        enrichment=enrich, models=models, trees=trees,
        semantic_features=semantic_features_from_audit(enrichment),
        semantic_splits=semantic_splits_from_audit(ev.get("semantic_split_audit")),
        controls=controls_from_report(models_ev, attribution),
        config={k: cfg.get(k) for k in ("max_nodes", "max_depth", "max_queries", "min_sample", "random_state") if k in cfg},
    )
    from core.execution_mode import ExecutionMode
    mode = str(report.get("execution_mode") or ExecutionMode.SCIENTIFIC_BENCHMARK.value)
    result.provenance.execution_mode = mode
    result.scientific = scientific_diagnostics_from_tuning(
        ev.get("trepan_scientific_tuning"), ev.get("oracle_contract"), seed=manifest.get("seed"), execution_mode=mode)
    return result


def controls_from_report(models_ev: Optional[Mapping[str, Any]], attribution: Optional[Mapping[str, Any]]) -> List[ControlRow]:
    """Controlo negativo lado a lado: sem semântica / OWL real / controlo aleatório (teste final, só relatado).

    Só existe quando o gate de atribuição correu. ``random_control`` são as features derivadas aleatórias
    do próprio gate (não uma OWL baralhada); a precisão não é reportada para esse braço e fica 'não reportada'.
    """
    test = (attribution or {}).get("test_attribution")
    if not attribution or not test:
        return []
    models_ev = dict(models_ev or {})
    rows: List[ControlRow] = []
    orig, rel = models_ev.get("original"), models_ev.get("reloaded")
    if orig:
        rows.append(ControlRow("no_semantics", accuracy=Measure.of(orig.get("accuracy")), fidelity=Measure.of(orig.get("oracle_fidelity")),
                               nodes=Measure.of(orig.get("nodes"))))
    if rel:
        rows.append(ControlRow("real_owl", accuracy=Measure.of(rel.get("accuracy")), fidelity=Measure.of(rel.get("oracle_fidelity")),
                               nodes=Measure.of(rel.get("nodes"))))
    ctrls = list(test.get("controls") or [])
    if ctrls:
        fid = sum(c.get("oracle_fidelity", 0.0) for c in ctrls) / len(ctrls)
        rows.append(ControlRow("random_control", accuracy=Measure.na(Reason.NOT_REPORTED), fidelity=Measure.of(fid),
                               nodes=Measure.na(Reason.NOT_REPORTED), n_runs=len(ctrls)))
    return rows


def reloaded_space_mode(ev: Mapping[str, Any]) -> Optional[str]:
    return _get(ev, "reloaded_feature_space", "mode") or _get(ev, "reloaded_feature_space", "space")
