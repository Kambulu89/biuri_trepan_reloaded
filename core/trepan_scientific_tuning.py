"""Selecção científica train-only da capacidade TREPAN e força semântica.

O módulo nunca recebe o teste externo. Primeiro selecciona uma capacidade comum
para Original/Reloaded usando o TREPAN Original; depois, mantendo exactamente
essa capacidade, selecciona apenas parâmetros da extensão semântica do Reloaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from core.training_config import required_query_budget
from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


@dataclass(frozen=True)
class ScientificTrepanSearchConfig:
    cv_folds: int = 3
    max_capacity_candidates: int = 6
    max_semantic_candidates: int = 6
    fidelity_target: float = 0.95
    fidelity_weight: float = 0.65
    balanced_accuracy_weight: float = 0.20
    macro_f1_weight: float = 0.15
    complexity_penalty: float = 0.015


def _folds(y, requested: int, seed: int):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    n = max(2, min(int(requested), int(counts.min())))
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=int(seed)), n


def _objective(model, X_val, y_val, oracle, cfg: ControlledTrepanConfig, search: ScientificTrepanSearchConfig):
    pred = np.asarray(model.predict(X_val))
    teacher = np.asarray(oracle.predict(X_val))
    fidelity = float(accuracy_score(teacher, pred))
    ba = float(balanced_accuracy_score(y_val, pred))
    f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
    complexity = float(getattr(model, "node_count_", 0)) / max(1.0, float(cfg.max_nodes))
    score = (
        search.fidelity_weight * fidelity
        + search.balanced_accuracy_weight * ba
        + search.macro_f1_weight * f1
        - search.complexity_penalty * complexity
    )
    return {"score": float(score), "fidelity": fidelity, "balanced_accuracy": ba, "macro_f1": f1, "complexity": complexity}


def _capacity_candidates(base: ControlledTrepanConfig, p: int, limit: int):
    def cfg(**kw):
        c = replace(base, **kw)
        # Mais nós permitidos exigem mais queries: sem isto o candidato "maior" ficava truncado pelo orçamento.
        need = required_query_budget(c.max_nodes, c.min_sample)
        return replace(c, max_queries=max(int(c.max_queries), need)) if c.max_nodes != base.max_nodes else c
    q2 = max(int(base.max_queries), min(12000, max(int(base.max_queries) * 2, 1000)))
    nodes2 = max(int(base.max_nodes), min(127, max(31, int(base.max_nodes) * 2 - 1)))
    depth2 = max(int(base.max_depth), min(nodes2, max(int(base.max_depth), 12)))
    n2 = min(5, max(2, int(base.max_n) + 1))
    f2 = min(int(p), max(int(base.max_features_per_node), min(32, max(12, int(base.max_features_per_node) * 2))))
    candidates = [
        base,
        cfg(max_nodes=nodes2, max_depth=depth2),
        cfg(max_n=n2),
        cfg(max_features_per_node=f2),
        cfg(max_queries=q2, min_sample=max(int(base.min_sample), min(2000, max(200, int(base.min_sample) * 2)))),
        cfg(max_nodes=nodes2, max_depth=depth2, max_n=n2, max_features_per_node=f2, max_queries=q2),
    ]
    out=[]; seen=set()
    for c in candidates:
        key=(c.max_nodes,c.max_depth,c.max_n,c.max_features_per_node,c.max_queries,c.min_sample,c.beam_width)
        if key not in seen:
            seen.add(key); out.append(c)
    return out[:max(1,int(limit))]


def _semantic_candidates(base: ControlledTrepanConfig, limit: int):
    # (ganho_semântico, coerência, fracção_active_query, budget_candidatos,
    #  força_EFSR, min_disagreement, min_ganho_fidelidade_local)
    presets = [
        (0.50, 0.10, 0.35, 12, 0.75, 0.08, 0.004),
        (1.00, 0.15, 0.50, 24, 1.00, 0.05, 0.002),
        (1.50, 0.20, 0.65, 32, 1.25, 0.05, 0.002),
        (2.00, 0.25, 0.75, 40, 1.50, 0.03, 0.001),
        (1.25, 0.35, 0.80, 48, 1.75, 0.03, 0.001),
        (2.50, 0.35, 0.85, 48, 2.00, 0.02, 0.001),
    ]
    return [
        replace(
            base,
            semantic_gain_strength=g,
            semantic_group_strength=grp,
            semantic_active_query_fraction=active,
            semantic_candidate_budget=budget,
            error_focused_refinement=True,
            error_focus_strength=focus_strength,
            error_focus_min_disagreement=min_disagreement,
            error_focus_min_local_fidelity_gain=min_local_gain,
        )
        for g,grp,active,budget,focus_strength,min_disagreement,min_local_gain in presets[:max(1,int(limit))]
    ]


def _mean_metrics(rows):
    keys=("score","fidelity","balanced_accuracy","macro_f1","complexity")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def tune_scientific_trepan(
    X_train,
    y_train,
    *,
    oracle,
    feature_names: Sequence[str],
    base_config: ControlledTrepanConfig,
    semantic_feature_weights=None,
    semantic_feature_groups=None,
    semantic_relatedness_matrix=None,
    query_projector=None,
    search: ScientificTrepanSearchConfig = ScientificTrepanSearchConfig(),
    ontology_graph=None,
    semantic_feature_entities=None,
) -> dict[str, Any]:
    X=np.asarray(X_train,dtype=float); y=np.asarray(y_train)
    if X.ndim != 2 or len(X)!=len(y):
        raise ValueError("X_train/y_train inválidos para tuning científico TREPAN.")
    cv, folds = _folds(y, search.cv_folds, base_config.random_state)
    splits=list(cv.split(X,y))

    capacity_history=[]
    for candidate in _capacity_candidates(base_config, X.shape[1], search.max_capacity_candidates):
        rows=[]; failed=None
        for tr,va in splits:
            try:
                model=TrepanOriginalClassifier(**candidate.common_tree_kwargs()).fit(
                    X[tr], oracle=oracle, feature_names=feature_names
                )
                rows.append(_objective(model,X[va],y[va],oracle,candidate,search))
            except (ValueError, RuntimeError) as exc:
                failed=str(exc); break
        if rows and failed is None:
            mean=_mean_metrics(rows)
        else:
            mean={"score":float("-inf"),"fidelity":0.0,"balanced_accuracy":0.0,"macro_f1":0.0,"complexity":1.0}
        capacity_history.append({"config":asdict(candidate),"mean":mean,"failed":failed})

    valid=[r for r in capacity_history if np.isfinite(r["mean"]["score"])]
    if not valid:
        common=base_config
    else:
        # Preferir a configuração mais simples que alcance a meta de fidelidade;
        # caso contrário, maior score composto.
        target=[r for r in valid if r["mean"]["fidelity"] >= search.fidelity_target]
        if target:
            chosen=min(target,key=lambda r:(r["config"]["max_nodes"],r["config"]["max_queries"],-r["mean"]["score"]))
        else:
            chosen=max(valid,key=lambda r:r["mean"]["score"])
        common=ControlledTrepanConfig(**chosen["config"])

    semantic_history=[]
    weights=np.ones(X.shape[1],dtype=float) if semantic_feature_weights is None else np.asarray(semantic_feature_weights,dtype=float)
    for candidate in _semantic_candidates(common, search.max_semantic_candidates):
        rows=[]; failed=None; usage=[]
        for tr,va in splits:
            try:
                model=TrepanReloadedClassifier(
                    **candidate.common_tree_kwargs(),
                    semantic_gain_strength=candidate.semantic_gain_strength,
                    semantic_group_strength=candidate.semantic_group_strength,
                    semantic_candidate_budget=candidate.semantic_candidate_budget,
                    semantic_relation_threshold=candidate.semantic_relation_threshold,
                    semantic_active_query_fraction=candidate.semantic_active_query_fraction,
                    semantic_active_pool_multiplier=candidate.semantic_active_pool_multiplier,
                    error_focused_refinement=candidate.error_focused_refinement,
                    error_focus_min_disagreement=candidate.error_focus_min_disagreement,
                    error_focus_strength=candidate.error_focus_strength,
                    error_focus_semantic_weight=candidate.error_focus_semantic_weight,
                    error_focus_uncertainty_weight=candidate.error_focus_uncertainty_weight,
                    error_focus_min_local_fidelity_gain=candidate.error_focus_min_local_fidelity_gain,
                    error_focus_min_real_fidelity_gain=candidate.error_focus_min_real_fidelity_gain,
                    error_focus_anchor_k=candidate.error_focus_anchor_k,
                    error_focus_top_k=candidate.error_focus_top_k,
                    error_focus_min_regions=candidate.error_focus_min_regions,
                    mirror_when_no_semantic_effect=candidate.mirror_when_no_semantic_effect,
                    alpha=candidate.alpha,
                    beta=candidate.beta,
                    gain_criterion=candidate.gain_criterion,
                    semantic_query_projection=candidate.semantic_query_projection,
                    error_focus_fidelity_tolerance=candidate.error_focus_fidelity_tolerance,
                ).fit(
                    X[tr], oracle=oracle, feature_names=feature_names,
                    semantic_feature_weights=weights,
                    semantic_feature_groups=semantic_feature_groups,
                    semantic_relatedness_matrix=semantic_relatedness_matrix,
                    query_projector=query_projector,
                    ontology_graph=ontology_graph,
                    semantic_feature_entities=semantic_feature_entities,
                )
                rows.append(_objective(model,X[va],y[va],oracle,candidate,search))
                usage.append(float(model.semantic_audit_summary_.get("ontology_usage_rate",0.0)))
            except (ValueError, RuntimeError) as exc:
                failed=str(exc); break
        if rows and failed is None:
            mean=_mean_metrics(rows); mean["ontology_usage_rate"]=float(np.mean(usage)) if usage else 0.0
            # pequeno desempate a favor de semântica realmente utilizada, sem
            # substituir fidelidade/performance pelo simples uso da ontologia.
            mean["selection_score"] = float(mean["score"] + 0.01 * mean["ontology_usage_rate"])
        else:
            mean={"score":float("-inf"),"selection_score":float("-inf"),"fidelity":0.0,"balanced_accuracy":0.0,"macro_f1":0.0,"complexity":1.0,"ontology_usage_rate":0.0}
        semantic_history.append({"config":asdict(candidate),"mean":mean,"failed":failed})
    valid_sem=[r for r in semantic_history if np.isfinite(r["mean"]["selection_score"])]
    selected=ControlledTrepanConfig(**max(valid_sem,key=lambda r:r["mean"]["selection_score"])["config"]) if valid_sem else common
    return {
        "selection_scope":"training_cv_only",
        "test_used_for_selection":False,
        "cv_folds":int(folds),
        "search_config":asdict(search),
        "common_capacity":asdict(common),
        "selected_config":asdict(selected),
        "capacity_history":capacity_history,
        "semantic_history":semantic_history,
    }


__all__=["ScientificTrepanSearchConfig","tune_scientific_trepan"]
