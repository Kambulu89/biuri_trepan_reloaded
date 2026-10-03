"""Ponte entre a sessão da GUI/serviço e o cfkit.

* usa APENAS o dataset activo e o modelo seleccionado (``model_explained`` explícito);
* constraints, densidade e escalas são aprendidas na partição de TREINO quando disponível;
* categóricas LabelEncoder -> kind 'code' com rótulos humanos; one-hot detectado só com prova no treino;
* features derivadas (onto_*) nunca são alteradas directamente (recalculadas por ``derive_fn`` ou congeladas);
* constraints OWL são usadas para plausibilidade MESMO que o enriquecimento preditivo do MLP tenha sido rejeitado.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from counterfactuals.cfkit.api import CFContext, explain_text, generate_counterfactual, local_robustness
from counterfactuals.cfkit.result import CounterfactualResult, CounterfactualStatus
from counterfactuals.cfkit.rules import ConstraintSet, OntologyConstraintExtractor

LEGACY_METHOD_MAP = {"AUTO": "AUTO", "CLEAR": "CLEAR", "COGS": "COGS", "LORE-LOCAL": "LORE", "LORE": "LORE", "LORE-GLOBAL": "TREE", "TREE": "TREE"}
CFKIT_METHODS = tuple(LEGACY_METHOD_MAP)


def _ontology_hash(ontology: Any) -> str:
    if ontology is None:
        return "none"
    names = []
    for accessor in ("classes", "data_properties", "object_properties"):
        fn = getattr(ontology, accessor, None)
        if callable(fn):
            try:
                names += sorted(str(getattr(e, "name", e)) for e in fn())
            except Exception:
                pass
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def constraint_config(session: Mapping[str, Any], names: List[str], X_fit: np.ndarray) -> Dict[str, Any]:
    cfg = dict(session.get("constraints") or session.get("config") or {})
    features = {str(k): dict(v) for k, v in (cfg.get("features") or {}).items()}
    codes = dict(cfg.get("categorical_values") or {})
    for name, info in (session.get("feature_encoders_info") or {}).items():
        if name in names:
            labels = list(info.get("labels") or [])
            features[name] = {**features.get(name, {}), "kind": "code", "labels": {float(i): str(l) for i, l in enumerate(labels)}}
            codes[name] = tuple(float(i) for i in range(len(labels)))
    if features:
        cfg["features"] = features
    if codes:
        cfg["categorical_values"] = codes
    cfg.setdefault("auto_detect_onehot", True)
    original = set(session.get("feature_names_original") or [])
    derived = [n for n in names if n.startswith("onto_") and n not in original]     # derivadas: nunca alteradas directamente
    if derived:
        cfg["derived_features"] = list(set(cfg.get("derived_features") or []) | set(derived))
    if session.get("augmented_derive_fn") is not None and derived:
        cfg["derive_fn"] = session["augmented_derive_fn"]
    return cfg


def build_cf_context(session: Mapping[str, Any], context: Mapping[str, Any], options: Mapping[str, Any]) -> Tuple[CFContext, List[str]]:
    notes: List[str] = []
    names = list(context["feature_names"])
    augmented = bool(context.get("ontology_active"))
    X_fit = session.get("X_cf_train_augmented" if augmented else "X_cf_train_original")
    if X_fit is None or np.asarray(X_fit).ndim != 2 or np.asarray(X_fit).shape[1] != len(names):
        X_fit = context["X"]
        notes.append("partição de treino indisponível: constraints/densidade aprendidas nos dados carregados (podem incluir linhas de teste)")
        fit_source = "loaded_data_no_split"
    else:
        X_fit = np.asarray(X_fit, dtype=float)
        fit_source = "train_partition"
    X_fit = np.asarray(X_fit, dtype=float)
    cfg = constraint_config(session, names, X_fit)
    ontology = session.get("ontology") if bool(options.get("validate_ontology", True)) else None
    extracted = None
    if ontology is not None:
        extracted = OntologyConstraintExtractor(ontology, names, session.get("ontology_feature_mapping")).extract()
    constraints = ConstraintSet.from_config(X_fit, names, cfg, extracted)
    ctx = CFContext.build(context["oracle"], X_fit, names, constraints=constraints, model_name=str(context["target_model"]),
                          dataset_name=str(session.get("dataset_name", "dataset_carregado")), class_labels=session.get("class_labels"),
                          ontology_hash=_ontology_hash(ontology), seed=int(options.get("seed", 42)))
    ctx.constraints.ontology_notes.append(f"referência de treino: {fit_source}")
    return ctx, notes


def _aggregate(cands: List[Dict[str, Any]], diversity: Mapping[str, Any]) -> Dict[str, Any]:
    if not cands:
        return {"validity": 0.0, "proximity": None, "sparsity": None, "diversity": None, "stability": None, "robustness": None, "plausibility": 0.0,
                "actionability": None, "ontology_consistency": None, "causal_consistency": None, "global_fidelity": None}

    def mean(key):
        vals = [c["metrics"][key] for c in cands if c["metrics"].get(key) is not None]
        return float(np.mean(vals)) if vals else None
    sets = [set(ch["feature"] for ch in c["changes"]) for c in cands]
    jac = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            jac.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
    return {"validity": mean("validity"), "proximity": mean("proximity"), "sparsity": mean("sparsity"), "diversity": diversity.get("mean_pairwise") or 0.0,
            "stability": float(np.mean(jac)) if jac else 1.0, "robustness": mean("robustness"), "plausibility": mean("plausibility"),
            "plausibility_score": mean("plausibility_score"), "actionability": mean("actionability"), "ontology_consistency": mean("ontology_consistency"),
            "causal_consistency": None, "coverage": mean("coverage"), "local_fidelity": mean("local_fidelity"), "global_fidelity": None}


def to_engine_dict(result: CounterfactualResult, ctx: CFContext, instance: np.ndarray, *, robustness_samples: int, robustness_epsilon: float, seed: int) -> Dict[str, Any]:
    """Resultado cfkit no formato consumido pela GUI/avaliação/árvore CF (e campos novos auditáveis)."""
    cands: List[Dict[str, Any]] = []
    for k, c in enumerate(result.candidates):
        vec = np.asarray(c.vector)
        rob = local_robustness(ctx, instance, vec, result.target_class, n=robustness_samples, epsilon=robustness_epsilon, seed=seed + k)
        n_changes = max(len(c.changes), 1)
        meta = dict(c.metadata)
        cands.append({
            "method": result.method_label if result.method != "AUTO" else c.method, "prediction": c.predicted_class, "vector": c.vector,
            "changes": [{**ch, "delta": ch["delta"]} for ch in c.changes], "human": c.human, "rule": c.rule or None,
            "metrics": {"validity": bool(c.model_valid), "model_valid": bool(c.model_valid), "semantic_valid": c.semantic_valid,
                        "proximity": c.proximity, "sparsity": c.sparsity, "plausibility": bool(c.plausible) if c.plausible is not None else None,
                        "plausibility_score": c.plausibility, "actionability": sum(1 for ch in c.changes if ch["actionable"]) / n_changes if c.changes else 1.0,
                        "causal_consistency": None, "ontology_consistency": (None if c.semantic_valid is None else (1.0 if c.semantic_valid else 0.0)),
                        "robustness": float(rob["score"]), "coverage": meta.get("support"), "local_fidelity": meta.get("local_fidelity"),
                        "global_fidelity": None, "original_probability": c.original_probability, "counterfactual_probability": c.counterfactual_probability},
            "domain_validation": {"plausible": not c.hard_violations, "violations": c.hard_violations, "changed_features": [ch["feature"] for ch in c.changes]},
            "ontology_validation": {"applicable": c.semantic_status != "NOT_AVAILABLE", "status": c.semantic_status, "consistent": c.semantic_valid,
                                    "warnings": c.soft_warnings},
            "robustness_details": rob, "metadata": meta, "cross_model": c.cross_model, "dominated_by": c.dominated_by})
    best = cands[0] if cands else None
    return {
        "status": result.status.value, "status_legacy": "success" if result.status == CounterfactualStatus.SUCCESS else result.status.value.lower(),
        "message": result.message, "method": result.method, "methods_executed": [result.method], "method_label": result.method_label,
        "canonical": result.canonical, "factual_prediction": result.original_class, "desired_class": result.target_class,
        "target_class": result.target_class, "original_instance": list(result.original_instance), "original_human": result.original_human,
        "feature_names": list(ctx.feature_names), "candidates": cands, "best_candidate": best, "aggregate_metrics": _aggregate(cands, result.diversity),
        "rst_feature_scores": {}, "narrative": result.explanation, "explanation": result.explanation, "warnings": list(result.warnings),
        "model_explained": result.model_explained, "semantic_validation": result.semantic_validation, "runtime": result.runtime, "seed": result.seed,
        "iterations": result.iterations, "terminated_by": result.terminated_by, "diversity": result.diversity, "rejected": result.rejected,
        "diagnostics": result.diagnostics, "provenance": result.provenance, "pipeline": "cfkit",
    }


def generate_with_cfkit(session: Mapping[str, Any], context: Mapping[str, Any], options: Mapping[str, Any], index: int, desired: Any) -> Dict[str, Any]:
    ctx, notes = build_cf_context(session, context, options)
    method = LEGACY_METHOD_MAP[str(options.get("method", "AUTO")).upper().replace("_", "-")]
    seed = int(options.get("seed", 42))
    tree = context.get("global_tree") if method in ("TREE",) or context.get("model_type") == "tree" else None
    X = np.asarray(context["X"], dtype=float)
    result = generate_counterfactual(
        X[index], desired, context["oracle"], method, context=ctx, random_state=seed, n_cfs=int(options.get("total_cfs", 5)),
        max_iterations=int(options.get("max_iterations", 40)), max_time=options.get("max_time", 30.0), instance_id=int(index),
        tree_model=tree if method == "TREE" else None, other_models=options.get("other_models"), require_plausible=bool(options.get("require_plausible", False)),
        local_agreement_samples=int(options.get("local_agreement_samples", 0)))
    result.warnings.extend(notes)
    out = to_engine_dict(result, ctx, X[index], robustness_samples=int(options.get("robustness_samples", 100)),
                         robustness_epsilon=float(options.get("robustness_epsilon", 0.02)), seed=seed)
    out["warnings"] = list(result.warnings)
    return out
