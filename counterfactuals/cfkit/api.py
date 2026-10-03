"""API unificada de contrafactuais (LORE-/CLEAR-/CoGS-inspired + tree-path), com validação obrigatória no modelo."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from counterfactuals.cfkit import methods as M
from counterfactuals.cfkit.metrics import DensityModel, GowerMetric, diversity, drop_near_duplicates, mark_dominance
from counterfactuals.cfkit.result import CounterfactualCandidate, CounterfactualResult, CounterfactualStatus
from counterfactuals.cfkit.rules import ConstraintSet, OntologyConstraintExtractor
from counterfactuals.cfkit.schema import EPS, FeatureSchema

PIPELINE_VERSION = "cfkit-1.0"
LOG = logging.getLogger("cfkit")


class CounterfactualRequestError(ValueError):
    """Pedido inválido (por exemplo, contexto inconsistente)."""


# ----------------------------------------------------------------------------- contexto
def _model_n_features(model: Any) -> Optional[int]:
    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        value = getattr(obj, "n_features_in_", None)
        if value is not None:
            return int(value)
        tree = getattr(obj, "tree_", None)
        if tree is not None and getattr(tree, "n_features", None) is not None:
            return int(tree.n_features)
    return None


def is_tree_model(model: Any) -> bool:
    for cand in (model, getattr(model, "tree_model", None), getattr(model, "explainer_tree", None)):
        if cand is not None and (hasattr(cand, "tree_") or (hasattr(cand, "root_") and hasattr(cand, "export_rules"))):
            return True
    return False


def model_fingerprint(model: Any, X_ref: np.ndarray, k: int = 31) -> str:
    """Hash do comportamento do modelo (predições numa amostra fixa) + classe + hiperparâmetros."""
    X = np.asarray(X_ref, dtype=float)
    idx = np.linspace(0, len(X) - 1, min(k, len(X))).astype(int)
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    payload = {"class": f"{type(model).__module__}.{type(model).__qualname__}", "pred": np.asarray(model.predict(X[idx])).astype(str).tolist(),
               "params": {k_: str(v) for k_, v in params.items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class CFContext:
    """Tudo o que é ajustado SÓ no treino (constraints, escala Gower, densidade, doadores)."""

    model: M.ModelAdapter
    model_name: str
    dataset_name: str
    X_train: np.ndarray
    feature_names: List[str]
    constraints: ConstraintSet
    metric: GowerMetric
    density: DensityModel
    weights: M.ObjectiveWeights
    class_labels: Dict[Any, str] = field(default_factory=dict)
    ontology_hash: str = "none"
    fingerprint: str = ""

    @classmethod
    def build(cls, model: Any, X_train, feature_names: Sequence[str], *, constraints: Optional[ConstraintSet] = None,
              config: Optional[Mapping[str, Any]] = None, model_name: str = "modelo", dataset_name: str = "dataset_carregado",
              class_labels: Optional[Mapping[Any, str]] = None, ontology: Any = None, ontology_mapping: Optional[Mapping[str, str]] = None,
              ontology_hash: str = "none", weights: Optional[M.ObjectiveWeights] = None, density_k: int = 5, density_quantile: float = 0.95,
              seed: int = 0) -> "CFContext":
        X = np.asarray(X_train, dtype=float)
        names = [str(n) for n in feature_names]
        if X.ndim != 2 or X.shape[1] != len(names):
            raise CounterfactualRequestError("X_train e feature_names são incompatíveis.")
        if not np.isfinite(X).all():
            raise CounterfactualRequestError("X_train contém NaN/Inf.")
        n_model = _model_n_features(model)
        if n_model is not None and n_model != X.shape[1]:
            raise CounterfactualRequestError(f"O modelo espera {n_model} features; o espaço contrafactual tem {X.shape[1]}.")
        if constraints is None:
            extracted = OntologyConstraintExtractor(ontology, names, ontology_mapping).extract() if ontology is not None else None
            constraints = ConstraintSet.from_config(X, names, config, extracted)
        elif constraints.schema.names != names:
            raise CounterfactualRequestError("As constraints pertencem a um schema de colunas diferente.")
        metric = GowerMetric(constraints.schema).fit(X)
        density = DensityModel(metric, k=density_k, quantile=density_quantile, seed=seed).fit(X)
        adapter = M.ModelAdapter(model, model_name)
        return cls(adapter, model_name, dataset_name, X, names, constraints, metric, density, weights or M.ObjectiveWeights(),
                   {k: str(v) for k, v in (class_labels or {}).items()}, ontology_hash, model_fingerprint(model, X))


# ----------------------------------------------------------------------------- alvo
def resolve_target(classes: Optional[np.ndarray], original_class: Any, target: Any) -> Tuple[Any, str, List[Any]]:
    """Devolve (alvo, erro, opções). Multiclasse nunca assume 'classe oposta'."""
    if classes is None or len(classes) == 0:
        return None, "o modelo não expõe classes_: indique target_class explícita e verifique-a no modelo", []
    labels = list(classes.tolist())
    options = [c for c in labels if str(c) != str(original_class)]
    if target is None or str(target).lower() in ("opposite", "oposta", "opuesta"):
        if len(labels) == 2:
            return options[0], "", options
        return None, ("classificação multiclasse: indique a classe alvo explicitamente. "
                      f"Opções válidas: {options}"), options
    for c in labels:
        if str(c) == str(target):
            if str(c) == str(original_class):
                return None, f"a instância já é prevista como {c}; escolha outra classe. Opções: {options}", options
            return c, "", options
    return None, f"classe alvo {target!r} inexistente. Classes do modelo: {labels}", options


# ----------------------------------------------------------------------------- cache / provenance
def cache_key(ctx: CFContext, instance, target, method: str, seed: int, options: Mapping[str, Any]) -> str:
    payload = {"model": ctx.fingerprint, "instance": np.asarray(instance, dtype=float).round(12).tolist(), "target": str(target),
               "method": method, "constraints": ctx.constraints.provenance(), "seed": int(seed), "ontology": ctx.ontology_hash,
               "pipeline": PIPELINE_VERSION, "weights": ctx.weights.to_dict(), "options": {k: str(v) for k, v in sorted(options.items())}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class CFCache:
    def __init__(self):
        self._store: Dict[str, CounterfactualResult] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[CounterfactualResult]:
        if key in self._store:
            self.hits += 1
            return copy.deepcopy(self._store[key])
        self.misses += 1
        return None

    def put(self, key: str, result: CounterfactualResult) -> None:
        self._store[key] = copy.deepcopy(result)


# ----------------------------------------------------------------------------- texto determinístico
CAUSAL_DISCLAIMER = {"pt": "Isto descreve o comportamento do modelo, não uma relação causal nem uma garantia sobre o mundo real.",
                     "es": "Esto describe el comportamiento del modelo, no una relación causal ni una garantía sobre el mundo real."}


def _fmt_value(v: Any) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def format_change(change: Mapping[str, Any]) -> str:
    before, after = _fmt_value(change["original"]), _fmt_value(change["counterfactual"])
    if change.get("delta") is not None and change["kind"] in ("continuous", "integer"):
        return f"{change['feature']}: {before} → {after} ({change['delta']:+.6g})"
    return f"{change['feature']}: {before} → {after}"


def explain_text(result: CounterfactualResult, lang: str = "pt") -> str:
    """Texto determinístico. Nunca afirma causalidade; diz o que o MODELO passaria a prever."""
    orig, tgt, model = result.original_class, result.target_class, result.model_explained
    if result.status != CounterfactualStatus.SUCCESS or result.best is None:
        return f"Não foi encontrado um contrafactual válido para {model} mudar a previsão de {orig} para {tgt}: [{result.status.value}] {result.message}"
    best = result.best
    parts = [format_change(c) for c in best.changes]
    text = (f"Para o modelo {model} prever {tgt} em vez de {orig}, o contrafactual encontrado ({result.method_label or result.method}) "
            f"altera {len(parts)} característica(s): " + "; ".join(parts) + f". Com estas alterações o modelo passaria a prever {best.predicted_class}.")
    if best.soft_warnings:
        text += " Avisos (restrições soft): " + "; ".join(best.soft_warnings) + "."
    if not result.canonical:
        text += f" Método {result.method_label} é uma aproximação inspirada no método original (não a implementação canónica)."
    return text + " " + CAUSAL_DISCLAIMER.get(lang, CAUSAL_DISCLAIMER["pt"])


def candidate_table(result: CounterfactualResult, candidate_index: int = 0, show_all: bool = False) -> List[Dict[str, Any]]:
    """Por defeito só as features alteradas; ``show_all=True`` mostra todas (espaço humano)."""
    cand = result.candidates[candidate_index]
    changed = {c["feature"]: c for c in cand.changes}
    rows = []
    for name, original in result.original_human.items():
        if name in changed:
            c = changed[name]
            rows.append({"feature": name, "original": c["original"], "counterfactual": c["counterfactual"], "delta": c["delta"], "changed": True})
        elif show_all:
            rows.append({"feature": name, "original": original, "counterfactual": cand.human.get(name, original), "delta": None, "changed": False})
    return rows


# ----------------------------------------------------------------------------- núcleo
def _canonical_method(method: str) -> str:
    key = str(method or "AUTO").upper().replace("_", "-")
    aliases = {"LORE-LOCAL": "LORE", "LORE-INSPIRED": "LORE", "CLEAR-INSPIRED": "CLEAR", "COGS-INSPIRED": "COGS", "TREE-PATH": "TREE", "LORE-GLOBAL": "TREE"}
    return aliases.get(key, key)


def _fail(ctx: CFContext, method: str, status: CounterfactualStatus, message: str, *, instance, original_class=None, target=None,
          seed=None, instance_id=None, runtime=0.0, warnings=None, canonical=False) -> CounterfactualResult:
    inst = np.asarray(instance, dtype=float).reshape(-1) if instance is not None else np.array([])
    try:
        human = ctx.constraints.schema.decode(inst) if len(inst) == ctx.constraints.schema.n_features and np.isfinite(inst).all() else {}
    except Exception:
        human = {}
    info = M.METHOD_REGISTRY.get(method, {})
    res = CounterfactualResult(method=method, status=status, model_explained=ctx.model_name, dataset=ctx.dataset_name,
                               original_instance=inst.tolist(), original_human=human, original_class=original_class, target_class=target,
                               predicted_class=None, runtime=runtime, seed=seed, warnings=list(warnings or []), message=message,
                               instance_id=instance_id, method_label=info.get("label", method), canonical=bool(info.get("canonical", False)))
    res.explanation = explain_text(res)
    res.provenance = _provenance(ctx, res, {})
    LOG.info("cfkit method=%s instance=%s status=%s target=%s seed=%s", method, instance_id, status.value, target, seed)
    return res


def _provenance(ctx: CFContext, res: CounterfactualResult, options: Mapping[str, Any]) -> Dict[str, Any]:
    return {"pipeline_version": PIPELINE_VERSION, "model_fingerprint": ctx.fingerprint, "dataset": ctx.dataset_name, "ontology_hash": ctx.ontology_hash,
            "constraints": ctx.constraints.provenance(), "objective_weights": ctx.weights.to_dict(), "options": {k: str(v) for k, v in options.items()},
            "method_registry": M.METHOD_REGISTRY.get(res.method, {}),
            "log": {"method": res.method, "instance_id": res.instance_id, "target": str(res.target_class), "seed": res.seed, "iterations": res.iterations,
                    "runtime": res.runtime, "status": res.status.value, "terminated_by": res.terminated_by},
            "ranges_learned_from": "X_train only", "causality": "NOT_CLAIMED: constraints semânticas não são causais"}


def _finalize(ctx: CFContext, instance: np.ndarray, orig_cls, target, raw: List[Dict[str, Any]], *, budget: M.Budget, n_cfs: int, sparsify_post: bool,
              require_plausible: bool, search_ctx: M.SearchContext, rejected: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reparação -> constraints hard -> RE-VALIDAÇÃO no modelo -> semântica -> métricas. Só entram CFs plenamente válidos."""
    cs, schema = ctx.constraints, ctx.constraints.schema
    seen, prepared = set(), []
    for item in raw:
        v = cs.repair(instance, item["vector"])
        key = tuple(np.round(v, 9))
        if key in seen:
            continue
        seen.add(key)
        prepared.append({**item, "vector": v})
    diagnostics["raw_candidates"] = len(raw)
    diagnostics["unique_candidates"] = len(prepared)
    cap = max(12, n_cfs * 8)                      # custo limitado: só os melhores candidatos (ordem do gerador) são validados/sparsificados
    if len(prepared) > cap:
        diagnostics["candidates_truncated_to"] = cap
        prepared = prepared[:cap]
    blocked_by_search = int(search_ctx.stats.get("hard_blocked", 0))
    for v in search_ctx.blocked_examples:
        if len(rejected) < 25:
            rejected.append({"reason": "violação hard (barrado durante a pesquisa)", "violations": [h["message"] for h in cs.hard_violations(instance, v)],
                             "human": schema.decode(v)})
    if not prepared:
        diagnostics.update(rejected_model_invalid=0, rejected_hard_constraints=blocked_by_search)
        if blocked_by_search:
            diagnostics["only_hard_constraint_blockers"] = True
        return []
    V = np.asarray([p["vector"] for p in prepared])
    preds = ctx.model.predict(V)                       # RE-VALIDAÇÃO OBRIGATÓRIA (Parte 19)
    out = []
    n_model_invalid = n_hard = 0
    for p, pred in zip(prepared, preds):
        v = p["vector"]
        hard = cs.hard_violations(instance, v)
        if pred != target:
            n_model_invalid += 1
            continue
        if hard:
            n_hard += 1
            if len(rejected) < 25:
                rejected.append({"reason": "violação hard", "violations": [h["message"] for h in hard], "human": schema.decode(v)})
            continue
        if sparsify_post:
            v2 = M.sparsify(search_ctx, v)
            if ctx.model.predict(v2)[0] == target and not cs.hard_violations(instance, v2):
                v = v2
        out.append({**p, "vector": v})
    diagnostics.update(rejected_model_invalid=n_model_invalid, rejected_hard_constraints=n_hard + blocked_by_search)
    if (n_hard or blocked_by_search) and not out:
        diagnostics["only_hard_constraint_blockers"] = True
    final = []
    for p in out:
        v = p["vector"]
        pred = ctx.model.predict(v)[0]
        sem = cs.semantic_validation(instance, v)
        dens = ctx.density.score(v)
        prox = ctx.metric.distance(instance, v)
        probs_o = ctx.model.proba(instance)
        probs_c = ctx.model.proba(v)
        p_orig = p_cf = p_orig_target = None
        if probs_o is not None and ctx.model.classes is not None and probs_o.shape[1] == len(ctx.model.classes):
            ci = {str(c): i for i, c in enumerate(ctx.model.classes)}
            p_orig = float(probs_o[0, ci[str(orig_cls)]])                 # P(classe original | x)
            p_orig_target = float(probs_o[0, ci[str(target)]])             # P(alvo | x)
            p_cf = float(probs_c[0, ci[str(target)]])                      # P(alvo | cf)
        if require_plausible and dens.get("plausible") is False:
            if len(rejected) < 25:
                rejected.append({"reason": "implausível (kNN)", "human": schema.decode(v)})
            continue
        soft_msgs, _ = cs.soft_violations(instance, v)
        meta = dict(p["metadata"])
        meta["density"] = dens
        final.append({"vector": v, "human": schema.decode(v), "changes": schema.describe_change(instance, v), "predicted": pred,
                      "model_valid": bool(pred == target), "semantic_valid": sem["valid"], "semantic_status": sem["status"],
                      "hard": [h["message"] for h in cs.hard_violations(instance, v)], "soft": sem["soft_warnings"] or soft_msgs,
                      "proximity": prox, "sparsity": len(schema.changed_units(instance, v)), "plausibility": dens.get("plausibility"),
                      "plausible": dens.get("plausible"), "p_orig": p_orig, "p_cf": p_cf, "p_orig_target": p_orig_target,
                      "cost_weighted": ctx.metric.distance(instance, v, weighted=True), "method": p["method"], "rule": meta.get("rule"), "metadata": meta})
    return final


def generate_counterfactual(instance, target_class, model, method: str, constraints: Optional[ConstraintSet] = None, random_state: int = 0, *,
                            context: Optional[CFContext] = None, X_train=None, feature_names: Optional[Sequence[str]] = None, config=None,
                            n_cfs: int = 3, max_iterations: int = 40, max_time: Optional[float] = None, include_rejected: bool = True,
                            keep_dominated: bool = True, other_models: Optional[Mapping[str, Any]] = None, tree_model: Any = None,
                            instance_id: Any = None, cache: Optional[CFCache] = None, sparsify_post: bool = True, require_plausible: bool = False,
                            duplicate_tol: float = 1e-3, weights: Optional[M.ObjectiveWeights] = None, model_name: str = "modelo",
                            dataset_name: str = "dataset_carregado", class_labels=None, lore_surrogate: str = "c45", lang: str = "pt",
                            local_agreement_samples: int = 0, **extra) -> CounterfactualResult:
    """Interface comum: ``generate_counterfactual(instance, target_class, model, method, constraints, random_state)``.

    Devolve SEMPRE um ``CounterfactualResult`` com status explícito. Todo CF devolvido foi re-passado ao ``model``
    (``model.predict(cf) == target_class``) e cumpre as constraints hard."""
    t0 = time.perf_counter()
    seed = int(random_state)
    if context is None:
        if X_train is None or feature_names is None:
            raise CounterfactualRequestError("Forneça `context` (CFContext) ou X_train + feature_names.")
        context = CFContext.build(model, X_train, feature_names, constraints=constraints, config=config, model_name=model_name, dataset_name=dataset_name,
                                  class_labels=class_labels, weights=weights, seed=seed)
    elif constraints is not None and constraints is not context.constraints:
        context = copy.copy(context)
        context.constraints = constraints
        context.metric = GowerMetric(constraints.schema).fit(context.X_train)
        context.density = DensityModel(context.metric, seed=seed).fit(context.X_train)
    ctx = context
    ctx.model = M.ModelAdapter(model, ctx.model_name) if ctx.model.model is not model else ctx.model
    mkey = _canonical_method(method)
    options = dict(n_cfs=n_cfs, max_iterations=max_iterations, max_time=max_time, sparsify_post=sparsify_post, require_plausible=require_plausible,
                   lore_surrogate=lore_surrogate, duplicate_tol=duplicate_tol)
    schema = ctx.constraints.schema
    inst = np.asarray(instance, dtype=float).reshape(-1)
    # --- validação de entrada
    if mkey != "AUTO" and mkey not in M.SUPPORTED_METHODS:
        return _fail(ctx, mkey, CounterfactualStatus.METHOD_FAILURE, f"método desconhecido {method!r}; opções: {('AUTO',) + M.SUPPORTED_METHODS}", instance=inst, seed=seed, instance_id=instance_id)
    if len(inst) != schema.n_features or not np.isfinite(inst).all():
        why = "NaN/Inf na instância" if len(inst) == schema.n_features else f"instância com {len(inst)} valores; o modelo/schema usa {schema.n_features}"
        return _fail(ctx, mkey, CounterfactualStatus.UNSUPPORTED_FEATURE_SPACE, why, instance=inst, seed=seed, instance_id=instance_id)
    try:
        orig_cls = ctx.model.predict(inst)[0]
    except Exception as exc:
        return _fail(ctx, mkey, CounterfactualStatus.UNSUPPORTED_FEATURE_SPACE, f"o modelo não aceita a instância: {exc}", instance=inst, seed=seed, instance_id=instance_id)
    target, err, opts = resolve_target(ctx.model.classes, orig_cls, target_class)
    if err:
        res = _fail(ctx, mkey, CounterfactualStatus.INVALID_TARGET, err, instance=inst, original_class=orig_cls, target=target_class, seed=seed, instance_id=instance_id)
        res.diagnostics["valid_targets"] = [str(o) for o in opts]
        return res
    feasible, why = ctx.constraints.feasible()
    if not feasible:
        return _fail(ctx, mkey, CounterfactualStatus.CONSTRAINT_INFEASIBLE, why, instance=inst, original_class=orig_cls, target=target, seed=seed, instance_id=instance_id)
    tree = tree_model if tree_model is not None else (model if is_tree_model(model) else None)
    key = None
    if cache is not None:
        key = cache_key(ctx, inst, target, mkey, seed, options)
        cached = cache.get(key)
        if cached is not None:
            cached.provenance["cache"] = "hit"
            return cached
    methods_to_run = [mkey] if mkey != "AUTO" else (["TREE"] if tree is not None else []) + ["LORE", "CLEAR", "COGS"]
    if "TREE" in methods_to_run and tree is None:
        return _fail(ctx, mkey, CounterfactualStatus.UNSUPPORTED_FEATURE_SPACE, "TREE exige um modelo em árvore (C4.5/TREPAN/sklearn)", instance=inst, original_class=orig_cls, target=target, seed=seed, instance_id=instance_id)
    rng = np.random.default_rng(seed)
    budget = M.Budget(max_iterations=max_iterations, max_time=max_time)
    sctx = M.SearchContext(ctx.model, ctx.constraints, ctx.metric, ctx.density, ctx.X_train, inst, orig_cls, target, rng, budget, ctx.weights, n_wanted=n_cfs)
    raw: List[Dict[str, Any]] = []
    warnings: List[str] = []
    failures: List[str] = []
    for name in methods_to_run:
        try:
            if name == "TREE":
                raw += M.tree_path(sctx, tree, ctx.feature_names)
            elif name == "LORE":
                raw += M.lore_inspired(sctx, ctx.feature_names, surrogate=lore_surrogate)
            elif name == "CLEAR":
                raw += M.clear_inspired(sctx)
            elif name == "COGS":
                raw += M.cogs_inspired(sctx)
        except Exception as exc:  # falha de método nunca escapa sem diagnóstico
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if mkey == "AUTO":
            budget.iterations = 0                        # cada método do AUTO tem o seu orçamento de iterações
    warnings += sctx.notes
    diagnostics: Dict[str, Any] = {"method_stats": dict(sctx.stats), "model_calls": int(ctx.model.calls)}
    rejected: List[Dict[str, Any]] = []
    final = _finalize(ctx, inst, orig_cls, target, raw, budget=budget, n_cfs=n_cfs, sparsify_post=sparsify_post, require_plausible=require_plausible,
                      search_ctx=sctx, rejected=rejected, diagnostics=diagnostics)
    final, n_dup = drop_near_duplicates(ctx.metric, sorted(final, key=lambda c: (c["sparsity"], c["proximity"])), duplicate_tol)
    diagnostics["near_duplicates_removed"] = n_dup
    mark_dominance(final)
    # seleção diversa guloso (máx-mín distância) sem perder o melhor
    chosen: List[Dict[str, Any]] = []
    pool = list(final)
    if pool:
        chosen.append(pool.pop(0))
        while pool and len(chosen) < n_cfs:
            def gain(c):
                return min(ctx.metric.distance(c["vector"], k["vector"]) for k in chosen) - 0.2 * c["proximity"] - 0.02 * c["sparsity"] - (1e3 if c["dominated_by"] and not keep_dominated else 0)
            pool.sort(key=gain, reverse=True)
            chosen.append(pool.pop(0))
    chosen.sort(key=lambda c: (c["sparsity"], c["proximity"]))
    cands = []
    for c in chosen:
        cross = {}
        for name, other in (other_models or {}).items():
            try:
                other_pred = M.ModelAdapter(other, name).predict(c["vector"])[0]
                entry = {"predicted": other_pred.item() if isinstance(other_pred, np.generic) else other_pred, "target_achieved": bool(other_pred == target)}
                if local_agreement_samples > 0:
                    entry["local_agreement"] = _local_agreement(ctx, other, c["vector"], inst, rng, local_agreement_samples)
                cross[name] = entry
            except Exception as exc:
                cross[name] = {"error": f"{type(exc).__name__}: {exc}"}
        pred_py = c["predicted"].item() if isinstance(c["predicted"], np.generic) else c["predicted"]
        cands.append(CounterfactualCandidate(
            vector=c["vector"].tolist(), human=c["human"], changes=c["changes"], predicted_class=pred_py, model_valid=c["model_valid"],
            semantic_valid=c["semantic_valid"], semantic_status=c["semantic_status"], hard_violations=c["hard"], soft_warnings=c["soft"],
            proximity=c["proximity"], sparsity=c["sparsity"], plausibility=c["plausibility"], plausible=c["plausible"],
            original_probability=c["p_orig"], counterfactual_probability=c["p_cf"], original_target_probability=c["p_orig_target"], cost_weighted_distance=c["cost_weighted"], method=c["method"],
            rule=c["rule"], dominated_by=c["dominated_by"], cross_model=cross, metadata=c["metadata"]))
    runtime = time.perf_counter() - t0
    info = M.METHOD_REGISTRY.get(mkey, {"label": "AUTO (" + "+".join(methods_to_run) + ")", "canonical": False})
    orig_py = orig_cls.item() if isinstance(orig_cls, np.generic) else orig_cls
    tgt_py = target.item() if isinstance(target, np.generic) else target
    # --- estado explícito
    if cands:
        status, msg = CounterfactualStatus.SUCCESS, f"{len(cands)} contrafactual(is) validado(s) no modelo explicado"
    elif failures and not raw:
        status, msg = CounterfactualStatus.METHOD_FAILURE, "; ".join(failures)
    elif budget.reason == "max_time" or (max_time is not None and budget.elapsed() >= max_time):
        status, msg = CounterfactualStatus.TIMEOUT, f"orçamento de tempo ({max_time}s) esgotado sem CF válido"
    elif diagnostics.get("only_hard_constraint_blockers"):
        status, msg = CounterfactualStatus.CONSTRAINT_INFEASIBLE, "existem vectores que mudam a classe mas todos violam constraints hard"
    else:
        status, msg = CounterfactualStatus.NO_COUNTERFACTUAL_FOUND, f"nenhum CF válido encontrado (terminação: {budget.reason})"
    if status == CounterfactualStatus.NO_COUNTERFACTUAL_FOUND and budget.reason == "max_iterations":
        warnings.append("orçamento de iterações esgotado")
    if failures:
        warnings += failures
    best = cands[0] if cands else None
    res = CounterfactualResult(
        method=mkey, status=status, model_explained=ctx.model_name, dataset=ctx.dataset_name, original_instance=inst.tolist(),
        original_human=schema.decode(inst), original_class=orig_py, target_class=tgt_py, predicted_class=best.predicted_class if best else None,
        candidates=cands, changed_features=[c["feature"] for c in best.changes] if best else [], proximity=best.proximity if best else None,
        sparsity=best.sparsity if best else None, plausibility=best.plausibility if best else None,
        semantic_validation=best.semantic_status if best else ("NOT_AVAILABLE" if not ctx.constraints.semantic_available else "UNDETERMINED"),
        runtime=runtime, seed=seed, warnings=warnings, message=msg, instance_id=instance_id, iterations=budget.iterations, terminated_by=budget.reason,
        diversity=diversity(ctx.metric, [np.asarray(c.vector) for c in cands]), rejected=rejected if include_rejected else [], diagnostics=diagnostics,
        canonical=bool(info.get("canonical", False)), method_label=info.get("label", mkey))
    res.explanation = explain_text(res, lang)
    res.provenance = _provenance(ctx, res, options)
    if cache is not None and key is not None:
        res.provenance["cache"] = "miss"
        cache.put(key, res)
    LOG.info("cfkit method=%s instance=%s status=%s target=%s seed=%s iterations=%s runtime=%.3f", mkey, instance_id, status.value, tgt_py, seed, budget.iterations, runtime)
    return res


def local_robustness(ctx: CFContext, instance, vector, target, *, n: int = 100, epsilon: float = 0.02, seed: int = 0) -> Dict[str, Any]:
    """P(f(cf+δ)=alvo) com perturbações reproduzíveis sobre as features pesquisáveis, sempre reparadas pelo schema."""
    cols = ctx.constraints.schema.searchable_columns(ctx.constraints.nonactionable_policy)
    vec = np.asarray(vector, dtype=float)
    if not cols:
        ok = bool(ctx.model.predict(vec)[0] == target)
        return {"score": float(ok), "valid": int(ok), "total": 1, "epsilon": float(epsilon)}
    rng = np.random.default_rng(seed)
    scale = np.where(np.nanstd(ctx.X_train, axis=0) > EPS, np.nanstd(ctx.X_train, axis=0), 1.0)
    pts = []
    for _ in range(max(1, int(n))):
        v = vec.copy()
        v[cols] = v[cols] + rng.normal(0, epsilon, len(cols)) * scale[cols]
        pts.append(ctx.constraints.repair(np.asarray(instance, dtype=float), v))
    hits = int(np.sum(ctx.model.predict(np.asarray(pts)) == target))
    return {"score": hits / len(pts), "valid": hits, "total": len(pts), "epsilon": float(epsilon)}


def _local_agreement(ctx: CFContext, other: Any, cf: np.ndarray, original: np.ndarray, rng: np.random.Generator, n: int) -> float:
    """Concordância modelo explicado vs outro modelo numa vizinhança do CF (estudo de fidelidade local; não substitui a validação)."""
    cols = ctx.constraints.schema.searchable_columns(ctx.constraints.nonactionable_policy)
    scale = np.where(np.nanstd(ctx.X_train, axis=0) > EPS, np.nanstd(ctx.X_train, axis=0), 1.0)
    pts = []
    for _ in range(n):
        v = cf.copy()
        v[cols] = v[cols] + rng.normal(0, 0.05, len(cols)) * scale[cols]
        pts.append(ctx.constraints.repair(original, v))
    P = np.asarray(pts)
    return float(np.mean(ctx.model.predict(P) == M.ModelAdapter(other).predict(P)))


# ----------------------------------------------------------------------------- comparação / benchmark / export
def compare_methods(instance, target_class, model, methods: Sequence[str], *, context: CFContext, random_state: int = 0, **kw) -> Dict[str, Any]:
    """Executa os métodos na MESMA instância (mesma seed/orçamento). Sem ranking escondido: só métricas lado a lado."""
    results = {m: generate_counterfactual(instance, target_class, model, m, context=context, random_state=random_state, **kw) for m in methods}
    table = []
    for m, r in results.items():
        b = r.best
        table.append({"method": r.method_label, "status": r.status.value, "success": r.status == CounterfactualStatus.SUCCESS,
                      "proximity": r.proximity, "sparsity": r.sparsity, "plausibility": r.plausibility, "semantic_validation": r.semantic_validation,
                      "n_cfs": len(r.candidates), "runtime": r.runtime, "terminated_by": r.terminated_by, "canonical": r.canonical})
    return {"results": results, "table": table, "note": "sem ranking agregado: compare as métricas por objectivo"}


def select_instances(predictions: np.ndarray, protocol: str = "random", n: int = 20, seed: int = 0, indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Protocolo de selecção de instâncias DEFINIDO A PRIORI (nunca depende do sucesso de um método)."""
    N = len(predictions)
    rng = np.random.default_rng(seed)
    if protocol == "all":
        sel = list(range(N))
    elif protocol == "indices":
        sel = [int(i) for i in (indices or [])]
    elif protocol == "stratified":
        sel = []
        classes = np.unique(predictions)
        for c in classes:
            pool = np.where(predictions == c)[0]
            k = max(1, int(round(n * len(pool) / N)))
            sel += rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
    elif protocol == "random":
        sel = rng.choice(N, size=min(n, N), replace=False).tolist()
    else:
        raise ValueError(f"protocolo de selecção desconhecido: {protocol!r}")
    return {"protocol": protocol, "seed": int(seed), "n_requested": int(n), "indices": sorted(int(i) for i in sel)}


def run_cf_benchmark(model, X_instances, context: CFContext, methods: Sequence[str] = ("LORE", "CLEAR", "COGS"), *, protocol: str = "random",
                     n_instances: int = 20, seed: int = 0, target_policy: str = "auto", indices: Optional[Sequence[int]] = None,
                     **gen_kw) -> Dict[str, Any]:
    """Várias instâncias × métodos. ``X_instances`` pode ser teste, mas constraints/densidade/escalas vêm do TREINO (contexto)."""
    X = np.asarray(X_instances, dtype=float)
    preds = np.asarray(model.predict(X))
    selection = select_instances(preds, protocol, n_instances, seed, indices)
    classes = ctx_classes = context.model.classes
    rows: List[Dict[str, Any]] = []
    for i in selection["indices"]:
        orig = preds[i]
        targets: List[Any]
        if target_policy == "auto":
            others = [c for c in classes.tolist() if str(c) != str(orig)] if classes is not None else [None]
            targets = others            # binário: 1 alvo (oposta); multiclasse: um pedido por classe alvo (sem assumir "oposta")
        else:
            targets = [target_policy]
        for tgt in targets:
            for m in methods:
                r = generate_counterfactual(X[i], tgt, model, m, context=context, random_state=seed + i, instance_id=int(i), **gen_kw)
                b = r.best
                rows.append({"instance_id": int(i), "method": r.method_label, "method_key": r.method, "target": str(tgt), "original_class": str(orig),
                             "status": r.status.value, "success": r.status == CounterfactualStatus.SUCCESS,
                             "proximity": r.proximity, "sparsity": r.sparsity, "plausibility": r.plausibility,
                             "semantic_valid": (b.semantic_valid if b else None), "semantic_status": r.semantic_validation,
                             "model_valid": bool(b.model_valid) if b else False, "runtime": r.runtime, "terminated_by": r.terminated_by,
                             "seed": r.seed, "n_cfs": len(r.candidates)})
    summary: Dict[str, Any] = {}
    for m in dict.fromkeys(r["method"] for r in rows):
        g = [r for r in rows if r["method"] == m]
        ok = [r for r in g if r["success"]]
        sem_avail = [r for r in ok if r["semantic_valid"] is not None]
        summary[m] = {"n_tasks": len(g), "success_rate": len(ok) / len(g) if g else 0.0,
                      "mean_proximity": float(np.mean([r["proximity"] for r in ok])) if ok else None,
                      "mean_sparsity": float(np.mean([r["sparsity"] for r in ok])) if ok else None,
                      "mean_plausibility": float(np.mean([r["plausibility"] for r in ok if r["plausibility"] is not None])) if any(r["plausibility"] is not None for r in ok) else None,
                      "semantic_validity_rate": (float(np.mean([bool(r["semantic_valid"]) for r in sem_avail])) if sem_avail else None),
                      "semantic_available_rate": (len(sem_avail) / len(ok) if ok else None),
                      "mean_runtime": float(np.mean([r["runtime"] for r in g])),
                      "status_counts": {s: sum(1 for r in g if r["status"] == s) for s in sorted({r["status"] for r in g})}}
    return {"selection": selection, "rows": rows, "summary": summary, "model_explained": context.model_name, "dataset": context.dataset_name,
            "seed": seed, "pipeline_version": PIPELINE_VERSION, "constraints": context.constraints.provenance(),
            "note": "sem ranking agregado; instâncias seleccionadas pelo protocolo declarado, não por sucesso de método"}


def export_results(results: Sequence[CounterfactualResult], out_dir, *, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """counterfactuals.csv + counterfactual_report.json + summary.md (com provenance)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path, md_path = out / "counterfactuals.csv", out / "counterfactual_report.json", out / "summary.md"
    cols = ["instance_id", "method", "method_label", "canonical", "status", "model_explained", "dataset", "original_class", "target_class", "rank",
            "model_valid", "semantic_status", "proximity", "sparsity", "plausibility", "original_probability", "counterfactual_probability",
            "changes", "soft_warnings", "runtime", "seed", "terminated_by", "message"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            base = {"instance_id": r.instance_id, "method": r.method, "method_label": r.method_label, "canonical": r.canonical, "status": r.status.value,
                    "model_explained": r.model_explained, "dataset": r.dataset, "original_class": r.original_class, "target_class": r.target_class,
                    "runtime": f"{r.runtime:.4f}", "seed": r.seed, "terminated_by": r.terminated_by, "message": r.message}
            if not r.candidates:
                w.writerow(base)
            for k, c in enumerate(r.candidates):
                w.writerow({**base, "rank": k + 1, "model_valid": c.model_valid, "semantic_status": c.semantic_status, "proximity": c.proximity,
                            "sparsity": c.sparsity, "plausibility": c.plausibility, "original_probability": c.original_probability,
                            "counterfactual_probability": c.counterfactual_probability, "changes": "; ".join(format_change(x) for x in c.changes),
                            "soft_warnings": "; ".join(c.soft_warnings)})
    report = {"pipeline_version": PIPELINE_VERSION, "n_results": len(results), "extra": dict(extra or {}),
              "results": [r.to_dict() for r in results]}
    json_path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    lines = ["# Contrafactuais — resumo", "", f"- pipeline: `{PIPELINE_VERSION}` · resultados: {len(results)}", ""]
    for r in results:
        lines += [f"## {r.method_label} · instância {r.instance_id} · {r.status.value}", f"- modelo explicado: **{r.model_explained}** · dataset: {r.dataset}",
                  f"- {r.explanation}", f"- semântica: {r.semantic_validation} · seed {r.seed} · {r.runtime:.3f}s · terminação: {r.terminated_by}", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}
