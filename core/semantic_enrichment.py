"""Avaliação do enriquecimento semântico (headless, agnóstica ao dataset).

Fluxo (cada etapa é um estado independente no relatório)::

    A. Quality gate        ontologia estruturalmente válida, matching, ABox segura
    B. Novidade            constantes / duplicados / quase-duplicados (OntologyProcessor)
    C. Triagem barata      estabilidade por fold: informação mútua + ganho add-one
                           com um modelo linear (nunca decide sozinha o MLP)
    D. Comparação real     MLP base  vs  MLP base + features OWL selecionadas,
                           mesmas folds, mesmos dados, mesmo orçamento de otimização

Regras de ausência de leakage:

* em cada fold o ``OntologyProcessor`` é **refeito** (``fit``) só com o treino do fold;
  validação/teste só passam por ``transform``;
* a seleção de features semânticas de cada fold usa apenas o treino desse fold;
* o MLP base e o MLP ontológico são otimizados **separadamente**, com a mesma lista de
  candidatos, o mesmo número de folds internos e a mesma semente (orçamento igual);
* o conjunto de teste externo nunca entra aqui.

A decisão devolve estados granulares (não binários). Rejeitar o enriquecimento do MLP
**não** desliga a utilidade semântica para o TREPAN (``semantic_trepan_available``).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.ontology_processor import OntologyProcessor
from core.semantic_version import SEMANTIC_PIPELINE_VERSION



@dataclass(frozen=True)
class EnrichmentConfig:
    """Configuração congelada **antes** de ver resultados (não ajustar a posteriori)."""

    cv_folds: int = 5
    random_state: int = 42
    # Estabilidade (Parte 14): fração de folds em que a feature tem de ser escolhida.
    min_selection_frequency: float = 0.40
    top_k_per_fold: Optional[int] = None  # por omissão ceil(sqrt(n_candidatas))
    screen_features: int = 24  # máximo de features com ganho add-one calculado por fold
    # Otimização igual para os dois MLP (Parte 16).
    tuning_candidates: int = 6
    tuning_inner_folds: int = 3
    max_iter: int = 300
    # Decisão (Parte 17). Utilidade ponderada documentada, igual à do SemanticUtilityGate.
    balanced_accuracy_weight: float = 0.35
    macro_f1_weight: float = 0.35
    accuracy_weight: float = 0.15
    minority_recall_weight: float = 0.15
    complexity_penalty: float = 0.02
    noninferiority_margin: float = 0.01
    min_secondary_gain: float = 0.005
    n_bootstrap: int = 1000
    confidence: float = 0.95
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------- métricas ---
def _confusion(y_idx: np.ndarray, p_idx: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(y_idx * k + p_idx, minlength=k * k).reshape(k, k)


def _metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "precision_macro": float(precision[present].mean()) if present.any() else 0.0,
        "recall_macro": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "minority_recall": float(recall[present].min()) if present.any() else 0.0,
    }


def classification_metrics_np(y_idx: np.ndarray, p_idx: np.ndarray, k: int) -> Dict[str, float]:
    return _metrics_from_confusion(_confusion(y_idx, p_idx, k))


def utility(metrics: Dict[str, float], cfg: EnrichmentConfig, complexity: float = 0.0) -> float:
    return float(
        cfg.balanced_accuracy_weight * metrics["balanced_accuracy"]
        + cfg.macro_f1_weight * metrics["macro_f1"]
        + cfg.accuracy_weight * metrics["accuracy"]
        + cfg.minority_recall_weight * metrics["minority_recall"]
        - cfg.complexity_penalty * complexity
    )


# ------------------------------------------------------------------- modelos ---
def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(parts)


def candidate_grid(n: int, seed: int) -> List[Dict[str, Any]]:
    """Mesma lista (e mesma ordem) para os dois braços: orçamento igual por construção."""
    rng = np.random.default_rng(seed)
    hidden = [(32,), (64,), (16,), (32, 16), (64, 32), (24,)]
    grid = [{"hidden_layer_sizes": (32,), "alpha": 1e-4, "learning_rate_init": 1e-3}]
    while len(grid) < max(n, 1):
        grid.append({
            "hidden_layer_sizes": hidden[int(rng.integers(len(hidden)))],
            "alpha": float(10 ** rng.uniform(-5, -1)),
            "learning_rate_init": float(10 ** rng.uniform(-3, -1.7)),
        })
    return grid[: max(n, 1)]


def _mlp_pipeline(frame: pd.DataFrame, params: Dict[str, Any], cfg: EnrichmentConfig, seed: int):
    return Pipeline([
        ("prep", _preprocessor(frame)),
        ("mlp", MLPClassifier(max_iter=cfg.max_iter, random_state=seed, **params)),
    ])


def _fit_predict(frame_tr, y_tr, frame_te, params, cfg, seed):
    model = _mlp_pipeline(frame_tr, params, cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(frame_tr, y_tr)
    return model.predict(frame_te)


def _tune(frame_tr: pd.DataFrame, y_tr: np.ndarray, cfg: EnrichmentConfig, seed: int) -> Dict[str, Any]:
    """Escolhe hiperparâmetros por CV interna (só treino). Mesmo orçamento nos dois braços."""
    grid = candidate_grid(cfg.tuning_candidates, cfg.random_state)
    if len(grid) == 1:
        return grid[0]
    _, counts = np.unique(y_tr, return_counts=True)
    folds = int(min(cfg.tuning_inner_folds, counts.min()))
    if folds < 2:
        return grid[0]
    inner = StratifiedKFold(folds, shuffle=True, random_state=seed)
    k = len(np.unique(y_tr))
    scores = []
    for params in grid:
        fold_scores = []
        for a, b in inner.split(frame_tr, y_tr):
            pred = _fit_predict(frame_tr.iloc[a], y_tr[a], frame_tr.iloc[b], params, cfg, seed)
            fold_scores.append(
                _metrics_from_confusion(_confusion(y_tr[b], pred, k))["balanced_accuracy"]
            )
        scores.append(float(np.mean(fold_scores)))
    return grid[int(np.argmax(scores))]


def _cheap_model(frame: pd.DataFrame, seed: int):
    return Pipeline([
        ("prep", _preprocessor(frame)),
        ("lr", LogisticRegression(max_iter=500, random_state=seed)),
    ])


def _cheap_ba(frame_tr, y_tr, frame_va, y_va, seed, k) -> float:
    model = _cheap_model(frame_tr, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame_tr, y_tr)
    return _metrics_from_confusion(_confusion(y_va, model.predict(frame_va), k))["balanced_accuracy"]


# ------------------------------------------------------------------ resultado ---
@dataclass
class SemanticEnrichmentResult:
    report: Dict[str, Any]
    processor: Optional[OntologyProcessor] = None
    selected_features: List[str] = field(default_factory=list)
    input_features: List[str] = field(default_factory=list)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Features originais + apenas as semânticas selecionadas (parâmetros do treino)."""
        if self.processor is None:
            return X.copy()
        full = self.processor.transform(X)
        return full[self.input_features + [c for c in self.selected_features if c in full.columns]]


def _quality_dict(quality) -> Dict[str, Any]:
    if quality is None:
        return {}
    return quality.to_dict() if hasattr(quality, "to_dict") else dict(quality)


def _spec_index(processor: OntologyProcessor) -> Dict[str, Dict[str, Any]]:
    return {spec["name"]: spec for spec in processor.feature_specs_}


# ----------------------------------------------------------------- principal ---
def evaluate_semantic_enrichment(
    X: pd.DataFrame,
    y: Sequence[Any],
    ontology,
    *,
    quality_report=None,
    reasoner_report: Optional[Dict[str, Any]] = None,
    config: EnrichmentConfig = EnrichmentConfig(),
    kinds: Optional[Sequence[str]] = None,
    reasoner_inferred_only: bool = False,
    matches_transform: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
) -> SemanticEnrichmentResult:
    """Avalia (sem leakage) se as features OWL melhoram o MLP e o que oferecem ao TREPAN.

    ``X`` são apenas as features originais do conjunto de desenvolvimento (nunca o teste).
    ``kinds`` restringe os tipos de feature (ablação por componente);
    ``reasoner_inferred_only`` mantém só features cuja hierarquia foi inferida pelo reasoner;
    ``matches_transform`` permite controlos negativos (ex. matching baralhado).
    """
    cfg = config
    X = X.reset_index(drop=True).copy()
    classes, y_idx = np.unique(np.asarray(y), return_inverse=True)
    k = len(classes)
    quality = _quality_dict(quality_report)
    reasoner = dict(reasoner_report or quality.get("reasoner") or {})
    base_cols = [str(c) for c in X.columns]
    stages: Dict[str, Any] = {}
    report: Dict[str, Any] = {
        "semantic_pipeline_version": SEMANTIC_PIPELINE_VERSION,
        "config": {key: val for key, val in asdict(cfg).items()},
        "n_samples": int(len(X)), "n_original_features": len(base_cols),
        "stages": stages, "test_used": False,
        "ablation": {"kinds": list(kinds) if kinds else None,
                     "reasoner_inferred_only": bool(reasoner_inferred_only)},
        "reasoner": {key: reasoner.get(key) for key in (
            "reasoner_used", "reasoning_mode", "executed", "consistent",
            "duration_seconds", "inferred_axioms_count", "fallback") if key in reasoner},
    }

    def finish(status: str, reason: str, *, trepan: bool, selected=None, processor=None, extra=None):
        report.update({
            "decision": status, "decision_reason": reason,
            "semantic_mlp_accepted": status.startswith("ACCEPT"),
            "semantic_trepan_available": bool(trepan),
            "selected_semantic_features": list(selected or []),
        })
        if extra:
            report.update(extra)
        return SemanticEnrichmentResult(report, processor, list(selected or []), base_cols)

    # ---- Etapa A: quality gate -------------------------------------------------
    abox = quality.get("abox") or {}
    structural_ok = bool(quality.get("accepted"))
    metrics_q = quality.get("metrics") or {}
    stages["A_quality"] = {
        "status": quality.get("status", "NOT_EVALUATED"),
        "structural_valid": structural_ok,
        "abox_status": abox.get("status", "NOT_EVALUATED"),
        "mapping_coverage": metrics_q.get("feature_coverage"),
        "matched_features": metrics_q.get("mapped_features"),
        "total_features": metrics_q.get("total_features"),
        "ambiguous_matches": metrics_q.get("ambiguous_matches"),
        "generic_match_ratio": metrics_q.get("generic_match_ratio"),
        "reasoner_consistent": metrics_q.get("reasoner_consistent", reasoner.get("consistent")),
    }
    if not quality:
        return finish("REJECT_INVALID_ONTOLOGY", "quality_report_missing", trepan=False)
    if abox and abox.get("accepted") is False:
        return finish("REJECT_LEAKAGE_RISK", f"abox:{abox.get('status')}", trepan=False)
    if not structural_ok:
        return finish("REJECT_INVALID_ONTOLOGY", f"quality:{quality.get('status')}", trepan=False)

    matches = [m for m in quality.get("matches", []) if m.get("accepted")]
    if matches_transform is not None:
        matches = matches_transform([dict(m) for m in matches])
    proc_kwargs = dict(cfg.processor_kwargs)
    proc_kwargs.setdefault("reasoner_report", reasoner)

    def new_processor():
        return OntologyProcessor(ontology, **proc_kwargs)

    def keep(processor: OntologyProcessor, names: Sequence[str]) -> List[str]:
        specs = _spec_index(processor)
        out = []
        for name in names:
            spec = specs.get(name, {})
            if kinds and spec.get("kind") not in set(kinds):
                continue
            if reasoner_inferred_only and not spec.get("reasoner_inferred"):
                continue
            out.append(name)
        return out

    # ---- Etapa B: novidade (ajuste final no conjunto de desenvolvimento) --------
    try:
        full = new_processor().fit(X, accepted_matches=matches, log=False)
    except Exception as exc:
        return finish("NOT_EVALUATED", f"processor_error:{type(exc).__name__}: {exc}", trepan=True)
    pool_all = [n for n in full.output_features_ if n not in base_cols]
    pool = keep(full, pool_all)
    summary = dict(full.generation_summary_)
    stages["B_novelty"] = {**summary, "pool_after_ablation_filter": len(pool),
                           "status": "VALID" if pool else "NO_NOVEL_FEATURES"}
    ontology_has_structure = (metrics_q.get("semantic_richness") or {}).get("level") != "poor"
    trepan_ok = True  # estrutura/ABox/matching já validados na etapa A
    if not pool:
        return finish("REJECT_NO_NOVEL_FEATURES", "all_candidates_removed_by_novelty_filters",
                      trepan=trepan_ok, processor=full,
                      extra={"feature_audit": _feature_rows(full, {}, {}, set(), pool_all, X)})

    # ---- Etapas C+D: por fold, fit só no treino --------------------------------
    min_class = int(np.bincount(y_idx).min())
    if min_class < 2:
        raise ValueError("Cada classe precisa de pelo menos 2 amostras para validação cruzada.")
    cv = StratifiedKFold(n_splits=int(min(cfg.cv_folds, min_class)), shuffle=True,
                         random_state=cfg.random_state)
    oof_base = np.full(len(X), -1, dtype=int)
    oof_onto = np.full(len(X), -1, dtype=int)
    sel_counts: Dict[str, int] = {}
    mi_values: Dict[str, List[float]] = {}
    gain_values: Dict[str, List[float]] = {}
    fold_rows: List[Dict[str, Any]] = []
    n_folds = cv.get_n_splits()
    for fold, (tr, va) in enumerate(cv.split(X, y_idx)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y_idx[tr], y_idx[va]
        proc = new_processor().fit(Xtr, accepted_matches=matches, log=False)  # TRAIN ONLY
        fold_error = None
        try:
            Ztr, Zva = proc.transform(Xtr), proc.transform(Xva)
            fold_pool = keep(proc, [n for n in proc.output_features_ if n not in base_cols])
        except Exception as exc:  # ex.: categoria da validação sem mapping aprendido no treino
            Ztr, Zva, fold_pool = Xtr, Xva, []
            fold_error = f"{type(exc).__name__}: {exc}"
        selected: List[str] = []
        if fold_pool:
            numeric_pool = [c for c in fold_pool if pd.api.types.is_numeric_dtype(Ztr[c])]
            mi = mutual_info_classif(Ztr[numeric_pool].to_numpy(float), ytr,
                                     random_state=cfg.random_state) if numeric_pool else []
            mi_map = dict(zip(numeric_pool, map(float, mi)))
            for name, value in mi_map.items():
                mi_values.setdefault(name, []).append(value)
            top_k = cfg.top_k_per_fold or max(1, int(math.ceil(math.sqrt(len(numeric_pool)))))
            ranked = sorted(mi_map, key=lambda n: (-mi_map[n], n))
            selected = ranked[:top_k]
            for name in selected:
                sel_counts[name] = sel_counts.get(name, 0) + 1
            base_ba = _cheap_ba(Ztr[base_cols], ytr, Zva[base_cols], yva, cfg.random_state, k)
            for name in ranked[: cfg.screen_features]:
                gain = _cheap_ba(Ztr[base_cols + [name]], ytr, Zva[base_cols + [name]], yva,
                                 cfg.random_state, k) - base_ba
                gain_values.setdefault(name, []).append(float(gain))
        seed = cfg.random_state + fold
        p_base = _tune(Xtr, ytr, cfg, seed)
        onto_cols = base_cols + selected
        p_onto = _tune(Ztr[onto_cols], ytr, cfg, seed)
        oof_base[va] = _fit_predict(Xtr, ytr, Xva, p_base, cfg, seed)
        oof_onto[va] = _fit_predict(Ztr[onto_cols], ytr, Zva[onto_cols], p_onto, cfg, seed)
        fold_rows.append({"fold": fold, "selected_semantic_features": selected,
                          "transform_error": fold_error,
                          "base_params": _jsonable(p_base), "onto_params": _jsonable(p_onto)})

    frequency = {n: sel_counts.get(n, 0) / n_folds for n in pool}
    stable = [n for n in pool if frequency[n] >= cfg.min_selection_frequency]
    stages["C_screening"] = {
        "status": "VALID" if stable else "UNSTABLE",
        "min_selection_frequency": cfg.min_selection_frequency,
        "stable_features": stable, "folds": n_folds,
        "selection_frequency": frequency,
    }
    audit = _feature_rows(full, {"mi": mi_values, "gain": gain_values}, frequency, set(stable), pool_all, X)
    complexity = len(stable) / max(1, len(base_cols) + len(stable))

    # ---- Etapa D: comparação real ----------------------------------------------
    m_base = classification_metrics_np(y_idx, oof_base, k)
    m_onto = classification_metrics_np(y_idx, oof_onto, k)
    u_base, u_onto = utility(m_base, cfg), utility(m_onto, cfg, complexity)
    lo, hi = _paired_bootstrap_ci(y_idx, oof_base, oof_onto, k, cfg, complexity)
    delta = {name: m_onto[name] - m_base[name] for name in m_base}
    stages["D_mlp_comparison"] = {
        "protocol": "oof_stratified_cv_same_folds_per_fold_refit_equal_tuning_budget",
        "tuning_candidates_per_arm": cfg.tuning_candidates,
        "tuning_inner_folds": cfg.tuning_inner_folds,
        "base": {**m_base, "utility": u_base},
        "with_owl": {**m_onto, "utility": u_onto},
        "evaluation_note": (
            "A estimativa OOF usa a seleção de features feita só com o treino de cada fold; "
            "o conjunto final (selected_semantic_features) são as features com "
            "selection_frequency >= min_selection_frequency, ajustadas em todo o desenvolvimento."
        ),
        "delta": delta, "utility_gain": u_onto - u_base,
        "utility_gain_ci": [lo, hi], "confidence": cfg.confidence,
        "per_fold": fold_rows,
    }
    status, reason = _decide(stable, pool, delta, u_onto - u_base, lo, hi, cfg)
    # Transparência (não altera a decisão): aceitar por não-inferioridade com o IC da utilidade
    # a incluir zero é evidência fraca; só um IC acima de zero é evidência forte de ganho.
    strength = ("n/a" if not status.startswith("ACCEPT") else "strong" if lo > 0 else "weak")
    return finish(status, reason, trepan=trepan_ok, selected=stable if status.startswith("ACCEPT") else [],
                  processor=full, extra={
                      "evidence_strength": strength,
                      "feature_audit": audit,
                      "candidate_stable_features": stable,
                      "semantic_richness_level": (metrics_q.get("semantic_richness") or {}).get("level"),
                      "ontology_knowledge_available_for_trepan": bool(ontology_has_structure or pool),
                  })


def _decide(stable, pool, delta, gain, lo, hi, cfg: EnrichmentConfig):
    if not stable:
        return "REJECT_UNSTABLE_FEATURES", "no_feature_reached_min_selection_frequency"
    if hi < 0 or gain < -cfg.noninferiority_margin:
        return "REJECT_DEGRADATION", "utility_ci_below_zero_or_loss_above_noninferiority_margin"
    partial = len(stable) < len(pool)
    if lo > 0:
        return ("ACCEPT_PARTIAL_FEATURE_SET" if partial else "ACCEPT_SIGNIFICANT_GAIN",
                "utility_gain_ci_excludes_zero")
    secondary = max(delta["minority_recall"], delta["macro_f1"])
    if lo >= -cfg.noninferiority_margin and delta["balanced_accuracy"] >= -cfg.noninferiority_margin \
            and secondary >= cfg.min_secondary_gain:
        return ("ACCEPT_PARTIAL_FEATURE_SET" if partial else "ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN",
                "non_inferior_with_secondary_metric_gain")
    return "REJECT_NO_INFORMATIONAL_GAIN", "ci_includes_zero_and_no_secondary_gain"


def _paired_bootstrap_ci(y, p_base, p_onto, k, cfg: EnrichmentConfig, complexity: float):
    rng = np.random.default_rng(cfg.random_state)
    n = len(y)
    diffs = np.empty(cfg.n_bootstrap)
    for b in range(cfg.n_bootstrap):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        diffs[b] = (utility(classification_metrics_np(yb, p_onto[idx], k), cfg, complexity)
                    - utility(classification_metrics_np(yb, p_base[idx], k), cfg))
    alpha = (1.0 - cfg.confidence) / 2.0
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1.0 - alpha))


def _jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    return {key: (list(val) if isinstance(val, tuple) else val) for key, val in params.items()}


def _feature_rows(proc: OntologyProcessor, stats, frequency, stable: set, pool_all, X) -> List[Dict[str, Any]]:
    """Relatório por feature (Parte 26): origem, fontes, estatísticas, decisão e motivo."""
    mi = stats.get("mi", {}) if stats else {}
    gain = stats.get("gain", {}) if stats else {}
    specs = _spec_index(proc)
    try:
        enriched = proc.transform(X)
    except Exception:
        enriched = None
    rows = []
    for audit in proc.feature_audit_:
        name = audit["feature"]
        spec = specs.get(name, {})
        sources = list(audit["owl_origin"]["source_columns"])
        variance = corr = None
        if enriched is not None and name in enriched.columns:
            col = pd.to_numeric(enriched[name], errors="coerce")
            variance = float(col.var(ddof=0))
            cs = []
            for src in sources:
                if src in X.columns and pd.api.types.is_numeric_dtype(X[src]) and col.std(ddof=0) > 0 \
                        and X[src].std(ddof=0) > 0:
                    cs.append(abs(float(np.corrcoef(col, X[src])[0, 1])))
            corr = max(cs) if cs else None
        if not audit["accepted"]:
            decision, reason = "REJECTED", f"novelty_filter:{audit['rejection_reason']}"
        elif name in stable:
            decision, reason = "SELECTED", "stable_selection_across_folds"
        else:
            decision, reason = "REJECTED", "unstable_or_not_selected"
        rows.append({
            "feature": name, "origin": audit["kind"], "knowledge_source": audit["knowledge_source"],
            "source_features": sources, "ontology_family": spec.get("family"),
            "ontology_roles": spec.get("roles"),
            "ontology_entities": audit["owl_origin"]["owl_entities_or_properties"],
            "formula": audit["derivation_rule"], "provenance": audit["provenance"],
            "reasoner_inferred": audit.get("reasoner_inferred", False),
            "variance": variance, "max_abs_corr_with_sources": corr,
            "linear_redundancy_r2": audit.get("linear_redundancy_r2"),
            "mutual_information_mean": float(np.mean(mi[name])) if name in mi else None,
            "mutual_information_std": float(np.std(mi[name])) if name in mi else None,
            "selection_frequency": frequency.get(name),
            "predictive_gain_mean": float(np.mean(gain[name])) if name in gain else None,
            "predictive_gain_std": float(np.std(gain[name])) if name in gain else None,
            "decision": decision, "reason": reason,
        })
    return rows


def export_feature_audit(report: Dict[str, Any], path) -> str:
    """Exporta a tabela por feature (CSV ou JSON conforme a extensão)."""
    import json
    from pathlib import Path
    rows = report.get("feature_audit") or []
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    else:
        pd.DataFrame(rows).to_csv(target, index=False)
    return str(target)
