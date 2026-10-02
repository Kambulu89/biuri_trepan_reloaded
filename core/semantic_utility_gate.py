"""Gate preditivo para aceitar apenas enriquecimento OWL com utilidade real.

O gate é ajustado exclusivamente no conjunto de desenvolvimento. As mesmas
partições OOF são usadas para o baseline original e para cada subconjunto
semântico. Em produção, quando o ``OntologyProcessor`` e o frame bruto do
subconjunto de desenvolvimento são fornecidos, as estatísticas aprendidas pelo
feature engineering são recalibradas *dentro de cada fold*, evitando leakage.

A decisão final usa um MLP. Se o MLP Original estiver disponível, a sua família
e hiperparâmetros são clonados (desembrulhando apenas calibração) para que a
comparação Base vs Base+OWL não seja decidida por um proxy linear diferente.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.ontology_semantic_graph import normalize_ontology_name
from core.scientific_errors import (
    OntologyConsistencyError, OntologyLeakageError, OntologyQualityError,
)


def _resolve_names(requested: Sequence[str], available: Sequence[str]) -> Dict[str, Optional[str]]:
    """Casa nomes exactos primeiro e só depois pela chave tolerante.

    A chave ignora caixa, sublinhados e prefixos URI. Colisões ambíguas (duas
    colunas disponíveis com a mesma chave) não são resolvidas por tolerância.
    """
    exact = set(available)
    by_key: Dict[str, list] = {}
    for name in available:
        by_key.setdefault(normalize_ontology_name(name), []).append(name)
    resolved: Dict[str, Optional[str]] = {}
    for name in requested:
        if name in exact:
            resolved[name] = name
            continue
        hits = by_key.get(normalize_ontology_name(name), [])
        resolved[name] = hits[0] if len(hits) == 1 else None
    return resolved


@dataclass(frozen=True)
class SemanticUtilityConfig:
    cv_folds: int = 5
    random_state: int = 42
    min_predictive_gain: float = 0.002
    max_noninferiority_loss: float = 0.0
    min_selection_stability: float = 0.40
    max_semantic_features: int = 40
    near_constant_std: float = 1e-6
    duplicate_correlation: float = 0.999999
    complexity_penalty: float = 0.02
    max_complexity_ratio: float = 0.60
    min_matching_confidence: float = 0.72
    min_quality_coverage: float = 0.35
    balanced_accuracy_weight: float = 0.35
    macro_f1_weight: float = 0.35
    accuracy_weight: float = 0.15
    minority_recall_weight: float = 0.15
    max_single_feature_audits: int = 16


class SemanticUtilityGate:
    """Seleciona features OWL por estabilidade e ganho preditivo OOF com MLP."""

    VALID_QUALITY_STATES = {"VALID_DOMAIN_ONTOLOGY", "PARTIALLY_USABLE", "accepted"}

    def __init__(self, config: SemanticUtilityConfig = SemanticUtilityConfig()):
        self.config = config

    @staticmethod
    def _folds(y: np.ndarray, requested: int) -> int:
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or int(counts.min()) < 2:
            raise ValueError("SemanticUtilityGate exige duas classes e duas amostras por classe.")
        return max(2, min(int(requested), int(counts.min())))

    @staticmethod
    def _unwrap_reference_estimator(reference_estimator):
        """Retorna o estimador MLP clonável por baixo de wrappers de calibração."""
        estimator = reference_estimator
        seen = set()
        for _ in range(4):
            if estimator is None or id(estimator) in seen:
                break
            seen.add(id(estimator))
            if isinstance(estimator, CalibratedClassifierCV):
                estimator = getattr(estimator, "estimator", None)
                continue
            break
        if isinstance(estimator, Pipeline):
            if any(isinstance(step, MLPClassifier) for _, step in estimator.steps):
                return estimator
        if isinstance(estimator, MLPClassifier):
            return Pipeline([("scaler", StandardScaler()), ("mlp", estimator)])
        return None

    @classmethod
    def _model(cls, random_state: int, reference_estimator=None):
        reference = cls._unwrap_reference_estimator(reference_estimator)
        if reference is not None:
            return clone(reference), {
                "family": "MLPClassifier",
                "source": "mlp_original_hyperparameters",
                "reference_reused": True,
            }
        # Fallback apenas para chamadas unitárias/CLI sem MLP Original disponível.
        # Continua a ser MLP; a arquitectura permanece fixa em todos os candidatos.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs",
                alpha=1e-3, max_iter=700, random_state=random_state,
            )),
        ])
        return model, {
            "family": "MLPClassifier",
            "source": "semantic_gate_fixed_fallback",
            "reference_reused": False,
        }

    def _metrics(
        self,
        y,
        predictions,
        semantic_count: int,
        total_semantic: int,
        *,
        base_feature_count: int = 0,
    ):
        recalls = recall_score(y, predictions, average=None, zero_division=0)
        cfg = self.config
        denominator = max(1, int(base_feature_count) + int(semantic_count))
        complexity = semantic_count / denominator
        values = {
            "accuracy": float(accuracy_score(y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "minority_recall": float(np.min(recalls)) if len(recalls) else 0.0,
            "semantic_feature_count": int(semantic_count),
            "candidate_semantic_pool_size": int(total_semantic),
            "base_feature_count": int(base_feature_count),
            "complexity_ratio": float(complexity),
        }
        values["complexity_penalty_value"] = float(cfg.complexity_penalty * complexity)
        values["utility_before_complexity"] = float(
            cfg.balanced_accuracy_weight * values["balanced_accuracy"]
            + cfg.macro_f1_weight * values["macro_f1"]
            + cfg.accuracy_weight * values["accuracy"]
            + cfg.minority_recall_weight * values["minority_recall"]
        )
        values["utility"] = float(
            values["utility_before_complexity"] - values["complexity_penalty_value"]
        )
        return values

    def _decision_status(
        self,
        *,
        accepted: bool,
        complexity_excess: bool,
        utility_gain: float,
        balanced_accuracy_gain: float,
        macro_f1_gain: float,
        complexity_penalty_delta: float,
    ):
        if accepted:
            return "ACCEPT_PARTIAL_FEATURES", "validated_predictive_gain"
        if complexity_excess:
            return "REJECT_COMPLEXITY_COST", "semantic_complexity_ratio_exceeded"
        positive_raw_gain = balanced_accuracy_gain > 0 and macro_f1_gain > 0
        if positive_raw_gain and complexity_penalty_delta > 0 and utility_gain < self.config.min_predictive_gain:
            return (
                "REJECT_GAIN_BELOW_COMPLEXITY_COST",
                "positive_predictive_gain_below_complexity_cost",
            )
        if utility_gain >= 0 and utility_gain < self.config.min_predictive_gain:
            return "REJECT_INSUFFICIENT_NET_UTILITY", "net_utility_gain_below_threshold"
        return "REJECT_NO_INFORMATIONAL_GAIN", "no_oof_predictive_gain"

    @staticmethod
    def _quality_rejection(quality_report: Optional[Dict[str, Any]]) -> Optional[str]:
        if not quality_report:
            return None
        if quality_report.get("accepted") is True:
            return None
        status = str(quality_report.get("status", ""))
        mapping = {
            "CONTAMINATED_ONTOLOGY": "REJECT_LEAKAGE",
            "LOGICALLY_INCONSISTENT": "REJECT_INCONSISTENT",
            "REASONER_NOT_VALIDATED": "REJECT_ONTOLOGY_REASONER",
            "INSUFFICIENT_SCHEMA_COVERAGE": "REJECT_ONTOLOGY_MATCHING",
            "GENERIC_ONTOLOGY": "REJECT_ONTOLOGY_QUALITY",
            "INVALID_ONTOLOGY": "REJECT_ONTOLOGY_QUALITY",
            "rejected": "REJECT_ONTOLOGY_QUALITY",
        }
        if status in mapping:
            return mapping[status]
        if not (quality_report.get("reasoner") or {}).get("consistent", True):
            return "REJECT_INCONSISTENT"
        return "REJECT_ONTOLOGY_QUALITY"

    @staticmethod
    def _oof_predict(estimator, y, splits, selected_indices, fold_views):
        predictions = np.empty(len(y), dtype=np.asarray(y).dtype)
        coverage = np.zeros(len(y), dtype=int)
        for fold_id, (fit_idx, val_idx) in enumerate(splits):
            X_fit_full, X_val_full = fold_views[fold_id]
            model = clone(estimator)
            model.fit(X_fit_full[:, selected_indices], np.asarray(y)[fit_idx])
            predictions[val_idx] = model.predict(X_val_full[:, selected_indices])
            coverage[val_idx] += 1
        if not np.all(coverage == 1):
            raise RuntimeError("Predições OOF sem cobertura exactamente uma vez por amostra.")
        return predictions

    @staticmethod
    def _static_fold_views(X, splits):
        return [(X[fit_idx], X[val_idx]) for fit_idx, val_idx in splits]

    def _fold_local_semantic_views(
        self,
        *,
        X,
        names,
        base_names,
        base_indices,
        semantic_indices,
        splits,
        semantic_processor,
        raw_base_frame,
    ):
        """Reconstrói apenas onto_* em cada fold; base numérica fica exactamente igual."""
        if semantic_processor is None or raw_base_frame is None:
            return self._static_fold_views(X, splits), {
                "enabled": False,
                "scope": "precomputed_semantic_matrix",
                "reason": "processor_or_raw_frame_not_provided",
            }
        raw = pd.DataFrame(raw_base_frame).copy()
        if len(raw) != len(X):
            raise ValueError("raw_base_frame não está alinhado ao X do gate semântico.")
        if [normalize_ontology_name(c) for c in raw.columns] != [
            normalize_ontology_name(c) for c in base_names
        ]:
            raise ValueError(
                "raw_base_frame deve conter exactamente as features ARFF originais na mesma ordem."
            )
        semantic_names = [names[i] for i in semantic_indices]
        views = []
        fold_audit = []
        for fold_id, (fit_idx, val_idx) in enumerate(splits):
            train_enriched, val_enriched, meta = semantic_processor.transform_fold_pair(
                raw.iloc[fit_idx].copy(), raw.iloc[val_idx].copy()
            )
            columns = list(map(str, train_enriched.columns))
            resolved = _resolve_names(semantic_names, columns)
            missing = [name for name, hit in resolved.items() if hit is None]
            if missing:
                raise ValueError(f"Processor fold-local não reproduziu features: {missing}")
            fold_columns = [resolved[name] for name in semantic_names]
            train_sem = train_enriched.loc[:, fold_columns].to_numpy(dtype=float)
            val_sem = val_enriched.loc[:, fold_columns].to_numpy(dtype=float)
            X_fit_full = np.empty((len(fit_idx), len(names)), dtype=float)
            X_val_full = np.empty((len(val_idx), len(names)), dtype=float)
            X_fit_full[:, base_indices] = X[fit_idx][:, base_indices]
            X_val_full[:, base_indices] = X[val_idx][:, base_indices]
            X_fit_full[:, semantic_indices] = train_sem
            X_val_full[:, semantic_indices] = val_sem
            if not np.isfinite(X_fit_full).all() or not np.isfinite(X_val_full).all():
                raise ValueError("Feature engineering fold-local produziu NaN/inf.")
            views.append((X_fit_full, X_val_full))
            fold_audit.append({"fold": int(fold_id), **meta})
        return views, {
            "enabled": True,
            "scope": "inner_fold_training_only",
            "reason": "ontology_train_statistics_recomputed_per_fold",
            "folds": fold_audit,
        }

    def evaluate(
        self,
        X_enriched,
        y,
        feature_names: Sequence[str],
        base_feature_names: Sequence[str],
        *,
        quality_report: Optional[Dict[str, Any]] = None,
        reference_estimator=None,
        semantic_processor=None,
        raw_base_frame: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        X = np.asarray(X_enriched, dtype=float)
        y = np.asarray(y)
        names = list(map(str, feature_names))
        base_names = list(map(str, base_feature_names))
        if X.ndim != 2 or len(X) != len(y) or X.shape[1] != len(names):
            raise ValueError("SemanticUtilityGate recebeu X/y/schema incompatíveis.")
        if not np.isfinite(X).all():
            raise ValueError("SemanticUtilityGate não aceita zero-fill/NaN/inf artificiais.")
        if len(set(names)) != len(names):
            raise ValueError("SemanticUtilityGate rejeitou nomes de features duplicados.")
        resolved_base = _resolve_names(base_names, names)
        missing_base = [name for name, hit in resolved_base.items() if hit is None]
        if missing_base:
            raise ValueError(f"Features originais ausentes no gate semântico: {missing_base}")

        quality_rejection = self._quality_rejection(quality_report)
        base_names = [resolved_base[name] for name in base_names]
        base_indices = [names.index(name) for name in base_names]
        base_set = set(base_names)
        semantic_indices = [i for i, name in enumerate(names) if name not in base_set]
        audit: Dict[str, Any] = {
            "scope": "development_oof_only", "test_used": False,
            "config": asdict(self.config), "base_indices": base_indices,
            "candidate_semantic_indices": semantic_indices,
            "candidate_semantic_names": [names[i] for i in semantic_indices],
            "rejections": [],
        }
        feature_audit = {}
        for index in semantic_indices:
            column = X[:, index]
            _, counts = np.unique(column, return_counts=True)
            feature_audit[index] = {
                "feature": names[index],
                "index": int(index),
                "constant_rate": float(counts.max() / len(column)) if len(counts) else 1.0,
                "accepted_by_semantic_gate": False,
                "rejection_reason": None,
                "duplicate_of": None,
                "fold_stability": None,
                "mean_mutual_information": None,
                "effect_on_metrics": None,
            }
        audit["feature_audit"] = list(feature_audit.values())
        if quality_report:
            quality_metrics = dict(quality_report.get("metrics") or {})
            accepted_matches = [
                row for row in (quality_report.get("matches") or []) if row.get("accepted")
            ]
            mean_confidence = float(np.mean([
                float(row.get("score", 0.0)) for row in accepted_matches
            ])) if accepted_matches else 0.0
            audit["ontology_quality"] = {
                "accepted": bool(quality_report.get("accepted")),
                "status": quality_report.get("status"),
                "reasoner_executed": bool((quality_report.get("reasoner") or {}).get("executed")),
                "reasoner_consistent": (quality_report.get("reasoner") or {}).get("consistent"),
                "feature_coverage": float(quality_metrics.get("feature_coverage", 0.0)),
                "mean_matching_confidence": mean_confidence,
                "generic_entity_ratio": float(quality_metrics.get("generic_entity_ratio", 0.0)),
            }
            if (
                accepted_matches and mean_confidence < self.config.min_matching_confidence
            ) or (
                "feature_coverage" in quality_metrics
                and float(quality_metrics["feature_coverage"]) < self.config.min_quality_coverage
            ):
                quality_rejection = "REJECT_ONTOLOGY_MATCHING"
        if quality_rejection:
            for row in feature_audit.values():
                row["rejection_reason"] = "ontology_quality_gate_rejected"
            audit["feature_audit"] = list(feature_audit.values())
            audit.update({
                "accepted": False, "status": quality_rejection,
                "selected_indices": base_indices,
                "selected_feature_names": base_names,
                "selected_semantic_indices": [], "selected_semantic_names": [],
                "reason": "ontology_quality_gate_rejected",
            })
            return audit
        if not semantic_indices:
            audit.update({
                "accepted": False, "status": "REJECT_NO_DERIVED_SEMANTIC_FEATURES",
                "selected_indices": base_indices,
                "selected_feature_names": base_names,
                "selected_semantic_indices": [], "selected_semantic_names": [],
                "reason": "no_semantic_features",
            })
            return audit

        # Filtro não supervisionado: constantes e duplicações determinísticas.
        retained = []
        reference_indices = list(base_indices)
        for index in semantic_indices:
            column = X[:, index]
            if float(np.std(column)) <= self.config.near_constant_std:
                audit["rejections"].append({"feature": names[index], "reason": "near_constant"})
                feature_audit[index]["rejection_reason"] = "near_constant"
                continue
            duplicate = None
            for ref in reference_indices + retained:
                other = X[:, ref]
                if np.allclose(column, other, rtol=0.0, atol=1e-12):
                    duplicate = names[ref]
                    break
                if np.std(other) > self.config.near_constant_std:
                    corr = abs(float(np.corrcoef(column, other)[0, 1]))
                    if np.isfinite(corr) and corr >= self.config.duplicate_correlation:
                        duplicate = names[ref]
                        break
            if duplicate is not None:
                audit["rejections"].append({
                    "feature": names[index], "reason": "deterministic_duplicate",
                    "duplicate_of": duplicate,
                })
                feature_audit[index]["rejection_reason"] = "deterministic_duplicate"
                feature_audit[index]["duplicate_of"] = duplicate
                continue
            retained.append(index)

        if not retained:
            audit["feature_audit"] = list(feature_audit.values())
            audit.update({
                "accepted": False, "status": "REJECT_NO_INFORMATIONAL_NOVELTY",
                "selected_indices": base_indices,
                "selected_feature_names": base_names,
                "selected_semantic_indices": [], "selected_semantic_names": [],
                "reason": "all_semantic_features_constant_or_duplicate",
            })
            return audit

        folds = self._folds(y, self.config.cv_folds)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_state)
        splits = [(fit_idx, val_idx) for fit_idx, val_idx in cv.split(X, y)]
        try:
            fold_views, fold_local_audit = self._fold_local_semantic_views(
                X=X, names=names, base_names=base_names, base_indices=base_indices,
                semantic_indices=semantic_indices, splits=splits,
                semantic_processor=semantic_processor, raw_base_frame=raw_base_frame,
            )
        except Exception as exc:
            for idx in retained:
                feature_audit[idx]["rejection_reason"] = "fold_local_semantic_refit_failed"
            audit.update({
                "accepted": False,
                "status": "REJECT_FOLD_LOCAL_SEMANTIC_REFIT",
                "reason": f"fold_local_semantic_refit_failed:{type(exc).__name__}:{exc}",
                "selected_indices": base_indices,
                "selected_feature_names": base_names,
                "selected_semantic_indices": [],
                "selected_semantic_names": [],
                "feature_audit": list(feature_audit.values()),
                "fold_local_semantic_refit": {"enabled": False, "error": str(exc)},
            })
            return audit
        audit["fold_local_semantic_refit"] = fold_local_audit

        selection_counts = {index: 0 for index in retained}
        mi_totals = {index: 0.0 for index in retained}
        top_per_fold = max(1, min(
            self.config.max_semantic_features,
            int(np.ceil(np.sqrt(len(retained)))),
        ))
        for fold_id, (fit_idx, _) in enumerate(splits):
            X_fit_full, _ = fold_views[fold_id]
            scores = mutual_info_classif(
                X_fit_full[:, retained], y[fit_idx], random_state=self.config.random_state,
            )
            order = np.argsort(-scores, kind="stable")[:top_per_fold]
            for local_idx, score in enumerate(scores):
                mi_totals[retained[local_idx]] += float(score)
            for local_idx in order:
                selection_counts[retained[int(local_idx)]] += 1
        stability = {idx: count / folds for idx, count in selection_counts.items()}
        for idx in retained:
            feature_audit[idx]["fold_stability"] = float(stability[idx])
            feature_audit[idx]["mean_mutual_information"] = float(mi_totals[idx] / folds)
        stable = [idx for idx in retained if stability[idx] >= self.config.min_selection_stability]
        if not stable:
            for idx in retained:
                feature_audit[idx]["rejection_reason"] = "unstable_across_folds"
            audit["feature_audit"] = list(feature_audit.values())
            audit.update({
                "accepted": False, "status": "REJECT_UNSTABLE",
                "selected_indices": base_indices,
                "selected_feature_names": base_names,
                "selected_semantic_indices": [], "selected_semantic_names": [],
                "selection_stability": {names[i]: float(stability[i]) for i in retained},
                "reason": "semantic_features_not_stable_across_folds",
            })
            return audit
        stable.sort(key=lambda idx: (-stability[idx], -mi_totals[idx], names[idx]))
        stable = stable[: self.config.max_semantic_features]

        candidate_sizes = sorted(set(
            size for size in (1, 3, 5, 10, 20, len(stable)) if 0 < size <= len(stable)
        ))
        model, model_audit = self._model(self.config.random_state, reference_estimator)
        audit["gate_estimator"] = model_audit
        baseline_pred = self._oof_predict(model, y, splits, base_indices, fold_views)
        baseline = self._metrics(
            y, baseline_pred, 0, len(semantic_indices), base_feature_count=len(base_indices)
        )

        # Diagnóstico individual limitado por custo; seleção final continua a avaliar subconjuntos.
        single_feature_effects = {}
        audited_features = stable[: max(0, int(self.config.max_single_feature_audits))]
        for index in audited_features:
            singleton_pred = self._oof_predict(
                model, y, splits, base_indices + [index], fold_views
            )
            singleton = self._metrics(
                y, singleton_pred, 1, len(semantic_indices), base_feature_count=len(base_indices)
            )
            effect = {
                "accuracy_delta": float(singleton["accuracy"] - baseline["accuracy"]),
                "balanced_accuracy_delta": float(singleton["balanced_accuracy"] - baseline["balanced_accuracy"]),
                "macro_f1_delta": float(singleton["macro_f1"] - baseline["macro_f1"]),
                "minority_recall_delta": float(singleton["minority_recall"] - baseline["minority_recall"]),
            }
            single_feature_effects[names[index]] = effect
            feature_audit[index]["effect_on_metrics"] = effect

        rows = [{"candidate": "original_only", "indices": base_indices, **baseline}]
        for size in candidate_sizes:
            chosen_semantic = stable[:size]
            indices = base_indices + chosen_semantic
            predictions = self._oof_predict(model, y, splits, indices, fold_views)
            row = self._metrics(
                y, predictions, size, len(semantic_indices), base_feature_count=len(base_indices)
            )
            row.update({
                "candidate": f"semantic_top_{size}", "indices": indices,
                "semantic_indices": chosen_semantic,
                "semantic_names": [names[i] for i in chosen_semantic],
            })
            rows.append(row)
        best = max(
            rows[1:], key=lambda row: (
                row["utility"], row["balanced_accuracy"], row["macro_f1"],
                -row["semantic_feature_count"],
            ),
        )
        utility_gain = float(best["utility"] - baseline["utility"])
        ba_gain = float(best["balanced_accuracy"] - baseline["balanced_accuracy"])
        f1_gain = float(best["macro_f1"] - baseline["macro_f1"])
        predictive_gain = min(ba_gain, f1_gain)
        complexity_excess = bool(best["complexity_ratio"] > self.config.max_complexity_ratio)
        accepted = bool(
            utility_gain >= self.config.min_predictive_gain
            and predictive_gain >= -self.config.max_noninferiority_loss
            and not complexity_excess
        )
        selected_semantic = list(best.get("semantic_indices", [])) if accepted else []
        selected_set = set(selected_semantic)
        for idx in retained:
            feature_audit[idx]["accepted_by_semantic_gate"] = idx in selected_set
            if feature_audit[idx]["rejection_reason"] is None and idx not in selected_set:
                feature_audit[idx]["rejection_reason"] = (
                    "not_in_best_oof_subset" if accepted else "semantic_gate_rejected_candidate"
                )
        selected_indices = base_indices + selected_semantic
        complexity_penalty_delta = float(
            best.get("complexity_penalty_value", 0.0) - baseline.get("complexity_penalty_value", 0.0)
        )
        status, decision_reason = self._decision_status(
            accepted=accepted,
            complexity_excess=complexity_excess,
            utility_gain=utility_gain,
            balanced_accuracy_gain=ba_gain,
            macro_f1_gain=f1_gain,
            complexity_penalty_delta=complexity_penalty_delta,
        )
        if accepted and len(selected_semantic) == len(semantic_indices):
            status = "ACCEPT_ONTOLOGY"
        audit.update({
            "accepted": accepted, "status": status, "cv_folds": folds,
            "baseline": baseline, "selected_candidate": best,
            "candidate_results": rows, "utility_gain": utility_gain,
            "balanced_accuracy_gain": ba_gain, "macro_f1_gain": f1_gain,
            "complexity_penalty_delta": complexity_penalty_delta,
            "informational_novelty_ratio": float(len(retained) / len(semantic_indices)),
            "complexity_excess": complexity_excess,
            "selected_indices": selected_indices,
            "selected_feature_names": [names[i] for i in selected_indices],
            "selected_semantic_indices": selected_semantic,
            "selected_semantic_names": [names[i] for i in selected_semantic],
            "selection_stability": {names[i]: float(stability[i]) for i in retained},
            "mean_mutual_information": {names[i]: float(mi_totals[i] / folds) for i in retained},
            "single_feature_effects": single_feature_effects,
            "single_feature_audit_limit": int(self.config.max_single_feature_audits),
            "feature_audit": list(feature_audit.values()),
            "reason": decision_reason,
        })
        return audit

    def assert_usable(self, report: Dict[str, Any]) -> None:
        if not report.get("accepted"):
            status = str(report.get("status"))
            message = "Ontologia não utilizável pelo gate semântico: " + status
            if status == "REJECT_LEAKAGE":
                raise OntologyLeakageError(message)
            if status == "REJECT_INCONSISTENT":
                raise OntologyConsistencyError(message)
            raise OntologyQualityError(message)


__all__ = ["SemanticUtilityConfig", "SemanticUtilityGate"]
