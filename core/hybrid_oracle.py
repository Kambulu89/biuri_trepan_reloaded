"""Oráculo híbrido validado e probabilidades OOF alinhadas por classe."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from core.scientific_errors import ClassOrderMismatchError, FeatureSpaceMismatchError


def _aligned_probabilities(model, X, classes: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(np.asarray(X, dtype=float)), dtype=float)
    model_classes = np.asarray(model.classes_)
    aligned = np.zeros((len(raw), len(classes)), dtype=float)
    mapping = {label: idx for idx, label in enumerate(classes)}
    for local, label in enumerate(model_classes):
        if label not in mapping:
            raise ClassOrderMismatchError(f"Classe inesperada no oráculo: {label!r}")
        aligned[:, mapping[label]] = raw[:, local]
    aligned = np.clip(aligned, 1e-12, None)
    return aligned / aligned.sum(axis=1, keepdims=True)


class FeatureProjectedOracle:
    """Expõe um modelo original no schema enriquecido por índices explícitos."""

    ORACLE_TYPE = "projected_original"
    ORACLE_LABEL = "MLP Original projetado no espaço enriquecido"

    def __init__(self, model, original_indices: Sequence[int], n_enriched_features: int):
        self.model = model
        self.original_indices = tuple(int(i) for i in original_indices)
        self.n_features_in_ = int(n_enriched_features)
        self.classes_ = np.asarray(model.classes_)

    def _project(self, X):
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise FeatureSpaceMismatchError(
                f"Schema híbrido incompatível: esperado={self.n_features_in_}; "
                f"recebido={matrix.shape[1] if matrix.ndim == 2 else None}."
            )
        return matrix[:, self.original_indices]

    def predict_proba(self, X):
        return self.model.predict_proba(self._project(X))

    def predict(self, X):
        return self.model.predict(self._project(X))


class WeightedHybridOracle:
    """Combina probabilidades de professores ajustados apenas no desenvolvimento."""

    ORACLE_TYPE = "validated_hybrid_ontological"
    ORACLE_LABEL = "Oráculo Híbrido Ontológico OOF"

    def __init__(
        self,
        components: Mapping[str, object],
        weights: Mapping[str, float],
        classes: Sequence,
        n_features_in: int,
        *,
        training_oof_probabilities: Optional[np.ndarray] = None,
        audit: Optional[dict] = None,
    ):
        self.components = dict(components)
        self.weights = {name: float(weights.get(name, 0.0)) for name in self.components}
        total = sum(self.weights.values())
        if not self.components or total <= 0:
            raise ValueError("Oráculo híbrido exige componentes e pesos positivos.")
        self.weights = {name: value / total for name, value in self.weights.items()}
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in)
        self.training_oof_probabilities_ = (
            None if training_oof_probabilities is None
            else np.asarray(training_oof_probabilities, dtype=float)
        )
        self.training_oof_row_count_ = (
            None if self.training_oof_probabilities_ is None
            else int(len(self.training_oof_probabilities_))
        )
        self.hybrid_audit_ = dict(audit or {})

    def predict_proba(self, X):
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise FeatureSpaceMismatchError("Schema incompatível no oráculo híbrido.")
        combined = np.zeros((len(matrix), len(self.classes_)), dtype=float)
        for name, model in self.components.items():
            combined += self.weights[name] * _aligned_probabilities(
                model, matrix, self.classes_,
            )
        combined = np.clip(combined, 1e-12, None)
        return combined / combined.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@dataclass(frozen=True)
class HybridOracleConfig:
    weight_step: float = 0.10
    minimum_gain_over_original: float = 0.002
    balanced_accuracy_weight: float = 0.40
    macro_f1_weight: float = 0.35
    accuracy_weight: float = 0.15
    calibration_weight: float = 0.10


def _metrics_from_probs(y, probabilities, classes, config: HybridOracleConfig):
    predictions = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    loss = float(log_loss(y, probabilities, labels=np.asarray(classes)))
    result = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "log_loss": loss,
    }
    result["utility"] = float(
        config.balanced_accuracy_weight * result["balanced_accuracy"]
        + config.macro_f1_weight * result["macro_f1"]
        + config.accuracy_weight * result["accuracy"]
        + config.calibration_weight * (1.0 / (1.0 + loss))
    )
    return result


def _simplex_weights(names: Sequence[str], step: float):
    units = int(round(1.0 / float(step)))
    if units < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("weight_step deve dividir 1,0 exatamente.")
    for allocation in itertools.product(range(units + 1), repeat=len(names)):
        if sum(allocation) == units:
            yield {name: value / units for name, value in zip(names, allocation)}


def fit_validation_weighted_hybrid(
    components: Mapping[str, object],
    X_validation,
    y_validation,
    *,
    original_component: str = "original",
    config: HybridOracleConfig = HybridOracleConfig(),
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Aprende pesos só na validação interna e mantém original se não houver ganho."""
    if original_component not in components:
        raise ValueError("Componente original ausente no gate do oráculo híbrido.")
    X = np.asarray(X_validation, dtype=float)
    y = np.asarray(y_validation)
    classes = np.asarray(components[original_component].classes_)
    probabilities = {
        name: _aligned_probabilities(model, X, classes)
        for name, model in components.items()
    }
    names = list(components)
    rows = []
    for weights in _simplex_weights(names, config.weight_step):
        combined = sum(weights[name] * probabilities[name] for name in names)
        metrics = _metrics_from_probs(y, combined, classes, config)
        rows.append({"weights": weights, **metrics})
    original_weights = {name: float(name == original_component) for name in names}
    original_row = next(row for row in rows if row["weights"] == original_weights)
    best = max(
        rows, key=lambda row: (
            row["utility"], row["balanced_accuracy"], row["macro_f1"],
            row["weights"].get(original_component, 0.0),
        ),
    )
    gain = float(best["utility"] - original_row["utility"])
    ontological_mass = float(sum(
        value for name, value in best["weights"].items() if name != original_component
    ))
    accepted = bool(
        gain >= config.minimum_gain_over_original and ontological_mass > 0
    )
    selected = best if accepted else original_row
    return dict(selected["weights"]), {
        "scope": "internal_validation_only", "test_used": False,
        "accepted_hybrid": accepted,
        "status": "ACCEPT_HYBRID" if accepted else "USE_ORIGINAL_TEACHER",
        "original": original_row, "best_candidate": best,
        "selected": selected, "utility_gain": gain,
        "candidate_count": len(rows), "classes": classes.tolist(),
    }


def cross_fitted_probabilities(
    estimator,
    X,
    y,
    *,
    classes: Optional[Sequence] = None,
    cv_folds: int = 5,
    random_state: int = 42,
):
    """Produz exatamente uma probabilidade OOF por linha, com classes alinhadas."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.asarray(classes if classes is not None else np.unique(y))
    _, counts = np.unique(y, return_counts=True)
    folds = max(2, min(int(cv_folds), int(counts.min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    output = np.full((len(y), len(classes)), np.nan, dtype=float)
    coverage = np.zeros(len(y), dtype=int)
    fold_rows = []
    for fold, (fit_idx, hold_idx) in enumerate(cv.split(X, y)):
        model = clone(estimator).fit(X[fit_idx], y[fit_idx])
        output[hold_idx] = _aligned_probabilities(model, X[hold_idx], classes)
        coverage[hold_idx] += 1
        fold_rows.append({"fold": fold, "fit": len(fit_idx), "holdout": len(hold_idx)})
    if not np.all(coverage == 1) or not np.isfinite(output).all():
        raise RuntimeError("Probabilidades híbridas OOF sem cobertura exata.")
    return output, {
        "source": "cross_fitted_predict_proba", "test_used": False,
        "folds": folds, "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()), "fold_rows": fold_rows,
    }


__all__ = [
    "FeatureProjectedOracle", "WeightedHybridOracle", "HybridOracleConfig",
    "fit_validation_weighted_hybrid", "cross_fitted_probabilities",
]
