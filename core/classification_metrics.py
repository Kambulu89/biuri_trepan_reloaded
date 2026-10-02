"""Métricas canónicas de classificação para o protocolo BIURI/TREPAN.

Mantém Precision, Accuracy e fidelidade semanticamente separadas. A UI pode
escolher qual métrica mostrar, mas nunca reutiliza ``accuracy_score`` para
preencher um campo denominado Precision/Precisão.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true, y_pred, *, labels: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true e y_pred devem ter o mesmo número de linhas.")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = []
    total = int(cm.sum())
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        predicted_positive = int(cm[:, i].sum())
        per_class.append({
            "class": label.item() if isinstance(label, np.generic) else label,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "predicted_positive": predicted_positive,
        })

    _, counts = np.unique(y_pred, return_counts=True)
    degeneracy_rate = float(counts.max() / len(y_pred)) if len(y_pred) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "labels": [v.item() if isinstance(v, np.generic) else v for v in labels],
        "per_class": per_class,
        "unique_predictions": int(len(np.unique(y_pred))),
        "degenerate_prediction_rate": degeneracy_rate,
        "metric_semantics": {
            "accuracy": "accuracy_score_against_real_labels",
            "precision_macro": "precision_score_average_macro_against_real_labels",
            "precision_weighted": "precision_score_average_weighted_against_real_labels",
            "recall_macro": "recall_score_average_macro_against_real_labels",
            "macro_f1": "f1_score_average_macro_against_real_labels",
            "balanced_accuracy": "balanced_accuracy_score_against_real_labels",
        },
    }


def metric_value(metrics: Dict[str, Any], name: str, default: float = 0.0) -> float:
    aliases = {
        "precision": "precision_macro",
        "f1": "macro_f1",
        "recall": "recall_macro",
    }
    key = aliases.get(name, name)
    value = metrics.get(key, default)
    return float(default if value is None else value)


__all__ = ["compute_classification_metrics", "metric_value"]
