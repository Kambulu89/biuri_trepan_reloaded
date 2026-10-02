"""Validação biomédica/clinicamente orientada para classificadores.

Não certifica uso clínico. Produz métricas, verificações de independência e um
guardião de alegações alinhado com TRIPOD+AI, DECIDE-AI e princípios GMLP.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)


def _binary_metrics(y_true, y_pred, positive_label) -> Dict[str, float]:
    labels = [label for label in np.unique(np.r_[y_true, y_pred]) if label != positive_label]
    negative = labels[0] if labels else 0
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[negative, positive_label]
    ).ravel()
    return {
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "ppv": float(tp / max(1, tp + fp)),
        "npv": float(tn / max(1, tn + fn)),
        "prevalence": float(np.mean(np.asarray(y_true) == positive_label)),
    }


def _ece(y_binary: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if np.any(mask):
            result += mask.mean() * abs(float(y_binary[mask].mean() - probability[mask].mean()))
    return float(result)


def _decision_curve(y_binary: np.ndarray, probability: np.ndarray) -> list[dict]:
    n = len(y_binary)
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        predicted = probability >= threshold
        tp = int(np.sum(predicted & (y_binary == 1)))
        fp = int(np.sum(predicted & (y_binary == 0)))
        net_benefit = tp / n - fp / n * threshold / (1 - threshold)
        rows.append({"threshold": float(threshold), "net_benefit": float(net_benefit)})
    return rows


def validate_biomedical_model(
    y_true,
    y_pred,
    *,
    y_proba=None,
    classes: Optional[Sequence] = None,
    positive_label=None,
    patient_ids_test: Optional[Sequence] = None,
    patient_ids_train: Optional[Sequence] = None,
    site_ids_test: Optional[Sequence] = None,
    subgroup_values: Optional[Mapping[str, Sequence]] = None,
    timestamps_train: Optional[Sequence] = None,
    timestamps_test: Optional[Sequence] = None,
    external_validation: bool = False,
    random_state: int = 42,
) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true/y_pred devem ser vetores compatíveis.")
    unique = np.unique(np.r_[y_true, y_pred])
    positive = positive_label if positive_label is not None else unique[-1]
    result = {
        "n": int(len(y_true)),
        "external_validation": bool(external_validation),
        "guidelines": ["TRIPOD+AI", "DECIDE-AI", "FDA/IMDRF GMLP"],
        "patient_split": "unknown",
        "site_coverage": int(len(np.unique(site_ids_test))) if site_ids_test is not None else None,
        "temporal_split": "unknown",
    }
    if patient_ids_test is not None:
        test_ids = np.asarray(patient_ids_test)
        if len(test_ids) != len(y_true):
            raise ValueError("patient_ids_test incompatível com y_true.")
        overlap = set(test_ids.tolist()) & set(np.asarray(patient_ids_train).tolist()) \
            if patient_ids_train is not None else set()
        result["patient_split"] = "independent" if not overlap else "leakage"
        result["patient_overlap_count"] = len(overlap)
    if timestamps_train is not None and timestamps_test is not None:
        train_t = np.asarray(timestamps_train)
        test_t = np.asarray(timestamps_test)
        result["temporal_split"] = (
            "prospective" if train_t.max() < test_t.min() else "overlap"
        )

    if len(unique) == 2:
        result["diagnostic"] = _binary_metrics(y_true, y_pred, positive)
        if y_proba is not None:
            probs = np.asarray(y_proba, dtype=float)
            class_list = list(classes) if classes is not None else list(unique)
            if probs.ndim == 2:
                if positive not in class_list:
                    raise ValueError("positive_label não existe em classes.")
                probability = probs[:, class_list.index(positive)]
            elif probs.ndim == 1:
                probability = probs
            else:
                raise ValueError("y_proba deve ser vetor ou matriz 2D.")
            binary = (y_true == positive).astype(int)
            result["discrimination"] = {
                "auroc": float(roc_auc_score(binary, probability)),
                "auprc": float(average_precision_score(binary, probability)),
            }
            result["calibration"] = {
                "brier": float(brier_score_loss(binary, probability)),
                "ece_10": _ece(binary, probability),
            }
            clipped = np.clip(probability, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            if len(np.unique(binary)) == 2:
                calibration = LogisticRegression(C=1e6, max_iter=500).fit(logits, binary)
                result["calibration"].update({
                    "intercept": float(calibration.intercept_[0]),
                    "slope": float(calibration.coef_[0, 0]),
                })
            result["decision_curve"] = _decision_curve(binary, probability)
    else:
        result["diagnostic"] = {
            "macro_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_ppv": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    subgroup_report = {}
    for name, values in (subgroup_values or {}).items():
        arr = np.asarray(values)
        if len(arr) != len(y_true):
            raise ValueError(f"Subgrupo {name!r} incompatível com y_true.")
        rows = {}
        for value in np.unique(arr):
            mask = arr == value
            rows[str(value)] = {
                "n": int(mask.sum()),
                "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
                "sensitivity_macro": float(
                    recall_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
                ),
            }
        subgroup_report[str(name)] = rows
    result["subgroups"] = subgroup_report

    blockers = []
    if result["patient_split"] == "leakage":
        blockers.append("patient_overlap")
    if patient_ids_test is None:
        blockers.append("patient_ids_missing")
    if y_proba is None:
        blockers.append("calibration_not_evaluated")
    if not external_validation:
        blockers.append("external_validation_missing")
    if site_ids_test is None:
        blockers.append("site_validation_missing")
    if any(row["n"] < 20 for groups in subgroup_report.values() for row in groups.values()):
        blockers.append("small_subgroup")
    result["clinical_claim_guard"] = {
        "clinical_superiority_claim_allowed": not blockers,
        "blockers": blockers,
        "statement": (
            "Avaliação clínica mínima satisfeita; ainda requer governança regulatória."
            if not blockers else
            "Não alegar superioridade clínica; requisitos de validação permanecem pendentes."
        ),
    }
    return result
