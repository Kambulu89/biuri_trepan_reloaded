"""Validação auditável de lotes contrafactuais por professor e feature space."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import numpy as np

REQUIRED_BATCH_FIELDS = {
    "teacher_id", "teacher_label", "feature_space", "schema_hash", "ontology_hash",
    "dataset_hash", "seed", "method",
}


def validate_counterfactual_batch_metadata(metadata: Dict[str, Any]) -> None:
    missing = sorted(k for k in REQUIRED_BATCH_FIELDS if metadata.get(k) in {None, ""})
    if missing:
        raise ValueError(f"Metadados contrafactuais incompletos: {missing}")


def validate_counterfactual_candidate(
    teacher,
    x_original,
    x_counterfactual,
    *,
    expected_n_features: int,
    minimum_confidence: float = 0.60,
    semantic_valid: bool = True,
    density_ok: bool = True,
    duplicate: bool = False,
    test_derived: bool = False,
) -> Dict[str, Any]:
    xo = np.asarray(x_original, dtype=float).reshape(1, -1)
    xc = np.asarray(x_counterfactual, dtype=float).reshape(1, -1)
    checks = {
        "schema_compatible": xo.shape[1] == expected_n_features and xc.shape[1] == expected_n_features,
        "semantic_valid": bool(semantic_valid),
        "density_ok": bool(density_ok),
        "not_duplicate": not bool(duplicate),
        "not_test_derived": not bool(test_derived),
    }
    if checks["schema_compatible"]:
        before = teacher.predict(xo)[0]
        after = teacher.predict(xc)[0]
        checks["changes_teacher_prediction"] = bool(before != after)
        if hasattr(teacher, "predict_proba"):
            confidence = float(np.max(teacher.predict_proba(xc)))
            checks["minimum_confidence"] = confidence >= float(minimum_confidence)
        else:
            confidence = None
            checks["minimum_confidence"] = True
    else:
        before = after = confidence = None
        checks["changes_teacher_prediction"] = False
        checks["minimum_confidence"] = False
    accepted = bool(all(checks.values()))
    return {
        "accepted": accepted,
        "checks": checks,
        "teacher_prediction_before": before,
        "teacher_prediction_after": after,
        "teacher_confidence": confidence,
        "reason": "valid_counterfactual" if accepted else "rejected:" + ",".join(k for k, v in checks.items() if not v),
    }


def assert_counterfactual_context_compatible(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    keys = ("teacher_id", "feature_space", "schema_hash", "ontology_hash", "dataset_hash")
    mismatches = [k for k in keys if source.get(k) != target.get(k)]
    if mismatches:
        raise ValueError(
            "Lote contrafactual incompatível com o alvo; reutilização entre professor/espaço/ontologia é proibida: "
            + ", ".join(mismatches)
        )


__all__ = [
    "validate_counterfactual_batch_metadata", "validate_counterfactual_candidate",
    "assert_counterfactual_context_compatible", "REQUIRED_BATCH_FIELDS",
]
