"""Auditoria imutável do protocolo usado na comparação de modelos.

Os hashes não são critérios de seleção: servem apenas para provar que os modelos
foram avaliados nas mesmas linhas lógicas, schema e matrizes previamente
congeladas. O teste final nunca deve ser usado para ajustar hiperparâmetros.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def stable_sha256_array(values) -> str:
    """Hash SHA-256 determinístico de forma, dtype e conteúdo completo."""
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": array.shape, "dtype": str(array.dtype)}, sort_keys=True).encode("utf-8"))
    if array.dtype.kind in {"O", "U", "S"}:
        payload = json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":"), default=_json_default)
        digest.update(payload.encode("utf-8"))
    else:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def stable_sha256_strings(values: Optional[Sequence[Any]]) -> Optional[str]:
    if values is None:
        return None
    payload = json.dumps([str(v) for v in values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _missing_count(values) -> int:
    array = np.asarray(values)
    if array.dtype.kind in {"f", "c"}:
        return int(np.isnan(array).sum())
    try:
        return int(np.sum(array != array))
    except Exception:
        return 0


def build_protocol_audit(
    X_test,
    y_test,
    *,
    X_test_reloaded=None,
    feature_names: Optional[Sequence[Any]] = None,
    feature_names_reloaded: Optional[Sequence[Any]] = None,
    class_order_original: Optional[Sequence[Any]] = None,
    class_order_active_oracle: Optional[Sequence[Any]] = None,
    test_row_ids=None,
    seed: Optional[int] = None,
    repeat_count: int = 1,
    preprocessing_id: str = "caller_provided_preprocessed_matrix",
) -> Dict[str, Any]:
    X = np.asarray(X_test)
    y = np.asarray(y_test)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Protocolo inválido: X_test/y_test têm dimensões incompatíveis.")

    Xr = None if X_test_reloaded is None else np.asarray(X_test_reloaded)
    if Xr is not None and (Xr.ndim != 2 or len(Xr) != len(y)):
        raise ValueError(
            "Protocolo inválido: X_test_reloaded deve representar exatamente as mesmas linhas de y_test."
        )

    if test_row_ids is None:
        row_ids = np.arange(len(y), dtype=np.int64)
        row_id_source = "positional_ids_generated_after_locked_split"
    else:
        row_ids = np.asarray(test_row_ids)
        if len(row_ids) != len(y):
            raise ValueError("test_row_ids não corresponde ao número de linhas de y_test.")
        row_id_source = "caller_provided"

    logical_rows_digest = hashlib.sha256()
    logical_rows_digest.update(stable_sha256_array(row_ids).encode("ascii"))
    logical_rows_digest.update(stable_sha256_array(y).encode("ascii"))

    return {
        "protocol_version": "BIURI_V9_1_AUDITFIX_1",
        "role": "locked_test_reporting_only",
        "test_used_for_selection": False,
        "n_test_rows": int(len(y)),
        "same_logical_row_count": bool(Xr is None or len(Xr) == len(X)),
        "row_id_source": row_id_source,
        "logical_test_rows_sha256": logical_rows_digest.hexdigest(),
        "test_row_ids_sha256": stable_sha256_array(row_ids),
        "y_test_sha256": stable_sha256_array(y),
        "X_test_original_sha256": stable_sha256_array(X),
        "X_test_reloaded_sha256": stable_sha256_array(Xr) if Xr is not None else None,
        "schema_original_sha256": stable_sha256_strings(feature_names),
        "schema_reloaded_sha256": stable_sha256_strings(feature_names_reloaded),
        "class_order_original": [str(v) for v in class_order_original] if class_order_original is not None else None,
        "class_order_active_oracle": [str(v) for v in class_order_active_oracle] if class_order_active_oracle is not None else None,
        "class_order_original_sha256": stable_sha256_strings(class_order_original),
        "class_order_active_oracle_sha256": stable_sha256_strings(class_order_active_oracle),
        "missing_count_original": _missing_count(X),
        "missing_count_reloaded": _missing_count(Xr) if Xr is not None else None,
        "preprocessing_id": str(preprocessing_id),
        "seed": None if seed is None else int(seed),
        "repeat_count": int(repeat_count),
    }


__all__ = ["stable_sha256_array", "stable_sha256_strings", "build_protocol_audit"]
