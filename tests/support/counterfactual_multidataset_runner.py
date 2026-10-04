"""Validação experimental multi-dataset - exclusiva para testes P9."""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np

from counterfactuals._paths import RESULTADOS_CONSOLIDADOS_DIR, models_dir
from validation.counterfactual_research.dataset_registry import ALL_DATASETS
from counterfactuals.transfer import evaluate_transfer_protocol


def load_test_session(dataset_name: str) -> Dict[str, Any]:
    """Carrega artefactos dos datasets de referência para testes, não para a GUI."""
    directory = models_dir(dataset_name)
    paths = {
        "mlp_oracle": directory / f"mlp_{dataset_name}.pkl",
        "X_train_enc": directory / f"X_train_enc_{dataset_name}.pkl",
        "y_train_enc": directory / f"y_train_enc_{dataset_name}.pkl",
        "tree_a": directory / f"trepan_{dataset_name}.pkl",
        "tree_b": directory / f"trepan_reloaded_{dataset_name}.pkl",
        "state": directory / f"pipeline_state_{dataset_name}.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Artefactos de teste ausentes para {dataset_name}: " + ", ".join(missing)
        )
    state_path = paths.pop("state")
    loaded = {key: joblib.load(path) for key, path in paths.items()}
    for key in ("tree_a", "tree_b"):
        if not hasattr(loaded[key], "monotonic_cst"):
            loaded[key].monotonic_cst = None
    try:
        state = joblib.load(state_path)
    except Exception as exc:
        warnings.warn(
            f"Metadados de teste indisponíveis para {dataset_name}: {exc}; "
            "serão usados nomes posicionais.",
            RuntimeWarning,
        )
        state = {}
    X = np.asarray(loaded["X_train_enc"], dtype=float)
    y = np.asarray(loaded["y_train_enc"])
    names = list(state.get("transformed_feature_names") or [])
    if not names:
        names = [f"feature_{index}" for index in range(X.shape[1])]
    if len(names) != X.shape[1]:
        raise ValueError(
            f"Schema de teste inválido em {dataset_name}: {X.shape[1]} colunas "
            f"e {len(names)} nomes."
        )
    return {
        **loaded,
        "dataset_name": dataset_name,
        "mlp_original": loaded["mlp_oracle"],
        "X_train_original": X,
        "y_train_original": y,
        "transformed_feature_names": names,
        "feature_names_original": names,
        "tree_a_feature_names": names,
        "tree_b_feature_names": names,
        "X_tree_a_reference": X,
        "X_tree_b_reference": X,
        "config": {
            "class_labels": state.get("class_labels") or {},
            "category_prefix": state.get("category_prefix") or [],
        },
        "ontology_acceptance": {"accepted": False},
    }


def evaluate_six_dataset_test(
    sessions: Mapping[str, Mapping[str, Any]],
    *,
    minimum_datasets: int = 6,
    **kwargs,
) -> Dict[str, Any]:
    """P9 experimental: nunca é chamado pelo serviço ou pela interface."""
    if len(sessions) < minimum_datasets:
        raise ValueError(
            f"O teste P9 exige {minimum_datasets} datasets; recebidos {len(sessions)}."
        )
    results = {
        name: evaluate_transfer_protocol({**session, "dataset_name": name}, **kwargs)
        for name, session in sessions.items()
    }
    method_names = sorted({
        method
        for result in results.values()
        for method in result.get("summary_by_method", {})
    })
    by_method = {}
    for method in method_names:
        summaries = [
            result["summary_by_method"][method]
            for result in results.values()
            if method in result.get("summary_by_method", {})
        ]
        by_method[method] = {
            "dataset_count": len(summaries),
            "mean_valid_mlp_cf_rate": _mean(
                row.get("valid_mlp_cf_rate") for row in summaries
            ),
            "mean_strong_transfer_rate": _mean(
                row.get("strong_transfer_rate") for row in summaries
            ),
            "mean_joint_robustness": _mean(
                row.get("mean_joint_robustness") for row in summaries
            ),
        }
    return {
        "status": "success",
        "protocol": "P9_TEST_ONLY",
        "test_scope": "six_reference_datasets",
        "datasets": results,
        "global_summary": {
            "dataset_count": len(results),
            "mean_strong_transfer_rate": _mean(
                item["summary"].get("strong_transfer_rate") for item in results.values()
            ),
            "mean_joint_robustness": _mean(
                item["summary"].get("mean_joint_robustness") for item in results.values()
            ),
            "by_method": by_method,
        },
    }


def run_six_dataset_test(
    datasets: Optional[Sequence[str]] = None,
    *,
    methods: Iterable[str] = ("LORE-LOCAL", "CLEAR", "COGS"),
    fraction: float = 0.33,
    seed: int = 42,
    robustness_samples: int = 100,
    robustness_epsilon: float = 0.02,
    output_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    names = list(datasets or ALL_DATASETS)
    sessions = {name: load_test_session(name) for name in names}
    result = evaluate_six_dataset_test(
        sessions,
        methods=tuple(methods),
        fraction=fraction,
        seed=seed,
        robustness_samples=robustness_samples,
        robustness_epsilon=robustness_epsilon,
    )
    result["exported_files"] = export_test_results(
        result,
        output_dir or (RESULTADOS_CONSOLIDADOS_DIR / "testes_p9_multidataset"),
    )
    return result


def export_test_results(result: Mapping[str, Any], output_dir: Any) -> Dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "test_p9_multidataset.json"
    rows_path = directory / "test_p9_rows.csv"
    summary_path = directory / "test_p9_summary.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    rows, summaries = [], []
    for dataset, item in (result.get("datasets") or {}).items():
        rows.extend(
            {"dataset": dataset, **_flat(row)} for row in item.get("rows") or []
        )
        summaries.append({
            "dataset": dataset, "method": "ALL", **_flat(item.get("summary") or {}),
        })
        summaries.extend(
            {"dataset": dataset, "method": method, **_flat(summary)}
            for method, summary in (item.get("summary_by_method") or {}).items()
        )
    _write_csv(rows_path, rows)
    _write_csv(summary_path, summaries)
    return {"json": str(json_path), "rows_csv": str(rows_path), "summary_csv": str(summary_path)}


def _mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def _flat(values: Mapping[str, Any]) -> Dict[str, Any]:
    output = {}
    for key, value in values.items():
        if isinstance(value, (Mapping, list, tuple, np.ndarray)):
            output[key] = json.dumps(value, ensure_ascii=False, default=_json_default)
        else:
            output[key] = _json_default(value) if isinstance(value, np.generic) else value
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Tipo não serializável: {type(value).__name__}")


__all__ = [
    "load_test_session", "evaluate_six_dataset_test", "run_six_dataset_test",
    "export_test_results",
]
