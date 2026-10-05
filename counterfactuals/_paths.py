"""Caminhos partilhados pelos pipelines de contrafactuais."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

CF_ROOT = Path(__file__).resolve().parent
TREPA_ROOT = CF_ROOT.parent
EXPERIMENTOS_DIR = CF_ROOT / "experimentos"
CLEAR_OUTPUT_DIR = CF_ROOT / "clear_output"
RESULTADOS_CONSOLIDADOS_DIR = CF_ROOT / "resultados_consolidados"
RESULTADOS_MEJORA_DIR = CF_ROOT / "resultados_mejora"


def experiment_dir(dataset_name: str) -> Path:
    return EXPERIMENTOS_DIR / dataset_name


def models_dir(dataset_name: str) -> Path:
    return experiment_dir(dataset_name) / "modelos"


def results_dir(dataset_name: str) -> Path:
    return experiment_dir(dataset_name) / "resultados"


def mejora_dir(dataset_name: str) -> Path:
    return experiment_dir(dataset_name) / "mejora"


def clear_output_dir(dataset_name: str) -> Path:
    return CLEAR_OUTPUT_DIR / dataset_name


def dataset_paths(dataset_name: str) -> Dict[str, Path]:
    return {
        'experiment': experiment_dir(dataset_name),
        'models': models_dir(dataset_name),
        'resultados': results_dir(dataset_name),
        'mejora': mejora_dir(dataset_name),
        'clear_output': clear_output_dir(dataset_name),
    }


def ensure_dirs(dataset_name: str) -> None:
    for path in dataset_paths(dataset_name).values():
        path.mkdir(parents=True, exist_ok=True)
