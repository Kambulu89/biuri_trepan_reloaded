"""API em lote dos pipelines de contrafactuais sobre datasets NOMEADOS (validação; não usada pela GUI nem pela produção).

As fases usam o registo ``dataset_registry`` (configuração por dataset). O serviço de produção
(``counterfactuals.service``) só trabalha a partir da sessão treinada e dos metadados observados.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from counterfactuals._paths import CF_ROOT, TREPA_ROOT
from validation.counterfactual_research.dataset_registry import ALL_DATASETS, MLP_ACCURACY, get_dataset_config


def _ensure_paths() -> None:
    for path in (str(TREPA_ROOT), str(CF_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def train_oracle_and_surrogates(dataset_name: str) -> Dict[str, Any]:
    """Fase 1: MLP + árvores TREPAN/TREPAN-Reloaded + seleção 33%."""
    _ensure_paths()
    from validation.counterfactual_research.pipeline_train import run_training_pipeline
    return run_training_pipeline(dataset_name)


def generate_counterfactuals(dataset_name: str) -> Dict[str, Any]:
    """Fase 2: CLEAR + COGS + indicadores A/B/C."""
    _ensure_paths()
    from counterfactuals.pipelines.pipeline_counterfactuals import run_cf_pipeline
    df, summary = run_cf_pipeline(dataset_name, get_dataset_config(dataset_name))
    return {'dataframe': df, 'summary': summary}


def improve_surrogates(
    dataset_name: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Fase 3: grid search + improve_surrogate."""
    _ensure_paths()
    from validation.counterfactual_research.pipeline_improve import run_experiment
    targets = datasets or ([dataset_name] if dataset_name else ALL_DATASETS)
    return run_experiment(seed, targets)


def run_full_pipeline(dataset_name: str, seed: int = 42) -> Dict[str, Any]:
    """Executa as três fases sequencialmente."""
    train_summary = train_oracle_and_surrogates(dataset_name)
    cf_result = generate_counterfactuals(dataset_name)
    improve_entries = improve_surrogates(dataset_name=dataset_name, seed=seed)
    return {
        'train': train_summary,
        'counterfactuals': cf_result,
        'improve': improve_entries,
    }


def evaluate_consistency(dataset_name: str) -> Optional[Any]:
    """Carrega indicadores de consistência guardados."""
    import pandas as pd
    from counterfactuals._paths import results_dir
    path = results_dir(dataset_name) / 'consistency_indicators.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_counterfactuals(dataset_name: str, method: str = 'cogs') -> List[Dict[str, Any]]:
    """Carrega CFs de disco (clear ou cogs)."""
    _ensure_paths()
    from validation.counterfactual_research.pipeline_improve import cargar_cfs
    return cargar_cfs(dataset_name, method)


__all__ = ['ALL_DATASETS', 'MLP_ACCURACY', 'train_oracle_and_surrogates', 'generate_counterfactuals', 'improve_surrogates',
           'run_full_pipeline', 'evaluate_consistency', 'load_counterfactuals']
