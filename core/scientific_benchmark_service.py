"""Serviço do modo SCIENTIFIC / BENCHMARK: o ÚNICO ponto de entrada científico, partilhado pela GUI e pela produção.

Não reimplementa nada: delega em ``train_production_dataframe`` (mesmo código da produção) e verifica o contrato do
oráculo congelado. Fluxo:

    Dataset -> split -> treino/calibração do MLP (só treino) -> FrozenOracle -> oracle_id -> tuning (só treino)
            -> TREPAN Original -> TREPAN Reloaded -> avaliação final (teste usado uma só vez)

O modo interativo/exploratório da GUI fica separado (``core/execution_mode.py``) e os seus resultados nunca são
tratados como benchmark.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from core.execution_mode import ExecutionMode
from core.experiment_builders import from_production_report
from core.experiment_result import ExperimentResult
from core.production_training import train_production_dataframe
from core.scientific_experiment_contract import OracleContractViolation
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig

# Configuração científica principal: CV repetida 5×3 (Repeated Stratified K-Fold, só treino).
SCIENTIFIC_CV_REPEATS = 5
SCIENTIFIC_CV_FOLDS = 3


def scientific_search_config(**overrides) -> ScientificTrepanSearchConfig:
    """Configuração científica principal do tuning (CV 5×3); ``overrides`` só para testes/ablações."""
    values = dict(cv_folds=SCIENTIFIC_CV_FOLDS, cv_repeats=SCIENTIFIC_CV_REPEATS)
    values.update(overrides)
    return ScientificTrepanSearchConfig(**values)


@dataclass
class BenchmarkOutcome:
    execution_mode: str
    report: Mapping[str, Any]
    result: ExperimentResult
    oracle_id: str
    artifacts: Optional[Mapping[str, Any]] = None   # objetos exatos avaliados (árvores, oráculo congelado, nomes)

    def tree_view(self) -> dict:
        """Dados para os visualizadores: as MESMAS árvores avaliadas (nunca retreinadas), com oracle_id e configuração."""
        a = dict(self.artifacts or {})
        return {"trepan_original": a.get("trepan_original"), "trepan_reloaded": a.get("trepan_reloaded"),
                "feature_names_original": a.get("feature_names_original"), "feature_names_reloaded": a.get("feature_names_reloaded"),
                "class_names": a.get("class_names"), "oracle_id": self.oracle_id, "selected_config": a.get("selected_config")}

    @property
    def usable_as_benchmark(self) -> bool:
        return bool(self.result.scientific and self.result.scientific.benchmark_eligible)


def frame_from_arrays(X, y, feature_names: Sequence[str], target_name: str) -> pd.DataFrame:
    """DataFrame (colunas brutas + alvo) a partir das matrizes carregadas pela GUI, sem recodificar nada."""
    X = np.asarray(X, dtype=object)
    if X.ndim != 2 or X.shape[1] != len(feature_names) or len(X) != len(y):
        raise ValueError("X/y/feature_names inconsistentes para o modo benchmark.")
    df = pd.DataFrame(X, columns=[str(c) for c in feature_names]).infer_objects()
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().all():
                df[col] = converted
    df[str(target_name)] = np.asarray(y)
    return df


def verify_oracle_contract(report: Mapping[str, Any]) -> str:
    """Garante que Original e Reloaded consultaram exatamente o mesmo oráculo congelado; devolve o ``oracle_id``."""
    contract = (report.get("evaluation") or {}).get("oracle_contract") or {}
    ids = contract.get("tree_oracle_ids") or {}
    oracle_id = (contract.get("oracle") or {}).get("oracle_id")
    if not oracle_id or set(ids) != {"trepan_original", "trepan_reloaded"} or set(ids.values()) != {oracle_id}:
        raise OracleContractViolation(f"Contrato do oráculo não cumprido: {ids} vs {oracle_id}")
    if not contract.get("unchanged_after") or not contract.get("single_oracle_for_all_trees"):
        raise OracleContractViolation("O oráculo mudou ou não é único para todas as árvores.")
    return str(oracle_id)


def run_scientific_benchmark(
    df: pd.DataFrame, *, target: str, seed: int = 42, out_dir: Optional[str | Path] = None,
    owl_path: Optional[str | Path] = None, ontology=None, reasoner_report: Optional[dict] = None,
    require_reasoner: bool = True, search: Optional[ScientificTrepanSearchConfig] = None,
    oracle_builder: str = "factory", progress_fn: Optional[Callable[[str, int, str], None]] = None,
    **production_kwargs,
) -> BenchmarkOutcome:
    """Executa o pipeline científico completo (o MESMO ``train_production_dataframe`` da produção)."""
    if progress_fn:
        progress_fn("init", 5, "scientific benchmark: dataset -> split -> oráculo congelado")

    def _run(directory):
        return train_production_dataframe(
            df, target=target, out_dir=directory, seed=seed, owl_path=owl_path, ontology=ontology,
            reasoner_report=reasoner_report, require_reasoner=require_reasoner, scientific_tuning=True,
            trepan_search=search or scientific_search_config(), oracle_builder=oracle_builder, return_artifacts=True, **production_kwargs)

    if out_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            report = _run(tmp)
    else:
        report = _run(out_dir)
    oracle_id = verify_oracle_contract(report)
    report["execution_mode"] = ExecutionMode.SCIENTIFIC_BENCHMARK.value
    result = from_production_report(report)
    result.provenance.source = "gui_benchmark"
    result.provenance.training_mode = ExecutionMode.SCIENTIFIC_BENCHMARK.value
    if progress_fn:
        progress_fn("done", 100, f"oracle_id={oracle_id[:8]}")
    artifacts = report.pop("artifacts", None)         # fora do relatório serializável
    if artifacts is not None and artifacts.get("oracle_id") != oracle_id:
        raise OracleContractViolation("As árvores devolvidas não pertencem ao oráculo congelado verificado.")
    return BenchmarkOutcome(ExecutionMode.SCIENTIFIC_BENCHMARK.value, report, result, oracle_id, artifacts)


__all__ = ["BenchmarkOutcome", "SCIENTIFIC_CV_FOLDS", "SCIENTIFIC_CV_REPEATS", "frame_from_arrays",
           "run_scientific_benchmark", "scientific_search_config", "verify_oracle_contract"]
