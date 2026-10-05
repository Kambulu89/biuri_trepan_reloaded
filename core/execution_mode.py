"""Modos de execução: o exploratório (interativo) nunca é benchmark científico por omissão.

- ``SCIENTIFIC_BENCHMARK``: usa exatamente o pipeline científico (``core/scientific_benchmark_service.py`` ->
  ``train_production_dataframe``) com o contrato do oráculo congelado. Os seus resultados podem ser usados como benchmark.
- ``INTERACTIVE_EXPLORATORY``: o modo normal da GUI (MLP com Optuna, interação, cache). Resultados exploratórios:
  não são utilizados automaticamente como benchmark científico.
"""
from __future__ import annotations

import enum


class ExecutionMode(str, enum.Enum):
    SCIENTIFIC_BENCHMARK = "SCIENTIFIC_BENCHMARK"
    INTERACTIVE_EXPLORATORY = "INTERACTIVE_EXPLORATORY"

    @property
    def label(self) -> str:
        return {"SCIENTIFIC_BENCHMARK": "SCIENTIFIC / BENCHMARK",
                "INTERACTIVE_EXPLORATORY": "INTERACTIVE / EXPLORATORY"}[self.value]

    @property
    def usable_as_benchmark(self) -> bool:
        return self is ExecutionMode.SCIENTIFIC_BENCHMARK


def parse_mode(value) -> ExecutionMode:
    if isinstance(value, ExecutionMode):
        return value
    try:
        return ExecutionMode(str(value))
    except ValueError as exc:
        raise ValueError(f"Modo de execução desconhecido: {value!r}") from exc


DEFAULT_MODE = ExecutionMode.INTERACTIVE_EXPLORATORY

__all__ = ["ExecutionMode", "parse_mode", "DEFAULT_MODE"]
