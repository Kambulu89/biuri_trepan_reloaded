"""Acesso somente-leitura ao benchmark confirmatório V7 congelado.

A execução única confirmatória já foi consumida. A build de produção não
reconstrói nem volta a executar aquele protocolo; preserva apenas os artefactos
em ``results/confirmatory_v7`` para auditoria e reprodutibilidade documental.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from core.scientific_errors import ScientificProtocolError


@dataclass(frozen=True)
class ConfirmatoryConfig:
    repeats: int = 3
    test_size: float = 0.25
    validation_size: float = 0.22
    max_samples: int = 600
    random_state: int = 20260830
    mlp_search_iterations: int = 5
    mlp_cv_folds: int = 3
    active_iterations: int = 2
    active_budget: int = 32
    reasoner_engine: str = "hermit"
    oracle_noninferiority_tolerance: float = 0.01
    surrogate_noninferiority_tolerance: float = 0.01
    surrogate_fidelity_floor: float = 0.90
    surrogate_complexity_ratio_limit: float = 1.50


def load_frozen_confirmatory_results(root: str | Path = "results/confirmatory_v7"):
    root = Path(root)
    path = root / "confirmatory_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Resultado confirmatório congelado não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_confirmatory_once(*args, **kwargs):
    raise ScientificProtocolError(
        "A execução confirmatória V7 está congelada e não pode ser repetida na build de produção. "
        "Use load_frozen_confirmatory_results() para auditoria dos resultados existentes."
    )


__all__ = ["ConfirmatoryConfig", "run_confirmatory_once", "load_frozen_confirmatory_results"]
