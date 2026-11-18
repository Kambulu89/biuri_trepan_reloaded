"""Executor da FASE 7 da auditoria (EXPERIMENTAL): igual a ``run_fase6`` (mesmos braços, orçamentos e Reloaded O congelado na fase 5), com o
MLP/oráculo substituído pelo MLP otimizado de ``mlp_selection_fase7`` (seleção só no treino). Campanha nova; não substitui a fase 6.

    python -m validation.benchmark.run_fase7 --unit phys_bmi 42 --out DIR --reloaded O [--reps 3]
"""
from __future__ import annotations

import sys

from validation.benchmark import run as R      # noqa: F401  PRIMEIRO: fixa o ambiente numérico
from core.benchmark import runner as RN
from validation.benchmark import run_fase6 as F6
from validation.benchmark.mlp_selection_fase7 import select_and_fit


def fit_mlp_fase7(X, y, seed: int, trials: int = 0, inner_splits: int = 3):
    return select_and_fit(X, y, seed)


if __name__ == "__main__":
    RN.fit_mlp = fit_mlp_fase7
    sys.exit(F6.main())
