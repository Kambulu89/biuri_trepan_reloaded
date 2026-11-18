"""Executor da FASE 8 da auditoria (EXPERIMENTAL): o pipeline da fase 7 (MLP otimizado congelado + Reloaded O + amostrador conjunto,
orçamentos «tuning» e 7 nós) em datasets REAIS. Só acrescenta o registo do Adult; tudo o resto é ``run_fase7`` sem alterações.

    python -m validation.benchmark.run_fase8 --unit adult 42 --out DIR --reloaded O [--reps 3]
"""
from __future__ import annotations

import sys

from validation.benchmark import run as R      # noqa: F401  PRIMEIRO: fixa o ambiente numérico
from validation.benchmark import datasets_adult  # noqa: F401  (regista o Adult)
from core.benchmark import runner as RN
from validation.benchmark import run_fase6 as F6
from validation.benchmark.run_fase7 import fit_mlp_fase7

if __name__ == "__main__":
    RN.fit_mlp = fit_mlp_fase7
    sys.exit(F6.main())
