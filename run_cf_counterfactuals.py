#!/usr/bin/env python3
"""Fase 2: geração de contrafactuais (CLEAR + COGS) e indicadores de consistência."""
import sys

from counterfactuals._bootstrap import setup
from validation.counterfactual_research.dataset_registry import ALL_DATASETS
from validation.counterfactual_research.service_batch import generate_counterfactuals

setup()

if __name__ == "__main__":
    datasets = [sys.argv[1]] if len(sys.argv) > 1 else ALL_DATASETS
    if len(sys.argv) > 1 and sys.argv[1] not in ALL_DATASETS:
        print(f"Dataset '{sys.argv[1]}' não reconhecido. Opções: {ALL_DATASETS}")
        sys.exit(1)

    for ds in datasets:
        generate_counterfactuals(ds)
