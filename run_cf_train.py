#!/usr/bin/env python3
"""Fase 1: treino MLP + extração de árvores sustitutos."""
import sys

from counterfactuals._bootstrap import setup
from validation.counterfactual_research.dataset_registry import ALL_DATASETS
from validation.counterfactual_research.service_batch import train_oracle_and_surrogates

setup()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        datasets = [sys.argv[1]]
        if sys.argv[1] not in ALL_DATASETS:
            print(f"Dataset '{sys.argv[1]}' não reconhecido. Opções: {ALL_DATASETS}")
            sys.exit(1)
    else:
        datasets = ALL_DATASETS

    for ds in datasets:
        train_oracle_and_surrogates(ds)
