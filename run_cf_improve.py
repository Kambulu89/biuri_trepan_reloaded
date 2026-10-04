#!/usr/bin/env python3
"""Fase 3: melhoria de árvores sustitutos com contrafactuais."""
import sys

from counterfactuals._bootstrap import setup
from validation.counterfactual_research.dataset_registry import ALL_DATASETS
from validation.counterfactual_research.service_batch import improve_surrogates

setup()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] not in ALL_DATASETS:
            print(f"Dataset '{sys.argv[1]}' não reconhecido. Opções: {ALL_DATASETS}")
            sys.exit(1)
        improve_surrogates(dataset_name=sys.argv[1])
    else:
        improve_surrogates()
