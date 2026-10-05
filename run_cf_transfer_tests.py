#!/usr/bin/env python3
"""Executa a suíte experimental P9; não faz parte do fluxo da aplicação."""
from __future__ import annotations

import argparse

from validation.counterfactual_research.dataset_registry import ALL_DATASETS
from tests.support.counterfactual_multidataset_runner import run_six_dataset_test


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teste P9 isolado em pelo menos seis datasets de referência."
    )
    parser.add_argument("datasets", nargs="*", default=ALL_DATASETS)
    parser.add_argument("--methods", nargs="+", default=["LORE-LOCAL", "CLEAR", "COGS"])
    parser.add_argument("--fraction", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robustness-samples", type=int, default=100)
    parser.add_argument("--robustness-epsilon", type=float, default=0.02)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run_six_dataset_test(
        args.datasets,
        methods=args.methods,
        fraction=args.fraction,
        seed=args.seed,
        robustness_samples=args.robustness_samples,
        robustness_epsilon=args.robustness_epsilon,
        output_dir=args.output_dir,
    )
    print(
        f"Teste P9 concluído em {result['global_summary']['dataset_count']} datasets."
    )
    for label, path in result["exported_files"].items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
