"""Executor da avaliação DEV de B1 (EXPERIMENTAL): igual a ``run.py --unit`` mas com os braços B1 (calibrações × λ) e respetivos controlos permutados
acrescentados aos braços A–F, no MESMO split, MLP/FrozenOracle, ontologia e tuning estrutural ontology-blind. Só datasets DEV.

    python -m validation.benchmark.run_b1 --unit iris 42 --out DIR [--reuse-checkpoint-from REF_DIR]
``--reuse-checkpoint-from``: copia o checkpoint do tuning estrutural da execução de referência (chaves incluem oracle_id/hash do treino/código;
se não coincidirem o tuning repete-se, com resultados idênticos). Reutilização legítima de resultados de baseline congelados.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

from validation.benchmark import run as R
from validation.benchmark import manifest as mf

CALIBRATIONS = ("tiebreak", "bounded", "calibrated")
LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)       # grelha pré-registada (D-8)


def run_unit_b1(dataset: str, seed: int, out: Path, reuse_checkpoint_from: Path | None = None) -> Path:
    if reuse_checkpoint_from is not None:
        src = Path(reuse_checkpoint_from) / "checkpoints" / f"{dataset}_seed{seed}.ckpt"
        dst = out / "checkpoints" / f"{dataset}_seed{seed}.ckpt"
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    orig_cfg, orig_groups = R.benchmark_config, mf.MAIN_ARM_GROUPS

    def cfg_b1(s: int):
        return dataclasses.replace(orig_cfg(s), b1_calibrations=CALIBRATIONS, b1_lambdas=LAMBDAS)

    R.benchmark_config, mf.MAIN_ARM_GROUPS = cfg_b1, tuple(orig_groups) + ("B1",)
    try:
        return R.run_unit(dataset, seed, out)
    finally:
        R.benchmark_config, mf.MAIN_ARM_GROUPS = orig_cfg, orig_groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    a = ap.parse_args()
    run_unit_b1(a.unit[0], int(a.unit[1]), Path(a.out), Path(a.reuse_checkpoint_from) if a.reuse_checkpoint_from else None)


if __name__ == "__main__":
    sys.exit(main())
