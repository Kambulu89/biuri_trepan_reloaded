"""Executor P1 (Doc. 26 §5): avaliação do amostrador C em datasets DEV com atributos categóricos reais (registados em ``datasets_categorical``).
Igual a ``run_sampler_c`` mas (i) regista os datasets categóricos e (ii) declara ao amostrador as features ordinais/contagens dos METADADOS do dataset.
    python -m validation.benchmark.run_sampler_c_cat --unit german_credit 42 --out DIR [--reps 3]
"""
from __future__ import annotations

import functools
import sys

from validation.benchmark import datasets_categorical as dc
from validation.benchmark import datasets as ds_mod
from validation.benchmark import run_sampler_c as SCM
import core.query_sampling as QS


def main() -> None:
    argv = sys.argv
    dataset = argv[argv.index("--unit") + 1]
    names = ds_mod.REGISTRY[dataset].loader()[2]
    cols = dc.ordinal_columns(dataset, names)
    SCM.joint_sampler_context = functools.partial(QS.joint_sampler_context, ordinal_columns=cols)
    SCM.main()


if __name__ == "__main__":
    sys.exit(main())
