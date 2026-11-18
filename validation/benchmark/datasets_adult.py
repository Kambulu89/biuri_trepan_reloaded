"""Adult Census (UCI) para a fase 8 da auditoria (DADOS + METADADOS; não altera ``datasets.py`` nem o manifesto v1).

Fonte: ``data/dataset.arff`` do repositório (48 842 linhas). Por limite computacional (seleção do MLP com validação cruzada aninhada e TREPAN
com 6 braços × 3 réplicas por unidade), usa-se uma AMOSTRA FIXA de 2000 linhas, estratificada pela classe, com semente fixa, definida antes de
qualquer execução. Pré-processamento genérico (igual ao do German Credit): numéricas como estão; nominais → one-hot com níveis ordenados,
ajustados na amostra (sem rótulos); o valor em falta «?» é um nível próprio.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from validation.benchmark import datasets as ds_mod
from validation.benchmark.datasets_categorical import OneHotSchema

ROOT = Path(__file__).resolve().parents[2]
ADULT_ARFF = ROOT / "data" / "dataset.arff"
SAMPLE_ROWS = 2000
SAMPLE_SEED = 20261124
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def adult_full() -> pd.DataFrame:
    lines = ADULT_ARFF.read_text(encoding="utf-8", errors="replace").splitlines()
    cols, data = [], []
    in_data = False
    for line in lines:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if s.lower().startswith("@attribute"):
            cols.append(re.split(r"\s+", s, maxsplit=2)[1])
        elif s.lower().startswith("@data"):
            in_data = True
        elif in_data:
            data.append([v.strip() for v in s.split(",")])
    return pd.DataFrame(data, columns=cols)


def adult_sample() -> pd.DataFrame:
    df = adult_full()
    rng = np.random.default_rng(SAMPLE_SEED)
    parts = []
    for label, g in df.groupby("class", sort=True):
        k = int(round(SAMPLE_ROWS * len(g) / len(df)))
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=k, replace=False))])
    return pd.concat(parts).sort_index().reset_index(drop=True)


def adult_loader():
    df = adult_sample()
    y = (df["class"] == ">50K").astype(int).to_numpy()
    X = df.drop(columns=["class"])
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="raise")
    nominal = [c for c in X.columns if c not in NUMERIC]
    enc = OneHotSchema(nominal).fit(X)
    return enc.transform(X), y, enc.names(list(X.columns)), "income", ["<=50K", ">50K"]


ds_mod.REGISTRY.setdefault("adult", ds_mod.DatasetSpec(
    "adult", ds_mod.REAL_OFFLINE, f"UCI Adult Census, data/dataset.arff; amostra estratificada fixa de {SAMPLE_ROWS} linhas (semente {SAMPLE_SEED})",
    f"sample-{SAMPLE_ROWS}-seed-{SAMPLE_SEED}", adult_loader, "adult", None, str(ADULT_ARFF), ranking_eligible=False))
