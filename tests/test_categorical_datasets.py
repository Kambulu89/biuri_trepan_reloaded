"""Datasets categóricos DEV (P1): codificação genérica, categorias desconhecidas, dicionário de dados e ontologia (Doc. 26 §5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.benchmark import datasets_categorical as dc
from validation.benchmark import datasets as ds_mod


def test_onehot_unknown_category_is_all_zero_not_invented():
    df = pd.DataFrame({"c": ["a", "b", "a"], "x": [1.0, 2.0, 3.0]})
    enc = dc.OneHotSchema(["c"]).fit(df)
    new = pd.DataFrame({"c": ["b", "ZZZ"], "x": [5.0, 6.0]})
    out = enc.transform(new)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [0.0, 1.0, 5.0] and out[1].tolist() == [0.0, 0.0, 6.0]
    assert enc.names(["c", "x"]) == ["c__a", "c__b", "x"]


def test_german_loader_and_groups_cover_all_features():
    X, y, names, target, classes = dc.german_loader()
    assert X.shape == (1000, 61) and len(names) == 61 and set(np.unique(y)) == {0, 1} and classes == ["good", "bad"]
    assert not np.isnan(X).any()
    g = dc.german_groups(names)
    assert sorted(f for v in g.values() for f in v) == sorted(names)         # cada feature codificada em exatamente um grupo temático
    assert ds_mod.REGISTRY["german_credit"].ranking_eligible is False


def test_german_ordinal_metadata_only_names_existing_features():
    names = dc.german_loader()[2]
    idx = dc.ordinal_columns("german_credit", names)
    assert len(idx) == 4 and all(names[i] in dc.GERMAN_ORDINAL for i in idx)


def test_german_ontology_is_frozen_tbox_without_individuals():
    p = dc.freeze_german_ontology()
    txt = open(p, encoding="utf-8").read()
    assert "NamedIndividual" not in txt and "DomainMeasurement" in txt
