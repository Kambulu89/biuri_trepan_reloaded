"""Regressão (auditoria): o tipo inferido de uma coluna não pode depender de haver marcadores de ausente.

Antes da correção, uma coluna numérica de um CSV com '?' era lida como texto e classificada como categórica
(one-hot), perdendo a ordem dos valores.
"""
import numpy as np
import pandas as pd
import pytest

from core.data_contract import build_data_contract
from core.preprocessing import DataPreprocessor


def _frame(missing_token):
    rows = 60
    idade = [str(20 + (i * 7) % 41) for i in range(rows)]
    if missing_token is not None:
        for i in range(0, rows, 7):
            idade[i] = missing_token
    return pd.DataFrame({
        "idade": idade if missing_token is not None else [int(v) for v in idade],
        "sexo": ["M" if i % 2 else "F" for i in range(rows)],
        "classe": ["sim" if i % 3 == 0 else "nao" for i in range(rows)],
    })


def _column(contract, name):
    return next(c for c in contract.columns if c.name == name)


@pytest.mark.parametrize("token", ["?", "NA", "", " ? "])
def test_numeric_column_with_missing_tokens_stays_numeric(token):
    with_missing = build_data_contract(_frame(token), "classe")
    without_missing = build_data_contract(_frame(None), "classe")
    assert _column(with_missing, "idade").treatment == "usar"
    assert _column(with_missing, "idade").inferred_type == _column(without_missing, "idade").inferred_type
    assert _column(with_missing, "idade").missing_rate > 0


def test_preprocessor_keeps_one_ordered_numeric_feature():
    df = _frame("?")
    contract = build_data_contract(df, "classe")
    X = df.drop(columns=["classe"])
    pre = DataPreprocessor(contract).fit(X, df["classe"])
    names = list(pre.get_feature_names_out())
    assert "idade" in names and not any(n.startswith("idade_") for n in names)
    Z = pre.transform(X)
    col = Z[:, names.index("idade")]
    present = df["idade"] != "?"
    assert np.allclose(col[present.to_numpy()], df.loc[present, "idade"].astype(float).to_numpy())


def test_genuinely_categorical_text_is_untouched():
    df = _frame("?")
    df["codigo"] = ["A1" if i % 2 else "7" for i in range(len(df))]          # mistura texto/números: continua categórica
    contract = build_data_contract(df, "classe")
    assert _column(contract, "codigo").treatment == "categorica"
    assert _column(contract, "sexo").treatment == "categorica"


def test_target_is_never_coerced():
    df = _frame("?")
    df["classe"] = ["1" if i % 3 == 0 else "0" for i in range(len(df))]
    contract = build_data_contract(df, "classe")
    assert _column(contract, "classe").treatment == "target"
