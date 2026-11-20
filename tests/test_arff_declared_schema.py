"""Esquema ARFF declarado: categorias que parecem números e ordem declarada das classes.

Regressão (auditoria, commit ``af02d5d``): a conversão de colunas de texto «todas numéricas» em números, pensada para CSV com
marcadores de ausente, era aplicada também a atributos ARFF nominais. ``@attribute codigo {01,1}`` com os valores 01, 1 e ?
ficava com uma só categoria (1.0), era classificado como constante e descartado: o atributo nunca chegava ao modelo.

O tipo declarado no ARFF tem de prevalecer sobre a inferência; um CSV sem esquema mantém a inferência existente.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from core.data_contract import build_data_contract
from core.data_loading import load_tabular
from core.preprocessing import DataPreprocessor

N = 64


def _rows(target_values, *, missing_target_rows=()):
    rng = np.random.default_rng(7)
    out = []
    for i in range(N):
        codigo = "?" if i % 9 == 4 else ("01" if i % 2 else "1")
        x = float(np.round(rng.normal(), 4))
        cls = target_values[int(x > 0) ^ int(codigo == "01")]
        if i in missing_target_rows:
            cls = "?"
        out.append((x, codigo, cls))
    return out


def _write_arff(path, *, target_decl, target_values, missing_target_rows=()):
    lines = ["@relation esquema", "@attribute x numeric", "@attribute codigo {01,1}", f"@attribute classe {target_decl}", "@data"]
    lines += [f"{x},{c},{t}" for x, c, t in _rows(target_values, missing_target_rows=missing_target_rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def feature_arff(tmp_path):
    return _write_arff(tmp_path / "nominal.arff", target_decl="{sim,nao}", target_values=("sim", "nao"))


@pytest.fixture
def target_arff(tmp_path):
    # alvo declarado {1,01}: a ordem declarada é o inverso da ordenação lexicográfica ('01' < '1')
    return _write_arff(tmp_path / "alvo.arff", target_decl="{1,01}", target_values=("1", "01"), missing_target_rows=(3, 17, 40))


def _col(contract, name):
    return next(c for c in contract.columns if c.name == name)


# ----------------------------------------------------------------------------------------- 1. atributo nominal {01,1}
def test_loader_exposes_the_declared_arff_schema(feature_arff):
    df, meta = load_tabular(feature_arff)
    assert set(df["codigo"].unique()) == {"01", "1", "?"}                          # o scipy mantém os valores distintos
    schema = df.attrs.get("arff_schema")
    assert schema is not None and meta.get("arff_schema") == schema
    assert schema["codigo"] == {"type": "nominal", "values": ["01", "1"]}
    assert schema["classe"] == {"type": "nominal", "values": ["sim", "nao"]}
    assert schema["x"]["type"] == "numeric"


def test_declared_nominal_keeps_two_distinct_categories_in_contract_and_preprocessing(feature_arff):
    df, _ = load_tabular(feature_arff)
    contract = build_data_contract(df, "classe")
    codigo = _col(contract, "codigo")
    assert codigo.treatment == "categorica" and codigo.cardinality == 2 and codigo.missing_rate > 0
    X, y = df.drop(columns=["classe"]), df["classe"]
    pre = DataPreprocessor(contract).fit(X, y)
    names = list(pre.get_feature_names_out())
    assert "codigo_01" in names and "codigo_1" in names                           # duas categorias distintas
    assert "codigo___em_falta__" in names                                         # o ausente é uma categoria própria, nunca 01 nem 1
    Z = pre.transform(X)
    i01, i1, imiss = names.index("codigo_01"), names.index("codigo_1"), names.index("codigo___em_falta__")
    for row, value in zip(Z, df["codigo"]):
        assert (row[i01], row[i1], row[imiss]) == {"01": (1, 0, 0), "1": (0, 1, 0), "?": (0, 0, 1)}[value]


def test_declared_nominal_reaches_the_model_in_pipeline_v92(feature_arff):
    from core.pipeline_v92 import train_dataset
    df, _ = load_tabular(feature_arff)
    res = train_dataset(df, target="classe", seed=3, max_trepan_nodes=7)
    names = list(res.preprocessor.get_feature_names_out())
    assert {"codigo_01", "codigo_1"} <= set(names)
    assert res.mlp.n_features_in_ == len(names) and list(res.trepan_original.feature_names_in_) == names


def test_declared_nominal_reaches_the_model_in_production_training(feature_arff, tmp_path):
    from core.production_training import train_production_dataframe
    df, _ = load_tabular(feature_arff)
    rep = train_production_dataframe(df, target="classe", out_dir=tmp_path / "out", seed=3, scientific_tuning=False,
                                     require_reasoner=False, return_artifacts=True)
    names = rep["artifacts"]["feature_names_original"]
    assert {"codigo_01", "codigo_1"} <= set(names)
    assert rep["artifacts"]["mlp_original"].n_features_in_ == len(names)
    assert next(c for c in rep["contract"]["columns"] if c["name"] == "codigo")["treatment"] == "categorica"


def test_csv_without_schema_keeps_the_existing_inference(feature_arff, tmp_path):
    """Sem esquema não há como saber que '01' e '1' são categorias: a inferência anterior mantém-se (e fica documentada)."""
    df, _ = load_tabular(feature_arff)
    csv = tmp_path / "nominal.csv"
    df.to_csv(csv, index=False)
    raw, meta = load_tabular(csv)
    assert "arff_schema" not in raw.attrs and "arff_schema" not in meta
    contract = build_data_contract(raw, "classe")
    assert _col(contract, "codigo").treatment == "descartar"                      # comportamento de a0ec5dc para CSV
    assert _col(contract, "x").treatment == "usar"
    assert contract.class_order == []                                             # sem ordem declarada


# ----------------------------------------------------------------------------------------- 2. alvo declarado {1,01} com ausentes
def test_contract_carries_the_declared_class_order_and_excludes_missing_target(target_arff):
    df, _ = load_tabular(target_arff)
    contract = build_data_contract(df, "classe")
    assert contract.class_order == ["1", "01"]
    assert sorted(contract.removed_target_missing_rows) == [3, 17, 40]
    assert set(contract.class_counts) == {"1", "01"}


def _check_declared_order(mlp, Z, y_codes, class_order, metrics):
    assert class_order == ["1", "01"] and "?" not in class_order
    assert list(mlp.classes_) == [0, 1]                                            # índice j ↔ class_order[j]
    proba = mlp.predict_proba(Z)
    assert proba.shape[1] == 2
    assert np.array_equal(np.argmax(proba, axis=1), mlp.predict(Z))                 # coluna j da probabilidade = class_order[j]
    cm = np.asarray(metrics["confusion_matrix"])
    assert cm.shape == (2, 2) and int(cm.sum()) == len(y_codes)
    assert cm[0].sum() == int(np.sum(y_codes == 0)) and cm[1].sum() == int(np.sum(y_codes == 1))   # linha 0 = '1', linha 1 = '01'
    assert metrics["class_order"] == ["1", "01"]


def test_pipeline_v92_uses_the_declared_order_end_to_end(target_arff):
    from core.pipeline_v92 import train_dataset
    df, _ = load_tabular(target_arff)
    res = train_dataset(df, target="classe", seed=3, max_trepan_nodes=7)
    clean = df[df["classe"] != "?"].reset_index(drop=True)
    te = np.asarray(res.split["test_indices"])
    Zte = res.preprocessor.transform(clean.drop(columns=["classe"]).iloc[te])
    y_codes = clean["classe"].iloc[te].map({"1": 0, "01": 1}).to_numpy()
    _check_declared_order(res.mlp, Zte, y_codes, res.class_order, res.metrics["mlp"])
    labels = res.decode(res.mlp.predict(Zte))
    assert set(labels) <= {"1", "01"}
    assert np.mean(labels == clean["classe"].iloc[te].to_numpy()) == pytest.approx(res.metrics["mlp"]["accuracy"])


def test_production_training_uses_the_declared_order_end_to_end(target_arff, tmp_path):
    from core.production_training import train_production_dataframe
    df, _ = load_tabular(target_arff)
    rep = train_production_dataframe(df, target="classe", out_dir=tmp_path / "out", seed=3, scientific_tuning=False,
                                     require_reasoner=False, return_artifacts=True)
    art = rep["artifacts"]
    clean = df[df["classe"] != "?"].reset_index(drop=True)
    te = np.asarray(rep["split"]["test_indices"])
    Zte = art["preprocessor"].transform(clean.drop(columns=["classe"]).iloc[te])
    y_codes = clean["classe"].iloc[te].map({"1": 0, "01": 1}).to_numpy()
    _check_declared_order(art["mlp_original"], Zte, y_codes, art["class_names"], rep["evaluation"]["models"]["mlp_original"])
    assert rep["manifest"]["class_order"] == ["1", "01"]
    assert rep["contract"]["removed_target_missing_rows"] == [3, 17, 40]


# ----------------------------------------------------------------------------------------- GUI (percurso aplicável)
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")          # sem Qt/libEGL no ambiente: os testes da GUI são ignorados
    QApplication = QtWidgets.QApplication
    return QApplication.instance() or QApplication([])          # a referência tem de viver durante todo o módulo


@pytest.fixture
def gui_app(qapp, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    from gui.biuri_app_complete import BiuriApp
    monkeypatch.setenv("BIURI_DATASET_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    w = BiuriApp()
    yield w
    w.close()


def test_gui_keeps_numeric_looking_nominal_categories_distinct(gui_app, feature_arff):
    gui_app.load_data_from_file(str(feature_arff))
    assert "codigo" in gui_app.arff_meta["forced_categorical_columns"]
    tr = gui_app.trepan.mlp_trainer
    X, _ = gui_app.original_data
    tr._fit_encoders(X)
    j = list(gui_app.arff_meta["original_features"]).index("codigo")
    assert j in tr.feature_encoders and {"01", "1"} <= set(tr.feature_encoders[j].classes_)
    enc = tr._encode_features(X)[:, j]
    assert len({v for v, raw in zip(enc, X[:, j]) if raw == "01"} & {v for v, raw in zip(enc, X[:, j]) if raw == "1"}) == 0


def test_gui_uses_the_declared_target_order_and_drops_missing_target(gui_app, target_arff):
    gui_app.load_data_from_file(str(target_arff))
    assert gui_app.arff_meta["classes"] == ["1", "01"]
    X, y = gui_app.original_data
    assert "?" not in set(map(str, y)) and len(y) == N - 3
    tr = gui_app.trepan.mlp_trainer
    assert tr.arff_meta["classes"] == ["1", "01"]
    tr.train(X, y, optimize=False)                                                  # o mesmo treino que a GUI dispara
    assert list(tr.label_encoder.classes_) == ["1", "01"] and list(tr.model.classes_) == [0, 1]
    proba, codes = tr.predict_proba(X), tr.predict(X)
    assert proba.shape == (len(y), 2) and np.array_equal(np.argmax(proba, axis=1), codes)   # coluna j = classes_[j]
    labels = tr.label_encoder.inverse_transform(codes)
    assert set(labels) <= {"1", "01"}
    from core.evaluation_protocol import classification_metrics
    m = classification_metrics(tr.label_encoder.transform(y), codes, class_order=list(tr.label_encoder.classes_))
    assert m["class_order"] == ["1", "01"] and np.asarray(m["confusion_matrix"])[0].sum() == int(np.sum(np.asarray(y) == "1"))


# ----------------------------------------------------------------------------------------- benchmark exclusivamente numérico
def test_numeric_only_benchmark_rejects_declared_nominal_attributes_explicitly(feature_arff):
    from core.benchmark.runner import Dataset
    from core.scientific_errors import DataContractError
    with pytest.raises(DataContractError, match="codigo"):
        Dataset.from_file(str(feature_arff), "classe")


def test_numeric_only_benchmark_rejects_missing_target_explicitly(tmp_path):
    from core.benchmark.runner import Dataset
    from core.scientific_errors import DataContractError
    p = tmp_path / "num.arff"
    rows = [f"{i / 10},{'?' if i == 5 else ('1' if i % 2 else '01')}" for i in range(20)]
    p.write_text("@relation n\n@attribute x numeric\n@attribute classe {1,01}\n@data\n" + "\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="ausente"):
        Dataset.from_file(str(p), "classe")


def test_numeric_only_benchmark_still_preserves_declared_order_without_missing(tmp_path):
    from core.benchmark.runner import Dataset
    p = tmp_path / "num.arff"
    rows = [f"{i / 10},{'1' if i % 2 else '01'}" for i in range(20)]
    p.write_text("@relation n\n@attribute x numeric\n@attribute classe {1,01}\n@data\n" + "\n".join(rows) + "\n", encoding="utf-8")
    d = Dataset.from_file(str(p), "classe")
    assert list(d.y[:2]) == [1, 0]                                                # '01' → 1, '1' → 0 (ordem declarada)
