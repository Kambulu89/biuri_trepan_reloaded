"""ARFF com valores nominais Unicode (acentos), lido em UTF-8 em qualquer sistema.

Reprodução: o ``scipy.io.arff.loadarff`` guarda nominais como bytes ASCII e falha com ``'ascii' codec can't encode characters``
(no Windows, o UTF-8 lido como cp1252 dá «position 9-10»). ``core.arff_io.read_arff`` substitui-o nos três pontos de leitura
(``load_tabular``, GUI, ``MLPTrainer.load_arff``) e dá o mesmo resultado do scipy nos ficheiros ASCII.
"""
from __future__ import annotations

import glob
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import arff as scipy_arff

from core.arff_io import read_arff
from core.data_loading import load_tabular

ROOT = Path(__file__).resolve().parents[1]
ACCENTED = (
    "@relation clinica\r\n"
    "% biuri: target = desfecho\r\n"
    "@attribute idade numeric\r\n"
    "@attribute antecedente {'Cardiopatía isquémica','Salud Aparente','Hipertensión, controlada'}\r\n"
    "@attribute hora {'02:30','13:10'}\r\n"
    "@attribute desfecho {Sí,No}\r\n"
    "@data\r\n"
    "52,'Cardiopatía isquémica','02:30',Sí\r\n"
    "60,'Salud Aparente',?,No\r\n"
    "?,'Hipertensión, controlada','13:10',No\r\n"
    "71,'Cardiopatía isquémica','13:10',Sí\r\n"
)


def _write(tmp_path, text, name="d.arff", encoding="utf-8", bom=False):
    p = tmp_path / name
    p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode(encoding))
    return p


def test_scipy_reader_cannot_read_accented_nominal_values(tmp_path):
    """Causa-raiz documentada: é o leitor do scipy que falha, não o ficheiro."""
    with pytest.raises(UnicodeEncodeError):
        scipy_arff.loadarff(io.StringIO(ACCENTED))


@pytest.mark.parametrize("encoding,bom", [("utf-8", False), ("utf-8", True), ("cp1252", False)])
def test_accented_nominal_values_are_read_intact(tmp_path, encoding, bom):
    df, meta = read_arff(_write(tmp_path, ACCENTED, encoding=encoding, bom=bom))
    assert list(df.columns) == ["idade", "antecedente", "hora", "desfecho"]
    assert list(df["antecedente"]) == ["Cardiopatía isquémica", "Salud Aparente", "Hipertensión, controlada", "Cardiopatía isquémica"]
    assert list(df["desfecho"]) == ["Sí", "No", "No", "Sí"]
    assert list(df["hora"]) == ["02:30", "?", "13:10", "13:10"]                 # nominal ausente = '?' (como o scipy)
    assert df["idade"].dtype == float and np.isnan(df["idade"].iloc[2])       # numérico ausente = NaN
    assert meta["desfecho"] == ("nominal", ("Sí", "No")) and meta["idade"] == ("numeric", None)
    assert meta.encoding == ("cp1252" if encoding == "cp1252" else "utf-8")


def test_load_tabular_and_contract_keep_accents_and_declared_order(tmp_path):
    from core.data_contract import build_data_contract
    df, meta = load_tabular(_write(tmp_path, ACCENTED))
    assert meta["arff_schema"]["desfecho"] == {"type": "nominal", "values": ["Sí", "No"]}
    c = build_data_contract(df, "desfecho")
    assert c.class_order == ["Sí", "No"]
    assert next(col for col in c.columns if col.name == "antecedente").treatment == "categorica"


def test_mlp_trainer_reads_accented_arff(tmp_path):
    from core.mlp_trainer import MLPTrainer
    tr = MLPTrainer()
    X, y = tr.load_arff(str(_write(tmp_path, ACCENTED)))
    assert X.shape == (4, 3) and list(y) == ["Sí", "No", "No", "Sí"]
    assert tr.arff_meta["classes"] == ["Sí", "No"] and "Cardiopatía isquémica" in set(X[:, 1])


def test_gui_loads_accented_arff(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gui.biuri_app_complete import BiuriApp
    monkeypatch.setenv("BIURI_DATASET_CONFIG_DIR", str(tmp_path / "cfg"))
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: errors.append(a[2]) or QtWidgets.QMessageBox.StandardButton.Ok)
    w = BiuriApp()
    try:
        w.load_data_from_file(str(_write(tmp_path, ACCENTED)))
        X, y = w.original_data
        assert not errors and X.shape == (4, 3) and w.arff_meta["classes"] == ["Sí", "No"]
        assert "Cardiopatía isquémica" in set(X[:, 1])
    finally:
        w.close()
    assert app is not None


def test_quotes_commas_and_escapes(tmp_path):
    text = ("@relation q\n@attribute a {'x, y',\"z w\",'o\\'k'}\n@attribute b numeric\n@data\n"
            "'x, y',1\n\"z w\",2\n'o\\'k',3\n  'x, y' , 4 \n")
    df, _ = read_arff(_write(tmp_path, text))
    assert list(df["a"]) == ["x, y", "z w", "o'k", "x, y"] and list(df["b"]) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("data,msg", [
    ("52,'Desconhecido','02:30',Sí", "não declarado"),
    ("52,'Salud Aparente','02:30'", "valores para 4 atributos"),
    ("abc,'Salud Aparente','02:30',Sí", "não numérico"),
    ("{0 52, 3 Sí}", "esparso"),
])
def test_invalid_rows_fail_with_the_line_number(tmp_path, data, msg):
    header = ACCENTED.split("@data")[0] + "@data\n"
    with pytest.raises(ValueError, match=msg) as exc:
        read_arff(_write(tmp_path, header + data + "\n"))
    assert "linha" in str(exc.value)


def test_relational_attributes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="relacionais"):
        read_arff(_write(tmp_path, "@relation r\n@attribute bag relational\n@end bag\n@data\n"))


def test_same_result_as_scipy_on_every_ascii_arff_of_the_repository():
    files = sorted(set(glob.glob(str(ROOT / "**" / "*.arff"), recursive=True)))
    assert files
    for f in files:
        data, smeta = scipy_arff.loadarff(f)
        ref = pd.DataFrame(data)
        for c in ref.columns:
            if ref[c].dtype == object:
                ref[c] = ref[c].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
        new, meta = read_arff(f)
        assert list(ref.columns) == list(new.columns), f
        for c in ref.columns:
            assert ref[c].dtype == new[c].dtype, (f, c)
            if ref[c].dtype.kind == "f":
                assert np.allclose(ref[c], new[c], equal_nan=True), (f, c)
            else:
                assert list(ref[c]) == list(new[c]), (f, c)
        assert list(smeta.names()) == meta.names()
        assert all(tuple(smeta[n][1] or ()) == tuple(meta[n][1] or ()) for n in smeta.names()), f
