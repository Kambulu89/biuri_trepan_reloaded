"""Integração GUI: CSV/ARFF com esquema de dataset (alvo, ordem das classes, tipos), bloqueio pré-treino, retrocompatibilidade."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication, QMessageBox
from scipy.io import arff as scipy_arff

from core.dataset_schema import load_saved_schema
from core.mlp_trainer import MLPTrainer
from core.model_cache import build_cache_key
from core.training_config import get_training_preset
from gui.biuri_app_complete import BiuriApp
from gui.dataset_config_support import DatasetConfigDialog

CSV = ("code,size,flag,label\n"
       "1,1.5,yes,hi\n2,2.5,no,lo\n3,3.5,yes,hi\n1,4.5,no,lo\n2,5.5,yes,hi\n3,6.5,no,lo\n")
ARFF = ("@relation t\n@attribute a numeric\n@attribute b {x,y}\n@attribute cls {zeta,alpha}\n@data\n"
        "1.0,x,zeta\n2.0,y,alpha\n3.0,x,zeta\n4.0,y,alpha\n5.0,x,zeta\n6.0,y,alpha\n")


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv('BIURI_DATASET_CONFIG_DIR', str(tmp_path / 'cfg'))
    box = {"critical": [], "warning": []}
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: box["critical"].append(a[2]) or QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: box["warning"].append(a[2]) or QMessageBox.StandardButton.Ok)
    w = BiuriApp()
    w._box = box
    yield w
    w.close()


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_csv_loads_with_schema_and_numeric_codes_stay_categorical(app, tmp_path):
    app.load_data_from_file(_write(tmp_path, "d.csv", CSV))
    meta = app.arff_meta
    assert meta['target'] == 'label' and meta['classes'] == ['hi', 'lo']      # ordenação determinística
    assert meta['class_order_source'] == 'sorted' and meta['class_mapping'] == {'hi': 0, 'lo': 1}
    assert meta['column_types']['code'] == 'categorical' and meta['column_types']['flag'] == 'boolean'
    assert 'code' in meta['forced_categorical_columns']
    assert app.trepan.mlp_trainer.arff_meta['forced_categorical_columns'] == ['code']
    X, y = app.original_data
    assert X.shape == (6, 3) and list(y[:2]) == ['hi', 'lo']
    assert load_saved_schema(app.dataset_schema.dataset_id) is not None       # persistido por dataset


def test_trainer_encodes_forced_categorical_numeric_codes():
    tr = MLPTrainer()
    X = np.array([["1", 0.5], ["2", 1.5], ["3", 2.5]], dtype=object)
    assert tr._column_needs_encoder(X, 0) is False                            # comportamento anterior: parece número
    tr.arff_meta = {'features': ['code', 'v'], 'forced_categorical_columns': ['code']}
    assert tr._column_needs_encoder(X, 0) is True and tr._column_needs_encoder(X, 1) is False


def test_default_arff_behaviour_is_unchanged(app, tmp_path):
    path = _write(tmp_path, "a.arff", ARFF)
    app.load_data_from_file(path)
    assert app.dataset_schema.is_default_arff() and app.dataset_schema.cache_signature() is None
    assert app.arff_meta['target'] == 'cls' and app.arff_meta['classes'] == ['zeta', 'alpha']   # ordem declarada
    assert app.arff_meta['class_order_source'] == 'arff_header'
    data, _ = scipy_arff.loadarff(path)
    X, y = app.original_data
    assert X.shape == (6, 2) and list(y) == [v.decode() for v in data['cls']]
    assert load_saved_schema(app.dataset_schema.dataset_id) is None           # nada é gravado para o ARFF por omissão


def test_arff_explicit_target_directive_reorders_columns(app, tmp_path):
    path = _write(tmp_path, "a.arff", "% biuri:target = b\n" + ARFF)
    app.load_data_from_file(path)
    assert app.arff_meta['target'] == 'b' and app.arff_meta['target_source'] == 'arff_explicit'
    assert app.arff_meta['features'] == ['a', 'cls']


def test_pre_training_gate_blocks_critical(app, tmp_path, monkeypatch):
    app.load_data_from_file(_write(tmp_path, "m.csv", CSV.replace("2,2.5,no", "2,,no")))
    launched = []
    monkeypatch.setattr(app, "_launch_training_worker", lambda preset: launched.append(preset))
    app.train_model()
    assert launched == [] and app._box["critical"], "treino devia ficar bloqueado por valores ausentes"
    assert any(i.code == "MISSING_VALUES" for i in app.dataset_validation.critical)


def test_pre_training_gate_allows_valid_dataset(app, tmp_path, monkeypatch):
    app.load_data_from_file(_write(tmp_path, "d.csv", CSV))
    launched = []
    monkeypatch.setattr(app, "_launch_training_worker", lambda preset: launched.append(preset))
    app.train_model()
    assert len(launched) == 1 and not app._box["critical"]


def test_manual_request_changes_target_order_and_types_and_persists(app, tmp_path):
    path = _write(tmp_path, "d.csv", CSV)
    app.load_data_from_file(path)
    app._pending_schema_request = {"manual_class_order": ["lo", "hi"], "type_overrides": {"code": {"kind": "numeric"}}}
    app.load_data_from_file(path)
    assert app.arff_meta['classes'] == ['lo', 'hi'] and app.arff_meta['class_order_source'] == 'manual'
    assert app.arff_meta['forced_categorical_columns'] == []
    app.load_data_from_file(path)                                              # recarregar aplica a configuração guardada
    assert app.arff_meta['classes'] == ['lo', 'hi'] and app.arff_meta['column_types']['code'] == 'numeric'


def test_trained_model_order_is_protected_and_conflict_blocks(app, tmp_path):
    path = _write(tmp_path, "d.csv", CSV)
    app.load_data_from_file(path)
    tr = app.trepan.mlp_trainer
    tr.model = object()                                                        # "modelo treinado"
    tr.label_encoder.classes_ = np.asarray(['lo', 'hi'], dtype=object)
    app.load_data_from_file(path)                                              # recarga: herda a ordem do modelo
    assert app.arff_meta['classes'] == ['lo', 'hi'] and app.arff_meta['class_order_source'] == 'trained_model'
    app._pending_schema_request = {"manual_class_order": ["hi", "lo"]}         # pedido divergente: nunca em silêncio
    app.load_data_from_file(path)
    assert any(i.code == "CLASS_ORDER_CHANGES_TRAINED_MODEL" for i in app.dataset_validation.critical)
    assert app._validate_dataset_before_training() is False


def test_cache_key_unchanged_by_default_and_changes_with_schema():
    preset = get_training_preset('scientific')
    kw = dict(model_role='mlp_original', dataset_hash='d', ontology_hash='o', preprocessing_hash='p', preset=preset)
    base = build_cache_key(**kw)
    assert build_cache_key(**kw, schema_signature=None) == base
    assert build_cache_key(**kw, schema_signature='abc') != base
    assert build_cache_key(**kw, schema_signature='abc') != build_cache_key(**kw, schema_signature='abd')


def test_config_dialog_changes_target_order_and_blocks_on_critical(qapp, app, tmp_path):
    app.load_data_from_file(_write(tmp_path, "d.csv", CSV))
    dlg = DatasetConfigDialog(app, app._dataset_raw_df, app._dataset_make_schema, app._dataset_validate,
                              app.dataset_schema, None)
    assert dlg.class_list.count() == 2
    dlg.class_list.setCurrentRow(0)
    dlg._move(1)
    assert dlg.request()["manual_class_order"] == ['lo', 'hi']
    dlg.target_combo.setCurrentText('flag')
    assert dlg.request()["manual_target"] == 'flag' and dlg.request()["manual_class_order"] is None
    from PyQt6.QtWidgets import QDialogButtonBox
    assert dlg.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dlg.close()


def test_load_dialog_mentions_arff_and_csv(qapp):
    from PyQt6.QtWidgets import QGroupBox, QLabel
    from gui.biuri_app_complete import DataLoadDialog
    dlg = DataLoadDialog()
    titles = [g.title() for g in dlg.findChildren(QGroupBox)]
    assert "📁 Archivo de Datos (ARFF/CSV)" in titles
    assert not any(t.endswith("(ARFF)") for t in titles)
    assert "ARFF/CSV" in " ".join(l.text() for l in dlg.findChildren(QLabel))
    dlg.close()


def test_id_like_column_blocks_training_instead_of_failing_late(app, tmp_path, monkeypatch):
    """Tabela com uma coluna de texto única por linha (ex.: dicionário de dados, não dados): bloqueada antes do treino."""
    rows = "\n".join(f"campo_{i},desc_{i},{'A' if i % 2 else 'B'}" for i in range(30))
    app.load_data_from_file(_write(tmp_path, "dic.csv", "campo,descricao,classe\n" + rows + "\n"))
    codes = [i.code for i in app.dataset_validation.critical]
    assert "ID_LIKE_COLUMN" in codes
    launched = []
    monkeypatch.setattr(app, "_launch_training_worker", lambda preset: launched.append(preset))
    app.train_model()
    assert launched == [] and app._box["critical"]


def test_ontology_type_info_is_computed_when_ontology_loaded(app, tmp_path):
    """Regressão: a verificação de compatibilidade OWL↔tipos não pode falhar em silêncio (import errado)."""
    owl = ('<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
           'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#" '
           'xmlns:xsd="http://www.w3.org/2001/XMLSchema#" xml:base="http://ex.org/o" xmlns="http://ex.org/o#">'
           '<owl:Ontology rdf:about="http://ex.org/o"/><owl:Class rdf:about="#Thing1"/>'
           '<owl:DatatypeProperty rdf:about="#hasSize"><rdfs:domain rdf:resource="#Thing1"/>'
           '<rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#double"/><rdfs:label>size</rdfs:label></owl:DatatypeProperty>'
           '</rdf:RDF>')
    from owlready2 import World
    world = World()
    onto = world.get_ontology((tmp_path / "o.owl").as_uri())
    (tmp_path / "o.owl").write_text(owl, encoding="utf-8")
    onto = world.get_ontology((tmp_path / "o.owl").as_uri()).load()
    app.loaded_ontology = onto
    app.load_data_from_file(_write(tmp_path, "d.csv", CSV.replace("1.5", "1.5").replace("size", "size")))
    info = app._ontology_type_info(app.dataset_schema)
    assert info is not None and getattr(app, "_ontology_check_error", None) is None
