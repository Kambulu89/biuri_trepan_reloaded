"""Integração GUI: esquema resolvido automaticamente (ARFF › ontologia › inferência) e validado antes do treino."""
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
    assert load_saved_schema(app.dataset_schema.dataset_id) is not None       # esquema resolvido registado


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




def test_trained_model_order_is_protected_for_same_dataset_only(app, tmp_path):
    path = _write(tmp_path, "d.csv", CSV)
    app.load_data_from_file(path)
    assert app._validate_dataset_before_training() is True                       # o treino arranca e regista o dataset
    tr = app.trepan.mlp_trainer
    tr.model = object()                                                        # "modelo treinado" neste dataset
    tr.label_encoder.classes_ = np.asarray(['lo', 'hi'], dtype=object)
    app.load_data_from_file(path)                                              # recarga: herda a ordem do modelo
    assert app.arff_meta['classes'] == ['lo', 'hi'] and app.arff_meta['class_order_source'] == 'trained_model'
    assert app._validate_dataset_before_training() is True
    # um modelo treinado noutro ficheiro nunca bloqueia dados novos
    other = _write(tmp_path, "other.csv", CSV.replace("hi", "up").replace("lo", "down"))
    app.load_data_from_file(other)
    assert app.arff_meta['classes'] == ['down', 'up'] and app._validate_dataset_before_training() is True


def test_cache_key_unchanged_by_default_and_changes_with_schema():
    preset = get_training_preset('scientific')
    kw = dict(model_role='mlp_original', dataset_hash='d', ontology_hash='o', preprocessing_hash='p', preset=preset)
    base = build_cache_key(**kw)
    assert build_cache_key(**kw, schema_signature=None) == base
    assert build_cache_key(**kw, schema_signature='abc') != base
    assert build_cache_key(**kw, schema_signature='abc') != build_cache_key(**kw, schema_signature='abd')




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
    onto = world.get_ontology(str((tmp_path / "o.owl").resolve()))
    (tmp_path / "o.owl").write_text(owl, encoding="utf-8")
    onto = world.get_ontology(str((tmp_path / "o.owl").resolve())).load()
    app.loaded_ontology = onto
    app.load_data_from_file(_write(tmp_path, "d.csv", CSV.replace("1.5", "1.5").replace("size", "size")))
    info = app._ontology_type_info(app.dataset_schema)
    assert info is not None and getattr(app, "_ontology_check_error", None) is None


# ---- a ontologia ativa define alvo, ordem das classes e tipos; tudo validado contra os dados
OWL_HEAD = ('<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#" '
            'xml:base="http://ex.org/o" xmlns="http://ex.org/o#"><owl:Ontology rdf:about="http://ex.org/o"/>'
            '<owl:Class rdf:about="#Atributo"/><owl:Class rdf:about="#EtiquetaObjetivo"/>'
            '<owl:DatatypeProperty rdf:about="#nombreAtributo"/><owl:DatatypeProperty rdf:about="#tipoDato"/>'
            '<owl:DatatypeProperty rdf:about="#valorEtiqueta"/>')


def _onto(tmp_path, labels=("P", "N"), attrs=(("m1", "numeric"), ("m2", "numeric"))):
    from owlready2 import World
    body = OWL_HEAD
    for n, k in attrs:
        body += (f'<rdf:Description rdf:about="#a_{n}"><rdf:type rdf:resource="#Atributo"/>'
                 f'<nombreAtributo>{n}</nombreAtributo><tipoDato>{k}</tipoDato></rdf:Description>')
    for i, v in enumerate(labels, 1):
        body += (f'<rdf:Description rdf:about="#e_{i}"><rdf:type rdf:resource="#EtiquetaObjetivo"/>'
                 f'<valorEtiqueta>{v}</valorEtiqueta></rdf:Description>')
    path = tmp_path / "onto.owl"
    path.write_text(body + "</rdf:RDF>", encoding="utf-8")
    return World().get_ontology(str(path.resolve())).load()


ONTO_CSV = "m1,outcome,m2\n1.5,N,2.5\n2.5,P,3.5\n3.5,N,4.5\n4.5,P,5.5\n5.5,N,6.5\n6.5,P,7.5\n"


def test_ontology_defines_target_and_class_order_without_manual_config(app, tmp_path):
    app.loaded_ontology = _onto(tmp_path)                                      # ontologia ativa antes dos dados
    app.load_data_from_file(_write(tmp_path, "d.csv", ONTO_CSV))
    meta = app.arff_meta
    assert meta['target'] == 'outcome' and meta['target_source'] == 'ontology'   # a coluna-alvo NÃO é a última
    assert meta['classes'] == ['P', 'N'] and meta['class_order_source'] == 'ontology'
    assert meta['class_mapping'] == {'P': 0, 'N': 1}
    assert meta['features'] == ['m1', 'm2'] and meta['ontology_schema']['class_labels'] == ['P', 'N']
    assert not app.dataset_validation.critical
    assert app.trepan.mlp_trainer.arff_meta['classes'] == ['P', 'N']          # o encoder do MLP recebe esta ordem


def test_ontology_loaded_after_data_re_resolves_schema(app, tmp_path):
    path = _write(tmp_path, "d.csv", ONTO_CSV)
    app.load_data_from_file(path)
    assert app.arff_meta['target'] == 'm2' and app.arff_meta['target_source'] == 'default_last'   # sem ontologia
    app.loaded_ontology = _onto(tmp_path)
    app._refresh_dataset_schema_after_ontology_change()
    assert app.arff_meta['target'] == 'outcome' and app.arff_meta['classes'] == ['P', 'N']
    app.loaded_ontology = None
    app._refresh_dataset_schema_after_ontology_change()
    assert app.arff_meta['target'] == 'm2'


def test_ontology_data_conflict_blocks_training(app, tmp_path, monkeypatch):
    app.loaded_ontology = _onto(tmp_path, labels=("X", "Y"))                   # classes que não existem no ficheiro
    app.load_data_from_file(_write(tmp_path, "d.csv", ONTO_CSV))
    assert "ONTOLOGY_CLASSES_MISMATCH" in [i.code for i in app.dataset_validation.critical]
    launched = []
    monkeypatch.setattr(app, "_launch_training_worker", lambda preset: launched.append(preset))
    app.train_model()
    assert launched == [] and app._box["critical"]


def test_ontology_of_another_domain_blocks_training(app, tmp_path, monkeypatch):
    app.loaded_ontology = _onto(tmp_path, attrs=(("foo", "numeric"), ("bar", "numeric")))
    app.load_data_from_file(_write(tmp_path, "d.csv", ONTO_CSV))
    assert "ONTOLOGY_DATA_UNRELATED" in [i.code for i in app.dataset_validation.critical]


def test_ontology_declared_types_reach_the_mlp_preprocessing(app, tmp_path):
    csv = "code,m1,outcome\n1,1.5,N\n2,2.5,P\n3,3.5,N\n1,4.5,P\n2,5.5,N\n3,6.5,P\n"
    app.loaded_ontology = _onto(tmp_path, attrs=(("code", "numeric"), ("m1", "numeric")))
    app.load_data_from_file(_write(tmp_path, "d.csv", csv))
    assert app.arff_meta['column_types']['code'] == 'numeric' and app.arff_meta['forced_categorical_columns'] == []
    app.loaded_ontology = None
    app._refresh_dataset_schema_after_ontology_change()
    assert 'code' in app.arff_meta['forced_categorical_columns']              # sem ontologia: códigos continuam categóricos


def test_configure_dataset_feature_is_gone(app):
    assert not hasattr(app, "btn_dataset_config") and not hasattr(app, "open_dataset_config_dialog")
    import gui.dataset_schema_support as mod
    assert not hasattr(mod, "DatasetConfigDialog")
