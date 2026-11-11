"""Resolução automática e validação do esquema de datasets CSV/ARFF (alvo, ordem das classes, tipos), com ontologia."""
import numpy as np
import pandas as pd
import pytest

from core.arff_schema import OrderedLabelEncoder
from core.dataset_schema import (
    DatasetSchema, apply_schema, build_schema, infer_kind, load_saved_schema, move_target_last, read_arff_header,
    read_tabular_text, save_schema, sort_class_labels, validate_schema,
)
from core.ontology_schema import OntologySchemaDeclaration

CSV = ("id_code,age,smoker,grade,visit,label\n1,34.5,yes,3,2020-01-05,b\n2,51.2,no,1,2021-03-09,a\n3,29.8,yes,2,2020-07-30,b\n"
       "4,62.1,no,3,2022-11-11,c\n5,45.0,yes,1,2021-05-01,a\n6,38.4,no,2,2022-02-02,c\n")


def _csv(tmp_path, text=CSV, name="d.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _schema(tmp_path, text=CSV, **kw):
    p = _csv(tmp_path, text)
    df, meta = read_tabular_text(p)
    return p, df, build_schema(df, file_path=p, source_format="csv", **kw)


def _decl(**kw):
    base = dict(declared_columns=[], column_types={}, type_evidence={}, ordinal_orders={})
    base.update(kw)
    return OntologySchemaDeclaration(**base)


def _codes(rep, severity="critical"):
    return [i.code for i in getattr(rep, severity)]


# ---- alvo
def test_default_target_is_last_column(tmp_path):
    _, df, s = _schema(tmp_path)
    assert s.target == "label" and s.target_source == "default_last"


def test_missing_target_is_blocked(tmp_path):
    _, df, s = _schema(tmp_path)
    s.target = "nao_existe"
    assert "TARGET_MISSING" in _codes(validate_schema(s, df))


def test_arff_explicit_target_directive(tmp_path):
    arff = tmp_path / "x.arff"
    arff.write_text("% biuri:target = cls\n@relation r\n@attribute cls {n,p}\n@attribute a numeric\n@data\nn,1\np,2\n")
    h = read_arff_header(arff)
    assert h["target_directive"] == "cls"
    df = pd.DataFrame({"cls": ["n", "p"], "a": [1.0, 2.0]})
    s = build_schema(df, file_path=arff, source_format="arff", arff_header=h)
    assert s.target == "cls" and s.target_source == "arff_explicit" and not s.is_default_arff()
    assert list(move_target_last(df, s.target).columns) == ["a", "cls"]


def test_default_arff_is_untouched(tmp_path):
    arff = tmp_path / "x.arff"
    arff.write_text("@relation r\n@attribute a numeric\n@attribute cls {z,y}\n@data\n1,z\n2,y\n")
    h = read_arff_header(arff)
    df = pd.DataFrame({"a": [1.0, 2.0], "cls": ["z", "y"]})
    s = build_schema(df, file_path=arff, source_format="arff", arff_header=h)
    assert s.is_default_arff() and s.cache_signature() is None
    assert s.class_order == ["z", "y"] and s.class_order_source == "arff_declared"        # ordem declarada, não alfabética


# ---- classes
def test_class_order_sorting_numeric_vs_lexicographic():
    assert sort_class_labels(["10", "2", "1"]) == ["1", "2", "10"]
    assert sort_class_labels(["b", "a", "B"]) == ["B", "a", "b"]


def test_trained_model_order_has_priority_and_conflict_is_critical(tmp_path):
    _, df, s0 = _schema(tmp_path)
    assert s0.class_order == ["a", "b", "c"] and s0.class_order_source == "sorted"
    _, _, s1 = _schema(tmp_path, trained_class_order=["b", "c", "a"])
    assert s1.class_order == ["b", "c", "a"] and s1.class_order_source == "trained_model"
    assert s1.class_mapping == {"b": 0, "c": 1, "a": 2}
    _, _, s2 = _schema(tmp_path, trained_class_order=["a", "b", "c"])                      # coincide com a derivação
    assert s2.class_order_source == "sorted"
    # uma derivação diferente da do modelo treinado nunca passa em silêncio
    drift = DatasetSchema(**{**s0.__dict__})
    rep = validate_schema(drift, df, trained={"class_order": ["c", "b", "a"], "target": "label"})
    assert "CLASS_ORDER_CHANGES_TRAINED_MODEL" in _codes(rep)


def test_invalid_class_order_blocked(tmp_path):
    p, df, s = _schema(tmp_path)
    s.class_order = ["a", "b"]; s.class_mapping = {"a": 0, "b": 1}
    assert "CLASS_ORDER_INVALID" in _codes(validate_schema(s, df))


def test_encoder_uses_exactly_schema_order(tmp_path):
    _, df, s = _schema(tmp_path, trained_class_order=["c", "b", "a"])
    enc = OrderedLabelEncoder(s.class_order)
    y = apply_schema(df, s)[s.target]
    idx = enc.fit_transform(y)
    assert [s.class_mapping[v] for v in y] == list(idx)
    assert list(enc.inverse_transform(range(3))) == ["c", "b", "a"]


# ---- tipos
def test_type_inference_does_not_treat_numeric_codes_as_numeric():
    assert infer_kind(["1", "2", "3", "1", "2"])[:2] == ("categorical", True)
    assert infer_kind([str(i) for i in range(40)])[0] == "numeric"
    assert infer_kind(["0.5", "1.5", "2.5"])[0] == "numeric"
    assert infer_kind(["yes", "no", "yes"])[0] == "boolean"
    assert infer_kind(["0", "1", "1"])[0] == "boolean"
    assert infer_kind(["2020-01-05", "2021-03-09"])[0] == "datetime"
    assert infer_kind(["M", "F", "M"])[0] == "categorical"


def test_kinds_inferred_and_converted(tmp_path):
    p, df, s = _schema(tmp_path)
    kinds = {c.name: c.kind for c in s.columns}
    assert kinds == {"id_code": "categorical", "age": "numeric", "smoker": "boolean", "grade": "categorical",
                     "visit": "datetime", "label": "categorical"}
    out = apply_schema(df, s)
    assert list(out.columns)[-1] == "label"
    assert list(out["smoker"]) == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]                           # false-like = 0
    assert out["visit"].iloc[0] == pd.Timestamp("2020-01-05", tz="UTC").timestamp()
    assert out["age"].dtype == float
    assert {"id_code", "grade"} <= set(s.forced_categorical)                               # códigos nunca viram numéricos em silêncio
    assert any(i.code == "AMBIGUOUS_NUMERIC_CODES" for i in validate_schema(s, df).warnings)


# ---- ausentes, identificadores
def test_missing_values_block_for_csv(tmp_path):
    p, df, s = _schema(tmp_path, CSV.replace("2,51.2,no", "2,,no"))
    assert s.column("age").n_missing == 1
    assert "MISSING_VALUES" in _codes(validate_schema(s, df))


def test_id_like_column_is_blocked(tmp_path):
    rows = "\n".join(f"campo_{i},desc_{i},{'A' if i % 2 else 'B'}" for i in range(30))
    p, df, s = _schema(tmp_path, "campo,descricao,classe\n" + rows + "\n")
    assert _codes(validate_schema(s, df)).count("ID_LIKE_COLUMN") == 2


def test_duplicate_columns_blocked(tmp_path):
    p = _csv(tmp_path, "a,a,y\n1,2,u\n3,4,v\n")
    df, meta = read_tabular_text(p)
    assert meta["duplicate_columns"] == ["a"]
    s = build_schema(df, file_path=p, source_format="csv")
    assert "DUPLICATE_COLUMN" in _codes(validate_schema(s, df, duplicate_columns=meta["duplicate_columns"]))


# ---- compatibilidades com ARFF de referência, modelo treinado e pré-processamento
def test_trained_features_mismatch_and_arff_reference(tmp_path):
    p, df, s = _schema(tmp_path)
    rep = validate_schema(s, df, trained={"feature_names": ["x", "y"], "class_order": s.class_order, "target": "label"})
    assert "TRAINED_FEATURES_MISMATCH" in _codes(rep)
    ref = {"attributes": [("id_code", "numeric"), ("age", "{a,b}"), ("label", "{a,b,c}")]}
    codes = _codes(validate_schema(s, df, arff_reference=ref))
    assert "ARFF_COLUMNS_MISMATCH" in codes and "ARFF_TYPE_MISMATCH" in codes


def test_numeric_codes_mapped_in_ontology_conflict(tmp_path):
    p, df, s = _schema(tmp_path)
    rep = validate_schema(s, df, ontology_info={"id_code": {"entity": "hasId", "range": "numeric"}})
    assert "ONTOLOGY_TYPE_CONFLICT" in _codes(rep)


# ---- ontologia: o esquema é lido da própria ontologia e validado contra os dados
def test_ontology_declares_target_by_label_values_and_class_order(tmp_path):
    csv = "x1,lab,x2\n1.5,N,2.5\n2.5,P,3.5\n3.5,N,4.5\n4.5,P,5.5\n"
    p = _csv(tmp_path, csv)
    df, _ = read_tabular_text(p)
    decl = _decl(class_labels=["P", "N"], class_order_evidence="indivíduos", declared_columns=["x1", "x2"])
    s = build_schema(df, file_path=p, source_format="csv", ontology_decl=decl)
    assert s.target == "lab" and s.target_source == "ontology"                              # NÃO é a última coluna
    assert s.class_order == ["P", "N"] and s.class_order_source == "ontology"
    assert s.class_mapping == {"P": 0, "N": 1}
    assert list(apply_schema(df, s).columns) == ["x1", "x2", "lab"]
    assert validate_schema(s, df, ontology_decl=decl).ok


def test_ontology_declared_target_column_and_types(tmp_path):
    p, df, _ = _schema(tmp_path)
    decl = _decl(target_column="grade", target_evidence="marcado", declared_columns=["age", "id_code"],
                 column_types={"id_code": "numeric", "age": "numeric"}, type_evidence={"id_code": "e1", "age": "e2"})
    s = build_schema(df, file_path=p, source_format="csv", ontology_decl=decl)
    assert s.target == "grade" and s.target_source == "ontology"
    assert s.column("id_code").kind == "numeric" and s.column("id_code").source == "ontology"   # ontologia > inferência
    assert "id_code" not in s.forced_categorical


def test_ontology_conflicts_are_blocking(tmp_path):
    p, df, s = _schema(tmp_path)
    # alvo declarado inexistente
    assert "ONTOLOGY_TARGET_NOT_IN_DATA" in _codes(validate_schema(s, df, ontology_decl=_decl(target_column="zzz")))
    # alvo declarado diferente do que o ficheiro define
    assert "ONTOLOGY_TARGET_CONFLICT" in _codes(validate_schema(s, df, ontology_decl=_decl(target_column="age")))
    # classes da ontologia diferentes das observadas
    assert "ONTOLOGY_CLASSES_MISMATCH" in _codes(validate_schema(s, df, ontology_decl=_decl(class_labels=["a", "b", "zz"])))
    # ontologia de outro domínio: nenhum atributo coincide
    assert "ONTOLOGY_DATA_UNRELATED" in _codes(validate_schema(s, df, ontology_decl=_decl(declared_columns=["foo", "bar"])))
    # ordem diferente com o mesmo conjunto: aviso (prevalece o ficheiro/modelo)
    rep = validate_schema(s, df, ontology_decl=_decl(class_labels=["c", "b", "a"]))
    assert not any(c.startswith("ONTOLOGY") for c in _codes(rep)) and "ONTOLOGY_CLASS_ORDER_DIFFERS" in _codes(rep, "warnings")


def test_arff_declarations_prevail_but_conflict_with_ontology_is_blocked(tmp_path):
    arff = tmp_path / "x.arff"
    arff.write_text("@relation r\n@attribute a numeric\n@attribute cls {z,y}\n@data\n1,z\n2,y\n")
    h = read_arff_header(arff)
    df = pd.DataFrame({"a": [1.0, 2.0], "cls": ["z", "y"]})
    decl = _decl(class_labels=["y", "z"], declared_columns=["a"], column_types={"a": "categorical"})
    s = build_schema(df, file_path=arff, source_format="arff", arff_header=h, ontology_decl=decl)
    assert s.class_order == ["z", "y"] and s.class_order_source == "arff_declared" and s.is_default_arff()
    rep = validate_schema(s, df, ontology_decl=decl)
    assert "ONTOLOGY_TYPE_CONFLICT_DATA" in _codes(rep) and "ONTOLOGY_CLASS_ORDER_DIFFERS" in _codes(rep, "warnings")


# ---- registo do esquema resolvido e assinatura
def test_resolved_schema_is_recorded_for_reproducibility(tmp_path):
    p, df, s = _schema(tmp_path)
    cfg = tmp_path / "cfg"
    save_schema(s, cfg)
    loaded = load_saved_schema(s.dataset_id, cfg)
    assert loaded is not None and loaded.signature() == s.signature() and loaded.target == s.target
    assert load_saved_schema("inexistente", cfg) is None


def test_signature_changes_with_order_and_types_but_not_with_notes(tmp_path):
    _, _, a = _schema(tmp_path)
    _, _, b = _schema(tmp_path, trained_class_order=["c", "b", "a"])
    _, _, c = _schema(tmp_path, ontology_decl=_decl(declared_columns=["age"], column_types={"age": "categorical"}))
    assert len({a.signature(), b.signature(), c.signature()}) == 3
    a.columns[0].note = "x"
    assert a.signature() == _schema(tmp_path)[2].signature()


def test_legacy_default_arff_keeps_previous_tolerance(tmp_path):
    """ARFF por omissão: classe declarada sem amostras e ausentes no alvo são avisos (o pipeline anterior já os tratava)."""
    arff = tmp_path / "x.arff"
    arff.write_text("@relation r\n@attribute a numeric\n@attribute cls {z,y,w}\n@data\n1,z\n2,y\n3,?\n")
    h = read_arff_header(arff)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cls": ["z", "y", np.nan]})
    s = build_schema(df, file_path=arff, source_format="arff", arff_header=h)
    assert s.is_default_arff() and s.class_order == ["z", "y", "w"]
    rep = validate_schema(s, df)
    assert {"CLASS_ORDER_INVALID", "TARGET_HAS_MISSING"} <= set(_codes(rep, "warnings"))
    assert "CLASS_ORDER_INVALID" not in _codes(rep)
