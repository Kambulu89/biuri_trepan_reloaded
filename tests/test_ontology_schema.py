"""O esquema do dataset (alvo, ordem das classes, tipos) é lido da própria ontologia OWL — vocabulário genérico, sem datasets."""
import pytest

pytest.importorskip("owlready2")
from owlready2 import World

from core.ontology_schema import load_vocabulary, norm, read_ontology_schema

HEAD = ('<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema#" xml:base="http://ex.org/o" xmlns="http://ex.org/o#">'
        '<owl:Ontology rdf:about="http://ex.org/o"/>')


def _load(tmp_path, body, name="o.owl"):
    p = tmp_path / name
    p.write_text(HEAD + body + "</rdf:RDF>", encoding="utf-8")
    return World().get_ontology(str(p.resolve())).load()


def _abox(attr_cls, attr_name_prop, type_prop, label_cls, value_prop, attrs, labels, extra=""):
    """Ontologia 'descritor de atributos' (indivíduos), com vocabulário parametrizável."""
    body = f'<owl:Class rdf:about="#{attr_cls}"/><owl:Class rdf:about="#{label_cls}"/>'
    for p in (attr_name_prop, type_prop, value_prop):
        body += f'<owl:DatatypeProperty rdf:about="#{p}"/>'
    for name, kind in attrs:
        body += (f'<rdf:Description rdf:about="#attr_{name}"><rdf:type rdf:resource="#{attr_cls}"/>'
                 f'<{attr_name_prop}>{name}</{attr_name_prop}><{type_prop}>{kind}</{type_prop}></rdf:Description>')
    for i, lab in enumerate(labels, 1):
        body += (f'<rdf:Description rdf:about="#lab_{i}"><rdf:type rdf:resource="#{label_cls}"/>'
                 f'<{value_prop}>{lab}</{value_prop}></rdf:Description>')
    return body + extra


def test_vocabulary_is_generic_and_free_of_dataset_names():
    text = open("core/ontology_schema_vocabulary.json", encoding="utf-8").read().lower()
    for forbidden in ("iris", "wine", "breast", "digits", "german", "diabetes", "cancer"):
        assert forbidden not in text
    assert norm("Etiqueta Objetivo") == norm("etiqueta_objetivo") == "etiquetaobjetivo"
    assert "numeric" in load_vocabulary()["kinds"].values()


@pytest.mark.parametrize("vocab", [
    ("Atributo", "nombreAtributo", "tipoDato", "EtiquetaObjetivo", "valorEtiqueta"),
    ("Attribute", "attributeName", "dataType", "TargetLabel", "classValue"),
    ("Variable", "biuriColumnName", "biuriDataType", "ClassLabel", "biuriClassValue"),
])
def test_reads_descriptors_in_several_vocabularies(tmp_path, vocab):
    o = _load(tmp_path, _abox(*vocab, attrs=[("s1", "numeric"), ("s2", "string"), ("s3", "boolean")], labels=["P", "N"]))
    d = read_ontology_schema(o)
    assert d.column_types == {"s1": "numeric", "s2": "categorical", "s3": "boolean"}
    assert sorted(d.declared_columns) == ["s1", "s2", "s3"]
    assert d.class_labels == ["P", "N"] and not d.class_order_explicit and "nomeação" in d.class_order_evidence


def test_explicit_class_index_overrides_naming_order(tmp_path):
    body = _abox("Atributo", "nombreAtributo", "tipoDato", "EtiquetaObjetivo", "valorEtiqueta", [("s1", "numeric")], ["P", "N"],
                 extra='<owl:DatatypeProperty rdf:about="#biuriClassIndex"/>'
                       '<rdf:Description rdf:about="#lab_1"><biuriClassIndex>2</biuriClassIndex></rdf:Description>'
                       '<rdf:Description rdf:about="#lab_2"><biuriClassIndex>1</biuriClassIndex></rdf:Description>')
    d = read_ontology_schema(_load(tmp_path, body))
    assert d.class_labels == ["N", "P"] and d.class_order_explicit


def test_target_marker_and_class_order_annotation(tmp_path):
    body = ('<owl:DatatypeProperty rdf:about="#biuriColumnName"/><owl:DatatypeProperty rdf:about="#biuriIsTarget"/>'
            '<owl:DatatypeProperty rdf:about="#biuriClassOrder"/><owl:Class rdf:about="#Attribute"/>'
            '<rdf:Description rdf:about="#a1"><rdf:type rdf:resource="#Attribute"/><biuriColumnName>outcome</biuriColumnName>'
            '<biuriIsTarget>true</biuriIsTarget><biuriClassOrder>hi, lo ,mid</biuriClassOrder></rdf:Description>'
            '<rdf:Description rdf:about="#a2"><rdf:type rdf:resource="#Attribute"/><biuriColumnName>x</biuriColumnName></rdf:Description>')
    d = read_ontology_schema(_load(tmp_path, body))
    assert d.target_column == "outcome" and d.class_labels == ["hi", "lo", "mid"] and d.class_order_explicit


def test_type_token_target_marks_the_target_column(tmp_path):
    body = _abox("Atributo", "nombreAtributo", "tipoDato", "EtiquetaObjetivo", "valorEtiqueta",
                 [("x", "numeric"), ("outcome", "target")], ["a", "b"])
    d = read_ontology_schema(_load(tmp_path, body))
    assert d.target_column == "outcome" and "outcome" not in d.column_types


def test_property_per_column_ontology_uses_range_and_labels(tmp_path):
    body = ('<owl:Class rdf:about="#Thing1"/>'
            '<owl:DatatypeProperty rdf:about="#hasA"><rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#double"/>'
            '<rdfs:label>col_a</rdfs:label></owl:DatatypeProperty>'
            '<owl:DatatypeProperty rdf:about="#hasB"><rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>'
            '<rdfs:label>col_b</rdfs:label></owl:DatatypeProperty>'
            '<owl:DatatypeProperty rdf:about="#hasD"><rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#dateTime"/>'
            '<rdfs:label>col_d</rdfs:label></owl:DatatypeProperty>')
    o = _load(tmp_path, body)
    d = read_ontology_schema(o, ["col_a", "col_b", "col_d", "outra"])
    assert d.column_types == {"col_a": "numeric", "col_b": "categorical", "col_d": "datetime"}
    assert "outra" not in d.declared_columns
    assert read_ontology_schema(o).column_types == {}           # sem as colunas, não há como casar propriedades-por-coluna


def test_empty_or_missing_ontology_declares_nothing(tmp_path):
    assert not read_ontology_schema(None).has_any
    assert not read_ontology_schema(_load(tmp_path, '<owl:Class rdf:about="#Foo"/>')).has_any


def test_schema_vocabulary_properties_never_describe_columns(tmp_path):
    """`tipoDato`/`nombreAtributo`… são vocabulário de esquema: uma coluna com esse nome não é um atributo descrito."""
    o = _load(tmp_path, _abox("Atributo", "nombreAtributo", "tipoDato", "EtiquetaObjetivo", "valorEtiqueta",
                              [("s1", "numeric")], ["P", "N"]))
    d = read_ontology_schema(o, ["tipo_dato", "nombre_atributo", "s1"])
    assert d.declared_columns == ["s1"]
