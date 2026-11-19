"""Relações entre grandezas declaradas na ontologia (anotação ``derivedRelation``): mecanismo genérico do OntologyProcessor."""
import numpy as np
import pandas as pd
import pytest

from core.ontology_processor import OntologyProcessor
from core.owl_runtime import load_ontology_isolated

XS = "http://www.w3.org/2001/XMLSchema#"


def _owl(tmp_path, declarations):
    """TBox mínima de um domínio inventado (reservatório): 4 propriedades numéricas, relações declaradas em ``declarations``."""
    props = {"hasInflow": "caudal_entrada", "hasOutflow": "caudal_saida", "hasDepth": "profundidade", "hasSurface": "superficie"}
    body = ""
    for entity, label in props.items():
        extra = "".join(f'<derivedRelation rdf:datatype="{XS}string">{d}</derivedRelation>' for d in declarations.get(entity, []))
        body += (f'<owl:DatatypeProperty rdf:about="#{entity}"><rdfs:domain rdf:resource="#Reservoir"/>'
                 f'<rdfs:range rdf:resource="{XS}decimal"/><rdfs:label rdf:datatype="{XS}string">{label}</rdfs:label>{extra}</owl:DatatypeProperty>\n')
    text = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#" xml:base="http://example.org/audit/reservoir" xmlns="http://example.org/audit/reservoir#">
<owl:Ontology rdf:about="http://example.org/audit/reservoir"/>
<owl:AnnotationProperty rdf:about="#derivedRelation"/>
<owl:Class rdf:about="#Reservoir"/>
{body}</rdf:RDF>'''
    path = tmp_path / "reservoir.owl"
    path.write_text(text, encoding="utf-8")
    return load_ontology_isolated(str(path)), props


def _frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"caudal_entrada": rng.uniform(1, 9, n), "caudal_saida": rng.uniform(1, 9, n),
                         "profundidade": rng.uniform(2, 30, n), "superficie": rng.uniform(10, 500, n)})


def _matches(props, mapping=None):
    mapping = mapping or {label: entity for entity, label in props.items()}
    return [{"feature": label, "entity_name": entity, "accepted": True} for label, entity in mapping.items()]


def _fit(tmp_path, declarations, mapping=None):
    onto, props = _owl(tmp_path, declarations)
    frame = _frame()
    proc = OntologyProcessor(onto).fit(frame, accepted_matches=_matches(props, mapping), log=False)
    return proc, frame


def test_declared_operations_produce_the_declared_quantities(tmp_path):
    proc, frame = _fit(tmp_path, {"hasInflow": ["ratio:hasOutflow", "difference:hasOutflow"],
                                  "hasDepth": ["product:hasSurface", "ratio:hasSurface:2:1"]})
    out = proc.transform(frame)
    assert np.allclose(out["onto_hasInflow_ratio_hasOutflow"], frame.caudal_entrada / frame.caudal_saida)
    assert np.allclose(out["onto_hasInflow_difference_hasOutflow"], frame.caudal_entrada - frame.caudal_saida)
    assert np.allclose(out["onto_hasDepth_product_hasSurface"], frame.profundidade * frame.superficie)
    assert np.allclose(out["onto_hasDepth_pow2_ratio_hasSurface"], frame.profundidade ** 2 / frame.superficie)
    rows = {r["feature"]: r for r in proc.feature_audit_ if r["accepted"]}
    row = rows["onto_hasDepth_pow2_ratio_hasSurface"]
    assert row["knowledge_source"] == "ontology" and row["provenance"] == "owl_declared_relation"
    assert row["owl_origin"]["source_columns"] == ["profundidade", "superficie"]


def test_no_declaration_generates_no_declared_relation(tmp_path):
    proc, _ = _fit(tmp_path, {})
    assert not [s for s in proc.feature_specs_ if s.get("provenance") == "owl_declared_relation"]


def test_relations_follow_the_column_to_entity_mapping(tmp_path):
    """Com o mapeamento coluna→entidade permutado (controlo negativo), a mesma operação aplica-se a outras colunas."""
    permuted = {"profundidade": "hasInflow", "superficie": "hasOutflow", "caudal_entrada": "hasDepth", "caudal_saida": "hasSurface"}
    proc, frame = _fit(tmp_path, {"hasInflow": ["ratio:hasOutflow"]}, mapping=permuted)
    out = proc.transform(frame)
    assert np.allclose(out["onto_hasInflow_ratio_hasOutflow"], frame.profundidade / frame.superficie)


@pytest.mark.parametrize("declaration,reason", [
    ("ratio", "malformed_declaration"), ("power:hasOutflow", "malformed_declaration"), ("ratio:hasOutflow:2", "malformed_declaration"),
    ("ratio:hasOutflow:a:1", "non_numeric_exponent"), ("ratio:hasOutflow:0:1", "invalid_exponent"),
    ("ratio:hasUnknown", "operand_not_mapped_to_a_column"), ("ratio:hasInflow", "operand_not_mapped_to_a_column"),
])
def test_invalid_declarations_are_recorded_and_generate_nothing(tmp_path, declaration, reason):
    proc, _ = _fit(tmp_path, {"hasInflow": [declaration]})
    assert not [s for s in proc.feature_specs_ if s.get("provenance") == "owl_declared_relation"]
    assert [i["reason"] for i in proc.declared_relation_issues_] == [reason]


def test_transform_on_new_rows_uses_only_the_declared_formula(tmp_path):
    proc, _ = _fit(tmp_path, {"hasInflow": ["ratio:hasOutflow"]})
    new = _frame(n=15, seed=9)
    assert np.allclose(proc.transform(new)["onto_hasInflow_ratio_hasOutflow"], new.caudal_entrada / new.caudal_saida)


def test_zero_denominator_stays_finite(tmp_path):
    onto, props = _owl(tmp_path, {"hasInflow": ["ratio:hasOutflow"]})
    frame = _frame()
    frame.loc[0, "caudal_saida"] = 0.0
    proc = OntologyProcessor(onto).fit(frame, accepted_matches=_matches(props), log=False)
    assert np.isfinite(proc.transform(frame)["onto_hasInflow_ratio_hasOutflow"]).all()
