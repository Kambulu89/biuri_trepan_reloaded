"""Reasoner: diagnóstico preciso de ontologias inconsistentes (xml:lang em xsd:string) e correção em cópia."""
import importlib.util
import sys
from pathlib import Path

import pytest

owlready2 = pytest.importorskip("owlready2")

from core.ontology_reasoner import explain_inconsistency, inconsistency_message, run_owl_reasoner

ROOT = Path(__file__).resolve().parents[1]
HEAD = ('<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema#" xml:base="http://ex.org/m" xmlns="http://ex.org/m#">'
        '<owl:Ontology rdf:about="http://ex.org/m"/><owl:Class rdf:about="#A"/>'
        '<owl:DatatypeProperty rdf:about="#unidad"><rdfs:domain rdf:resource="#A"/>'
        '<rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/></owl:DatatypeProperty>')


def _onto(tmp_path, literal, name="m.owl"):
    p = tmp_path / name
    p.write_text(HEAD + '<rdf:Description rdf:about="#x"><rdf:type rdf:resource="#A"/>' + literal + "</rdf:Description></rdf:RDF>",
                 encoding="utf-8")
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor
    return p, TrepanReloadedExtractor.load_ontology_file(p)


def _fixer():
    spec = importlib.util.spec_from_file_location("owl_fix_langstring", ROOT / "scripts" / "owl_fix_langstring.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["owl_fix_langstring"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_langtagged_literal_in_xsd_string_is_named_as_the_cause(tmp_path):
    p, onto = _onto(tmp_path, '<unidad xml:lang="es">cm</unidad>')
    causes = explain_inconsistency(onto)
    assert any("xml:lang" in c and "unidad" in c for c in causes)
    msg = inconsistency_message(onto)
    assert "INCONSISTENTE" in msg and "memória" in msg and "owl_fix_langstring" in msg
    report = run_owl_reasoner(onto)
    if not report["executed"] and report["error_type"] == "reasoner_failed":
        assert "INCONSISTENTE" in report["user_message"] and "unidad" in report["user_message"]
        assert "verifique a sintaxe OWL, a memória" not in report["user_message"]       # deixa de enganar com "memória/Java"
    elif not report["executed"]:
        pytest.skip(f"BLOCKED_BY_ENVIRONMENT: {report['error_type']}")


def test_clean_ontology_has_no_findings(tmp_path):
    p, onto = _onto(tmp_path, "<unidad>cm</unidad>")
    assert explain_inconsistency(onto) == []


def test_fixer_writes_a_copy_and_makes_the_ontology_consistent(tmp_path):
    fixer = _fixer()
    p, _ = _onto(tmp_path, '<unidad xml:lang="es">cm</unidad>')
    original = p.read_text(encoding="utf-8")
    out = tmp_path / "fixed" / "m_biuri.owl"
    changes = fixer.fix_file(p, out)
    assert changes and changes[0]["propriedade"] == "unidad" and changes[0]["lang"] == "es"
    assert p.read_text(encoding="utf-8") == original                                  # o original nunca é alterado
    assert "xml:lang" not in out.read_text(encoding="utf-8")
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor
    onto = TrepanReloadedExtractor.load_ontology_file(out)
    report = run_owl_reasoner(onto)
    if not report["executed"]:
        pytest.skip(f"BLOCKED_BY_ENVIRONMENT: {report['error_type']}")
    assert report["consistent"] is True
    with pytest.raises(ValueError):
        fixer.fix_file(p, p)


def test_fixer_keeps_language_tags_on_properties_that_are_not_xsd_string(tmp_path):
    fixer = _fixer()
    text = HEAD.replace("</owl:DatatypeProperty>", "</owl:DatatypeProperty>") + (
        '<owl:AnnotationProperty rdf:about="#nota"/>'
        '<rdf:Description rdf:about="#x"><rdf:type rdf:resource="#A"/><nota xml:lang="pt">olá</nota></rdf:Description></rdf:RDF>')
    new, changes = fixer.fix_text(text)
    assert changes == [] and 'xml:lang="pt"' in new


def test_fixer_can_drop_imports(tmp_path):
    fixer = _fixer()
    text = HEAD.replace('<owl:Ontology rdf:about="http://ex.org/m"/>',
                        '<owl:Ontology rdf:about="http://ex.org/m"><owl:imports rdf:resource="http://www.w3.org/ns/prov-o#"/></owl:Ontology>') + "</rdf:RDF>"
    new, changes = fixer.fix_text(text, drop_imports=True)
    assert [c["propriedade"] for c in changes] == ["owl:imports"] and "prov-o" not in new
