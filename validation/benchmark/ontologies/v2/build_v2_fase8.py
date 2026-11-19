"""Ontologias da fase 8: Adult (TBox de raiz, a partir só dos NOMES das colunas e do dicionário de dados UCI) e Digits (cópia literal da v1,
sem relações: não há relação entre píxeis com fundamento de domínio que o mecanismo exprima). Fixadas antes de qualquer execução da fase 8.
Uso: python validation/benchmark/ontologies/v2/build_v2_fase8.py
"""
from __future__ import annotations

import re
import shutil
from xml.sax.saxutils import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
XS = "http://www.w3.org/2001/XMLSchema#"
THEMES = {   # atributo de origem -> classe (dicionário de dados UCI)
    "age": "Demographics", "sex": "Demographics", "race": "Demographics", "native-country": "Demographics",
    "education": "EducationProfile", "education-num": "EducationProfile",
    "workclass": "EmploymentProfile", "occupation": "EmploymentProfile", "hours-per-week": "EmploymentProfile",
    "marital-status": "HouseholdProfile", "relationship": "HouseholdProfile",
    "capital-gain": "CapitalFinance", "capital-loss": "CapitalFinance",
    "fnlwgt": "SurveyDesign",
}
RELATIONS = {"capital-gain": [("difference:hasCapitalLoss", "PROVADO",
                               "ganho de capital − perda de capital é o saldo de capital do ano (definição do dicionário de dados UCI/censo)")]}


def entity(column: str) -> str:
    base, _, level = column.partition("__")
    tok = "".join(w.capitalize() for w in re.split(r"[^0-9A-Za-z]+", base) if w)
    lev = "_" + re.sub(r"[^0-9A-Za-z]+", "_", level).strip("_") if level else ""
    return "has" + tok + (lev or "")


def build_adult(columns) -> Path:
    base = "http://example.org/trepan-audit/adult"
    classes = sorted(set(THEMES.values()))
    body = [f'<owl:Class rdf:about="#DomainMeasurement"><rdfs:label rdf:datatype="{XS}string">DomainMeasurement</rdfs:label></owl:Class>']
    for i, c in enumerate(classes):
        dis = "".join(f'<owl:disjointWith rdf:resource="#{o}"/>' for o in classes[i + 1:])
        body.append(f'<owl:Class rdf:about="#{c}"><rdfs:subClassOf rdf:resource="#DomainMeasurement"/>{dis}<rdfs:label rdf:datatype="{XS}string">{c}</rdfs:label></owl:Class>')
    for col in columns:
        src = col.split("__", 1)[0]
        rel = "".join(f'  <derivedRelation rdf:datatype="{XS}string">{d}</derivedRelation>\n  <knowledgeStatus rdf:datatype="{XS}string">{d} — {s}</knowledgeStatus>\n'
                      f'  <rdfs:comment rdf:datatype="{XS}string">{d} ({s}): {w}.</rdfs:comment>\n' for d, s, w in RELATIONS.get(col, []))
        body.append(f'<owl:DatatypeProperty rdf:about="#{entity(col)}">\n  <rdfs:domain rdf:resource="#{THEMES[src]}"/>\n'
                    f'  <rdfs:range rdf:resource="{XS}decimal"/>\n  <rdfs:label rdf:datatype="{XS}string">{escape(col)}</rdfs:label>\n{rel}</owl:DatatypeProperty>')
    text = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#"
         xml:base="{base}" xmlns="{base}#">
<owl:Ontology rdf:about="{base}">
  <rdfs:comment rdf:datatype="{XS}string">TBox do Adult Census (auditoria, fase 8): agrupamento temático dos atributos segundo o dicionário de dados UCI e uma relação declarada (saldo de capital). Zero instâncias; rótulos e resultados não consultados.</rdfs:comment>
</owl:Ontology>
<owl:AnnotationProperty rdf:about="#derivedRelation"/>
<owl:AnnotationProperty rdf:about="#knowledgeStatus"/>
''' + "\n".join(body) + "\n</rdf:RDF>\n"
    out = HERE / "adult.owl"
    out.write_text(text, encoding="utf-8")
    return out


def build_digits() -> Path:
    out = HERE / "digits.owl"
    shutil.copyfile(HERE.parent / "digits.owl", out)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parents[3]))
    from validation.benchmark.datasets_adult import adult_loader
    print(build_adult(adult_loader()[2]))
    print(build_digits())
