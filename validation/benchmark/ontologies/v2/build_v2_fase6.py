"""Ontologias v2 da fase 6: German Credit (v1 + relação declarada) e os três controlos positivos físicos (TBox de raiz).

Fixadas antes de qualquer execução da fase 6. Uso: python validation/benchmark/ontologies/v2/build_v2_fase6.py
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
XS = "http://www.w3.org/2001/XMLSchema#"

# dataset -> (grupos {classe: [(entidade, coluna)]}, relações {entidade: [(declaração, estatuto, justificação)]}, descrição)
PHYSICS = {
    "phys_bmi": ({"BodyMeasurement": [("hasHeightM", "height_m"), ("hasMassKg", "mass_kg")],
                  "ClinicalContext": [("hasAgeYears", "age_years"), ("hasSystolicBp", "systolic_bp_mmhg"), ("hasRestingHeartRate", "resting_heart_rate_bpm")]},
                 {"hasMassKg": [("ratio:hasHeightM:1:2", "PROVADO", "massa / altura² é a definição do índice de massa corporal (OMS)")]},
                 "Medidas corporais e contexto clínico de um adulto."),
    "phys_kinetic": ({"MotionState": [("hasMassKg", "mass_kg"), ("hasSpeed", "speed_m_s")],
                      "ImpactContext": [("hasAmbientTemperature", "ambient_temperature_c"), ("hasImpactAngle", "impact_angle_deg"), ("hasContactArea", "contact_area_cm2")]},
                     {"hasMassKg": [("product:hasSpeed:1:2", "PROVADO", "massa × velocidade² é proporcional à energia cinética, E = ½ m v² (mecânica clássica)")]},
                     "Estado de movimento de um corpo e contexto do impacto."),
    "phys_power": ({"ElectricalState": [("hasVoltage", "voltage_v"), ("hasResistance", "resistance_ohm")],
                    "InstallationContext": [("hasCableLength", "cable_length_m"), ("hasAmbientTemperature", "ambient_temperature_c"), ("hasWireCrossSection", "wire_cross_section_mm2")]},
                   {"hasVoltage": [("ratio:hasResistance:2:1", "PROVADO", "tensão² / resistência é a potência dissipada, P = V²/R (lei de Ohm e de Joule)")]},
                   "Estado elétrico de uma carga resistiva e contexto da instalação."),
}
GERMAN = {"hasCreditAmount": [("ratio:hasDurationMonths", "PROVADO",
                               "montante do crédito / prazo em meses é o capital médio amortizado por mês (definição aritmética do plano de amortização)")]}


def _relations(declarations):
    out = ""
    for declaration, status, why in declarations:
        out += (f'  <derivedRelation rdf:datatype="{XS}string">{declaration}</derivedRelation>\n'
                f'  <knowledgeStatus rdf:datatype="{XS}string">{declaration} — {status}</knowledgeStatus>\n'
                f'  <rdfs:comment rdf:datatype="{XS}string">{declaration} ({status}): {why}.</rdfs:comment>\n')
    return out


def build_physics(name: str) -> Path:
    groups, relations, description = PHYSICS[name]
    base = f"http://example.org/trepan-audit/{name}"
    body = [f'<owl:Class rdf:about="#DomainMeasurement"><rdfs:label rdf:datatype="{XS}string">DomainMeasurement</rdfs:label></owl:Class>']
    classes = list(groups)
    for i, cls in enumerate(classes):
        disjoint = "".join(f'<owl:disjointWith rdf:resource="#{o}"/>' for o in classes[i + 1:])
        body.append(f'<owl:Class rdf:about="#{cls}"><rdfs:subClassOf rdf:resource="#DomainMeasurement"/>{disjoint}'
                    f'<rdfs:label rdf:datatype="{XS}string">{cls}</rdfs:label></owl:Class>')
    for cls, members in groups.items():
        for entity, column in members:
            body.append(f'<owl:DatatypeProperty rdf:about="#{entity}">\n  <rdfs:domain rdf:resource="#{cls}"/>\n'
                        f'  <rdfs:range rdf:resource="{XS}decimal"/>\n  <rdfs:label rdf:datatype="{XS}string">{column}</rdfs:label>\n'
                        f'{_relations(relations.get(entity, []))}</owl:DatatypeProperty>')
    text = f'''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#"
         xml:base="{base}" xmlns="{base}#">
<owl:Ontology rdf:about="{base}">
  <rdfs:comment rdf:datatype="{XS}string">TBox de CONTROLO POSITIVO (auditoria, fase 6). {description} Zero instâncias. A relação declarada é a lei física que gerou os dados; o dataset é sintético.</rdfs:comment>
</owl:Ontology>
<owl:AnnotationProperty rdf:about="#derivedRelation"/>
<owl:AnnotationProperty rdf:about="#knowledgeStatus"/>
''' + "\n".join(body) + "\n</rdf:RDF>\n"
    out = HERE / f"{name}.owl"
    out.write_text(text, encoding="utf-8")
    return out


def build_german() -> Path:
    text = (HERE.parent / "german_credit.owl").read_text(encoding="utf-8")
    note = (f'  <rdfs:comment rdf:datatype="{XS}string">Versão 2 (auditoria, fase 6): acrescenta a relação declarada montante / prazo '
            f'(anotação derivedRelation), com estatuto e justificação no comentário da propriedade.</rdfs:comment>\n')
    text = text.replace("</owl:Ontology>", note + "</owl:Ontology>\n\n"
                        '<owl:AnnotationProperty rdf:about="#derivedRelation"/>\n<owl:AnnotationProperty rdf:about="#knowledgeStatus"/>', 1)
    for entity, declarations in GERMAN.items():
        pattern = re.compile(r'(<owl:DatatypeProperty rdf:about="#' + re.escape(entity) + r'">.*?)(</owl:DatatypeProperty>)', re.S)
        assert len(pattern.findall(text)) == 1, entity
        text = pattern.sub(lambda m: m.group(1) + _relations(declarations) + m.group(2), text)
    out = HERE / "german_credit.owl"
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    for dataset in PHYSICS:
        print(build_physics(dataset))
    print(build_german())
