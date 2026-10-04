"""Gera uma versão enriquecida (apenas TBox) da OWL do Breast Cancer Wisconsin Diagnostic.

Parte da OWL original (mantém todas as entidades, labels e aliases, para o matching continuar igual) e
acrescenta conhecimento de domínio **documentado na descrição do dataset** (Street et al., 1993):

1. ``measurementFamily`` / ``statisticRole`` em cada uma das 30 propriedades -> o processador gera features
   relacionais (worst-mean, delta relativo, contraste, erro normalizado, razão do erro) por família;
2. famílias por propriedade medida (radius, texture, ...) como super-propriedades -> agregados por família;
3. conceitos de domínio (tamanho / forma / textura) como super-propriedades -> agregados por conceito.

Não contém instâncias, linhas do dataset, partições nem limiares aprendidos.

Uso: python scripts/build_breast_cancer_enriched_owl.py ENTRADA.owl SAIDA.owl
"""
from __future__ import annotations

import re
import sys

SIZE = ("radius", "perimeter", "area")
SHAPE = ("smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension")
TEXTURE = ("texture",)
CONCEPTS = {"Size": SIZE, "Shape": SHAPE, "Texture": TEXTURE}
ROLE_OF_SUFFIX = {"mean": "mean", "se": "error", "worst": "worst"}
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def camel(token: str) -> str:
    return "".join(part.capitalize() for part in token.split("_"))


def concept_of(family: str) -> str:
    for name, families in CONCEPTS.items():
        if family in families:
            return name
    raise ValueError(family)


def enrich(text: str) -> str:
    # 1) cabeçalho: propriedades de anotação
    text = text.replace(
        '<owl:ObjectProperty rdf:about="#hasDomainOutcome">',
        '<owl:AnnotationProperty rdf:about="#measurementFamily"/>\n'
        '<owl:AnnotationProperty rdf:about="#statisticRole"/>\n\n'
        '<owl:ObjectProperty rdf:about="#hasDomainOutcome">', 1)

    families_seen: set[str] = set()

    def patch(match: re.Match) -> str:
        block = match.group(0)
        labels = re.findall(r">([a-z_]+_(?:mean|se|worst))<", block)
        if not labels:
            return block
        family, _, suffix = labels[0].rpartition("_")
        role = ROLE_OF_SUFFIX[suffix]
        families_seen.add(family)
        extra = (
            f'  <rdfs:subPropertyOf rdf:resource="#has{camel(family)}Measurement"/>\n'
            f'  <rdfs:subPropertyOf rdf:resource="#hasNuclear{concept_of(family)}Measurement"/>\n'
            f'  <measurementFamily rdf:datatype="{XSD_STRING}">{family}</measurementFamily>\n'
            f'  <statisticRole rdf:datatype="{XSD_STRING}">{role}</statisticRole>\n'
        )
        return block.replace("</owl:DatatypeProperty>", extra + "</owl:DatatypeProperty>")

    text = re.sub(r"<owl:DatatypeProperty rdf:about=\"#has[A-Za-z]+\">.*?</owl:DatatypeProperty>", patch, text, flags=re.S)

    # 2) super-propriedades (família e conceito) e classes de domínio
    new = ['<owl:DatatypeProperty rdf:about="#hasNuclearMorphometricValue">\n'
           '  <rdfs:domain rdf:resource="#BreastMassCytologyProfile"/>\n'
           '  <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#decimal"/>\n'
           f'  <rdfs:label rdf:datatype="{XSD_STRING}">nuclear morphometric value</rdfs:label>\n'
           '</owl:DatatypeProperty>\n']
    for concept, fams in CONCEPTS.items():
        new.append(
            f'<owl:DatatypeProperty rdf:about="#hasNuclear{concept}Measurement">\n'
            f'  <rdfs:subPropertyOf rdf:resource="#hasNuclearMorphometricValue"/>\n'
            f'  <rdfs:domain rdf:resource="#Nuclear{concept}Profile"/>\n'
            '  <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#decimal"/>\n'
            f'  <rdfs:label rdf:datatype="{XSD_STRING}">nuclear {concept.lower()} measurement</rdfs:label>\n'
            f'  <rdfs:comment rdf:datatype="{XSD_STRING}">groups the {", ".join(f.replace("_", " ") for f in fams)} '
            'measurements (Street et al., 1993 feature description)</rdfs:comment>\n'
            '</owl:DatatypeProperty>\n')
    for family in sorted(families_seen):
        concept = concept_of(family)
        new.append(
            f'<owl:DatatypeProperty rdf:about="#has{camel(family)}Measurement">\n'
            f'  <rdfs:subPropertyOf rdf:resource="#hasNuclear{concept}Measurement"/>\n'
            '  <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#decimal"/>\n'
            f'  <rdfs:label rdf:datatype="{XSD_STRING}">{family.replace("_", " ")} measurement (any statistic)</rdfs:label>\n'
            '</owl:DatatypeProperty>\n')
    classes = []
    for concept in CONCEPTS:
        classes.append(
            f'<owl:Class rdf:about="#Nuclear{concept}Profile">\n'
            '  <rdfs:subClassOf rdf:resource="#BreastMassCytologyProfile"/>\n'
            f'  <rdfs:label rdf:datatype="{XSD_STRING}">nuclear {concept.lower()} profile</rdfs:label>\n'
            '</owl:Class>\n')
    anchor = '<owl:Class rdf:about="#BreastMassCytologyProfile">'
    text = text.replace(anchor, "\n".join(new) + "\n" + anchor, 1)
    text = text.replace('<owl:Class rdf:about="#BreastMassDiagnosis">', "\n".join(classes) + "\n" + '<owl:Class rdf:about="#BreastMassDiagnosis">', 1)
    text = text.replace("3.0-domain-rich", "4.0-domain-rich-relational", 1)
    text = text.replace("</owl:Ontology>", f'  <rdfs:comment rdf:datatype="{XSD_STRING}">v4: adds measurementFamily/statisticRole annotations, '
                        "per-family and per-concept (size/shape/texture) super-properties. TBox only.</rdfs:comment>\n</owl:Ontology>", 1)
    return text


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as fh:
        out = enrich(fh.read())
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"escrito {dst}")
