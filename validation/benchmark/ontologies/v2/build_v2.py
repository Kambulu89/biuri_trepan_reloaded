"""Gera as ontologias v2 a partir das v1: mesma TBox + relações entre grandezas declaradas (anotação ``derivedRelation``).

Cada relação tem estatuto e fonte no comentário da propriedade. Nenhuma foi escolhida com base em rótulos, resultados de
modelos ou métricas: vêm de definições (geometria, química dos compostos fenólicos, dicionário de dados do dataset).
Uso: python validation/benchmark/ontologies/v2/build_v2.py
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
XS = "http://www.w3.org/2001/XMLSchema#string"

# entidade -> [(declaração, estatuto, justificação)]
RELATIONS = {
    "iris": {
        "hasSepalLengthCm": [
            ("product:hasSepalWidthCm", "INFERIDO", "comprimento × largura: aproximação retangular da área da sépala (geometria elementar; descritor morfométrico corrente)"),
            ("ratio:hasSepalWidthCm", "INFERIDO", "comprimento / largura: razão de aspeto da sépala (descritor de forma independente do tamanho)"),
        ],
        "hasPetalLengthCm": [
            ("product:hasPetalWidthCm", "INFERIDO", "comprimento × largura: aproximação retangular da área da pétala"),
            ("ratio:hasPetalWidthCm", "INFERIDO", "comprimento / largura: razão de aspeto da pétala"),
            ("ratio:hasSepalLengthCm", "INFERIDO", "comprimento da pétala / comprimento da sépala: tamanho relativo de dois órgãos do mesmo perianto"),
        ],
        "hasPetalWidthCm": [
            ("ratio:hasSepalWidthCm", "INFERIDO", "largura da pétala / largura da sépala: tamanho relativo de dois órgãos do mesmo perianto"),
        ],
    },
    "wine": {
        "hasFlavanoids": [
            ("ratio:hasTotalPhenols", "PROVADO", "os flavonoides são uma subclasse dos compostos fenólicos do vinho; a razão é a fração flavonoide dos fenóis totais (parte/todo). Fonte: Ribéreau-Gayon et al., Handbook of Enology, vol. 2, cap. 6 (2006)"),
        ],
        "hasNonflavanoidPhenols": [
            ("ratio:hasTotalPhenols", "PROVADO", "fenóis não flavonoides / fenóis totais: fração complementar da anterior (parte/todo). Mesma fonte"),
        ],
        "hasProanthocyanins": [
            ("ratio:hasFlavanoids", "PROVADO", "as proantocianidinas (taninos condensados) são polímeros de flavan-3-óis, uma subclasse dos flavonoides; a razão é a sua fração (parte/todo). Mesma fonte"),
        ],
        "hasAlcalinityOfAsh": [
            ("ratio:hasAsh", "INFERIDO", "a alcalinidade das cinzas é medida sobre as cinzas; a razão dá a alcalinidade por unidade de cinza"),
        ],
        "hasMagnesium": [
            ("ratio:hasAsh", "INFERIDO", "o magnésio é um constituinte mineral das cinzas; a razão dá a fração de magnésio nas cinzas"),
        ],
    },
    "breast_cancer": {
        "hasMeanPerimeter": [
            ("ratio:hasMeanArea:2:1", "PROVADO", "perímetro² / área: é a definição de compacidade do dicionário de dados (compactness = perimeter² / area − 1.0). Fonte: Street, Wolberg, Mangasarian (1993), Nuclear feature extraction for breast tumor diagnosis"),
            ("ratio:hasMeanRadius", "INFERIDO", "perímetro / raio: vale 2π num círculo; mede o desvio da forma circular"),
        ],
        "hasMeanArea": [
            ("ratio:hasMeanRadius:1:2", "INFERIDO", "área / raio²: vale π num círculo; mede o desvio da forma circular"),
        ],
        "hasWorstPerimeter": [
            ("ratio:hasWorstArea:2:1", "PROVADO", "perímetro² / área nos piores valores: definição de compacidade do dicionário de dados. Fonte: Street, Wolberg, Mangasarian (1993)"),
            ("ratio:hasWorstRadius", "INFERIDO", "perímetro / raio nos piores valores: 2π num círculo"),
        ],
        "hasWorstArea": [
            ("ratio:hasWorstRadius:1:2", "INFERIDO", "área / raio² nos piores valores: π num círculo"),
        ],
    },
}


def build(name: str) -> Path:
    text = (V1 / f"{name}.owl").read_text(encoding="utf-8")
    note = (f'  <rdfs:comment rdf:datatype="{XS}">Versão 2: acrescenta relações entre grandezas declaradas por anotação derivedRelation '
            f'("operação:outraPropriedade[:expoenteEsquerdo:expoenteDireito]"). Cada relação tem estatuto (PROVADO/INFERIDO) e justificação '
            f'no comentário da propriedade. Nenhuma relação foi escolhida com base em rótulos, resultados ou métricas.</rdfs:comment>\n')
    assert text.count("</owl:Ontology>") == 1
    text = text.replace("</owl:Ontology>", note + "</owl:Ontology>\n\n"
                        '<owl:AnnotationProperty rdf:about="#derivedRelation"/>\n<owl:AnnotationProperty rdf:about="#knowledgeStatus"/>')
    for entity, declarations in RELATIONS[name].items():
        pattern = re.compile(r'(<owl:DatatypeProperty rdf:about="#' + re.escape(entity) + r'">.*?)(</owl:DatatypeProperty>)', re.S)
        assert len(pattern.findall(text)) == 1, (name, entity)
        extra = ""
        for declaration, status, why in declarations:
            extra += (f'  <derivedRelation rdf:datatype="{XS}">{declaration}</derivedRelation>\n'
                      f'  <knowledgeStatus rdf:datatype="{XS}">{declaration} — {status}</knowledgeStatus>\n'
                      f'  <rdfs:comment rdf:datatype="{XS}">{declaration} ({status}): {why}.</rdfs:comment>\n')
        text = pattern.sub(lambda m: m.group(1) + extra + m.group(2), text)
    out = HERE / f"{name}.owl"
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    for dataset in RELATIONS:
        print(build(dataset))
