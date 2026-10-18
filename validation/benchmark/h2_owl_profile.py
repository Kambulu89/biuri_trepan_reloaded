"""H2 (Doc. 62): VERIFICAÇÃO PARCIAL do subconjunto de construções OWL usado pelo mecanismo T1/T2 (classes nomeadas, ``subClassOf``, ``equivalentClass``+``unionOf`` de classes nomeadas,
``DatatypeProperty`` com ``domain``/``range``/``label``). NÃO é uma validação formal do perfil OWL 2 DL (não há OWL API neste ambiente): confirma apenas que o ficheiro NÃO usa construções fora da lista
branca e que respeita as restrições estruturais do subconjunto (sem punning classe/propriedade, uniões só de classes declaradas, ``range`` = datatype, ``subClassOf`` sem ciclos, domínios declarados).
A consistência e a satisfazibilidade são verificadas à parte pelo reasoner (HermiT)."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
ALLOWED = {"{%s}%s" % (ns, t) for ns, t in [(RDF, "RDF"), (OWL, "Ontology"), (OWL, "Class"), (RDFS, "subClassOf"), (OWL, "equivalentClass"), (OWL, "unionOf"), (OWL, "DatatypeProperty"), (RDFS, "domain"), (RDFS, "range"), (RDFS, "label"), (RDFS, "comment")]}


def _name(v: str) -> str:
    return (v or "").split("#")[-1]


def check(path) -> Dict[str, Any]:
    root = ET.parse(path).getroot()
    about = "{%s}about" % RDF
    res = "{%s}resource" % RDF
    tags = {e.tag for e in root.iter()}
    problems: List[str] = [f"construção fora da lista branca: {t}" for t in sorted(tags - ALLOWED)]
    named = {_name(e.get(about)) for e in root.findall("{%s}Class" % OWL) if e.get(about)}
    props = {_name(e.get(about)) for e in root.findall("{%s}DatatypeProperty" % OWL) if e.get(about)}
    punning = sorted(named & props)
    problems += [f"punning classe/propriedade: {p}" for p in punning]
    edges, unions = {}, 0
    for c in root.findall("{%s}Class" % OWL):
        n = _name(c.get(about))
        if not n:
            continue
        for sc in c.findall("{%s}subClassOf" % RDFS):
            t = _name(sc.get(res))
            if t not in named:
                problems.append(f"subClassOf para classe não declarada: {n} ⊑ {t}")
            edges.setdefault(n, set()).add(t)
        for ec in c.findall("{%s}equivalentClass" % OWL):
            u = ec.find("{%s}Class/{%s}unionOf" % (OWL, OWL))
            if u is None:
                problems.append(f"equivalentClass que não é união de classes: {n}")
                continue
            members = [_name(m.get(about)) for m in u]
            unions += 1
            if len(members) < 1 or any(m not in named for m in members):
                problems.append(f"união com membro não declarado em {n}: {members}")
    # ciclos em subClassOf
    def cyc(n, seen):
        for t in edges.get(n, ()):
            if t in seen or cyc(t, seen | {t}):
                return True
        return False
    problems += [f"ciclo em subClassOf a partir de {n}" for n in edges if cyc(n, {n})]
    for p in root.findall("{%s}DatatypeProperty" % OWL):
        pn = _name(p.get(about))
        d, r = p.find("{%s}domain" % RDFS), p.find("{%s}range" % RDFS)
        if d is None or _name(d.get(res)) not in named:
            problems.append(f"domínio não declarado: {pn}")
        if r is None or not (r.get(res) or "").startswith(XSD):
            problems.append(f"range não é um datatype XSD: {pn}")
    return {"file": Path(path).name, "named_classes": len(named), "datatype_properties": len(props), "defined_unions": unions, "problems": problems, "subset_check": "PASS" if not problems else "FAIL",
            "scope": "verificação parcial de subconjunto; NÃO é validação formal OWL 2 DL"}


def main(out: str, paths: List[str]):
    res = {"checks": [check(p) for p in paths]}
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for c in res["checks"]:
        print(c["file"], c["subset_check"], c["named_classes"], c["datatype_properties"], c["defined_unions"], c["problems"][:3])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
