"""B3.0 (Docs. 31/32): sonda de axiomas OWL. Constrói mini-ontologias OWL 2 controladas (RDF/XML) com UM tipo de axioma de cada vez e regista, SEM alterar o núcleo,
o que o sistema faz com ele: grafo (arestas por tipo), grupos, profundidades, relatedness, features derivadas (OntologyProcessor), arestas inferidas pelo reasoner.
Serve de evidência para «reconhece / usa no reasoner / mapeia para features / produz candidatos / influencia splits» (Doc. 31 §5)."""
from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HEADER = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#"
         xml:base="http://example.org/b30" xmlns="http://example.org/b30#">
<owl:Ontology rdf:about="http://example.org/b30"/>
"""
FOOTER = "</rdf:RDF>\n"


def dprop(name: str, domain: Optional[str] = None, extra: str = "", rng: str = "http://www.w3.org/2001/XMLSchema#decimal", label: Optional[str] = None) -> str:
    dom = f'<rdfs:domain rdf:resource="#{domain}"/>' if domain else ""
    lab = f'<rdfs:label rdf:datatype="http://www.w3.org/2001/XMLSchema#string">{label or name}</rdfs:label>'
    return f'<owl:DatatypeProperty rdf:about="#{name}">{dom}<rdfs:range rdf:resource="{rng}"/>{lab}{extra}</owl:DatatypeProperty>\n'


def klass(name: str, sub: Sequence[str] = (), body: str = "") -> str:
    subs = "".join(f'<rdfs:subClassOf rdf:resource="#{s}"/>' for s in sub)
    return f'<owl:Class rdf:about="#{name}">{subs}{body}</owl:Class>\n'


def restriction(kind: str, prop: str, filler: Optional[str] = None, card: Optional[int] = None) -> str:
    if kind in ("someValuesFrom", "allValuesFrom"):
        inner = f'<owl:{kind} rdf:resource="#{filler}"/>'
    else:
        inner = f'<owl:{kind} rdf:datatype="http://www.w3.org/2001/XMLSchema#nonNegativeInteger">{card}</owl:{kind}>'
    return (f'<rdfs:subClassOf><owl:Restriction><owl:onProperty rdf:resource="#{prop}"/>{inner}</owl:Restriction></rdfs:subClassOf>')


def write(tmp: Path, name: str, body: str) -> Path:
    p = Path(tmp) / f"{name}.owl"
    p.write_text(HEADER + body + FOOTER, encoding="utf-8")
    return p


def probe(path, feature_names: Sequence[str], X: Optional[np.ndarray] = None, *, reasoner: bool = False, seed: int = 0) -> Dict[str, object]:
    """Executa o caminho real (carga → [reasoner] → matching → grafo → processador) e devolve o que foi reconhecido/produzido."""
    import pandas as pd
    from core.owl_runtime import load_ontology_isolated
    from core.ontology_quality import OntologyQualityGate
    from core.ontology_processor import OntologyProcessor
    from core.ontology_semantic_graph import OntologySemanticGraph
    names = [str(n) for n in feature_names]
    if X is None:
        X = np.random.default_rng(seed).normal(size=(120, len(names)))
    onto = load_ontology_isolated(str(path))
    rep_r = None
    if reasoner:
        from core.ontology_reasoner import run_owl_reasoner
        rep_r = run_owl_reasoner(onto, engine="hermit", infer_property_values=True, debug=0)
    gate = OntologyQualityGate()
    decisions = gate.match_features(names, onto)
    accepted = [d.__dict__ for d in decisions if d.accepted]
    graph = OntologySemanticGraph.from_ontology(onto, accepted_matches=accepted)
    proc = OntologyProcessor(onto, quality_gate=gate)
    proc.fit(pd.DataFrame(X, columns=names), accepted_matches=accepted, log=False)
    edge_types: Dict[str, List[str]] = {}
    for (a, b), ts in graph.edge_types.items():
        if a < b:
            edge_types[f"{a}--{b}"] = sorted(ts)
    ent = {n: graph.entity_for_feature(n) for n in names}
    rel = {f"{a}--{b}": round(float(graph.relatedness(ent[a], ent[b])), 4) if ent[a] and ent[b] else None for i, a in enumerate(names) for b in names[i + 1:]}
    return {"accepted": {d["feature"]: d["entity_name"] for d in accepted}, "edge_types": edge_types, "edge_type_set": sorted({t for ts in edge_types.values() for t in ts}),
            "groups": {f: list(graph.feature_groups.get(f, [])) for f in names}, "parents": {k: sorted(v) for k, v in graph.parents.items()}, "domains": {k: sorted(v) for k, v in graph.domains.items()},
            "relatedness": rel, "depth": {n: round(float(graph.get_feature_depth(n)), 4) for n in names},
            "derived": [(s["kind"], s["name"], s.get("sources") or s.get("source")) for s in proc.feature_specs_], "builtin_nodes": sorted(graph.builtin_nodes),
            "reasoner": None if rep_r is None else {"consistent": rep_r.get("consistent"), "executed": rep_r.get("executed"), "inferred_subclass_relations": rep_r.get("inferred_subclass_relations"),
                                                      "inferred_axioms_count": rep_r.get("inferred_axioms_count"), "unsatisfiable": rep_r.get("unsatisfiable_classes")},
            "inferred_subclass_pairs": sorted(map(str, getattr(proc, "_inferred_subclass_pairs", []) or [])), "n_nodes": len(graph.nodes)}


def axiom_cases() -> Dict[str, str]:
    """Um corpo OWL por tipo de axioma. Propriedades de dados f1,f2 (domínio A) e f3 (domínio B); A, B ⊑ Root salvo indicação."""
    base = klass("Root") + klass("A", ["Root"]) + klass("B", ["Root"]) + dprop("f1", "A") + dprop("f2", "A") + dprop("f3", "B")
    obj = '<owl:ObjectProperty rdf:about="#hasPart"><rdfs:domain rdf:resource="#A"/><rdfs:range rdf:resource="#B"/></owl:ObjectProperty>\n'
    return {
        "subClassOf": base,
        "equivalentClass": klass("Root") + klass("A", ["Root"], '<owl:equivalentClass rdf:resource="#A2"/>') + klass("A2") + klass("B", ["Root"]) + dprop("f1", "A") + dprop("f2", "A2") + dprop("f3", "B"),
        "disjointWith": klass("Root") + klass("A", ["Root"], '<owl:disjointWith rdf:resource="#B"/>') + klass("B", ["Root"]) + dprop("f1", "A") + dprop("f2", "A") + dprop("f3", "B"),
        "domain_only_flat": klass("A") + klass("B") + dprop("f1", "A") + dprop("f2", "A") + dprop("f3", "B"),
        "no_domain": klass("A") + klass("B") + dprop("f1") + dprop("f2") + dprop("f3"),
        "someValuesFrom": obj + klass("Root") + klass("B", ["Root"]) + klass("A", ["Root"], restriction("someValuesFrom", "hasPart", "B")) + dprop("f1", "A") + dprop("f2", "A") + dprop("f3", "B"),
        "allValuesFrom": obj + klass("Root") + klass("B", ["Root"]) + klass("A", ["Root"], restriction("allValuesFrom", "hasPart", "B")) + dprop("f1", "A") + dprop("f2", "A") + dprop("f3", "B"),
        "cardinality": obj + klass("Root") + klass("B", ["Root"]) + klass("A", ["Root"], restriction("cardinality", "hasPart", card=1)) + dprop("f1", "A") + dprop("f2", "A") + dprop("f3", "B"),
        "dataSubPropertyOf": klass("Root") + klass("A", ["Root"]) + klass("B", ["Root"]) + dprop("fAll", "A") + dprop("f1", "A", '<rdfs:subPropertyOf rdf:resource="#fAll"/>') + dprop("f2", "A", '<rdfs:subPropertyOf rdf:resource="#fAll"/>') + dprop("f3", "B"),
        "objectPropertyInverse": obj.replace("</owl:ObjectProperty>", '<owl:inverseOf rdf:resource="#partOf"/></owl:ObjectProperty>') + '<owl:ObjectProperty rdf:about="#partOf"/>\n' + base,
        "propertyChain": obj + '<owl:ObjectProperty rdf:about="#hasPart2"/><owl:ObjectProperty rdf:about="#chain"><owl:propertyChainAxiom rdf:parseType="Collection"><owl:ObjectProperty rdf:about="#hasPart"/><owl:ObjectProperty rdf:about="#hasPart2"/></owl:propertyChainAxiom></owl:ObjectProperty>\n' + base,
        "datatypeFacets": klass("Root") + klass("A", ["Root"]) + dprop("f1", "A", rng="http://www.w3.org/2001/XMLSchema#decimal").replace(
            '<rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#decimal"/>',
            '<rdfs:range><rdfs:Datatype><owl:onDatatype rdf:resource="http://www.w3.org/2001/XMLSchema#decimal"/><owl:withRestrictions rdf:parseType="Collection">'
            '<rdf:Description><xsd:minInclusive rdf:datatype="http://www.w3.org/2001/XMLSchema#decimal">-1.0</xsd:minInclusive></rdf:Description>'
            '<rdf:Description><xsd:maxInclusive rdf:datatype="http://www.w3.org/2001/XMLSchema#decimal">1.0</xsd:maxInclusive></rdf:Description></owl:withRestrictions></rdfs:Datatype></rdfs:range>')
        + dprop("f2", "A") + dprop("f3", "A"),
    }
