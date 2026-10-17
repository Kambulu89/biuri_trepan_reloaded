"""B3.0 (Docs. 31/32): auditoria OBSERVACIONAL das ontologias DEV. Executa o caminho real (carga → reasoner → quality gate/matching → OntologyProcessor →
OntologySemanticGraph → SemanticContext) SÓ sobre dados de TREINO e produz o censo de axiomas, o que é reconhecido/ignorado, as features derivadas, os grupos,
a relatedness e as condições de ativação estrutural do Reloaded. Não altera o núcleo; não usa teste nem rótulos para nada além do particionamento estratificado."""
from __future__ import annotations
import argparse, json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#", "owl": "http://www.w3.org/2002/07/owl#"}
Q = lambda p, t: "{%s}%s" % (NS[p], t)


def axiom_census(path) -> dict:
    """Censo independente (XML) dos axiomas declarados no ficheiro OWL/RDF-XML (sem depender do parser do sistema)."""
    root = ET.parse(str(path)).getroot()
    c = Counter()
    for el in root.iter():
        tag = el.tag
        if tag in (Q("owl", "Class"), Q("owl", "DatatypeProperty"), Q("owl", "ObjectProperty"), Q("owl", "AnnotationProperty"), Q("owl", "NamedIndividual"),
                   Q("owl", "Restriction"), Q("owl", "FunctionalProperty"), Q("owl", "TransitiveProperty"), Q("owl", "SymmetricProperty"), Q("owl", "AllDisjointClasses")):
            if tag == Q("owl", "Class") and el.get(Q("rdf", "about")) is None:
                c["class_anonymous"] += 1
            else:
                c[tag.split("}")[1]] += 1
        elif tag in (Q("rdfs", "subClassOf"), Q("owl", "equivalentClass"), Q("owl", "disjointWith"), Q("rdfs", "domain"), Q("rdfs", "range"), Q("owl", "inverseOf"),
                     Q("rdfs", "subPropertyOf"), Q("owl", "propertyChainAxiom"), Q("owl", "someValuesFrom"), Q("owl", "allValuesFrom"), Q("owl", "hasValue"),
                     Q("owl", "cardinality"), Q("owl", "minCardinality"), Q("owl", "maxCardinality"), Q("owl", "qualifiedCardinality"), Q("owl", "unionOf"),
                     Q("owl", "intersectionOf"), Q("owl", "complementOf"), Q("owl", "oneOf"), Q("owl", "withRestrictions"), Q("owl", "equivalentProperty"),
                     Q("owl", "propertyDisjointWith"), Q("owl", "sameAs"), Q("owl", "differentFrom")):
            c[tag.split("}")[1]] += 1
        elif tag == Q("rdfs", "label"):
            c["label"] += 1
        elif tag == Q("rdfs", "comment"):
            c["comment"] += 1
        elif tag.startswith("{http://www.w3.org/2001/XMLSchema#}") and tag.split("}")[1] in ("minInclusive", "maxInclusive", "minExclusive", "maxExclusive"):
            c["xsd_facet_" + tag.split("}")[1]] += 1
    # propriedades de anotação usadas como dados (ex.: família/papel)
    custom = Counter()
    base_ns = set(NS.values())
    for el in root:
        for ch in el:
            ns = ch.tag.split("}")[0][1:] if ch.tag.startswith("{") else ""
            if ns and ns not in base_ns and ns != "http://www.w3.org/2001/XMLSchema#":
                custom[ch.tag.split("}")[1]] += 1
    c["custom_annotation_uses:" + ",".join(sorted(custom))] = sum(custom.values()) if custom else 0
    return {k: v for k, v in sorted(c.items()) if v}


def pipeline(dataset: str, seed: int = 42, relation_threshold: float = 0.35) -> dict:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from validation.benchmark import datasets as D
    from core.owl_runtime import load_ontology_isolated
    from core.ontology_reasoner import run_owl_reasoner
    from core.ontology_quality import OntologyQualityGate
    from core.ontology_processor import OntologyProcessor
    from core.ontology_semantic_graph import OntologySemanticGraph
    from core.benchmark.semantic import OwlSemanticProvider
    spec = D.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    Xtr, _Xte, ytr, _yte = train_test_split(ds.X, ds.y, test_size=0.25, stratify=ds.y, random_state=seed)   # só o treino é usado a partir daqui
    names = [str(n) for n in ds.feature_names]
    onto = load_ontology_isolated(str(spec.ontology_path))
    reasoner = run_owl_reasoner(onto, engine="hermit", infer_property_values=True, debug=0)
    gate = OntologyQualityGate()
    rep = gate.evaluate(names, onto, reasoner_report=reasoner, require_reasoner=True)
    accepted = [m for m in rep.matches if m.get("accepted")]
    proc = OntologyProcessor(onto, quality_gate=gate)
    proc.fit(pd.DataFrame(Xtr, columns=names), accepted_matches=accepted, log=False)
    graph = OntologySemanticGraph.from_ontology(onto, accepted_matches=accepted)
    ctx = OwlSemanticProvider(str(spec.ontology_path)).build(Xtr, names)
    rel_types = Counter(t for (a, b), ts in graph.edge_types.items() if a < b for t in ts)
    n = ctx.n_features
    rel = np.asarray(ctx.relatedness)
    iu = np.triu_indices(n, 1)
    vals = np.round(rel[iu], 4)
    groups = list(ctx.groups or [])
    same = np.array([[groups[i] is not None and groups[i] == groups[j] for j in range(n)] for i in range(n)])
    cross = ~same & ~np.eye(n, dtype=bool)
    w = np.asarray(ctx.weights, dtype=float); wn = w / w.mean()
    feats = {"hierarchical": [], "relational": [], "constraint": [], "categorical": []}
    for s in proc.feature_specs_:
        k = {"hierarchical_aggregate": "hierarchical", "relational": "relational", "constraint": "constraint", "categorical_group": "categorical"}.get(s["kind"], s["kind"])
        feats.setdefault(k, []).append({"name": s["name"], "sources": s.get("sources") or ([s["source"]] if "source" in s else []), "provenance": s.get("provenance")})
    rg = [g for g in groups if g]
    repeated = any(rg.count(g) > 1 for g in set(rg))
    related = bool(np.any((rel - np.eye(n))[iu] > relation_threshold)) if n > 1 else False
    depths = np.asarray(ctx.depths, dtype=float)
    out = {
        "dataset": dataset, "ontology": Path(spec.ontology_path).name, "n_train": int(len(Xtr)), "n_features": len(names),
        "reasoner": {"engine": reasoner.get("engine"), "consistent": reasoner.get("consistent"), "keys": sorted(k for k in reasoner),
                     "inferred_subclass_pairs": len(getattr(proc, "_inferred_subclass_pairs", []) or []),
                     "inferred": {k: (len(v) if hasattr(v, "__len__") else v) for k, v in (reasoner.get("inferred") or {}).items()} if isinstance(reasoner.get("inferred"), dict) else reasoner.get("inferred")},
        "quality": {"status": rep.status, "accepted": rep.accepted, "issues": list(rep.issues), "metrics": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:80]) for k, v in rep.metrics.items()}},
        "matching": {"n_features": len(names), "n_accepted": len(accepted), "scores": [round(float(m["score"]), 3) for m in rep.matches], "margins": [round(float(m["ambiguity_margin"]), 3) for m in rep.matches],
                     "tie_breaks": sum(1 for m in rep.matches if m.get("tie_break")), "rejected": [(m["feature"], m["status"]) for m in rep.matches if not m.get("accepted")]},
        "graph": {"nodes": len(graph.nodes), "edge_types": dict(rel_types), "builtin_nodes": sorted(graph.builtin_nodes), "parents": {k: sorted(v) for k, v in graph.parents.items()}, "domains": {k: sorted(v) for k, v in graph.domains.items()},
                  "entity_bounds": len(graph.entity_bounds), "integer_entities": len(graph.integer_entities)},
        "derived_features": {k: v for k, v in feats.items()}, "n_derived": len(ctx.onto_idx),
        "groups": {g: [enr for enr, gg in zip(ctx.enriched_names, groups) if gg == g] for g in sorted(set(rg))},
        "relatedness": {"distinct_values_offdiag": sorted(set(vals.tolist())), "n_pairs": int(len(vals)), "same_group_values": sorted(set(np.round(rel[same & np.triu(np.ones((n, n), bool), 1)], 4).tolist())),
                        "cross_group_values": sorted(set(np.round(rel[cross & np.triu(np.ones((n, n), bool), 1)], 4).tolist())), "pairs_ge_threshold": int((rel[iu] >= relation_threshold).sum())},
        "depths": {"distinct": sorted(set(np.round(depths, 4).tolist())), "orig_features": [round(float(d), 3) for d in depths[: len(names)]]},
        "weights_normalised": {"orig": round(float(wn[0]), 4), "derived": round(float(wn[-1]), 4) if len(wn) > len(names) else None},
        "structure_active_conditions": {"repeated_group": repeated, "related_above_threshold": related, "weighted": bool(not np.allclose(w, 1.0)), "deep": bool(np.any(depths > 0))},
        "context_available": bool(ctx.available), "context_reason": ctx.reason,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "digits"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from validation.benchmark import datasets as D
    res = {}
    for d in a.datasets:
        r = pipeline(d)
        r["axiom_census"] = axiom_census(D.REGISTRY[d].ontology_path)
        res[d] = r
        print(d, "ok", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
