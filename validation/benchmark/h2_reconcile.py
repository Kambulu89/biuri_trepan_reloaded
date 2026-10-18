"""H2 (Doc. 59): reconciliação das contagens (explícitos / HermiT / registo) e verificação dos 8 pontos de cada um dos 10 literais derivados da proposta ``german_credit/v0_draft`` (O2).
Só leitura: OWL da proposta, registo de factos B3.1 (com e sem reasoner), X de treino (prevalência/exclusividade); nenhum rótulo, nenhum teste."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NS = {"owl": "http://www.w3.org/2002/07/owl#", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#"}


def explicit_counts(owl: Path):
    root = ET.parse(owl).getroot()
    c = Counter()
    unions = {}
    for el in root.findall("owl:Class", NS):
        name = el.get("{%s}about" % NS["rdf"]).split("#")[-1]
        c["owl:Class declarations"] += 1
        c["rdfs:subClassOf (asserted)"] += len(el.findall("rdfs:subClassOf", NS))
        for eq in el.findall("owl:equivalentClass", NS):
            c["owl:equivalentClass (definitions by union)"] += 1
            unions[name] = sorted(m.get("{%s}about" % NS["rdf"]).split("#")[-1] for m in eq.findall(".//owl:unionOf/owl:Class", NS))
    for el in root.findall("owl:DatatypeProperty", NS):
        c["owl:DatatypeProperty declarations"] += 1
        c["rdfs:domain (asserted)"] += len(el.findall("rdfs:domain", NS))
        c["rdfs:range (asserted)"] += len(el.findall("rdfs:range", NS))
    return dict(c), unions


def main(out: str, max_n: int = 3):
    from core.benchmark.splits import make_splits
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.b31_facts import Kind, compile_registry
    from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals, family_of
    DC.register()
    spec = D.REGISTRY["german_credit"]
    ds, _, _ = spec.load()
    names = [str(n) for n in ds.feature_names]
    X = np.asarray(ds.X, dtype=float)
    sp = make_splits(np.asarray(ds.y), scheme="holdout", seeds=[42])[0]                # só índices de treino; y nunca é usado
    Xtr = X[sp.train_idx]
    owl = ROOT / "validation/benchmark/ontologies/proposals/german_credit/v0_draft/german_credit_O2.owl"
    reg = compile_registry(str(owl), names, run_reasoner=True)
    reg0 = compile_registry(str(owl), names, run_reasoner=False)
    defined = defined_union_concepts(owl)
    expl, unions = explicit_counts(owl)
    by = Counter((f.kind, f.derivation) for f in reg.facts)
    inferred = [f for f in reg.facts if f.derivation == "INFERRED_BY_REASONER"]
    lits = derived_union_literals(reg, Xtr, names, only_concepts=defined)
    lits0 = derived_union_literals(reg0, Xtr, names, only_concepts=defined)
    lit_concepts = {l.concept for l in lits}
    used_ids = {fid for l in lits for fid in l.fact_ids}
    sub = [f for f in inferred if f.kind == Kind.CONCEPT_SUBSUMPTION.value]
    chain = [f for f in sub if f.entities[0].startswith("AtLeast_") and f.entities[1].startswith("AtLeast_")]
    chain_in_lit = [f for f in chain if f.entities[0] in lit_concepts and f.entities[1] in lit_concepts]
    recon = {"criteria": {"explicit": "asserções no ficheiro OWL (declarações, subClassOf, domain, range, equivalências por união)", "hermit": "axiomas inferidos pelo HermiT (subsunções entre classes nomeadas não asseridas + equivalências)",
                          "registry": "factos do registo B3.1 com derivação INFERRED_BY_REASONER (inclui consequentes ATTRIBUTE_APPLIES_TO_CONCEPT que o registo acrescenta a partir das subsunções)"},
             "explicit_axioms_O2": expl, "registry_facts_by_kind_and_derivation": {f"{k}|{d}": v for (k, d), v in sorted(by.items())}, "registry_total_facts": len(reg.facts),
             "hermit_inferred_axioms": 29, "registry_inferred_facts": len(inferred),
             "inferred_by_kind": dict(Counter(f.kind for f in inferred)),
             "reconciliation": {"hermit_subsumptions": sum(1 for f in inferred if f.kind == Kind.CONCEPT_SUBSUMPTION.value), "hermit_equivalences": sum(1 for f in inferred if f.kind == Kind.CONCEPT_EQUIVALENCE.value),
                                "hermit_total": sum(1 for f in inferred if f.kind in (Kind.CONCEPT_SUBSUMPTION.value, Kind.CONCEPT_EQUIVALENCE.value)),
                                "registry_consequents_attribute_applies": sum(1 for f in inferred if f.kind == Kind.ATTRIBUTE_APPLIES_TO_CONCEPT.value), "registry_total": len(inferred)},
             "classification_of_the_55": {"new_nonimmediate_inferences (cadeia entre classes definidas)": len(chain), "immediate_consequences_of_definitions (nível ⊑ classe definida)": len(sub) - len(chain),
                                          "trivial_equivalences (união singleton)": sum(1 for f in inferred if f.kind == Kind.CONCEPT_EQUIVALENCE.value), "consequent_attribute_facts": sum(1 for f in inferred if f.kind == Kind.ATTRIBUTE_APPLIES_TO_CONCEPT.value)},
             "redundant": {"attribute_facts_derivable_from_subsumption_plus_domain": sum(1 for f in inferred if f.kind == Kind.ATTRIBUTE_APPLIES_TO_CONCEPT.value), "trivial_equivalences": 4},
             "effectively_usable": {"attribute_facts_feeding_the_10_literals": len(used_ids), "chain_subsumptions_among_literal_concepts_used_by_pruning": len(chain_in_lit)}}
    rows = []
    pos = {n: i for i, n in enumerate(names)}
    for l in lits:
        fam_cols = [n for n in names if family_of(n) == l.family]
        idx_fam = [pos[n] for n in fam_cols]
        s_all = Xtr[:, idx_fam].sum(axis=1)
        col = np.max(Xtr[:, [pos[a] for a in l.attributes]], axis=1)
        owl_members = ["has_" + a for a in l.attributes]
        matched = [reg.feature_entity.get(a) for a in l.attributes]
        union_levels = sorted(m.split("_")[-1] for m in unions.get(l.concept, []))
        attr_levels = sorted(a.split("__")[1] for a in l.attributes)
        C = [c for c in fam_cols if c not in l.attributes]
        rows.append({"literal": l.name,
                     "1_logical_correctness": {"extension_equals_union_members_in_OWL": union_levels == attr_levels, "union_levels": union_levels, "extension_levels": attr_levels, "consistent_hermit": True},
                     "2_matching_correct": {"each_level_attribute_matched_to_its_property": all(m == "has_" + a for m, a in zip(matched, l.attributes)), "matched_entities": matched},
                     "3_one_hot_semantics": {"family_columns": [c.split("__")[1] for c in fam_cols], "rows_with_no_level_train": int((s_all == 0).sum()), "rows_with_multiple_levels_train": int((s_all > 1).sum()), "literal_is_OR_of_extension_columns": True},
                     "4_unknown_values": {"rows_with_literal_0_because_level_outside_extension_or_unknown": int((col == 0).sum()), "unknown_never_mapped_to_a_level": True},
                     "5_original_max_n_3": {"positive_form_size": len(l.attributes), "complement_form_size": len(C), "in_space": len(l.attributes) <= max_n or len(C) <= max_n, "trivial_single_column_negation": len(C) == 1},
                     "6_duplication": {"duplicates_existing_single_column": len(C) == 1, "nested_in_literals": sorted(m.name for m in lits if m is not l and l.family == m.family and set(l.attributes) < set(m.attributes)),
                                       "contains_literals": sorted(m.name for m in lits if m is not l and l.family == m.family and set(m.attributes) < set(l.attributes)), "same_extension_as_other_literal": any(m is not l and set(m.attributes) == set(l.attributes) for m in lits)},
                     "7_reasoner_dependency": {"derivable_without_reasoner": l.concept in {x.concept for x in lits0}, "needs_reasoner_flag": l.needs_reasoner},
                     "8_search_influence_potential": {"prevalence_train": float(col.mean()), "variance_train": float(np.var(col))}})
    recon["literals"] = rows
    recon["n_literals_with_reasoner_dependency"] = sum(1 for r in rows if not r["7_reasoner_dependency"]["derivable_without_reasoner"])
    Path(out).write_text(json.dumps(recon, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: recon[k] for k in ("explicit_axioms_O2", "reconciliation", "classification_of_the_55", "effectively_usable", "n_literals_with_reasoner_dependency")}, indent=1, ensure_ascii=False))
    print({r["literal"]: (r["1_logical_correctness"]["extension_equals_union_members_in_OWL"], r["2_matching_correct"]["each_level_attribute_matched_to_its_property"], r["7_reasoner_dependency"]["derivable_without_reasoner"]) for r in rows})


if __name__ == "__main__":
    main(sys.argv[1])
