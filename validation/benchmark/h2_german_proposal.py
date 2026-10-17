"""H2 (Doc. 50): PROPOSTA experimental v0_draft de ontologias para o conjunto categórico documentado (German Credit, DEV exploratório; EXPOSTO, nunca confirmatório). Gera:
O1 plana (partição temática igual à congelada), O2 expressiva (classes definidas: ordem cumulativa e taxonomia de finalidade), O3 neutralizada (as mesmas classes mas primitivas),
H permutada (ordens/composições trocadas, OWL válido) e I aleatória (uniões aleatórias de tamanho igual). TODO o conteúdo de O2 é RASCUNHO DE IA, **não** conhecimento de domínio validado:
as fontes (dicionário de dados UCI Statlog) estão citadas de memória e NÃO foram verificadas (sem acesso de rede); revisão de domínio, ontologia e metodologia: PENDENTE. Nada é escolhido por desempenho.
Ficheiros em ``validation/benchmark/ontologies/proposals/german_credit/v0_draft/`` (as ontologias congeladas não se tocam)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from validation.benchmark.b30_axiom_probe import HEADER, FOOTER, dprop, klass

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/benchmark/ontologies/proposals/german_credit/v0_draft"

# (variável → níveis por ordem CRESCENTE documentada; níveis não ordinais excluídos). Rascunho de IA a rever; ver axioms/*.json
ORDINAL: Dict[str, List[str]] = {
    "status_checking_account": ["A11", "A12", "A13"],                    # A14 «sem conta» excluído (não ordinal)
    "savings_account": ["A61", "A62", "A63", "A64"],                     # A65 «desconhecido/sem poupança» excluído
    "employment_since": ["A71", "A72", "A73", "A74", "A75"],
    "job": ["A171", "A172", "A173", "A174"],
}
TAXONOMY: Dict[str, Dict[str, List[str]]] = {"purpose": {"Vehicles": ["A40", "A41"], "HouseholdGoods": ["A42", "A43", "A44"]}}


def _union(name: str, members: List[str], parent: str) -> str:
    mem = "".join(f'<owl:Class rdf:about="#{m}"/>' for m in members)
    return (f'<owl:Class rdf:about="#{name}"><rdfs:subClassOf rdf:resource="#{parent}"/><owl:equivalentClass><owl:Class><owl:unionOf rdf:parseType="Collection">{mem}</owl:unionOf>'
            f'</owl:Class></owl:equivalentClass></owl:Class>\n')


def build(variant: str, features: List[str], groups: Dict[str, List[str]], seed: int = 20260101) -> str:
    rng = np.random.default_rng(seed)
    theme_of = {f: t for t, fs in groups.items() for f in fs}
    body = klass("DomainMeasurement")
    for t in groups:
        body += klass(t, ["DomainMeasurement"])
    expressive = variant != "O1"
    defs = variant in ("O2", "H", "I")                                       # O3 mantém as classes mas sem definição
    level_class: Dict[str, str] = {}
    extra = ""
    if expressive:
        # classes de nível
        for var, levels in list(ORDINAL.items()) + [(v, sum(g.values(), [])) for v, g in TAXONOMY.items()]:
            th = theme_of[f"{var}__{levels[0]}"]
            for lv in levels:
                cn = f"Lvl_{var}_{lv}"
                body += klass(cn, [th])
                level_class[f"{var}__{lv}"] = cn
        # classes cumulativas
        for var, levels in ORDINAL.items():
            th = theme_of[f"{var}__{levels[0]}"]
            order = list(levels)
            if variant == "H":
                order = [order[i] for i in rng.permutation(len(order))]
            for k in range(1, len(levels)):
                cn = f"AtLeast_{var}_{levels[k]}"
                if variant == "I":
                    members = [f"Lvl_{var}_{x}" for x in rng.choice(levels, size=len(levels) - k, replace=False)]
                else:
                    members = [f"Lvl_{var}_{x}" for x in order[k:]]
                body += _union(cn, members, th) if defs else klass(cn, [th])
        # taxonomia
        for var, groups_ in TAXONOMY.items():
            th = theme_of[f"{var}__{sum(groups_.values(), [])[0]}"]
            allv = sorted(sum(groups_.values(), []))
            pool = list(allv)
            if variant in ("H", "I"):
                pool = [allv[i] for i in rng.permutation(len(allv))]
            pos = 0
            for gname, lv in groups_.items():
                chosen = lv if variant in ("O2", "O3") else pool[pos:pos + len(lv)]
                pos += len(lv)
                cn = f"Purpose_{gname}"
                body += _union(cn, [f"Lvl_{var}_{x}" for x in chosen], th) if defs else klass(cn, [th])
    for f in features:
        dom = level_class.get(f, theme_of[f])
        body += dprop("has_" + f, dom, label=f)
    return HEADER.replace("example.org/b30", "example.org/h2/german").replace("b30#", "h2german#") + body + extra + FOOTER


ATTR_NUM = {"status_checking_account": 1, "savings_account": 6, "employment_since": 7, "purpose": 4, "job": 17}
SRC = "UCI Statlog (German Credit Data), dicionário de dados `german.doc` (Prof. Hofmann, Univ. Hamburgo) — CITADO DE MEMÓRIA, NÃO VERIFICADO"


def cards():
    """Fichas de axioma (rascunho de IA, HYPOTHESIS, revisões pendentes). Validadas por ``h2_contract.validate_card``."""
    from validation.benchmark.h2_contract import AxiomCard
    out = []
    for var, lv in ORDINAL.items():
        for k in range(1, len(lv)):
            out.append(AxiomCard(id=f"AX-GC-ORD-{var}-{lv[k]}", statement_nl=f"Em «{var}», os níveis {', '.join(lv[k:])} são todos «pelo menos {lv[k]}» na ordem documentada ({' < '.join(lv)}).",
                                 owl_axiom=f"EquivalentClasses(AtLeast_{var}_{lv[k]} ObjectUnionOf({' '.join('Lvl_%s_%s' % (var, x) for x in lv[k:])}))", entities=[f"{var}__{x}" for x in lv[k:]],
                                 axiom_type="T2_ORDINAL_ORDER", source={"citation": SRC, "locator": f"Atributo {ATTR_NUM[var]}", "verified": False},
                                 scope_and_validity="só os níveis listados; níveis não ordinais (sem conta / desconhecido) excluídos; ordem implícita na lista do dicionário — a confirmar", universality="CONDITIONAL"))
    for var, g in TAXONOMY.items():
        for gname, lv in g.items():
            out.append(AxiomCard(id=f"AX-GC-TAX-{var}-{gname}", statement_nl=f"Em «{var}», {', '.join(lv)} formam a categoria «{gname}» (agrupamento taxonómico proposto).",
                                 owl_axiom=f"EquivalentClasses(Purpose_{gname} ObjectUnionOf({' '.join('Lvl_%s_%s' % (var, x) for x in lv)}))", entities=[f"{var}__{x}" for x in lv], axiom_type="T1_DEFINED_CLASS_UNION",
                                 source={"citation": SRC, "locator": f"Atributo {ATTR_NUM[var]}", "verified": False}, scope_and_validity="agrupamento proposto pelo rascunho; critério de pertença a validar por especialista", universality="CONDITIONAL"))
    return out


def main(evidence_out: str):
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.b31_facts import compile_registry
    from validation.benchmark.h2_axiom_mechanisms import derived_union_literals
    DC.register()
    spec = D.REGISTRY["german_credit"]
    ds, _, _ = spec.load()
    names = [str(n) for n in ds.feature_names]
    X = np.asarray(ds.X, dtype=float)
    groups = spec.groups(names)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for v in ("O1", "O2", "O3", "H", "I"):
        p = OUT / f"german_credit_{v}.owl"
        p.write_text(build(v, names, groups), encoding="utf-8")
        reg = compile_registry(str(p), names, run_reasoner=True)
        lits = derived_union_literals(reg, X, names)
        s = reg.summary()
        rows[v] = {"file": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "ontology_status": reg.ontology_status, "n_facts": s["n_facts"], "by_kind": s["by_kind"],
                   "n_inferred_by_reasoner": sum(1 for f in reg.facts if f.derivation == "INFERRED_BY_REASONER"), "unsupported_axioms": s["unsupported_axioms"],
                   "mapped": sum(1 for n in names if reg.feature_entity.get(n)), "n_features": len(names),
                   "derived_literals": [{"concept": l.concept, "size": len(l.attributes), "attributes": [a.split("__")[1] for a in l.attributes], "needs_reasoner": l.needs_reasoner} for l in lits]}
    Path(evidence_out).write_text(json.dumps({"purpose": "correção semântica das propostas (sem benchmark, sem rótulos de teste)", "variants": rows}, indent=1))
    for v, r in rows.items():
        print(v, r["ontology_status"], "inferidos", r["n_inferred_by_reasoner"], "mapeados", r["mapped"], "/", r["n_features"], "literais", len(r["derived_literals"]),
              "(reasoner:", sum(1 for l in r["derived_literals"] if l["needs_reasoner"]), ")")


if __name__ == "__main__":
    main(sys.argv[1])
