"""H2 (Doc. 54): auditoria das inferências do reasoner e dos literais derivados da proposta ``german_credit/v0_draft`` (O2). Só leitura: ficheiro OWL da proposta, registo de factos B3.1,
dicionário das fichas de axioma, X de treino (SÓ prevalência dos literais; nenhum rótulo, nenhum teste). Para cada literal: origem, entidades, fonte, estatuto epistémico, resultado do reasoner,
atributos ARFF, significado tabular, condição de aplicabilidade, candidato m-of-n correspondente, mecanismo, se o Original o pode gerar, controlo que o neutraliza."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from validation.benchmark.h2_axiom_mechanisms import derived_union_literals, family_of
from validation.benchmark.h2_german_proposal import OUT, build, cards, ORDINAL, TAXONOMY

ROOT = Path(__file__).resolve().parents[2]

# Auditoria por variável (rascunho de IA; NADA verificado documentalmente nesta sessão). «evaluable_now» é sempre False enquanto a fonte não for verificada por pessoas.
VARIABLE_AUDIT = {
    "status_checking_account": {"type_claim": "ORDINAL parcial (A11<A12<A13 por montante); A14 «sem conta» é NOMINAL e fica de fora", "depends_on_hypotheses": ["A13 agrega «≥ 200 DM» e «cessão de ordenado ≥ 1 ano»: heterogéneo, ordem total assumida", "A14 não ordenável"], "decision": "HOLD_PENDING_SOURCE"},
    "savings_account": {"type_claim": "ORDINAL (intervalos de montante A61<A62<A63<A64); A65 «desconhecido/sem poupança» NOMINAL e fora", "depends_on_hypotheses": ["intervalos tal como no dicionário (não verificados)", "A65 fora da ordem"], "decision": "HOLD_PENDING_SOURCE"},
    "employment_since": {"type_claim": "ORDINAL (duração: A71 desempregado < A72 <1 a < A73 1–4 < A74 4–7 < A75 ≥7)", "depends_on_hypotheses": ["A71 «desempregado» é um estado, não uma duração: colocá-lo abaixo de «<1 ano» é convenção a confirmar"], "decision": "HOLD_PENDING_SOURCE"},
    "job": {"type_claim": "ORDEM FRACA (qualificação) com dimensões misturadas (residência, qualificação, trabalho por conta própria)", "depends_on_hypotheses": ["a ordem A171<A172<A173<A174 é inferida da ordem da lista, não de uma definição", "A171 mistura «desempregado» e «não qualificado não residente»"], "decision": "HOLD_WEAK_SEMANTICS (tratar como NOMINAL salvo fonte)"},
    "purpose": {"type_claim": "NOMINAL; agrupamentos «Veículos» e «Bens domésticos» são proposta do rascunho", "depends_on_hypotheses": ["nenhuma base documental conhecida para os agrupamentos; critério de pertença por definir; categorias raras (A44: 10, A410: 6, A48: 5 em treino)"], "decision": "EXCLUDE_NO_DOCUMENTARY_BASIS"},
}


def main(out: str, max_n: int = 3):
    from core.benchmark.splits import make_splits
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.b31_facts import Kind, compile_registry
    DC.register()
    spec = D.REGISTRY["german_credit"]
    ds, _, _ = spec.load()
    names = [str(n) for n in ds.feature_names]
    X = np.asarray(ds.X, dtype=float)
    y = np.asarray(ds.y)
    sp = make_splits(y, scheme="holdout", seeds=[42])[0]                       # só os ÍNDICES de treino; y nunca é usado abaixo
    Xtr = X[sp.train_idx]
    owl = OUT / "german_credit_O2.owl"
    reg = compile_registry(str(owl), names, run_reasoner=True)
    lits = derived_union_literals(reg, Xtr, names)
    by_card = {c.id: c for c in cards()}
    fam_cols: Dict[str, List[str]] = {}
    for n in names:
        fam_cols.setdefault(family_of(n), []).append(n)

    # ---- inferências (55): classificação por tipo
    inferred = [f for f in reg.facts if f.derivation == "INFERRED_BY_REASONER"]
    kinds = Counter(f.kind for f in inferred)
    sub = [f for f in inferred if f.kind == Kind.CONCEPT_SUBSUMPTION.value]
    att = [f for f in inferred if f.kind == Kind.ATTRIBUTE_APPLIES_TO_CONCEPT.value]
    typ = Counter()
    for f in sub:
        a, b = f.entities
        if a.startswith("Lvl_") and (b.startswith("AtLeast_") or b.startswith("Purpose_")):
            typ["level_subsumed_by_defined_class"] += 1
        elif a.startswith("AtLeast_") and b.startswith("AtLeast_"):
            typ["defined_class_subsumed_by_defined_class (cadeia ordinal)"] += 1
        else:
            typ["other"] += 1
    typ["attribute_applies_to_defined_class (consequência das subsunções)"] = len(att)

    # ---- literais
    rows = []
    for l in lits:
        var = l.family
        cols = fam_cols[var]
        S = set(l.attributes)
        C = [c for c in cols if c not in S]
        pos = len(S)
        neg = len(C)
        in_space = pos <= max_n or neg <= max_n
        trivial_neg = neg == 1
        form = ("TRIVIAL: negação de uma única coluna (%s ≤ 0,5) — já é um literal simples do Original" % C[0].split("__")[1]) if trivial_neg else f"1-de-{pos} (OR de literais positivos)" if pos <= max_n else (f"{neg}-de-{neg} sobre literais negados (AND de «coluna = 0» para {', '.join(c.split('__')[1] for c in C)})" if neg <= max_n else "fora do espaço com max_n=%d" % max_n)
        col = np.max(Xtr[:, [names.index(a) for a in l.attributes]], axis=1)
        prev = float(col.mean())
        card_ids = [cid for cid, c in by_card.items() if set(c.entities) == set(l.attributes)]
        card = by_card[card_ids[0]] if card_ids else None
        support = [f for f in reg.facts if f.derivation == "INFERRED_BY_REASONER" and f.entities[-1:] == (l.concept,)]
        rows.append({
            "literal": l.name, "concept": l.concept, "family": var,
            "1_origin_axiom": {"card": card.id if card else None, "owl": card.owl_axiom if card else None},
            "2_entities": {"defined_class": l.concept, "level_classes": [a for a in l.attributes]},
            "3_source": {"citation": card.source["citation"] if card else None, "locator": card.source["locator"] if card else None, "verified": bool(card.source["verified"]) if card else False},
            "4_epistemic_status": card.epistemic_status if card else None, "4b_authored_by": card.authored_by if card else None, "4c_reviews": {k: v["status"] for k, v in card.reviews.items()} if card else None,
            "5_reasoner_result": {"needs_reasoner": l.needs_reasoner, "n_supporting_inferred_facts": len(support), "fact_ids": list(l.fact_ids)[:3]},
            "6_arff_attributes": list(l.attributes), "6b_variable_columns": cols,
            "7_tabular_meaning": f"a linha pertence a um dos níveis {', '.join(a.split('__')[1] for a in l.attributes)} da variável «{var}» (equivalente a: nível ∉ {{{', '.join(c.split('__')[1] for c in C)}}})",
            "8_applicability_condition": "colunas one-hot mutuamente exclusivas da mesma variável; categoria desconhecida (todas a 0) ⇒ literal 0; níveis não ordinais fora da extensão",
            "9_mofn_candidate": form, "9c_trivial_negation_of_single_column": trivial_neg, "9b_prevalence_train": prev,
            "10_mechanism": "M-R (coluna binária derivada acrescentada) + M-P (poda de candidatos redundantes por subsunção entre literais derivados e níveis)",
            "11_original_can_generate": {"in_hypothesis_space_with_max_n_%d" % max_n: bool(in_space), "note": "pertence ao espaço de hipóteses, mas o Original só o encontra se a pesquisa (pool de literais por ganho individual + beam) lá chegar" if in_space else "exige n > max_n"},
            "12_neutralizing_control": "O3 (classe primitiva, mesma estrutura sem equivalência) e ablação do axioma; H (ordem permutada) e I (união aleatória de tamanho igual)",
        })
    # ---- redundância entre literais
    pairs = [(a["literal"], b["literal"]) for a in rows for b in rows if a is not b and a["family"] == b["family"] and set(a["6_arff_attributes"]) < set(b["6_arff_attributes"])]
    card_rows = []
    for cid, c in by_card.items():
        var = c.entities[0].split("__")[0]
        concept = cid.replace("AX-GC-ORD-", "AtLeast_").replace("AX-GC-TAX-", "Purpose_")
        concept = concept.replace("AtLeast_" + var + "-", "AtLeast_" + var + "_") if cid.startswith("AX-GC-ORD") else ("Purpose_" + cid.split("-")[-1])
        va = VARIABLE_AUDIT[var]
        n_inf = sum(1 for f in reg.facts if f.derivation == "INFERRED_BY_REASONER" and f.entities and f.entities[-1] == concept)
        card_rows.append({"card": cid, "variable": var, "statement": c.statement_nl, "entities": c.entities, "type_claim": va["type_claim"], "depends_on_hypotheses": va["depends_on_hypotheses"],
                          "documentation_status": "UNVERIFIED_NO_LOCAL_DOCUMENT", "epistemic_status": c.epistemic_status, "authored_by": c.authored_by, "reviews": {k: v["status"] for k, v in c.reviews.items()},
                          "decision": ("TRIVIAL_SINGLETON_NO_CONTENT (união de um único nível ≡ o próprio nível; sem literal)" if len(c.entities) == 1 else va["decision"]), "evaluable_now": False, "n_inferred_facts_with_this_concept_as_target": n_inf})
    audit = {"cards": card_rows, "variable_audit": VARIABLE_AUDIT, "ontology_file": str(owl.relative_to(ROOT)), "reasoner": {"n_facts": len(reg.facts), "n_inferred": len(inferred), "by_kind": dict(kinds), "inferred_by_type": dict(typ), "ontology_status": reg.ontology_status,
                                                                   "unsupported_axioms": reg.summary()["unsupported_axioms"]},
             "n_literals": len(rows), "literals": rows, "nested_pairs_same_variable": pairs,
             "trivial_or_redundant": {"consequent_attribute_facts": len(att), "derivable_from": "subsunções inferidas + domínio asserido", "trivial_negation_literals": [r["literal"] for r in rows if r["9c_trivial_negation_of_single_column"]], "literals_with_prevalence_lt_0.05_or_gt_0.95": [r["literal"] for r in rows if r["9b_prevalence_train"] < 0.05 or r["9b_prevalence_train"] > 0.95]},
             "train_rows": int(len(Xtr)), "labels_used": False, "test_rows_used": False}
    Path(out).write_text(json.dumps(audit, indent=1, default=str))
    print(json.dumps({"n_inferred": len(inferred), "by_kind": dict(kinds), "by_type": dict(typ), "in_space": sum(r["11_original_can_generate"]["in_hypothesis_space_with_max_n_%d" % max_n] for r in rows),
                      "nested_pairs": len(pairs), "extreme_prev": audit["trivial_or_redundant"]["literals_with_prevalence_lt_0.05_or_gt_0.95"]}, indent=1))
    for r in rows:
        print(r["literal"], r["9_mofn_candidate"][:60], round(r["9b_prevalence_train"], 3), r["11_original_can_generate"])


if __name__ == "__main__":
    main(sys.argv[1])
