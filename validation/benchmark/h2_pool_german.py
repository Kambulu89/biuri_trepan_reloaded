"""H2 (Docs. 59/61/62): condutor da análise técnica do *pool* sobre o German Credit (rascunho de IA NÃO validado). ANÁLISE ESTRUTURAL: sem accuracy, sem fidelidade a rótulos reais, sem teste, sem
selecionar nada pelo desempenho; as ontologias O2/H/I são as do ``v0_draft`` (hashes registados). A maquinaria (``h2_pool_study``) é agnóstica; este ficheiro só liga um dataset de validação e NÃO é
uma avaliação causal do benefício ontológico (axiomas por validar, fontes não consultadas).

Braços (mesmas instâncias, mesmo MLP/oráculo, mesmos parâmetros exceto o pool): A pool limitado do Original; B pool alargado genericamente (+r vagas por variância); C pool com priorização por relações
ontológicas verdadeiras (+r vagas reservadas às derivadas); D priorização permutada (ontologia H); E priorização aleatória (E_BASE: colunas base aleatórias; E_RELATIONS: relações aleatórias, ontologia I);
F pool genérico cujos candidatos m-of-n AVALIADOS igualam os de C (quando viável); A+ representação enriquecida sem vagas extra (isola a representação)."""
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "validation/benchmark/ontologies/proposals/german_credit/v0_draft"
PARAMS = dict(min_sample=300, max_queries=3000, max_nodes=7, max_depth=3, max_n=3, beam_width=2, min_samples_leaf=5, max_features_per_node=12, random_state=42)
TOL = {"feature_slots": 0.0, "mofn_generated": 0.10, "mofn_evaluated": 0.10, "queries": 0.10, "n_internal_nodes": 0.10, "depth": 0.0, "beam_width": 0.0}      # PROPOSTA, não congelada
# Tetos AUTORIZADOS (proposta declarada antes de qualquer desempenho; não congelados). Os recursos consumidos reportam-se à parte.
CEILINGS = dict(eligible_literals=40, evaluated_candidates=1200, oracle_queries=3000, rows_consulted=None, nodes=7, cpu_seconds=None)
BEAMS = (1, 2, 4)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _summ(a: Dict[str, Any]) -> Dict[str, Any]:
    c = a["trace"]["pool"]["counters"]
    return {"feature_slots": a["feature_slots"], "beam_width": a["beam_width"], "n_columns": a["n_columns"], "n_derived": a["trace"]["n_derived"], "counters": c, "queries": a["queries"],
            "oracle_calls": a["oracle_calls"], "rows_consulted": a["rows_consulted"], "cpu_seconds": round(a["cpu_seconds"], 3), "peak_memory_mb": round(a["peak_memory_mb"], 2), "complexity": a["complexity"], "n_internal_nodes": a["n_internal_nodes"], "depth": a["depth"],
            "n_searches": a["trace"]["pool"]["n_searches"], "stop_reasons": a["stop_reasons"],
            "rules_sha": hashlib.sha256(json.dumps(a["model"].export_rules(), default=str).encode()).hexdigest()[:16]}


def run_seed(seed: int) -> Dict[str, Any]:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from core.benchmark.splits import make_splits
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.b31_facts import compile_registry
    from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals
    from validation.benchmark.h2_harness import instance_hash
    from validation.benchmark.h2_pool_study import (Ceilings, ablate_literal, cards_by_concept, comparability, fit_pool_arm, ladder_state, match_generic_by_evaluated,
                                                    original_buildable, resource_report)
    DC.register()
    ds, _, _ = D.REGISTRY["german_credit"].load()
    names = [str(n) for n in ds.feature_names]
    X, y = np.asarray(ds.X, dtype=float), np.asarray(ds.y)
    sp = make_splits(y, scheme="holdout", seeds=[seed])[0]
    Xtr, ytr = X[sp.train_idx], y[sp.train_idx]                                    # só treino; o teste nunca é lido
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)).fit(Xtr, ytr)

    class Orc:
        def __init__(self):
            self.calls, self.rows = 0, 0
        def predict(self, Z):
            self.calls += 1
            self.rows += len(Z)
            return mlp.predict(np.asarray(Z)[:, : len(names)])
    make_oracle = Orc
    ceil = Ceilings(**CEILINGS)
    cards = cards_by_concept(V0 / "axioms")

    lits, validated = {}, {}
    for key in ("O2", "H", "I"):
        owl = V0 / f"german_credit_{key}.owl"
        reg = compile_registry(str(owl), names, run_reasoner=True)
        validated[key] = reg.ontology_status == "VALID"
        lits[key] = derived_union_literals(reg, Xtr, names, only_concepts=defined_union_concepts(owl))
    r = len(lits["O2"])
    mf0 = PARAMS["max_features_per_node"]
    var_order = [int(j) for j in np.argsort(-np.nanvar(Xtr, axis=0), kind="stable")]
    rng = np.random.RandomState(seed + 1000)
    rest = [j for j in var_order if j not in set(var_order[:mf0])]
    rand_base = sorted(int(j) for j in rng.choice(rest, size=r, replace=False))
    arms: Dict[str, Dict[str, Any]] = {}

    def go(key, derived, extras, mf, params=PARAMS, **kw):
        arms[key] = fit_pool_arm(Xtr, names, derived, make_oracle(), params, extras=extras, mf=mf, **kw)
        return arms[key]

    go("A_LIMITED", [], "none", mf0)
    go("A_PLUS_ENRICHED", lits["O2"], "none", mf0)
    go("B_WIDE_GENERIC", [], "none", mf0 + r)
    go("C_OWL_PRIORITY", lits["O2"], "derived", mf0 + r)
    go("D_PERMUTED_PRIORITY", lits["H"], "derived", mf0 + len(lits["H"]))
    go("E_RANDOM_BASE", [], "base", mf0 + r, forced_base=rand_base)
    go("E_RANDOM_RELATIONS", lits["I"], "derived", mf0 + len(lits["I"]))
    target = arms["C_OWL_PRIORITY"]["trace"]["pool"]["counters"]["mofn_evaluated"]
    m = match_generic_by_evaluated(lambda mf: fit_pool_arm(Xtr, names, [], make_oracle(), PARAMS, extras="none", mf=mf), target, range(mf0, len(names) + 1), tolerance=TOL["mofn_evaluated"])
    arms["F_GENERIC_MATCHED_EVALUATED"] = m["arm"]
    out: Dict[str, Any] = {"seed": seed, "instance_hash": instance_hash(Xtr, names), "train_rows": int(len(Xtr)), "r_extra_slots": r, "random_base_extras": [names[j] for j in rand_base],
                           "F_matching": {"mf": m["mf"], "relative_gap_evaluated": round(m["relative_gap"], 4), "feasible": m["feasible"], "target_evaluated": target},
                           "ontology_status_valid": validated, "arms": {k: _summ(a) for k, a in arms.items()}}
    out["tree_identical_to_A"] = {k: v["rules_sha"] == out["arms"]["A_LIMITED"]["rules_sha"] for k, v in out["arms"].items()}
    out["ceilings"] = {"authorized_proposal": CEILINGS, "arms_exceeding": {k: [] for k in arms}}
    from validation.benchmark.h2_pool_study import check_ceilings
    for k, a in arms.items():
        out["ceilings"]["arms_exceeding"][k] = check_ceilings(a, ceil)["exceeded"]
    extra = {"F_GENERIC_MATCHED_EVALUATED": len(range(mf0, len(names) + 1)) - 1, "C_OWL_PRIORITY": r, "A_PLUS_ENRICHED": r}          # ajustes auxiliares (varrimento de F; ablações)
    for k, a in arms.items():
        out["arms"][k]["resources"] = resource_report(a, ceil, extra_fits=extra.get(k, 0))
    out["comparability"] = {}
    # O desenho de cada contraste declara o que IGUALA: vagas (B, D, E) ou candidatos avaliados (F). A_LIMITED serve de referência: tem MENOS vagas, logo C vs A não é atribuível à ontologia.
    for b_key, eq in (("B_WIDE_GENERIC", ("feature_slots",)), ("F_GENERIC_MATCHED_EVALUATED", ("mofn_evaluated",)), ("D_PERMUTED_PRIORITY", ("feature_slots",)), ("E_RANDOM_BASE", ("feature_slots",)),
                      ("E_RANDOM_RELATIONS", ("feature_slots",)), ("A_LIMITED", ("feature_slots",))):
        out["comparability"][f"C_OWL_PRIORITY vs {b_key}"] = comparability(arms["C_OWL_PRIORITY"], arms[b_key], same_instances=True, same_oracle=True, same_params_except_pool=True,
                                                                          tol=dict(TOL), ceilings=ceil, design_equalized=eq)
    out["original_equivalent_buildable"] = {k: original_buildable(arms[k]["model"], lits["O2"], names) for k in ("A_LIMITED", "B_WIDE_GENERIC", "F_GENERIC_MATCHED_EVALUATED", "C_OWL_PRIORITY")}
    # sensibilidade ao tamanho do feixe (paragem antecipada, candidatos inválidos/duplicados por feixe)
    out["beam_sensitivity"] = {}
    for bw in BEAMS:
        pb = dict(PARAMS, beam_width=bw)
        row = {}
        for key, derived, extras, mf in (("A_LIMITED", [], "none", mf0), ("B_WIDE_GENERIC", [], "none", mf0 + r), ("C_OWL_PRIORITY", lits["O2"], "derived", mf0 + r)):
            a = fit_pool_arm(Xtr, names, derived, make_oracle(), pb, extras=extras, mf=mf)
            c = a["trace"]["pool"]["counters"]
            row[key] = {"mofn_generated": c["mofn_generated"], "mofn_evaluated": c["mofn_evaluated"], "invalid_mofn": c["invalid_mofn"], "duplicates_removed": c["duplicates_removed"], "queries": a["queries"],
                        "n_internal_nodes": a["n_internal_nodes"], "stop_reasons": a["stop_reasons"], "literals_selected": sum(1 for l in a["trace"]["literals"] if l["chosen_in_nodes"])}
        out["beam_sensitivity"][str(bw)] = row
    # rastreio por literal (C e A+), com ablação contrafactual por máscara; operacional ≠ preditivo
    out["literals"] = {}
    for key, extras, mf in (("C_OWL_PRIORITY", "derived", mf0 + r), ("A_PLUS_ENRICHED", "none", mf0)):
        a, rows = arms[key], []
        for k, l in enumerate(lits["O2"]):
            ab = ablate_literal(Xtr, names, lits["O2"], make_oracle, PARAMS, k, extras=extras, mf=mf, full=a)
            plit, ltr = a["trace"]["pool"]["literals"][k], a["trace"]["literals"][k]
            card = cards.get(l.concept, {})
            rows.append({"1_axiom_source": card.get("card_id"), "2_validation_status": {"technical_valid": validated["O2"], "domain_validation": card.get("domain_validation", "UNKNOWN"), "authored_by": card.get("authored_by"),
                                                                                       "epistemic_status": card.get("epistemic_status"), "source_verified": card.get("source_verified"), "reviews": card.get("reviews")},
                         "3_reasoner_inference": {"fact_ids": list(l.fact_ids), "needs_reasoner": l.needs_reasoner}, "4_features": list(l.attributes), "literal": l.name, "concept": l.concept,
                         "5_pool_entry": {"root_variance_rank_0based": plit["root_var_rank"], "root_reason": plit["root_reason"], "reasons_by_search": plit["reasons_by_search"]},
                         "6_candidate_generated_searches": plit["reasons_by_search"].get("IN_LITERAL_POOL", 0) + plit["reasons_by_search"].get("BELOW_LITERAL_POOL_CUTOFF", 0),
                         "7_statistical_evaluations": ltr["candidate_evaluations"], "8_split_decision_nodes": ltr["chosen_in_nodes"], "9_ablation": ab, "original_equivalent": plit["original_equivalent"],
                         "10_operational_vs_predictive": {"operational_influence": bool(ab["changed"]), "predictive_benefit": "NOT_ASSESSED"},
                         "ladder": ladder_state(ltr, plit, valid=bool(validated["O2"] and l.fact_ids), selected=bool(ltr["chosen_in_nodes"]), changed_by_ablation=ab["changed"])})
        out["literals"][key] = rows
    return out


def main(out: str, seeds: List[int]):
    warnings.filterwarnings("ignore")
    res = {"purpose": "ANÁLISE ESTRUTURAL do pool; sem accuracy/fidelidade a rótulos reais; axiomas = rascunho de IA NÃO validado; NÃO é avaliação causal do benefício ontológico",
           "ontology_hashes_lf_canonical": {k: _sha(V0 / f"german_credit_{k}.owl") for k in ("O2", "H", "I")}, "params": PARAMS, "tolerances_proposal_not_frozen": TOL, "ceilings_proposal_not_frozen": CEILINGS,
           "seeds": seeds, "runs": []}
    for s in seeds:
        res["runs"].append(run_seed(s))
        print("seed", s, {k: (v["counters"]["mofn_evaluated"], v["queries"], v["n_internal_nodes"]) for k, v in res["runs"][-1]["arms"].items()}, flush=True)
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [42])
