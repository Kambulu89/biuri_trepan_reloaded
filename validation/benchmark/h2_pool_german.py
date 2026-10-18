"""H2 (Doc. 59/61): condutor da análise técnica do *pool* sobre o German Credit (rascunho de IA não validado). ANÁLISE ESTRUTURAL: sem accuracy, sem fidelidade a rótulos reais, sem teste, sem selecionar
nada pelo desempenho; as ontologias O2/H/I são as do ``v0_draft`` (hashes registados). A maquinaria (``h2_pool_study``) é agnóstica; este ficheiro só liga um dataset de validação.
Estratégias: A (pool do Original), B (pool alargado genericamente: ``+r`` vagas por variância), C (``+r`` vagas reservadas às derivadas de relações ontológicas), controlos aleatório (colunas base aleatórias)
e permutado (derivadas da ontologia H), aleatório-OWL (derivadas da ontologia I), A+ (representação enriquecida sem vagas extra) e B_MATCHED (B com ``mf`` escolhido para igualar os candidatos m-of-n de C)."""
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
TOL = {"feature_slots": 0.0, "mofn_generated": 0.10, "mofn_evaluated": 0.10, "queries": 0.10, "n_internal_nodes": 0.10, "depth": 0.0}      # PROPOSTA, não congelada


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_seed(seed: int) -> Dict[str, Any]:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from core.benchmark.splits import make_splits
    from validation.benchmark import datasets as D, datasets_categorical as DC
    from validation.benchmark.b31_facts import compile_registry
    from validation.benchmark.h2_axiom_mechanisms import defined_union_concepts, derived_union_literals
    from validation.benchmark.h2_harness import instance_hash
    from validation.benchmark.h2_pool_study import ablate_literal, comparability, fit_pool_arm, ladder_state, original_buildable
    DC.register()
    ds, _, _ = D.REGISTRY["german_credit"].load()
    names = [str(n) for n in ds.feature_names]
    X, y = np.asarray(ds.X, dtype=float), np.asarray(ds.y)
    sp = make_splits(y, scheme="holdout", seeds=[seed])[0]
    Xtr, ytr = X[sp.train_idx], y[sp.train_idx]                                    # só treino; o teste nunca é lido
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)).fit(Xtr, ytr)

    class Orc:
        calls = 0
        def predict(self, Z):
            Orc.calls += 1
            return mlp.predict(np.asarray(Z)[:, : len(names)])
    make_oracle = Orc

    lits = {}
    for key in ("O2", "H", "I"):
        owl = V0 / f"german_credit_{key}.owl"
        reg = compile_registry(str(owl), names, run_reasoner=True)
        lits[key] = derived_union_literals(reg, Xtr, names, only_concepts=defined_union_concepts(owl))
    r = len(lits["O2"])
    mf0 = PARAMS["max_features_per_node"]
    var_order = [int(j) for j in np.argsort(-np.nanvar(Xtr, axis=0), kind="stable")]
    rng = np.random.RandomState(seed + 1000)
    rest = [j for j in var_order if j not in set(var_order[:mf0])]
    rand_base = sorted(int(j) for j in rng.choice(rest, size=r, replace=False))
    ih = instance_hash(Xtr, names)
    arms: Dict[str, Dict[str, Any]] = {}

    def go(key, derived, extras, mf, **kw):
        res = fit_pool_arm(Xtr, names, derived, make_oracle(), PARAMS, extras=extras, mf=mf, **kw)
        arms[key] = res
        return res

    go("A", [], "none", mf0)
    go("A_PLUS_ENRICHED", lits["O2"], "none", mf0)
    go("B_WIDE_GENERIC", [], "none", mf0 + r)
    go("C_ONTOLOGY_PRIORITY", lits["O2"], "derived", mf0 + r)
    go("R_RANDOM_BASE", [], "base", mf0 + r, forced_base=rand_base)
    go("P_PERMUTED_ONTOLOGY", lits["H"], "derived", mf0 + len(lits["H"]))
    go("I_RANDOM_RELATIONS", lits["I"], "derived", mf0 + len(lits["I"]))
    target = arms["C_ONTOLOGY_PRIORITY"]["trace"]["pool"]["counters"]["mofn_generated"]
    best = None
    for mf in range(mf0, len(names) + 1):                                         # escolha por CONTAGENS de candidatos (sem desempenho)
        res = fit_pool_arm(Xtr, names, [], make_oracle(), PARAMS, extras="none", mf=mf)
        gap = abs(res["trace"]["pool"]["counters"]["mofn_generated"] - target)
        if best is None or gap < best[0]:
            best = (gap, mf, res)
        if res["trace"]["pool"]["counters"]["mofn_generated"] >= target:
            break
    arms["B_MATCHED_CANDIDATES"] = best[2]
    out: Dict[str, Any] = {"seed": seed, "instance_hash": ih, "train_rows": int(len(Xtr)), "r_extra_slots": r, "random_base_extras": [names[j] for j in rand_base], "b_matched_mf": best[1], "arms": {}}
    for key, a in arms.items():
        c = a["trace"]["pool"]["counters"]
        out["arms"][key] = {"feature_slots": a["feature_slots"], "n_columns": a["n_columns"], "n_derived": a["trace"]["n_derived"], "counters": c, "queries": a["queries"], "oracle_calls": a["oracle_calls"],
                            "n_internal_nodes": a["n_internal_nodes"], "depth": a["depth"], "n_searches": a["trace"]["pool"]["n_searches"],
                            "rules_sha": hashlib.sha256(json.dumps(a["model"].export_rules(), default=str).encode()).hexdigest()[:16]}
    out["original_equivalent_buildable"] = {k: original_buildable(arms[k]["model"], lits["O2"], names) for k in ("A", "B_WIDE_GENERIC", "B_MATCHED_CANDIDATES", "C_ONTOLOGY_PRIORITY")}
    out["tree_identical_to_A"] = {k: v["rules_sha"] == out["arms"]["A"]["rules_sha"] for k, v in out["arms"].items()}
    out["comparability"] = {}
    for a_key, b_key in [("C_ONTOLOGY_PRIORITY", "B_WIDE_GENERIC"), ("C_ONTOLOGY_PRIORITY", "B_MATCHED_CANDIDATES"), ("C_ONTOLOGY_PRIORITY", "R_RANDOM_BASE"), ("C_ONTOLOGY_PRIORITY", "P_PERMUTED_ONTOLOGY"),
                          ("C_ONTOLOGY_PRIORITY", "I_RANDOM_RELATIONS"), ("C_ONTOLOGY_PRIORITY", "A")]:
        tol = dict(TOL)
        if b_key == "B_MATCHED_CANDIDATES" or b_key == "A":
            tol.pop("feature_slots")                      # B_MATCHED/A igualam candidatos / não igualam vagas por construção: reporta-se, não se exige
        out["comparability"][f"{a_key} vs {b_key}"] = comparability(arms[a_key], arms[b_key], same_instances=True, same_oracle=True, same_params_except_pool=True, tol=tol)
    # rastreio por literal (braço C e A+), com ablação contrafactual por máscara
    out["literals"] = {}
    for key, extras, mf in (("C_ONTOLOGY_PRIORITY", "derived", mf0 + r), ("A_PLUS_ENRICHED", "none", mf0)):
        a, rows = arms[key], []
        for k, l in enumerate(lits["O2"]):
            ab = ablate_literal(Xtr, names, lits["O2"], make_oracle, PARAMS, k, extras=extras, mf=mf, full=a)
            plit = a["trace"]["pool"]["literals"][k]
            ltr = a["trace"]["literals"][k]
            rows.append({"literal": l.name, "concept": l.concept, "attributes": list(l.attributes), "fact_ids": list(l.fact_ids), "needs_reasoner": l.needs_reasoner,
                         "root_variance_rank": plit["root_var_rank"], "root_reason": plit["root_reason"], "reasons_by_search": plit["reasons_by_search"], "original_equivalent": plit["original_equivalent"],
                         "candidate_evaluations": ltr["candidate_evaluations"], "chosen_in_nodes": ltr["chosen_in_nodes"], "ablation": ab,
                         "ladder": ladder_state(ltr, plit, validated=True, changed_by_ablation=ab["changed"])})
        out["literals"][key] = rows
    return out


def main(out: str, seeds: List[int]):
    warnings.filterwarnings("ignore")
    res = {"purpose": "ANÁLISE ESTRUTURAL do pool; sem accuracy/fidelidade a rótulos reais; rascunho de IA não validado; não é avaliação",
           "ontology_hashes_lf": {k: _sha(V0 / f"german_credit_{k}.owl") for k in ("O2", "H", "I")}, "params": PARAMS, "tolerances_proposal_not_frozen": TOL, "seeds": seeds, "runs": []}
    for s in seeds:
        res["runs"].append(run_seed(s))
        print("seed", s, {k: (v["counters"]["mofn_generated"], v["queries"], v["n_internal_nodes"]) for k, v in res["runs"][-1]["arms"].items()}, flush=True)
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [42])
