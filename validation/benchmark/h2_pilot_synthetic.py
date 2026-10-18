"""H2 (Doc. 62 §6): PILOTO TÉCNICO SINTÉTICO de pequeno custo, só ``validation/``. NÃO é uma campanha de desempenho e NÃO autoriza conclusões de superioridade: usa relações sintéticas de validade
CONHECIDA e CONTROLADA (gerador genérico + ontologias TRUE/PERM/NEUTRAL/FLAT/ambígua) em datasets sintéticos desconhecidos (dimensionalidade, famílias, níveis e nº de classes diferentes) e verifica o
funcionamento do mecanismo genérico:
 (1) compilação de axiomas válidos (extensão exata) e rejeição de extensões incompletas por matching ambíguo; (2) entrada dos literais no pool; (3) avaliação dos candidatos; (4) aplicação dos tetos;
 (5) comparabilidade de configuração e correspondência realizada de recursos; (6) rastreabilidade (estados separados); (7) equivalência sem semântica; (8) funcionamento em datasets desconhecidos.
Limites de recursos DECLARADOS ANTES da execução (``LIMITS``); se qualquer um for ultrapassado o piloto interrompe-se e regista ``ABORTED_LIMIT``. Nenhuma accuracy nem fidelidade é calculada."""
import hashlib
import json
import sys
import tempfile
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

LIMITS = {"max_wall_seconds": 300, "max_fits": 300, "max_peak_memory_mb": 1500}            # declarados antes de executar
CEILINGS = {"eligible_literals": 40, "evaluated_candidates": 1200, "oracle_queries": 800, "nodes": 5}   # tetos autorizados do piloto (proposta; não congelados)
CONFIGS = [
    {"id": "U1", "families": {"a": 5}, "k_true": 2, "n_noise": 4, "n_num": 2, "n_classes": 2, "n": 240, "seed": 101},
    {"id": "U2", "families": {"q": 3, "r": 6}, "k_true": 1, "n_noise": 2, "n_num": 3, "n_classes": 3, "n": 240, "seed": 102},
    {"id": "U3", "families": {"x": 7, "y": 2, "z": 4}, "k_true": 4, "n_noise": 8, "n_num": 3, "n_classes": 2, "n": 300, "seed": 103},
]
BASE_PARAMS = dict(min_sample=150, max_queries=800, max_nodes=5, max_depth=3, beam_width=2, min_samples_leaf=3, max_features_per_node=3, random_state=7)
TOL = {"feature_slots": 0.0, "mofn_evaluated": 0.10, "queries": 0.10, "n_internal_nodes": 0.10, "depth": 0.0, "beam_width": 0.0}      # PROPOSTA


class LimitExceeded(Exception):
    pass


class Budget:
    def __init__(self, limits):
        self.limits, self.t0, self.fits = limits, time.time(), 0
        self.peak = 0.0

    def tick(self):
        self.fits += 1
        if tracemalloc.is_tracing():
            self.peak = max(self.peak, tracemalloc.get_traced_memory()[1] / 1e6)
        if time.time() - self.t0 > self.limits["max_wall_seconds"]:
            raise LimitExceeded("max_wall_seconds")
        if self.fits > self.limits["max_fits"]:
            raise LimitExceeded("max_fits")


def _rules_sha(m) -> str:
    return hashlib.sha256(json.dumps(m.export_rules(), default=str).encode()).hexdigest()[:16]


def run_config(cfg: Dict[str, Any], budget: Budget) -> Dict[str, Any]:
    from core.trepan_original import TrepanOriginalClassifier
    from validation.benchmark import h2_synthetic as S
    from validation.benchmark.b30_axiom_probe import dprop
    from validation.benchmark.b31_facts import compile_registry
    from validation.benchmark.h2_axiom_mechanisms import control_random_unions, defined_union_concepts, derived_union_literals
    from validation.benchmark.h2_pool_study import Ceilings, ablate_literal, check_ceilings, comparability, fit_pool_arm, ladder_state, match_generic_by_evaluated, resource_report
    fams, k_true = cfg["families"], cfg["k_true"]
    X, names, truth = S.make_generic_data(cfg["n"], cfg["seed"], fams, cfg["n_noise"], cfg["n_num"], k_true=k_true, n_classes=cfg["n_classes"])
    fam0, k0 = next(iter(fams.items()))
    want = frozenset(f"{fam0}__L{j}" for j in range(k_true + 1, k0 + 1))
    binary = cfg["n_classes"] == 2

    def build(variant, text=None):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "o.owl"
            p.write_text(text or S.build_generic_owl(names, fams, k_true, variant), encoding="utf-8")
            reg = compile_registry(str(p), names, run_reasoner=True)
            rej: List[dict] = []
            lits = derived_union_literals(reg, X, names, only_concepts=defined_union_concepts(p), rejected=rej)
            return reg, lits, rej
    reg_t, lt, _ = build("TRUE")
    _, lp, _ = build("PERM")
    _, ln, _ = build("NEUTRAL")
    _, lf, _ = build("FLAT")
    dup_text = S.build_generic_owl(names, fams, k_true, "TRUE").replace("</rdf:RDF>", dprop(f"has_dup_{fam0}_L{k0}", f"Lvl_{fam0}_L{k0}", label=f"{fam0}__L{k0}") + "</rdf:RDF>")
    _, ld, rej_d = build("TRUE", dup_text)
    checks: Dict[str, Any] = {"1_compile": {"ontology_valid": reg_t.ontology_status == "VALID", "matching_complete": all(reg_t.feature_entity.get(n) for n in names),
                                            "true_extension_expected": sorted(want), "true_literal_present": want in {frozenset(l.attributes) for l in lt} or len(want) < 2,
                                            "perm_does_not_express_truth": want not in {frozenset(l.attributes) for l in lp}, "neutral_and_flat_have_no_literals": ln == [] and lf == [],
                                            "ambiguous_match_rejects_incomplete_literals": bool(rej_d) and not any(l.family == fam0 for l in ld),
                                            "ambiguous_rejections": rej_d}}
    ceil = Ceilings(**CEILINGS)
    r = len(lt)
    out: Dict[str, Any] = {"config": {k: v for k, v in cfg.items()}, "n_features": len(names), "n_classes": cfg["n_classes"], "n_derived_true": r, "runs": {}, "checks": checks}
    for max_n in (1, 3):
        params = dict(BASE_PARAMS, max_n=max_n)
        mf0 = params["max_features_per_node"]
        rng = np.random.RandomState(cfg["seed"] + max_n)
        var_order = [int(j) for j in np.argsort(-np.nanvar(X, axis=0), kind="stable")]
        rest = [j for j in var_order if j not in set(var_order[:mf0])]
        rand_base = sorted(int(j) for j in rng.choice(rest, size=min(r, len(rest)), replace=False)) if r else []
        rnd_rel = control_random_unions(lt, names, cfg["seed"]) if r else []
        arms: Dict[str, Any] = {}

        def fit(key, derived, extras, mf, **kw):
            budget.tick()
            arms[key] = fit_pool_arm(X, names, derived, truth.reorder(names), params, extras=extras, mf=mf, **kw)
            return arms[key]
        fit("A_LIMITED", [], "none", mf0)
        fit("A_PLUS_ENRICHED", lt, "none", mf0)
        fit("B_WIDE_GENERIC", [], "none", mf0 + r)
        fit("C_OWL_PRIORITY", lt, "derived", mf0 + r)
        fit("D_PERMUTED_PRIORITY", lp, "derived", mf0 + len(lp))
        fit("E_RANDOM_BASE", [], "base", mf0 + len(rand_base), forced_base=rand_base)
        fit("E_RANDOM_RELATIONS", rnd_rel, "derived", mf0 + len(rnd_rel))
        target = arms["C_OWL_PRIORITY"]["trace"]["pool"]["counters"]["mofn_evaluated"]
        scan = list(range(mf0, min(len(names), mf0 + r + 10) + 1))

        def f_fit(mf):
            budget.tick()
            return fit_pool_arm(X, names, [], truth.reorder(names), params, extras="none", mf=mf)
        m = match_generic_by_evaluated(f_fit, target, scan, tolerance=TOL["mofn_evaluated"])
        arms["F_GENERIC_MATCHED_EVALUATED"] = m["arm"]
        cmp_ = {}
        for b_key, eq in (("B_WIDE_GENERIC", ("feature_slots",)), ("D_PERMUTED_PRIORITY", ("feature_slots",)), ("E_RANDOM_BASE", ("feature_slots",)), ("E_RANDOM_RELATIONS", ("feature_slots",)),
                          ("F_GENERIC_MATCHED_EVALUATED", ("mofn_evaluated",)), ("A_LIMITED", ("feature_slots",))):
            c = comparability(arms["C_OWL_PRIORITY"], arms[b_key], same_instances=True, same_oracle=True, same_params_except_pool=True, tol=dict(TOL), ceilings=ceil, design_equalized=eq)
            cmp_[f"C vs {b_key}"] = {"status": c["status"], "violations": c["configuration_violations"], "realized_resource_match": c["realized_resource_match"]}
        # (7) equivalência sem semântica
        o1, o2 = truth.reorder(names), truth.reorder(names)
        orig = TrepanOriginalClassifier(**params).fit(X, oracle=o1, feature_names=names)
        a = arms["A_LIMITED"]
        eq_none = _rules_sha(orig) == _rules_sha(a["model"]) and orig.membership_queries_ == a["queries"]
        neutral = fit_pool_arm(X, names, ln, truth.reorder(names), params, extras="derived", mf=mf0)
        budget.tick()
        eq_neutral = _rules_sha(orig) == _rules_sha(neutral["model"]) and orig.membership_queries_ == neutral["queries"]
        # (2,3,6) literal verdadeiro: pool, avaliação, estados e ablação
        trace_true = None
        if r and want in {frozenset(l.attributes) for l in lt}:
            k = [i for i, l in enumerate(lt) if frozenset(l.attributes) == want][0]
            ap, cc = arms["A_PLUS_ENRICHED"], arms["C_OWL_PRIORITY"]
            budget.tick()
            budget.tick()
            ab = ablate_literal(X, names, lt, lambda: truth.reorder(names), params, k, extras="derived", mf=mf0 + r, full=cc)
            plit, ltr = cc["trace"]["pool"]["literals"][k], cc["trace"]["literals"][k]
            trace_true = {"literal": lt[k].name, "axiom_fact_ids": list(lt[k].fact_ids), "needs_reasoner": lt[k].needs_reasoner,
                          "pool_reason_A_PLUS_root": ap["trace"]["pool"]["literals"][k]["root_reason"], "pool_reason_C_root": plit["root_reason"], "candidate_evaluations_C": ltr["candidate_evaluations"],
                          "chosen_in_nodes_C": ltr["chosen_in_nodes"], "ablation": ab, "ladder": ladder_state(ltr, plit, valid=True, selected=bool(ltr["chosen_in_nodes"]), changed_by_ablation=ab["changed"])}
        out["runs"][f"max_n={max_n}"] = {"params": params, "F_matching": {"mf": m["mf"], "relative_gap": round(m["relative_gap"], 4), "feasible": m["feasible"], "scan_fits": len(scan)},
                                          "equivalence": {"A_equals_Original": eq_none, "valid_ontology_without_active_structure_equals_Original": eq_neutral},
                                          "ceilings_exceeded": {k: check_ceilings(a_, ceil)["exceeded"] for k, a_ in arms.items()}, "comparability": cmp_, "true_literal_trace": trace_true,
                                          "resources": {k: resource_report(a_, ceil, extra_fits=(len(scan) - 1 if k == "F_GENERIC_MATCHED_EVALUATED" else (1 if k == "C_OWL_PRIORITY" else 0))) for k, a_ in arms.items()}}
    # controlo negativo dos tetos: um teto apertado de propósito TEM de produzir NOT_COMPARABLE
    tight = comparability(arms["C_OWL_PRIORITY"], arms["B_WIDE_GENERIC"], same_instances=True, same_oracle=True, same_params_except_pool=True, tol=dict(TOL), ceilings=Ceilings(eligible_literals=1),
                          design_equalized=("feature_slots",))
    checks["4_ceilings_control"] = {"tight_ceiling_gives_NOT_COMPARABLE": tight["status"] == "NOT_COMPARABLE"}
    return out


def run_pilot(out: str, configs=None, limits=None) -> Dict[str, Any]:
    warnings.filterwarnings("ignore")
    limits = limits or LIMITS
    budget = Budget(limits)
    res: Dict[str, Any] = {"purpose": "PILOTO TÉCNICO sintético; sem accuracy/fidelidade; não autoriza conclusões de superioridade", "limits_declared_before_execution": limits, "ceilings_proposal": CEILINGS,
                           "configs": [], "status": "COMPLETED"}
    tracemalloc.start()
    try:
        for cfg in (configs or CONFIGS):
            res["configs"].append(run_config(cfg, budget))
    except LimitExceeded as e:
        res["status"] = "ABORTED_LIMIT"
        res["aborted_on"] = str(e)
    budget.peak = max(budget.peak, tracemalloc.get_traced_memory()[1] / 1e6)
    tracemalloc.stop()
    if budget.peak > limits["max_peak_memory_mb"] and res["status"] == "COMPLETED":
        res["status"] = "ABORTED_LIMIT"
        res["aborted_on"] = "max_peak_memory_mb"
    res["consumption"] = {"wall_seconds": round(time.time() - budget.t0, 2), "fits": budget.fits, "peak_memory_mb": round(budget.peak, 1)}
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return res


if __name__ == "__main__":
    r = run_pilot(sys.argv[1])
    print(r["status"], r["consumption"])
