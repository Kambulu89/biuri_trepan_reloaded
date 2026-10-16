"""B3.1 (Doc. 37): verificador do protocolo de enriquecimento validado de ontologias. Faz automaticamente o que uma máquina PODE verificar (estrutura do pacote, hashes,
fontes presentes, consistência/classificação, proxy do perfil OWL 2 DL, matching) e deixa explicitamente PENDENTE o que exige humanos (domínio, engenharia de ontologias,
metodologia). NUNCA declara validação independente sem revisões humanas nomeadas e aprovadas; um revisor chamado «claude»/«assistant»/«ai» é rejeitado."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

PENDING = "DOMAIN_REVIEW_PENDING"
AI_NAMES = {"claude", "assistant", "ai", "model", "llm"}
ROLES = ("domain_expert", "ontology_engineer", "methodology")
STEPS = ("1_domain_sources_identified", "2_axioms_proposed_and_justified", "3_provenance_recorded", "4_domain_review", "5_owl2_dl_modelling", "6_consistency_and_inference",
         "7_attribute_matching", "8_frozen_before_evaluation", "9_hashes_and_versions", "10_comparison_plan_with_controls")


def _human(name) -> bool:
    return bool(name) and str(name).strip().lower() not in AI_NAMES and not any(w in str(name).lower() for w in ("claude", "anthropic", "gpt"))


def dl_profile_proxy(owl_path: str) -> Dict[str, Any]:
    """PROXY do perfil OWL 2 DL (não há verificador formal de perfil neste ambiente): (a) sem punning classe/propriedade; (b) o HermiT classifica sem erro. Não substitui uma
    verificação com a OWL API Profile Checker."""
    from core.owl_runtime import load_ontology_isolated
    from core.ontology_reasoner import run_owl_reasoner
    onto = load_ontology_isolated(owl_path)
    cls = {c.name for c in onto.classes()}
    props = {p.name for p in list(onto.data_properties()) + list(onto.object_properties())}
    punning = sorted(cls & props)
    rep = run_owl_reasoner(onto, engine="hermit", infer_property_values=True, debug=0)
    return {"punning_class_property": punning, "reasoner_executed": bool(rep.get("executed")), "consistent": rep.get("consistent"), "unsatisfiable": rep.get("unsatisfiable_classes"),
            "inferred_axioms_count": rep.get("inferred_axioms_count"), "error": rep.get("error"), "formal_profile_checked": False,
            "note": "proxy: ausência de punning + classificação HermiT sem erro; perfil DL formal NÃO verificado"}


def check_package(pkg_dir: str | Path, feature_names: List[str] | None = None) -> Dict[str, Any]:
    d = Path(pkg_dir)
    prop = json.loads((d / "proposal.json").read_text())
    res: Dict[str, Any] = {step: {"status": "PENDING", "detail": ""} for step in STEPS}
    axioms = prop.get("axioms", [])
    res["1_domain_sources_identified"] = {"status": "PASS" if axioms and all(a.get("sources") for a in axioms) else "FAIL", "detail": f"{len(axioms)} axiomas; fontes em {sum(bool(a.get('sources')) for a in axioms)}"}
    res["2_axioms_proposed_and_justified"] = {"status": "PASS" if all(a.get("axiom_text") and a.get("justification") and a.get("relation_class") for a in axioms) else "FAIL", "detail": ""}
    res["3_provenance_recorded"] = {"status": "PASS" if all(all(s.get("citation") and s.get("locator") for s in a.get("sources", [])) for a in axioms) else "FAIL", "detail": "citação + localizador por fonte"}
    rv = prop.get("reviews", {})
    st = {}
    for r in ROLES:
        x = rv.get(r, {})
        ok = x.get("status") == "APPROVED" and _human(x.get("reviewer_name")) and bool(x.get("date")) and bool(x.get("signature_ref"))
        st[r] = "APPROVED_BY_HUMAN" if ok else (PENDING if r == "domain_expert" else "REVIEW_PENDING")
        if x.get("status") == "APPROVED" and not _human(x.get("reviewer_name")):
            st[r] = "REJECTED_NON_HUMAN_REVIEWER"
    res["4_domain_review"] = {"status": "PASS" if st["domain_expert"] == "APPROVED_BY_HUMAN" else "PENDING", "detail": st}
    onto = prop.get("ontology", {})
    owl = d / onto.get("file", "ontology.owl") if onto.get("file") else None
    if owl and owl.exists():
        sha = hashlib.sha256(owl.read_bytes()).hexdigest()
        prof = dl_profile_proxy(str(owl))
        res["5_owl2_dl_modelling"] = {"status": "PROXY_PASS" if not prof["punning_class_property"] and prof["reasoner_executed"] and not prof["error"] else "FAIL", "detail": prof}
        res["6_consistency_and_inference"] = {"status": "PASS" if prof["consistent"] and not prof["unsatisfiable"] else "FAIL", "detail": {"consistent": prof["consistent"], "inferred_axioms_count": prof["inferred_axioms_count"]}}
        if feature_names:
            from validation.benchmark.b31_facts import compile_registry
            reg = compile_registry(owl, feature_names, run_reasoner=False)
            cov = sum(1 for f in feature_names if reg.feature_entity.get(f)) / max(len(feature_names), 1)
            res["7_attribute_matching"] = {"status": "PASS" if cov >= 0.35 and reg.ontology_status in ("VALID", "REASONER_NOT_EXECUTED") else "FAIL", "detail": {"coverage": cov, "status": reg.ontology_status}}
        else:
            res["7_attribute_matching"] = {"status": "PENDING", "detail": "lista de atributos não fornecida"}
        res["9_hashes_and_versions"] = {"status": "PASS" if sha == onto.get("sha256") else "FAIL", "detail": {"computed": sha, "declared": onto.get("sha256")}}
    else:
        for k in ("5_owl2_dl_modelling", "6_consistency_and_inference", "7_attribute_matching", "9_hashes_and_versions"):
            res[k] = {"status": "PENDING", "detail": "sem ficheiro OWL no pacote (proposta de axiomas apenas)"}
    pre = prop.get("preregistration", {})
    res["8_frozen_before_evaluation"] = {"status": "PASS" if pre.get("frozen_before_evaluation") and pre.get("freeze_commit") else "PENDING", "detail": pre}
    res["10_comparison_plan_with_controls"] = {"status": "PASS" if {"original", "enriched", "permuted", "random", "empty"} <= set(prop.get("comparison_arms", [])) else "PENDING", "detail": prop.get("comparison_arms", [])}
    independent = all(st[r] == "APPROVED_BY_HUMAN" for r in ROLES) and all(res[s]["status"] in ("PASS", "PROXY_PASS") for s in STEPS)
    res["_summary"] = {"ontology_declared_independently_validated": bool(independent), "domain_review_status": st["domain_expert"],
                       "blocking": [s for s in STEPS if res[s]["status"] not in ("PASS", "PROXY_PASS")],
                       "rule": "validação independente só com 3 revisões humanas nomeadas e aprovadas + todos os passos automáticos; Claude prepara propostas, não valida"}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("package"); ap.add_argument("--features", nargs="*")
    a = ap.parse_args()
    print(json.dumps(check_package(a.package, a.features), indent=1, default=str))


if __name__ == "__main__":
    main()
