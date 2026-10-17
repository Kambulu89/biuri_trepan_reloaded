"""B3.2 (Doc. 40): diagnóstico do matching ambíguo do quality gate (``core/ontology_quality.py``). SÓ LEITURA do núcleo: reproduz o empate, avalia políticas alternativas como
funções puras sobre as decisões já produzidas e mede o impacto nas ontologias congeladas. Nada é alterado em ``core/``.

Defeito: ``OntologyQualityGate.match_features`` (linhas 294–298) aceita ``best_score >= 0.98`` mesmo com ``margin == 0`` (a margem só é exigida abaixo de 0,98). Num empate
perfeito (duas entidades com score 1,0) a ordenação estável (linhas 280–284) devolve a 1.ª entidade da lista de candidatos (classes, depois propriedades de dados, depois de objetos,
pela ordem de ``owlready2``), i.e. uma escolha arbitrária que depende da ORDEM de declaração."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

from validation.benchmark import datasets as D
from validation.benchmark.b30_axiom_probe import dprop, klass, write

EPS = 1e-9


def decide(d: Dict[str, Any], policy: str, min_margin: float = 0.08, min_score: float = 0.72) -> bool:
    """Política aplicada às decisões do gate (campos ``score``, ``ambiguity_margin``, ``tie_break``, ``accepted``).
    P0 = comportamento atual. P1 = rejeitar EMPATES (margem ≤ ε) sem desempate de tipo declarado. P2 = P1 + exigir margem mínima também para 0,98 ≤ score < 1,0 (só um
    casamento exato, score = 1,0, dispensa a margem)."""
    if policy == "P0":
        return bool(d["accepted"])
    if not d["accepted"]:
        return False
    tie_by_type = d.get("tie_break") is not None
    if policy in ("P1", "P2") and d["ambiguity_margin"] <= EPS and not tie_by_type:
        return False
    if policy == "P2" and d["score"] < 1.0 and d["ambiguity_margin"] < min_margin and not tie_by_type:
        return False
    return True


def decisions_for(owl_path: str, features: Sequence[str]) -> List[Dict[str, Any]]:
    from core.ontology_quality import OntologyQualityGate
    from core.owl_runtime import load_ontology_isolated
    onto = load_ontology_isolated(str(owl_path))
    return [dict(d.__dict__) for d in OntologyQualityGate().match_features(list(features), onto)]


def tie_sets(owl_path: str, features: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Para cada atributo: entidades com score igual ao melhor (a menos de ε), por tipo. Usa as funções do gate (``_candidate_score``, ``ontology_entities``) sem as alterar."""
    from core.ontology_quality import OntologyQualityGate, entity_kind, ontology_entities
    from core.owl_runtime import load_ontology_isolated
    onto = load_ontology_isolated(str(owl_path))
    ents = ontology_entities(onto)
    cands = ents["classes"] + ents["datatype_properties"] + ents["object_properties"]
    gate = OntologyQualityGate()
    out = {}
    for f in features:
        sc = [(gate._candidate_score(f, e), e) for e in cands]
        best = max((s for s, _ in sc), default=0.0)
        top = [(entity_kind(e), e.name) for s, e in sc if s >= best - EPS]
        out[f] = {"best": best, "top": top, "n_top": len(top), "n_top_datatype_property": sum(1 for k, _ in top if k == "datatype_property")}
    return out


def p3_accepts(d: Dict[str, Any], ties: Dict[str, Any], min_score: float = 0.72) -> bool:
    """P3 (proposta completa): rejeitar qualquer empate ENTRE ENTIDADES ELEGÍVEIS. A preferência de tipo (propriedade de dados sobre classe) só resolve o empate se EXATAMENTE uma
    propriedade de dados empatar com o melhor; empates entre ≥ 2 propriedades de dados (ou ≥ 2 classes sem propriedade) são rejeitados. P3 é o único que cobre empates de 3 vias."""
    if not d["accepted"] or d["score"] < min_score:
        return bool(d["accepted"]) and False
    t = ties[d["feature"]]
    if t["n_top"] == 1:
        return True
    return t["n_top_datatype_property"] == 1


def reproduce(tmp: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # 1) empate perfeito entre duas propriedades; 2) mesma ontologia com a ORDEM de declaração trocada; 3) quase-empate 1,0 vs 0,99; 4) empate classe/propriedade (desempate de tipo)
    a = klass("Root") + klass("A", ["Root"]) + klass("B", ["Root"]) + dprop("thermal_mean_a", "A", label="thermal_mean") + dprop("thermal_mean_b", "B", label="thermal_mean")
    b = klass("Root") + klass("A", ["Root"]) + klass("B", ["Root"]) + dprop("thermal_mean_b", "B", label="thermal_mean") + dprop("thermal_mean_a", "A", label="thermal_mean")
    for name, body in (("tie_order1", a), ("tie_order2", b)):
        d = decisions_for(write(tmp, name, body), ["thermal_mean"])[0]
        out[name] = {k: d[k] for k in ("entity_name", "runner_up_entity", "score", "runner_up_score", "ambiguity_margin", "accepted", "status", "tie_break")}
    near = klass("Root") + klass("A", ["Root"]) + dprop("thermal_mean", "A") + dprop("thermal_means", "A")
    d = decisions_for(write(tmp, "near", near), ["thermal_mean"])[0]
    out["near_tie_1.0_vs_runner"] = {k: d[k] for k in ("entity_name", "runner_up_entity", "score", "runner_up_score", "ambiguity_margin", "accepted", "status")}
    ct = klass("Root") + klass("ThermalMean", ["Root"]) + dprop("hasThermalMean", "ThermalMean", label="ThermalMean")
    d = decisions_for(write(tmp, "classprop", ct), ["thermal_mean"])[0]
    out["class_vs_property_tie"] = {k: d[k] for k in ("entity_name", "entity_type", "score", "runner_up_score", "ambiguity_margin", "accepted", "tie_break")}
    three = klass("Root") + klass("ThermalMean", ["Root"]) + dprop("hasThermalMeanA", "ThermalMean", label="ThermalMean") + dprop("hasThermalMeanB", "ThermalMean", label="ThermalMean")
    p3 = write(tmp, "threeway", three)
    d = decisions_for(p3, ["thermal_mean"])[0]
    out["three_way_tie_class_and_two_properties"] = {**{k: d[k] for k in ("entity_name", "entity_type", "score", "runner_up_score", "ambiguity_margin", "accepted", "tie_break")},
                                                     "tie_sets": tie_sets(p3, ["thermal_mean"])["thermal_mean"], "P1_accepts": decide(d, "P1"), "P3_accepts": p3_accepts(d, tie_sets(p3, ["thermal_mean"]))}
    return out


def impact() -> Dict[str, Any]:
    from validation.benchmark import datasets_categorical as DC
    DC.register()
    res: Dict[str, Any] = {}
    for did, spec in D.REGISTRY.items():
        if spec.ontology_path is None or not Path(spec.ontology_path).exists():
            continue
        ds, _, _ = spec.load()
        names = [str(n) for n in ds.feature_names]
        decs = decisions_for(str(spec.ontology_path), names)
        row = {"n_features": len(names), "accepted_P0": sum(decide(d, "P0") for d in decs)}
        for pol in ("P1", "P2"):
            ch = [d["feature"] for d in decs if decide(d, "P0") != decide(d, pol)]
            row[f"changed_{pol}"] = len(ch)
            row[f"changed_features_{pol}"] = ch[:10]
        ts = tie_sets(str(spec.ontology_path), names)
        row["changed_P3"] = sum(1 for d in decs if bool(d["accepted"]) != p3_accepts(d, ts))
        row["min_margin_among_accepted"] = min((d["ambiguity_margin"] for d in decs if d["accepted"]), default=None)
        row["n_accepted_margin_below_0.08"] = sum(1 for d in decs if d["accepted"] and d["ambiguity_margin"] < 0.08)
        row["n_accepted_with_exact_tie"] = sum(1 for d in decs if d["accepted"] and d["ambiguity_margin"] <= EPS)
        res[did] = row
    return res


def main(out: str):
    with tempfile.TemporaryDirectory() as td:
        rep = reproduce(Path(td))
    imp = impact()
    Path(out).write_text(json.dumps({"reproduction": rep, "impact_on_frozen_ontologies": imp, "policies": {"P0": "atual", "P1": "rejeitar empates (margem ≤ 1e-9) sem desempate de tipo", "P2": "P1 + margem mínima para 0,98 ≤ score < 1,0"}}, indent=1))
    print(json.dumps({"reproduction": rep, "impact": imp}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
