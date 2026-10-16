"""B3.1 (Doc. 36): problemas sintéticos de IDENTIFICABILIDADE com verdade conhecida, orçamento de pesquisa limitado e casos negativos. Só ``validation/``.

Princípios (pedido de B3.1 §7):
* conhecimento semântico VERDADEIRO e relevante para a estrutura da regra (um m-of-n sobre os atributos de um conceito);
* pesquisa do Original limitada por ``max_features_per_node`` (o Original só considera os atributos de maior variância: orçamento natural, não um defeito inventado);
* a dificuldade (``max_features_per_node``) é calibrada SÓ com o Original, em sementes de calibração DISJUNTAS das de avaliação, antes de qualquer variante semântica ser avaliada;
* problemas, variantes, critérios de aceitação e hash do código ficam congelados em ``problems_frozen.json`` (``freeze``) antes da avaliação (``evaluate`` recusa se o código mudou);
* o mecanismo é o arnês de ``b31_steering`` (preferência: decide ONDE procurar; pontuação estatística original); o avaliador de fidelidade é uma amostra independente;
* casos negativos: ontologia verdadeira irrelevante, incompleta, ambígua, inconsistente, semanticamente errada, matching insuficiente, válida sem vantagem possível.
Cada variante usa o MESMO oráculo, dados, orçamento e semente do Original."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.trepan_original import TrepanOriginalClassifier
from validation.benchmark.b30_axiom_probe import dprop, klass, write
from validation.benchmark.b31_facts import compile_registry
from validation.benchmark.b31_relatedness import semantic_units
from validation.benchmark.b31_steering import SemanticSteeredTrepan

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
FROZEN = ROOT / "docs/evidence/b1b4/b31/problems_frozen.json"
CODE_FILES = ("validation/benchmark/b31_facts.py", "validation/benchmark/b31_relatedness.py", "validation/benchmark/b31_steering.py", "validation/benchmark/b31_identifiability.py")

TOPICS = ["thermal", "optical", "acoustic", "magnetic", "chemical", "kinetic", "electric", "spatial"]
MEASURES = ["mean", "range", "peak", "trend", "spread", "level"]
CAL_SEEDS = tuple(range(1000, 1030))
EVAL_SEEDS = tuple(range(0, 40))


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    n_groups: int = 8
    group_size: int = 4
    rule: str = "within_group"          # within_group: 2-de-4 num conceito | cross_group: 2 atributos de cada um de dois conceitos
    rule_m: int = 2
    n_train: int = 300
    scale_sigma: float = 0.7
    min_sample: int = 400
    max_queries: int = 4000
    max_nodes: int = 7
    max_depth: int = 3
    max_n: int = 4
    beam_width: int = 2
    min_samples_leaf: int = 3
    max_features_per_node: int = 16      # calibrado (ver ``calibrate``)
    n_test: int = 4000


def make_instance(spec: ProblemSpec, seed: int) -> Dict[str, Any]:
    """Dados, nomes (ordem de colunas aleatória), grupos verdadeiros, atributos da regra e oráculo de forma fechada. Determinístico por (spec, seed)."""
    rng = np.random.default_rng([seed, 31])
    p = spec.n_groups * spec.group_size
    names_by_group = [[f"{TOPICS[g]}_{MEASURES[j]}" for j in range(spec.group_size)] for g in range(spec.n_groups)]
    all_names = [n for grp in names_by_group for n in grp]
    order = rng.permutation(p)                                       # ordem de colunas aleatória: sem pista pela posição
    names = [all_names[i] for i in order]
    sd = np.exp(spec.scale_sigma * rng.normal(size=p))
    X = rng.normal(size=(spec.n_train, p)) * sd
    g_rule = rng.permutation(spec.n_groups)
    if spec.rule == "within_group":
        rule_groups = [int(g_rule[0])]
        rule_names = list(names_by_group[rule_groups[0]])
    else:
        rule_groups = [int(g_rule[0]), int(g_rule[1])]
        pa, pb = rng.permutation(spec.group_size)[:2], rng.permutation(spec.group_size)[:2]
        rule_names = [names_by_group[rule_groups[0]][i] for i in pa] + [names_by_group[rule_groups[1]][i] for i in pb]
    rule_idx = [names.index(n) for n in rule_names]

    class Oracle:
        """MLP substituto de forma fechada: ≥ m dos atributos da regra positivos (só vê as colunas)."""
        def predict(self, Z):
            Z = np.asarray(Z, dtype=float)
            return ((Z[:, rule_idx] > 0).sum(axis=1) >= spec.rule_m).astype(int)

        def predict_proba(self, Z):
            y = self.predict(Z)
            return np.column_stack([1 - y, y]).astype(float)

    Xtest = np.random.default_rng([seed, 32]).normal(size=(spec.n_test, p)) * sd
    return {"X": X, "Xtest": Xtest, "names": names, "names_by_group": names_by_group, "rule_names": rule_names, "rule_idx": rule_idx, "rule_groups": rule_groups, "oracle": Oracle(), "sd": sd}


# ------------------------------------------------------------------------------------------------------ ontologias (variantes)
def _owl(tmp: Path, name: str, groups: Dict[str, List[str]], *, parents: Optional[Dict[str, str]] = None, extra: str = "", prop_names: Optional[Dict[str, str]] = None,
         nodomain: bool = False) -> Path:
    body = klass("MeasurementProfile")
    parent_classes = sorted(set((parents or {}).values()))
    for pc in parent_classes:
        body += klass(pc, ["MeasurementProfile"])
    for cn in groups:
        body += klass(cn, [(parents or {}).get(cn, "MeasurementProfile")])
    for cn, feats in groups.items():
        for f in feats:
            pn = (prop_names or {}).get(f, f)
            body += dprop(pn, None if nodomain else cn, label=pn)
    return write(tmp, name, body + extra)


def _cls(g: int) -> str:
    return f"{TOPICS[g].capitalize()}Profile"


def true_groups(inst: Dict[str, Any]) -> Dict[str, List[str]]:
    return {_cls(g): list(inst["names_by_group"][g]) for g in range(len(inst["names_by_group"]))}


def _redraw_partition(inst, seed, rule_names, kind: str) -> Dict[str, List[str]]:
    """Partição com os MESMOS tamanhos de grupo; garante que o conjunto da regra NÃO fica contido num só grupo (critério estrutural, independente de resultados)."""
    rng = np.random.default_rng([seed, 33 if kind == "perm" else 34])
    feats = [n for grp in inst["names_by_group"] for n in grp]
    sizes = [len(g) for g in inst["names_by_group"]]
    for _ in range(1000):
        perm = list(rng.permutation(feats))
        groups, i = {}, 0
        for g, s in enumerate(sizes):
            groups[_cls(g)] = perm[i:i + s]; i += s
        if not any(set(rule_names) <= set(m) for m in groups.values()):
            return groups
    raise RuntimeError("não foi possível destruir a regra")


def build_ontology(variant: str, inst: Dict[str, Any], seed: int, tmp: Path) -> Optional[Path]:
    tg = true_groups(inst)
    rn = inst["rule_names"]
    rule_cls = [_cls(g) for g in inst["rule_groups"]]
    if variant == "TRUE":
        return _owl(tmp, f"{variant}_{seed}", tg)
    if variant == "HIERARCHY":                                   # os conceitos da regra partilham um pai informativo (os restantes pares de conceitos têm outro pai)
        order = list(tg)
        parents = {}
        rest = [c for c in order if c not in rule_cls]
        for c in rule_cls:
            parents[c] = "RuleRelevantFamily"
        for i, c in enumerate(rest):
            parents[c] = f"OtherFamily{i // 2}"
        return _owl(tmp, f"{variant}_{seed}", tg, parents=parents)
    if variant == "PERM":
        return _owl(tmp, f"{variant}_{seed}", _redraw_partition(inst, seed, rn, "perm"))
    if variant == "RAND":
        return _owl(tmp, f"{variant}_{seed}", _redraw_partition(inst, seed, rn, "rand"))
    if variant == "EMPTY":                                       # válida, sem axiomas úteis (propriedades sem domínio)
        return _owl(tmp, f"{variant}_{seed}", tg, nodomain=True)
    if variant == "IRRELEVANT":                                  # verdadeira, mas sobre outros atributos; os da regra ficam em conceitos singulares
        groups = {}
        others = [n for grp in inst["names_by_group"] for n in grp if n not in rn]
        for i in range(0, len(others), 4):
            groups[f"DecoyConcept{i // 4}"] = others[i:i + 4]
        for j, n in enumerate(rn):
            groups[f"IsolatedConcept{j}"] = [n]
        return _owl(tmp, f"{variant}_{seed}", groups)
    if variant == "INCOMPLETE":                                  # um atributo da regra omitido do seu conceito (passa a conceito singular)
        groups = {k: [f for f in v if f != rn[0]] for k, v in tg.items()}
        groups["OrphanConcept"] = [rn[0]]
        return _owl(tmp, f"{variant}_{seed}", {k: v for k, v in groups.items() if v})
    if variant == "NOISY_SUPERSET":                              # conceito da regra com 2 atributos espúrios (verdade + ruído)
        groups = {k: list(v) for k, v in tg.items()}
        donors = [c for c in groups if c not in rule_cls][:2]
        moved = [groups[d].pop(0) for d in donors]
        groups[rule_cls[0]] = groups[rule_cls[0]] + moved
        return _owl(tmp, f"{variant}_{seed}", {k: v for k, v in groups.items() if v})
    if variant == "WRONG_SPLIT":                                 # relação semanticamente errada: atributos da regra divididos 2+2 por dois conceitos com intrusos
        groups = {k: [f for f in v if f not in rn] for k, v in tg.items()}
        groups = {k: v for k, v in groups.items() if v}
        others = [c for c in groups if c not in rule_cls]
        a, b = others[0], others[1]
        groups[a] = groups[a] + rn[:2]
        groups[b] = groups[b] + rn[2:]
        return _owl(tmp, f"{variant}_{seed}", groups)
    if variant == "AMBIGUOUS":                                   # duas propriedades quase homónimas por atributo da regra ⇒ matching ambíguo
        extra = ""
        pn = {}
        groups = {k: list(v) for k, v in tg.items()}
        for f in rn:
            pn[f] = f + "_a"
            extra += dprop(f + "_b", [c for c, v in tg.items() if f in v][0], label=f + "_b")
        return _owl(tmp, f"{variant}_{seed}", groups, extra=extra, prop_names=pn)
    if variant == "INCONSISTENT":                                # classe insatisfazível
        extra = (klass("ConceptP") + klass("ConceptQ", [], '<owl:disjointWith rdf:resource="#ConceptP"/>')
                 + klass("ContradictionConcept", ["ConceptP", "ConceptQ"]))
        return _owl(tmp, f"{variant}_{seed}", tg, extra=extra)
    if variant == "BADMATCH":                                    # nomes sem relação lexical com os atributos ⇒ cobertura insuficiente
        pn = {f: "zorvan" + hashlib.md5(f.encode()).hexdigest()[:6] for f in inst["names"]}
        return _owl(tmp, f"{variant}_{seed}", tg, prop_names=pn)
    return None                                                  # ORIG, WIDE, CONV: sem ontologia


VARIANTS_ONTOLOGY = ("TRUE", "HIERARCHY", "PERM", "RAND", "EMPTY", "IRRELEVANT", "INCOMPLETE", "NOISY_SUPERSET", "WRONG_SPLIT", "AMBIGUOUS", "INCONSISTENT", "BADMATCH")
VARIANTS_CONTROL = ("ORIG", "WIDE", "CONV")
ALL_VARIANTS = VARIANTS_CONTROL + VARIANTS_ONTOLOGY


def conventional_units(X, names, n_groups: int, group_size: int):
    """Controlo de engenharia convencional (sem ontologia): grupos por agrupamento hierárquico de |Spearman| sobre o TREINO. Mesmo formato de unidades."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import rankdata
    from validation.benchmark.b31_relatedness import SemanticUnit
    R = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    d = np.clip(1.0 - np.abs(np.nan_to_num(np.corrcoef(R, rowvar=False))), 0, None)
    np.fill_diagonal(d, 0.0)
    lab = fcluster(linkage(squareform(d, checks=False), method="average"), t=n_groups, criterion="maxclust")
    units = []
    for c in sorted(set(lab)):
        mem = tuple(sorted(names[i] for i in np.where(lab == c)[0]))
        if len(mem) >= 2:
            units.append(SemanticUnit(f"conv{c}", mem, 1.0, ("DATA",), (), 1.0))
    return units


def fit_variant(spec: ProblemSpec, inst: Dict[str, Any], variant: str, seed: int, tmp: Path, *, unit_budget: int = 3) -> Dict[str, Any]:
    kw = dict(min_sample=spec.min_sample, max_queries=spec.max_queries, max_nodes=spec.max_nodes, max_depth=spec.max_depth, max_n=spec.max_n, beam_width=spec.beam_width,
              min_samples_leaf=spec.min_samples_leaf, max_features_per_node=spec.max_features_per_node, random_state=seed)
    p = inst["X"].shape[1]
    t0 = time.perf_counter()
    reg = None
    units: List[Any] = []
    if variant == "ORIG":
        model = TrepanOriginalClassifier(**kw)
    elif variant == "WIDE":
        model = TrepanOriginalClassifier(**dict(kw, max_features_per_node=p))
    elif variant == "CONV":
        units = conventional_units(inst["X"], inst["names"], spec.n_groups, spec.group_size)
        model = SemanticSteeredTrepan(units=units, unit_budget=unit_budget, **kw)
    else:
        owl = build_ontology(variant, inst, seed, tmp)
        reg = compile_registry(owl, inst["names"])
        units = semantic_units(reg, inst["names"]) if reg.ontology_status == "VALID" else []
        model = SemanticSteeredTrepan(units=units, unit_budget=unit_budget, registry=reg, **kw)
    model.fit(inst["X"], oracle=inst["oracle"], feature_names=inst["names"])
    nodes = [n for n in model.iter_nodes() if not n.is_leaf and n.test is not None]
    rule = set(inst["rule_idx"])
    recovered = any({int(l.feature) for l in n.test.literals} == rule for n in nodes)
    yt = inst["oracle"].predict(inst["Xtest"])
    fid = float(np.mean(np.asarray(model.predict(inst["X"] if False else inst["Xtest"])) == yt))
    out = {"variant": variant, "seed": seed, "recovered": bool(recovered), "fidelity_test": fid, "n_internal": len(nodes), "queries": int(model.membership_queries_), "time_s": time.perf_counter() - t0,
           "rules_sha": hashlib.sha256(json.dumps(model.export_rules(), default=str).encode()).hexdigest()[:16], "n_units": len(units),
           "ontology_status": reg.ontology_status if reg is not None else None, "n_facts_usable": len(reg.usable()) if reg is not None else None}
    if isinstance(model, SemanticSteeredTrepan) and units:
        rep = model.trace_report(reg)
        out.update(n_choice_changed=rep["n_choice_changed"], n_influenced_in_final_tree=rep["n_influenced_in_final_tree"], n_trace_calls=rep["n_calls"])
    else:
        out.update(n_choice_changed=0, n_influenced_in_final_tree=0, n_trace_calls=0)
    return out


# ------------------------------------------------------------------------------------------------------ calibração (SÓ Original) e congelamento
def calibrate(base: ProblemSpec, grid=(8, 12, 16, 20, 24, 28), seeds=CAL_SEEDS) -> Dict[str, Any]:
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for mf in grid:
            spec = ProblemSpec(**{**asdict(base), "max_features_per_node": mf})
            rec = [fit_variant(spec, make_instance(spec, s), "ORIG", s, Path(td))["recovered"] for s in seeds]
            res[mf] = float(np.mean(rec))
    ok = [(abs(r - 0.5), mf) for mf, r in res.items() if 0.2 <= r <= 0.8]
    chosen = min(ok)[1] if ok else None
    return {"rates": res, "chosen_max_features_per_node": chosen, "rule": "mf com taxa de recuperação do Original mais próxima de 0,5 e em [0,2; 0,8]; empate: menor mf", "calibration_seeds": [seeds[0], seeds[-1]]}


def code_hash() -> Dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES}


CRITERIA = {
    "version": 1,
    "evaluation_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
    "unit_budget": 3,
    "recovery_definition": "existe um nó interno cujo teste m-of-n usa EXATAMENTE o conjunto de atributos da regra",
    "fidelity_definition": "concordância da árvore com o oráculo numa amostra independente de n_test linhas (nunca usada no ajuste)",
    "P1_within_group": {
        "C1_true_beats_orig": "recovery(TRUE) - recovery(ORIG) >= 0.20",
        "C2_true_beats_perm": "recovery(TRUE) - recovery(PERM) >= 0.20",
        "C3_true_beats_rand": "recovery(TRUE) - recovery(RAND) >= 0.20",
        "C4_true_beats_conventional": "recovery(TRUE) - recovery(CONV) >= 0.20",
        "C5_abstention_equals_original": "EMPTY, INCONSISTENT e BADMATCH: árvore (rules_sha) idêntica à do Original em 100% das sementes",
        "C6_ambiguous_no_spurious_gain": "|recovery(AMBIGUOUS) - recovery(ORIG)| <= 0.10",
        "C7_no_spurious_advantage": "IRRELEVANT, INCOMPLETE, WRONG_SPLIT: recovery - recovery(ORIG) <= 0.10",
        "C8_bounded_harm": "para toda a variante semântica: recovery - recovery(ORIG) >= -0.20 e fidelidade média >= fidelidade média ORIG - 0.02",
        "C9_noisy_superset": "reportado, sem limiar (ganho parcial esperado)",
    },
    "P2_cross_group": {
        "D1_true_flat_no_advantage": "recovery(TRUE) - recovery(ORIG) <= 0.10 (ontologia válida; a regra atravessa conceitos: sem vantagem possível)",
        "D2_hierarchy_advantage": "recovery(HIERARCHY) - recovery(ORIG) >= 0.20 (pai informativo comum dos conceitos da regra)",
        "D3_perm_no_advantage": "recovery(PERM) - recovery(ORIG) <= 0.10",
    },
    "controls_reported_not_judged": ["WIDE (Original com max_features_per_node = p: não é ontologia; mede o efeito do orçamento de atributos)"],
    "interpretation": "critérios de aceitação técnica do arnês (identificabilidade); NÃO são evidência de superioridade do Reloaded",
}


def freeze(out: Path = FROZEN) -> Dict[str, Any]:
    p1 = ProblemSpec("P1_within_group", rule="within_group")
    p2 = ProblemSpec("P2_cross_group", rule="cross_group")
    c1, c2 = calibrate(p1), calibrate(p2)
    if c1["chosen_max_features_per_node"] is None or c2["chosen_max_features_per_node"] is None:
        raise RuntimeError(f"calibração sem dificuldade intermédia: {c1['rates']} / {c2['rates']}")
    specs = {"P1_within_group": asdict(ProblemSpec(**{**asdict(p1), "max_features_per_node": c1["chosen_max_features_per_node"]})),
             "P2_cross_group": asdict(ProblemSpec(**{**asdict(p2), "max_features_per_node": c2["chosen_max_features_per_node"]}))}
    doc = {"frozen": True, "purpose": "congelado ANTES de avaliar qualquer variante semântica", "problems": specs, "calibration": {"P1_within_group": c1, "P2_cross_group": c2},
           "variants": {"controls": VARIANTS_CONTROL, "ontology": VARIANTS_ONTOLOGY}, "criteria": CRITERIA, "code_sha256": code_hash(), "evaluation_seeds": list(EVAL_SEEDS),
           "calibration_used_only_original": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    return doc


def evaluate(allow_drift: bool = False, seeds=EVAL_SEEDS, variants=ALL_VARIANTS, out: Optional[Path] = None) -> Dict[str, Any]:
    fz = json.loads(FROZEN.read_text())
    drift = {f: (h, code_hash()[f]) for f, h in fz["code_sha256"].items() if code_hash()[f] != h}
    if drift and not allow_drift:
        raise SystemExit(f"código alterado desde o congelamento: {sorted(drift)} (use --allow-drift e registe o desvio)")
    res: Dict[str, Any] = {"frozen_code_drift": {k: list(v) for k, v in drift.items()}, "problems": {}}
    with tempfile.TemporaryDirectory() as td:
        for pname, sd in fz["problems"].items():
            spec = ProblemSpec(**sd)
            rows = []
            for s in seeds:
                inst = make_instance(spec, s)
                for v in variants:
                    rows.append(fit_variant(spec, inst, v, s, Path(td), unit_budget=fz["criteria"]["unit_budget"]))
            res["problems"][pname] = rows
    if out:
        out.write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    e = sub.add_parser("evaluate")
    e.add_argument("--out", required=True)
    e.add_argument("--allow-drift", action="store_true")
    e.add_argument("--n-seeds", type=int, default=len(EVAL_SEEDS))
    a = ap.parse_args()
    if a.cmd == "freeze":
        d = freeze()
        print(json.dumps({"calibration": {k: v["rates"] for k, v in d["calibration"].items()}, "chosen": {k: v["max_features_per_node"] for k, v in d["problems"].items()}}, indent=1))
    else:
        evaluate(a.allow_drift, EVAL_SEEDS[:a.n_seeds], ALL_VARIANTS, Path(a.out))


if __name__ == "__main__":
    main()
