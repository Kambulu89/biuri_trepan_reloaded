"""B3.0 (Doc. 31 §P4 / Doc. 32): teste de IDENTIFICABILIDADE da contribuição semântica sobre dados sintéticos com verdade conhecida.

Problema: p atributos em grupos; o oráculo («MLP» de forma fechada) é um m-of-n DENTRO de um grupo verdadeiro (ex.: 2-de-3 em G1) → a relação ontológica
verdadeira é útil e conhecida. Compara-se, com o MESMO oráculo, dados, orçamento e sementes:
  ORIG       TREPAN Original;
  OWL_true   Reloaded + ontologia OWL (OwlSemanticProvider real, caminho completo) com a partição VERDADEIRA;
  OWL_wrong  idem com a partição PERMUTADA (relação falsa);
  GRP_true / GRP_wrong   Reloaded com grupos explícitos e relatedness «limpa» (1 no grupo, 0 fora) — o que uma medida discriminativa daria (contrafactual de diagnóstico).
Mede onde a diferença verdadeiro×permutado se perde: estrutura (grupos, relatedness), candidatos, splits e árvore. Não toca o núcleo; sem datasets conhecidos."""
from __future__ import annotations
import argparse, hashlib, json, tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from validation.benchmark.b30_axiom_probe import dprop, klass, write

NAMES = ["thermal_mean", "thermal_range", "thermal_peak", "optical_mean", "optical_range", "optical_peak", "acoustic_mean", "acoustic_peak"]
CLASSES = ["ThermalProfile", "OpticalProfile", "AcousticProfile"]
TRUE_PARTITION = [NAMES[0:3], NAMES[3:6], NAMES[6:8]]
WRONG_PARTITION = [[NAMES[0], NAMES[3], NAMES[6]], [NAMES[1], NAMES[4], NAMES[7]], [NAMES[2], NAMES[5]]]
RULE = (0, 1, 2)            # 2-de-3 sobre f1,f2,f3 (índices)
BUDGET = dict(min_sample=600, max_queries=6000, max_nodes=15, max_depth=5, max_n=3, beam_width=2, min_samples_leaf=3, max_features_per_node=8, purity_epsilon=0.05)


def truth(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return ((X[:, RULE] > 0).sum(axis=1) >= 2).astype(int)


class Oracle:
    """Vê só as p colunas originais (como o MLP congelado); ignora agregados."""
    def __init__(self, p): self.p = p
    def predict(self, X): return truth(np.asarray(X)[:, : self.p])
    def predict_proba(self, X):
        y = self.predict(X); return np.column_stack([1 - y, y]).astype(float)


def owl_for(tmp: Path, name: str, partition: Sequence[Sequence[str]]) -> Path:
    body = klass("MeasurementProfile")
    for gi, g in enumerate(partition):
        body += klass(CLASSES[gi], ["MeasurementProfile"])
    for gi, g in enumerate(partition):
        for f in g:
            body += dprop(f, CLASSES[gi])
    return write(tmp, name, body)


def make_data(seed: int, n: int = 500, p: int = 8):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), list(NAMES[:p])


def fit_arm(arm: str, X, names, seed: int, tmp: Path):
    from core.trepan_original import TrepanOriginalClassifier
    from core.trepan_reloaded_historical import TrepanReloadedClassifier
    from core.benchmark.semantic import OwlSemanticProvider, GroupSemanticProvider
    p = X.shape[1]
    orc = Oracle(p)
    if arm == "ORIG":
        m = TrepanOriginalClassifier(**BUDGET, random_state=seed).fit(X, oracle=orc, feature_names=names)
        return m, None, X
    part = TRUE_PARTITION if arm.endswith("true") else WRONG_PARTITION
    if arm.startswith("OWL"):
        ctx = OwlSemanticProvider(str(owl_for(tmp, f"o_{arm}_{seed}", part))).build(X, names)
    else:
        ctx = GroupSemanticProvider({CLASSES[i]: g for i, g in enumerate(part)}).build(X, names)
    assert ctx.available, ctx.reason
    Xe = ctx.enrich(X)
    kw = dict(BUDGET, random_state=seed, alpha=0.35, beta=0.20, semantic_gain_strength=1.0, semantic_group_strength=0.15, mirror_when_no_semantic_effect=False)
    sl = list(range(ctx.n_features))
    m = TrepanReloadedClassifier(**kw).fit(Xe, oracle=orc, feature_names=ctx.enriched_names, semantic_feature_weights=np.asarray(ctx.weights)[sl],
                                           semantic_feature_groups=[ctx.groups[i] for i in sl], semantic_relatedness_matrix=np.asarray(ctx.relatedness)[np.ix_(sl, sl)],
                                           semantic_feature_depths=np.asarray(ctx.depths)[sl], query_projector=ctx.query_projector())
    return m, ctx, Xe


def tree_tests(model) -> List[dict]:
    out = []
    for n in model.iter_nodes():
        t = getattr(n, "test", None)
        if n.is_leaf or t is None:
            continue
        out.append({"node_id": n.node_id, "m": int(t.m), "features": sorted({int(l.feature) for l in t.literals}), "n": len(t.literals)})
    return out


def summarise(model, ctx, Xtest_enr, Xtest, names, p) -> dict:
    tests = tree_tests(model)
    orig_part = lambda fs: sorted(f for f in fs if f < p)
    root = tests[0] if tests else {"features": [], "m": 0, "n": 0}
    true_set = set(RULE)
    hit = any(set(orig_part(t["features"])) == true_set and t["n"] >= 2 for t in tests)
    rootin = bool(root["features"]) and set(orig_part(root["features"])) <= true_set
    pred = np.asarray(model.predict(Xtest_enr))
    fid = float(np.mean(pred == truth(Xtest)))
    rows = list(getattr(model, "semantic_split_audit_", []) or [])
    return {"root_features": [names[i] if i < len(names) else f"d{i}" for i in root["features"]], "root_m": root["m"], "root_inside_true_group": rootin, "true_rule_recovered": bool(hit),
            "fidelity_to_oracle": fid, "n_internal": len(tests), "n_semantic_audit_rows": len(rows),
            "n_decision_changed": int(sum(1 for r in rows if r.get("decision_changed"))), "n_bonus_nonzero": int(sum(1 for r in rows if abs(r.get("semantic_bonus", 0.0)) > 1e-12)),
            "semantic_candidates_generated": int(getattr(model, "semantic_candidate_generation_count_", 0)),
            "membership_queries": int(getattr(model, "membership_queries_", 0)),
            "rules_digest": hashlib.sha256(json.dumps(model.export_rules(), default=str).encode()).hexdigest()[:16]}


def candidate_components(groups: Sequence, rel, features: Sequence[int], threshold: float = 0.35) -> List[List[int]]:
    """Réplica OBSERVACIONAL (só leitura) da regra de ``TrepanReloadedClassifier._ontology_guided_candidates`` que forma os componentes semânticos: mesmo grupo
    (ignora «general»/«none») OU relatedness ≥ limiar ⇒ união. Os candidatos m-of-n só combinam features do MESMO componente."""
    parent = {f: f for f in features}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    by_group: Dict[str, List[int]] = {}
    for f in features:
        g = groups[f] if f < len(groups) else None
        if g and str(g).lower() not in {"general", "none"}:
            by_group.setdefault(str(g), []).append(f)
    for mem in by_group.values():
        for a, b in zip(mem, mem[1:]):
            parent[find(b)] = find(a)
    if rel is not None:
        rel = np.asarray(rel, dtype=float)
        for i, a in enumerate(features):
            for b in features[i + 1:]:
                if rel[a, b] >= threshold:
                    parent[find(b)] = find(a)
    comps: Dict[int, List[int]] = {}
    for f in features:
        comps.setdefault(find(f), []).append(f)
    return sorted((sorted(c) for c in comps.values()), key=lambda c: (-len(c), c))


def structure_summary(ctx) -> dict:
    if ctx is None:
        return {}
    rel = np.asarray(ctx.relatedness); n = len(ctx.orig_idx)
    sub = rel[:n, :n]; iu = np.triu_indices(n, 1)
    comps = candidate_components(list(ctx.groups), rel, list(range(n)))
    return {"relatedness_orig_pairs_distinct": sorted(set(np.round(sub[iu], 4).tolist())), "groups_orig": list(ctx.groups[:n]), "structure_signature": ctx.structure_signature(),
            "semantic_components_over_original_features": comps, "n_semantic_components": len(comps)}


def run(seeds: Sequence[int], arms=("ORIG", "OWL_true", "OWL_wrong", "GRP_true", "GRP_wrong")) -> dict:
    res: Dict[str, dict] = {a: {} for a in arms}
    struct: Dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for s in seeds:
            X, names = make_data(s)
            Xtest = np.random.default_rng(10_000 + s).normal(size=(3000, X.shape[1]))
            for a in arms:
                m, ctx, Xe = fit_arm(a, X, names, s, tmp)
                Xt = ctx.enrich(Xtest) if ctx is not None else Xtest
                nm = ctx.enriched_names if ctx is not None else names
                res[a][str(s)] = summarise(m, ctx, Xt, Xtest, nm, X.shape[1])
                if ctx is not None and s == seeds[0]:
                    struct[a] = structure_summary(ctx)
    agg = {}
    for a, d in res.items():
        v = list(d.values())
        agg[a] = {"n": len(v), "fidelity_mean": float(np.mean([x["fidelity_to_oracle"] for x in v])), "true_rule_recovered": int(sum(x["true_rule_recovered"] for x in v)),
                  "root_inside_true_group": int(sum(x["root_inside_true_group"] for x in v)), "decision_changed_mean": float(np.mean([x["n_decision_changed"] for x in v])),
                  "semantic_candidates_mean": float(np.mean([x["semantic_candidates_generated"] for x in v])), "internal_mean": float(np.mean([x["n_internal"] for x in v]))}
    # divergência por semente entre verdadeiro e permutado, por etapa
    div = {}
    for fam in ("OWL", "GRP"):
        if f"{fam}_true" in res and f"{fam}_wrong" in res:
            same_tree = [res[f"{fam}_true"][k]["rules_digest"] == res[f"{fam}_wrong"][k]["rules_digest"] for k in res[f"{fam}_true"]]
            div[fam] = {"identical_trees_true_vs_wrong": int(sum(same_tree)), "n": len(same_tree),
                        "same_root_features": int(sum(res[f"{fam}_true"][k]["root_features"] == res[f"{fam}_wrong"][k]["root_features"] for k in res[f"{fam}_true"]))}
    return {"seeds": list(seeds), "aggregate": agg, "divergence_true_vs_wrong": div, "structure": struct, "per_seed": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.seeds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({"aggregate": r["aggregate"], "divergence": r["divergence_true_vs_wrong"], "structure": r["structure"]}, indent=1))


if __name__ == "__main__":
    main()
