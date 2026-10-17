"""B3-Q (Docs. 45–48): auditoria da qualidade das ontologias DEV. Só ``validation/``; só leitura dos OWL congelados, dos resultados M29/sampler_c já existentes e dos dados de TREINO
(partição ``holdout`` seed 42, reconstruída por ``make_splits`` e verificada contra o hash gravado). Nenhuma execução de benchmark, nenhum teste usado, nenhuma ontologia alterada.

Saídas (``--out``): (1) censo de axiomas classificados por função (ANOTAÇÃO, DECLARAÇÃO, TIPAGEM, AGRUPAMENTO, TAXONOMIA, DISJUNÇÃO, RAIZ, SUBSTANTIVO_RELACIONAL), (2) factos compilados pelo
registo B3.1, (3) utilização medida nos resultados existentes (ontologia verdadeira vs controlos permutado/aleatório), (4) diagnóstico de suporte nos dados de treino (sinergia por pares
dentro vs fora dos grupos, teste de permutação das atribuições). O ponto (4) mede RELEVÂNCIA PARA A TAREFA e é separado da qualidade intrínseca da ontologia."""
from __future__ import annotations

import argparse
import glob
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#", "owl": "http://www.w3.org/2002/07/owl#"}
Q = lambda p, t: "{%s}%s" % (NS[p], t)
THING = "http://www.w3.org/2002/07/owl#Thing"
SUBSTANTIVE_TAGS = ("equivalentClass", "equivalentProperty", "subPropertyOf", "inverseOf", "propertyChainAxiom", "someValuesFrom", "allValuesFrom", "minCardinality", "maxCardinality",
                    "cardinality", "hasValue", "onProperty", "ObjectProperty", "TransitiveProperty", "FunctionalProperty", "SymmetricProperty", "DatatypeRestriction", "withRestrictions",
                    "unionOf", "intersectionOf", "complementOf", "oneOf", "NamedIndividual")


def census(owl: Path) -> Dict[str, Any]:
    """Classifica cada axioma/asserção por FUNÇÃO. Não decide verdade de domínio (isso é revisão humana)."""
    root = ET.parse(owl).getroot()
    c: Counter = Counter()
    groups: Dict[str, List[str]] = defaultdict(list)
    extra: Counter = Counter()
    for el in root:
        tag = el.tag.split("}")[1]
        if tag == "Ontology":
            c["ANOTACAO"] += sum(1 for ch in el if ch.tag.split("}")[1] in ("comment", "label"))
        elif tag in ("Class", "DatatypeProperty", "AnnotationProperty", "ObjectProperty"):
            c["DECLARACAO"] += 1
            for ch in el:
                t = ch.tag.split("}")[1]
                if t in ("label", "comment"):
                    c["ANOTACAO"] += 1
                elif t == "subClassOf":
                    tgt = ch.get(Q("rdf", "resource"))
                    if tgt is None:                                            # restrição / expressão complexa
                        c["SUBSTANTIVO_RELACIONAL"] += 1
                    elif tgt == THING:
                        c["RAIZ"] += 1
                    else:
                        c["TAXONOMIA"] += 1
                elif t == "disjointWith":
                    c["DISJUNCAO"] += 1
                elif t == "domain":
                    c["AGRUPAMENTO"] += 1
                    groups[ch.get(Q("rdf", "resource")).split("#")[-1]].append(el.get(Q("rdf", "about")).split("#")[-1])
                elif t == "range":
                    c["TIPAGEM"] += 1
                elif t in SUBSTANTIVE_TAGS:
                    c["SUBSTANTIVO_RELACIONAL"] += 1
                else:
                    c["ANOTACAO_PERSONALIZADA"] += 1
                    extra[t] += 1
            if tag == "ObjectProperty":
                c["SUBSTANTIVO_RELACIONAL"] += 1
        elif tag == "AllDisjointClasses":
            c["DISJUNCAO"] += 1
        elif tag in SUBSTANTIVE_TAGS:
            c["SUBSTANTIVO_RELACIONAL"] += 1
    return {"by_function": dict(c), "custom_annotation_properties_used": dict(extra), "groups": {k: sorted(v) for k, v in groups.items()}}


def usage_from_existing_results() -> Dict[str, Any]:
    """Utilização medida (leitura dos brutos existentes): ontologia verdadeira (sc_D, reloaded_owl_full) vs controlos permutado (M29 P) e aleatório (M29 D)."""
    base = ROOT / "docs/evidence/b1b4"
    keys = ["semantic_features_generated", "semantic_features_used", "ontology_influenced_splits", "internal_nodes", "m_of_n_count", "semantic_decision_changed_count", "mapped_feature_count"]
    agg: Dict[Any, List[dict]] = defaultdict(list)
    for f in glob.glob(str(base / "sampler_c/dev_numeric/raw/*/seed*/sc_D_r*.json")):
        d = json.load(open(f))
        agg[(d["dataset"], "TRUE_ONTOLOGY (sc_D)")].append(d["row"])
    for f in glob.glob(str(base / "m29_campaign/*/raw/*/seed*/m29_[DP]_k*.json")):
        d = json.load(open(f))
        agg[(d["dataset"], "CONTROL_RANDOM (M29 D)" if "_D_" in d["arm"] else "CONTROL_PERMUTED (M29 P)")].append(d["row"])
    out: Dict[str, Any] = {}
    for (ds, arm), rows in sorted(agg.items()):
        out.setdefault(ds, {})[arm] = {"n_units": len(rows), **{k: round(float(np.mean([float(r.get(k) or 0) for r in rows])), 2) for k in keys},
                                       "fraction_internal_nodes_influenced": round(float(np.mean([float(r.get("ontology_influenced_splits") or 0) / max(float(r.get("internal_nodes") or 1), 1) for r in rows])), 3),
                                       "ontology_category_counts": dict(Counter(str(r.get("ontology_category")) for r in rows))}
    return out


def data_support(ds_id: str, seed: int = 42, n_perm: int = 2000) -> Dict[str, Any]:
    """Relevância para a tarefa (TREINO apenas): sinergia por pares = acc_CV3(árvore prof. 3 com o par) − max(acc_CV3 de cada atributo). Compara pares do MESMO grupo da ontologia com
    pares de grupos diferentes; valor-p por permutação das atribuições (tamanhos de grupo preservados). NÃO é critério de qualidade da ontologia."""
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from core.benchmark.splits import make_splits
    from validation.benchmark import datasets as D
    spec = D.REGISTRY[ds_id]
    ds, _, _ = spec.load()
    X, y = np.asarray(ds.X, dtype=float), np.asarray(ds.y)
    sp = make_splits(y, scheme="holdout", seeds=[seed])[0]
    Xt, yt = X[sp.train_idx], y[sp.train_idx]
    rec = json.load(open(ROOT / f"docs/evidence/b1b4/m29_campaign/main/raw/{ds_id}/seed{seed}/split.json"))["splits"][0]["split_hash"]
    hash_ok = bool(rec == sp.split_hash)
    names = [str(n) for n in ds.feature_names]
    own = census(Path(spec.ontology_path))["groups"]
    from validation.benchmark.b31_facts import compile_registry
    reg = compile_registry(str(spec.ontology_path), names, run_reasoner=False)
    grp = {}
    for g, props in own.items():
        for p_ in props:
            for n in names:
                if reg.feature_entity.get(n) == p_:
                    grp[n] = g
    clf = lambda: DecisionTreeClassifier(max_depth=3, random_state=0)
    cv = lambda cols: float(np.mean(cross_val_score(clf(), Xt[:, cols], yt, cv=3)))
    single = [cv([j]) for j in range(len(names))]
    pairs, syn, acc = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = cv([i, j])
            pairs.append((i, j)); acc.append(a); syn.append(a - max(single[i], single[j]))
    syn, acc = np.array(syn), np.array(acc)
    lab = np.array([grp.get(n, f"_none_{n}") for n in names])
    same = np.array([lab[i] == lab[j] for i, j in pairs])
    rng = np.random.default_rng(3434)
    obs_syn, obs_acc = float(syn[same].mean() - syn[~same].mean()), float(acc[same].mean() - acc[~same].mean())
    ps, pa = 0, 0
    for _ in range(n_perm):
        pl = rng.permutation(lab)
        sm = np.array([pl[i] == pl[j] for i, j in pairs])
        ps += (syn[sm].mean() - syn[~sm].mean()) >= obs_syn
        pa += (acc[sm].mean() - acc[~sm].mean()) >= obs_acc
    return {"dataset": ds_id, "train_n": int(len(yt)), "split_hash_ok": hash_ok, "split_hash": sp.split_hash, "n_pairs": len(pairs), "n_same_group_pairs": int(same.sum()), "n_mapped_features": int(sum(1 for n in names if n in grp)),
            "mean_synergy_same": float(syn[same].mean()) if same.any() else None, "mean_synergy_cross": float(syn[~same].mean()), "diff_synergy": obs_syn, "p_perm_synergy": (ps + 1) / (n_perm + 1),
            "mean_pair_acc_same": float(acc[same].mean()) if same.any() else None, "mean_pair_acc_cross": float(acc[~same].mean()), "diff_pair_acc": obs_acc, "p_perm_pair_acc": (pa + 1) / (n_perm + 1),
            "best_single_acc": float(max(single)), "note": "diagnóstico de dados de TREINO; mede suporte estatístico da partição, não qualidade da ontologia"}


def registry_summary(ds_id: str) -> Dict[str, Any]:
    from validation.benchmark import datasets as D
    from validation.benchmark.b31_facts import compile_registry
    spec = D.REGISTRY[ds_id]
    ds, _, _ = spec.load()
    reg = compile_registry(str(spec.ontology_path), [str(n) for n in ds.feature_names], run_reasoner=True)
    s = reg.summary()
    s["n_usable"] = len(reg.usable())
    s["n_hard_constraints"] = sum(1 for f in reg.facts if f.permitted_use == "HARD_CONSTRAINT")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from validation.benchmark import datasets as D
    res: Dict[str, Any] = {"usage_existing_results": usage_from_existing_results(), "ontologies": {}}
    for ds_id in ("iris", "wine", "breast_cancer", "digits"):
        spec = D.REGISTRY[ds_id]
        res["ontologies"][ds_id] = {"owl": str(Path(spec.ontology_path).relative_to(ROOT)), "census": census(Path(spec.ontology_path)), "fact_registry": registry_summary(ds_id), "data_support": data_support(ds_id)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
