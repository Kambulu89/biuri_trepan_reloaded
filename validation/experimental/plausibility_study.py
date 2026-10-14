"""Estudo de plausibilidade/cobertura/custo de estratégias de geração de consultas (Doc. 24). EXPERIMENTAL; só DEV e sintéticos; nível de AMOSTRAS (não integra no TREPAN).

Por unidade: divisão treino/teste (o teste NÃO é usado); o treino divide-se em base B (de onde o amostrador aprende) e hold-out H (linhas reais que o amostrador nunca viu);
oráculo = MLP ajustado em B. Métricas: plausibilidade (AUC real-vs-sintético, razão de distâncias ao vizinho mais próximo), domínio/categorias, consistência semântica
(derivadas), fidelidade de um substituto treinado SÓ nas consultas e medida em H, cobertura de fronteira (fração de consultas junto à fronteira do oráculo e fração
de pontos de fronteira reais de H cobertos), diversidade (distância entre consultas, duplicados, cópias de B) e custo (s por 1000 consultas aceites; aceitação sob uma
restrição de caminho). Não otimiza a AUC: reporta o vetor completo.
    python -m validation.experimental.plausibility_study --dataset iris --seed 42 --out FILE.json
    python -m validation.experimental.plausibility_study --mixed --seed 3 --out FILE.json
"""
from __future__ import annotations

from validation.benchmark import run as _R  # noqa: F401  (ambiente numérico)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from validation.experimental.query_samplers import JointEmpiricalSampler, MarginalSampler, NeighborhoodSampler, infer_schema


def _margin(oracle, X):
    P = np.asarray(oracle.predict_proba(X))
    s = np.sort(P, axis=1)
    return s[:, -1] - s[:, -2]


def evaluate(strategies: dict, B, H, oracle, transform, orig_dim: int, n_q: int = 2000, seed: int = 0) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neighbors import NearestNeighbors
    from sklearn.tree import DecisionTreeClassifier
    sd = np.maximum(B[:, :orig_dim].std(axis=0), 1e-12)
    Zs = lambda A: (np.asarray(A)[:, :orig_dim] - B[:, :orig_dim].mean(0)) / sd
    nnB = NearestNeighbors(n_neighbors=2).fit(Zs(B))
    d_real_H = nnB.kneighbors(Zs(H), n_neighbors=1)[0][:, 0]
    d_BB = nnB.kneighbors(Zs(B), n_neighbors=2)[0][:, 1]
    enr = (lambda A: np.asarray(transform(np.asarray(A)[:, :orig_dim]))) if transform is not None else (lambda A: np.asarray(A))
    H_enr = enr(H)
    yH = np.asarray(oracle.predict(H))
    mH = _margin(oracle, H)
    ref_boundary = Zs(H)[mH < 0.3]
    rho = float(np.median(d_BB))
    schema = infer_schema(B[:, :orig_dim])
    # restrição de caminho: região esquerda do 1.º split de uma árvore rasa sobre B (rótulos do oráculo)
    tr = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(enr(B), np.asarray(oracle.predict(B)))
    feat, thr = int(tr.tree_.feature[0]), float(tr.tree_.threshold[0])

    class Cons:
        def accepts(self, A):
            return np.asarray(A)[:, feat] <= thr
    out = {}
    for name, smp in strategies.items():
        t0 = time.perf_counter()
        Q = smp.draw(n_q)
        sec_per_1000 = (time.perf_counter() - t0) / n_q * 1000
        t1 = time.perf_counter()
        try:
            smp.tries_, smp.accepted_ = 0, 0
            Qc = smp.draw(500, Cons())
            acc = smp.accepted_ / max(1, smp.tries_)
            cons_sec = (time.perf_counter() - t1) / 500 * 1000
        except Exception:
            acc, cons_sec = 0.0, float("nan")
        Qo = Q[:, :orig_dim]
        r = {"sec_per_1000_queries": float(sec_per_1000), "acceptance_under_path_constraint": float(acc), "sec_per_1000_under_constraint": float(cons_sec)}
        # plausibilidade
        k = min(len(H), n_q, 1000)
        for space, cols in (("orig", slice(0, orig_dim)), ("enr", slice(None))):
            Xs = np.vstack([H[:k] if space == "orig" else H_enr[:k], Q[:k, cols]])
            if space == "orig":
                Xs = np.vstack([H[:k][:, :orig_dim], Q[:k, :orig_dim]])
            y = np.r_[np.zeros(k), np.ones(min(k, len(Q)))]
            if space == "enr" and H_enr.shape[1] != Q.shape[1]:
                continue
            clf = RandomForestClassifier(n_estimators=60, min_samples_leaf=3, random_state=seed, n_jobs=1)
            pr = cross_val_predict(clf, Xs, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
            r[f"auc_real_vs_synth_{space}"] = float(roc_auc_score(y, pr))
        dQ = nnB.kneighbors(Zs(Q), n_neighbors=1)[0][:, 0]
        r["nn_distance_ratio_vs_holdout"] = float(np.median(dQ) / max(1e-12, np.median(d_real_H)))
        r["out_of_range_rate"] = float(((Qo < B[:, :orig_dim].min(0)) | (Qo > B[:, :orig_dim].max(0))).any(axis=1).mean())
        r["onehot_violation_rate"] = float(np.mean([np.any(Qo[:, g].sum(1) != 1.0) for g in schema.onehot_groups])) if schema.onehot_groups else 0.0
        r["onehot_violation_rate"] = float((np.any([Qo[:, g].sum(1) != 1.0 for g in schema.onehot_groups], axis=0)).mean()) if schema.onehot_groups else 0.0
        r["vocab_violation_rate"] = float((np.any([~np.isin(Qo[:, j], schema.vocab[j]) for j in schema.discrete], axis=0)).mean()) if schema.discrete else 0.0
        r["binary_violation_rate"] = float((np.any([~np.isin(Qo[:, j], [0.0, 1.0]) for j in schema.binary], axis=0)).mean()) if schema.binary else 0.0
        if transform is not None and Q.shape[1] > orig_dim:
            r["derived_violation_rate"] = float((np.abs(Q - enr(Q)).max(axis=1) > 1e-9).mean())
        # oráculo / fidelidade do substituto treinado só nas consultas
        yq = np.asarray(oracle.predict(Q[:, :orig_dim] if Q.shape[1] == orig_dim else Q[:, :orig_dim]))
        sur = DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=seed).fit(Q if Q.shape[1] == H_enr.shape[1] else enr(Q), yq)
        r["surrogate_fidelity_on_holdout"] = float(np.mean(sur.predict(H_enr) == yH))
        r["query_label_entropy"] = float(-(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(np.searchsorted(np.unique(yH), yq), minlength=len(np.unique(yH))) / len(yq)))
        # fronteira
        mQ = _margin(oracle, Q[:, :orig_dim])
        r["near_boundary_fraction_margin<0.2"] = float((mQ < 0.2).mean())
        if len(ref_boundary):
            nnQ = NearestNeighbors(n_neighbors=1).fit(Zs(Q))
            r["boundary_points_covered_fraction"] = float((nnQ.kneighbors(ref_boundary, n_neighbors=1)[0][:, 0] <= rho).mean())
        # diversidade
        nnq = NearestNeighbors(n_neighbors=2).fit(Zs(Q[: min(len(Q), 1000)]))
        r["diversity_nn_distance_ratio"] = float(np.median(nnq.kneighbors(Zs(Q[: min(len(Q), 1000)]), n_neighbors=2)[0][:, 1]) / max(1e-12, np.median(d_BB)))
        cont = schema.continuous
        r["exact_copy_of_base_rows"] = float(np.mean([tuple(np.round(q[cont], 9)) in {tuple(np.round(b[cont], 9)) for b in B} for q in Qo[:300]])) if cont else 0.0
        out[name] = r
    return out


def make_strategies(B, transform, seed):
    return {"A_marginal_historical": MarginalSampler(B, transform=transform, on_enriched=True, random_state=seed),
            "As_marginal_joint_groups": MarginalSampler(B, transform=transform, joint_groups=True, random_state=seed),
            "B_marginal_coherent": MarginalSampler(B, transform=transform, random_state=seed),
            "C_joint_empirical": JointEmpiricalSampler(B, transform=transform, random_state=seed),
            "D_neighborhood": NeighborhoodSampler(B, transform=transform, random_state=seed)}


def run_dev(dataset: str, seed: int) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from core.benchmark.semantic import OwlSemanticProvider
    from validation.benchmark import datasets as ds_mod
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    Xtr, _Xte, ytr, _yte = train_test_split(ds.X, ds.y, test_size=0.25, stratify=ds.y, random_state=seed)      # o teste não é usado
    B, H, yB, _ = train_test_split(Xtr, ytr, test_size=0.5, stratify=ytr, random_state=seed)
    oracle = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=600, random_state=seed)).fit(B, yB)
    ctx = OwlSemanticProvider(str(spec.ontology_path)).build(Xtr, ds.feature_names, seed)
    transform = ctx.enrich if ctx.available else None
    res = {"dataset": dataset, "seed": seed, "orig_dim": int(B.shape[1]), "n_derived": len(ctx.onto_idx) if ctx.available else 0, "split": "train only (B/H halves)"}
    res["strategies"] = evaluate(make_strategies(B, transform, seed), B, H, oracle, transform, B.shape[1], seed=seed)
    return res


def run_mixed(seed: int) -> dict:
    from sklearn.model_selection import train_test_split
    from validation.experimental.mixed_data import make_mixed
    X, y, names, T, oracle = make_mixed(seed)
    B, H = train_test_split(X, test_size=0.4, random_state=seed)

    class O:
        classes_ = oracle.classes_
        def predict(self, A): return oracle.predict(A)
        def predict_proba(self, A):
            # oráculo sintético sem probabilidades: margem aproximada pela distância ao limiar do score (monótona), só para cobertura de fronteira
            A = np.asarray(A, float)[:, :14]
            s = 1.2 * A[:, 0] - 0.8 * A[:, 2] + (A[:, 4] == 1) * 1.0 - (A[:, 8] == 1) * 1.0 + 0.6 * A[:, 10] + 0.3 * A[:, 12]
            q = np.quantile(1.2 * X[:, 0] - 0.8 * X[:, 2] + (X[:, 4] == 1) * 1.0 - (X[:, 8] == 1) * 1.0 + 0.6 * X[:, 10] + 0.3 * X[:, 12], [0.33, 0.66])
            d = np.abs(s[:, None] - q[None, :]).min(axis=1)
            conf = 1 / (1 + np.exp(-3 * d))
            lab = oracle.predict(A)
            P = np.full((len(A), 3), 0.0)
            P[np.arange(len(A)), lab] = conf
            P[np.arange(len(A)), (lab + 1) % 3] = 1 - conf
            return P
    res = {"dataset": "synthetic_mixed", "seed": seed, "orig_dim": 14, "n_derived": 2}
    res["strategies"] = evaluate(make_strategies(B, T, seed), B, H, O(), T, 14, seed=seed)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--mixed", action="store_true"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run_mixed(a.seed) if a.mixed else run_dev(a.dataset, a.seed)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
