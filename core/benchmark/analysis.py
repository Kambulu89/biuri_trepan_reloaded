"""Análise estatística pareada, controlo negativo, ablação e níveis de evidência.

Os contrastes são DECLARADOS A PRIORI (``DEFAULT_CONTRASTS``) e não dependem dos resultados.
Nada aqui impõe uma hierarquia Reloaded > Original > C4.5: os números decidem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from core.benchmark import stats as st

PRIMARY_METRICS = ("fidelity_to_oracle", "accuracy_real_labels")
AGG_METRICS = (
    "accuracy_real_labels", "balanced_accuracy_real_labels", "precision_macro_real_labels", "recall_macro_real_labels",
    "macro_f1_real_labels", "weighted_f1_real_labels", "minority_recall_real_labels", "fidelity_to_oracle",
    "node_count", "internal_nodes", "leaf_count", "depth", "average_leaf_depth", "average_rule_length", "features_used",
    "m_of_n_count", "semantic_split_count", "semantic_decision_changed_count", "membership_queries", "mlp_training_time", "tree_training_time", "query_time",
    "semantic_processing_time", "reasoner_time", "disagreement_rate_to_oracle", "fidelity_train", "rule_count",
    "average_rule_literals", "ontology_usage_rate",
)


@dataclass(frozen=True)
class Contrast:
    name: str
    arm_a: str
    arm_b: str
    metrics: Sequence[str]
    question: str
    group: str = "main"                 # main | negative_control | ablation | sensitivity | sanity
    semantic_claim: bool = False        # exige controlo negativo para evidência forte


DEFAULT_CONTRASTS: List[Contrast] = [
    # Comparações causais (arm_a - arm_b); o sinal NUNCA é assumido.
    Contrast("architectural_gain", "reloaded_core", "trepan_original", ("fidelity_to_oracle", "accuracy_real_labels", "node_count"),
             "C vs D: ganho arquitectural (mecanismos Reloaded não semânticos, sem ontologia). 0 se D == Original", group="main"),
    Contrast("incremental_ontology", "reloaded_owl_full", "reloaded_core", ("fidelity_to_oracle", "accuracy_real_labels", "node_count"),
             "D vs E: ganho incremental da ontologia sobre a mesma arquitectura", group="main", semantic_claim=True),
    Contrast("NEGATIVE_CONTROL_real_vs_shuffled", "reloaded_owl_full", "reloaded_owl_shuffled", ("fidelity_to_oracle", "accuracy_real_labels", "node_count"),
             "E vs F — CONTROLO NEGATIVO: OWL real > OWL permutada? (sem isto não se atribui o ganho à ontologia)", group="negative_control"),
    Contrast("total_reloaded_vs_original", "reloaded_owl_full", "trepan_original", ("fidelity_to_oracle", "accuracy_real_labels", "node_count"),
             "C vs E: ganho total do Reloaded+OWL face ao TREPAN Original (mesmo FrozenOracle)", group="main", semantic_claim=True),
    Contrast("trepan_original_vs_c45", "trepan_original", "c45", ("accuracy_real_labels", "balanced_accuracy_real_labels", "fidelity_to_oracle", "node_count"),
             "C − B: TREPAN Original vs C4.5 canónico (fidelity de ambos medida contra o mesmo FrozenOracle)", group="baseline"),
    Contrast("reloaded_owl_vs_c45", "reloaded_owl_full", "c45", ("accuracy_real_labels", "balanced_accuracy_real_labels", "fidelity_to_oracle", "node_count"),
             "E − B: sistema completo (Reloaded + OWL) vs C4.5 canónico", group="baseline"),
    # Ablações isoladas (mesma arquitectura D como referência).
    Contrast("semantic_score_effect", "reloaded_semantic_score", "reloaded_core", ("fidelity_to_oracle", "accuracy_real_labels", "node_count"),
             "Ablação: score semântico (sem features OWL) vs D", group="ablation", semantic_claim=True),
    Contrast("owl_features_effect", "reloaded_owl_features", "reloaded_core", ("fidelity_to_oracle", "accuracy_real_labels", "node_count"),
             "Ablação: features OWL (sem score semântico) vs D", group="ablation", semantic_claim=True),
    Contrast("ablation_original_no_mofn", "trepan_original_no_mofn", "trepan_original", ("fidelity_to_oracle", "node_count"),
             "Ablação: Original sem m-of-n", group="ablation"),
    Contrast("ablation_no_mofn", "reloaded_owl_full_no_mofn", "reloaded_owl_full", ("fidelity_to_oracle", "node_count"),
             "Ablação: Reloaded OWL sem m-of-n", group="ablation"),
    Contrast("ablation_no_active_queries", "reloaded_owl_full_no_active_queries", "reloaded_owl_full", ("fidelity_to_oracle", "node_count"),
             "Ablação: sem active queries", group="ablation"),
    Contrast("ablation_no_error_focus", "reloaded_owl_full_no_error_focus", "reloaded_owl_full", ("fidelity_to_oracle", "node_count"),
             "Ablação: sem refinamento focado no erro", group="ablation"),
    # Experiência separada (oráculo diferente): fora dos braços A–F.
    Contrast("owl_in_mlp", "mlp_ontological", "mlp_original", ("accuracy_real_labels", "balanced_accuracy_real_labels", "macro_f1_real_labels"),
             "different_oracle_experiment: o MLP Ontológico melhora o MLP Original?", group="different_oracle_experiment", semantic_claim=True),
    Contrast("end_to_end_vs_original", "reloaded_e2e", "trepan_original", ("accuracy_real_labels", "fidelity_to_oracle", "node_count"),
             "different_oracle_experiment: pipeline com oráculo ontológico vs Original", group="different_oracle_experiment"),
]


def aggregate(df: pd.DataFrame, *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Uma linha por (dataset, arm, oracle, métrica): n, mean, std, median, min, max, IC95% bootstrap."""
    out = []
    df = df.copy()
    df["oracle_name"] = df["oracle_name"].fillna("n/a")
    for (ds, arm, oracle), g in df.groupby(["dataset", "arm", "oracle_name"], sort=False):
        for m in AGG_METRICS:
            if m not in g or g[m].isna().all():
                continue
            vals = g[m].astype(float).to_numpy()
            d = st.describe(vals)
            lo, hi = st.bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed)
            out.append(dict(dataset=ds, arm=arm, oracle_name=oracle, metric=m, n=d["n"], mean=d["mean"], std=d["std"],
                            median=d["median"], min=d["min"], max=d["max"], ci_low=lo, ci_high=hi,
                            ci_method=f"percentile bootstrap of the mean ({n_boot})"))
    return pd.DataFrame(out)


def nadeau_bengio_p(diff: np.ndarray, n_train: int, n_test: int) -> float:
    """t reamostrado corrigido (Nadeau & Bengio) para folds/repetições NÃO independentes."""
    d = np.asarray(diff, float)
    k = d.size
    if k < 3 or d.std(ddof=1) == 0:
        return float("nan")
    var = (1.0 / k + n_test / n_train) * d.var(ddof=1)
    t = d.mean() / math.sqrt(var)
    return float(2 * sps.t.sf(abs(t), k - 1))


def run_contrasts(df: pd.DataFrame, *, scheme: str = "holdout", contrasts: Sequence[Contrast] = DEFAULT_CONTRASTS,
                  n_boot: int = 10000, alpha: float = 0.05, policy: st.EvidencePolicy = st.EvidencePolicy(), seed: int = 0) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ds, gd in df.groupby("dataset", sort=False):
        for c in contrasts:
            A = gd[gd["arm"] == c.arm_a].set_index("split_id")
            B = gd[gd["arm"] == c.arm_b].set_index("split_id")
            common = [i for i in A.index if i in B.index]
            if not common:
                rows.append(dict(dataset=ds, contrast=c.name, group=c.group, question=c.question, arm_a=c.arm_a, arm_b=c.arm_b,
                                 metric="-", status="not_run", reason="um dos braços não foi executado (ver skipped)"))
                continue
            for m in c.metrics:
                if m not in A or A.loc[common, m].isna().all() or B.loc[common, m].isna().all():
                    continue
                oa, ob = set(A.loc[common, "oracle_name"].fillna("n/a")), set(B.loc[common, "oracle_name"].fillna("n/a"))
                comparable, reason = True, ""
                if m == "fidelity_to_oracle" and oa != ob:
                    comparable, reason = False, f"oráculos diferentes ({sorted(oa)} vs {sorted(ob)}): fidelity não comparável"
                if not comparable:
                    rows.append(dict(dataset=ds, contrast=c.name, group=c.group, question=c.question, arm_a=c.arm_a, arm_b=c.arm_b,
                                     metric=m, status="not_comparable", reason=reason, oracle_a=";".join(sorted(oa)), oracle_b=";".join(sorted(ob))))
                    continue
                a = A.loc[common, m].astype(float).to_numpy()
                b = B.loc[common, m].astype(float).to_numpy()
                comp = st.paired_comparison(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
                n_tr = float(A.loc[common, "n_train"].mean()); n_te = float(A.loc[common, "n_test"].min())
                row = dict(dataset=ds, contrast=c.name, group=c.group, question=c.question, arm_a=c.arm_a, arm_b=c.arm_b, metric=m,
                           status="ok", reason="", oracle_a=";".join(sorted(oa)), oracle_b=";".join(sorted(ob)),
                           semantic_claim=c.semantic_claim, min_test_samples=int(n_te), scheme=scheme, **comp)
                if scheme != "holdout":
                    row["nadeau_bengio_p"] = nadeau_bengio_p(a - b, n_tr, n_te)
                rows.append(row)
    res = pd.DataFrame(rows)
    if res.empty or "wilcoxon_p" not in res:
        return res
    res["p_holm"] = np.nan
    res["p_bh"] = np.nan
    for ds, g in res[res["status"] == "ok"].groupby("dataset"):
        prim = g[g["metric"].isin(PRIMARY_METRICS)].index          # família primária: Holm
        res.loc[prim, "p_holm"] = st.holm(res.loc[prim, "wilcoxon_p"].tolist())
        res.loc[g.index, "p_bh"] = st.benjamini_hochberg(res.loc[g.index, "wilcoxon_p"].tolist())   # suplementar: todas
    # evidência (com controlo negativo)
    nc = negative_control_verdicts(res)
    levels, why = [], []
    for _, r in res.iterrows():
        if r["status"] != "ok":
            levels.append(None); why.append(""); continue
        passed = None
        needs = bool(r.get("semantic_claim", False))
        if needs:
            passed = nc.get((r["dataset"], r["metric"]), {}).get("passed")
        p_adj = r["p_holm"] if r["metric"] in PRIMARY_METRICS else float("nan")
        lvl, reasons = st.evidence_level(r.to_dict(), p_adjusted=p_adj, min_test_samples=int(r["min_test_samples"]),
                                         needs_negative_control=needs, negative_control_passed=passed, policy=policy)
        if int(r["ties"]) == int(r["n_pairs"]) and int(r["n_pairs"]) > 0:
            lvl, reasons = "IDENTICAL_RESULTS", ["todas as diferenças são 0 (resultados idênticos par a par)"]
        if r.get("scheme", "holdout") != "holdout" and lvl == st.SUPPORTED:
            lvl, reasons = st.INDICATIVE, reasons + ["folds repetidos não independentes: confirmar com Nadeau-Bengio / mais datasets"]
        levels.append(lvl); why.append("; ".join(reasons))
    res["evidence_level"], res["evidence_reasons"] = levels, why
    return res


def negative_control_verdicts(res: pd.DataFrame) -> Dict[tuple, Dict[str, Any]]:
    """real > shuffled de forma robusta? Por (dataset, métrica)."""
    out: Dict[tuple, Dict[str, Any]] = {}
    nc = res[(res.get("contrast") == "NEGATIVE_CONTROL_real_vs_shuffled") & (res.get("status") == "ok")] if "contrast" in res else res.iloc[0:0]
    for _, r in nc.iterrows():
        lo, hi, md = r["ci_low"], r["ci_high"], r["mean_diff"]
        if int(r["n_pairs"]) < st.EvidencePolicy().min_pairs_verdict or not (isinstance(lo, float) and not math.isnan(lo)):
            verdict, passed = "INSUFFICIENT_DATA", False
        elif lo > 0:
            verdict, passed = "REAL_GT_SHUFFLED", True
        elif hi < 0:
            verdict, passed = "SHUFFLED_GT_REAL", False
        else:
            verdict, passed = "REAL_APPROX_SHUFFLED", False
        out[(r["dataset"], r["metric"])] = dict(verdict=verdict, passed=passed, mean_diff=md, ci_low=lo, ci_high=hi, n_pairs=int(r["n_pairs"]))
    return out


def semantic_attribution(res: pd.DataFrame, metric: str = "fidelity_to_oracle") -> List[Dict[str, Any]]:
    """Decompõe o ganho do Reloaded: arquitectura (D-C) | ontologia incremental (E-D) | total (E-C) | real vs permutada (E-F)."""
    out = []
    nc = negative_control_verdicts(res)
    for ds in res["dataset"].unique():
        g = res[(res["dataset"] == ds) & (res["metric"] == metric) & (res["status"] == "ok")].set_index("contrast")
        def md(name):
            return float(g.loc[name, "mean_diff"]) if name in g.index else None
        entry = dict(dataset=ds, metric=metric, architectural_gain=md("architectural_gain"),
                     owl_features_effect=md("owl_features_effect"), semantic_score_effect=md("semantic_score_effect"),
                     incremental_ontology=md("incremental_ontology"), total_vs_original=md("total_reloaded_vs_original"),
                     real_vs_shuffled=md("NEGATIVE_CONTROL_real_vs_shuffled"),
                     negative_control=nc.get((ds, metric), {}).get("verdict", "NOT_RUN"))
        v = entry["negative_control"]
        entry["attribution"] = ("ganho atribuível à semântica real (real > shuffled)" if v == "REAL_GT_SHUFFLED" else
                                "NÃO atribuível à ontologia: real ≈ shuffled (efeito de arquitectura/features/ruído)" if v == "REAL_APPROX_SHUFFLED" else
                                "a OWL permutada foi MELHOR que a real: sem evidência de contribuição semântica" if v == "SHUFFLED_GT_REAL" else
                                "controlo negativo não executado/insuficiente: não atribuir à ontologia")
        out.append(entry)
    return out


def cross_dataset(frames: Dict[str, pd.DataFrame], res_all: pd.DataFrame, metric: str = "fidelity_to_oracle") -> Dict[str, Any]:
    """Resumo entre datasets: por contraste, diferença média por dataset + Wilcoxon entre datasets (n datasets)."""
    out: Dict[str, Any] = {"n_datasets": len(frames), "contrasts": []}
    ok = res_all[(res_all["status"] == "ok") & (res_all["metric"] == metric)]
    for name, g in ok.groupby("contrast"):
        diffs = g["mean_diff"].astype(float).to_numpy()
        entry = dict(contrast=name, n_datasets=int(len(diffs)), per_dataset=dict(zip(g["dataset"], diffs.round(5).tolist())))
        if len(diffs) >= 5 and np.any(diffs != 0):
            entry["wilcoxon_p_across_datasets"] = float(sps.wilcoxon(diffs).pvalue)
        else:
            entry["wilcoxon_p_across_datasets"] = None
            entry["note"] = "menos de 5 datasets: sem teste entre datasets (apenas descritivo)"
        out["contrasts"].append(entry)
    # Friedman entre braços principais (blocos = datasets)
    arms = ["trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled"]
    mat = []
    for ds, f in frames.items():
        means = f.groupby("arm")[metric].mean()
        if all(a in means.index for a in arms):
            mat.append([means[a] for a in arms])
    out["friedman"] = st.friedman_posthoc(np.array(mat), arms) if len(mat) >= 3 else {"valid": False, "reason": "Friedman exige >=3 datasets completos"}
    return out
