"""Tabelas e relatórios reconstruídos EXCLUSIVAMENTE a partir dos resultados brutos (``raw/``).

Ordem do relatório principal (como exigido): 1) invariantes do protocolo; 2) tabelas individuais por dataset×braço;
3) contrastes causais por dataset; 4) tabela da ontologia (D/E/F + categoria do quality gate); 5) só no fim, o ranking agregado.
Os datasets sintéticos aparecem numa secção SEPARADA de validação controlada e nunca entram no ranking nem em contagens de datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.benchmark import analysis as an
from validation.benchmark import datasets as ds_mod
from validation.benchmark import invariants as inv
from validation.benchmark import manifest as mf
from validation.benchmark.raw_store import load_raw

ARM_ORDER = ["mlp_original", "c45", "trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled"]
ARM_LABEL = {"mlp_original": "A MLP/FrozenOracle", "c45": "B C4.5", "trepan_original": "C TREPAN Original", "reloaded_core": "D Reloaded sem OWL",
             "reloaded_owl_full": "E Reloaded + OWL", "reloaded_owl_shuffled": "F Reloaded + OWL permutada"}
CAUSAL = {"architectural_gain": "D − C (ganho arquitetural)", "incremental_ontology": "E − D (ganho ontológico)",
          "NEGATIVE_CONTROL_real_vs_shuffled": "E − F (real vs permutada)", "total_reloaded_vs_original": "E − C (ganho total)",
          "trepan_original_vs_c45": "C − B (Original vs C4.5)", "reloaded_owl_vs_c45": "E − B (sistema completo vs C4.5)"}
CONTRASTS = [c for c in an.DEFAULT_CONTRASTS if c.name in CAUSAL]
INDIVIDUAL_COLS = [("accuracy_real_labels", "Accuracy"), ("fidelity_to_oracle", "Fidelity"), ("macro_f1_real_labels", "Macro-F1"),
                   ("node_count", "Nós"), ("leaf_count", "Folhas"), ("depth", "Profundidade"), ("features_used", "Features"),
                   ("membership_queries", "Queries"), ("tree_training_time", "Tempo (s)")]


def _fmt(m, s) -> str:
    return "—" if pd.isna(m) else (f"{m:.4f} ± {s:.4f}" if not pd.isna(s) else f"{m:.4f}")


def individual_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, arm), g in df.groupby(["dataset", "arm"], sort=False):
        row = {"Dataset": ds, "Arm": ARM_LABEL.get(arm, arm), "n_seeds": int(g["seed"].nunique()), "_arm": arm}
        for col, label in INDIVIDUAL_COLS:
            vals = g[col].astype(float) if col in g else pd.Series(dtype=float)
            if col == "tree_training_time" and arm == "mlp_original":
                vals = g["mlp_training_time"].astype(float)
            row[label] = _fmt(vals.mean(), vals.std(ddof=1)) if vals.notna().any() else "—"
        out.append(row)
    t = pd.DataFrame(out)
    t["_o"] = t["_arm"].map({a: i for i, a in enumerate(ARM_ORDER)})
    return t.sort_values(["Dataset", "_o"]).drop(columns=["_arm", "_o"]).reset_index(drop=True)


def contrast_table(df: pd.DataFrame, *, n_boot: int = 10000) -> pd.DataFrame:
    res = an.run_contrasts(df, scheme="holdout", contrasts=CONTRASTS, n_boot=n_boot)
    if res.empty:
        return res
    rows = []
    for (ds, name), g in res[res["status"] == "ok"].groupby(["dataset", "contrast"], sort=False):
        m = g.set_index("metric")
        fid = m.loc["fidelity_to_oracle"] if "fidelity_to_oracle" in m.index else None
        acc = m.loc["accuracy_real_labels"] if "accuracy_real_labels" in m.index else None
        cx = m.loc["node_count"] if "node_count" in m.index else None
        main = fid if fid is not None else acc
        direction = ("sem pares" if main is None else "zero (idêntico par a par)" if int(main["ties"]) == int(main["n_pairs"]) else
                     "positivo" if main["ci_low"] > 0 else "negativo" if main["ci_high"] < 0 else
                     ("positivo (IC inclui 0)" if main["mean_diff"] > 0 else "negativo (IC inclui 0)" if main["mean_diff"] < 0 else "nulo"))
        rows.append({"Dataset": ds, "Contraste": CAUSAL[name], "contrast": name, "n_seeds": int(main["n_pairs"]) if main is not None else 0,
                     "Δ Fidelity": None if fid is None else fid["mean_diff"], "Δ Fidelity (mediana)": None if fid is None else fid["median_diff"],
                     "IC95% Fidelity": None if fid is None else f"[{fid['ci_low']:.4f}, {fid['ci_high']:.4f}]",
                     "Δ Accuracy": None if acc is None else acc["mean_diff"],
                     "IC95% Accuracy": None if acc is None else f"[{acc['ci_low']:.4f}, {acc['ci_high']:.4f}]",
                     "Δ Complexidade (nós)": None if cx is None else cx["mean_diff"],
                     "Effect size (rank-biserial)": None if main is None else main["rank_biserial"], "Cliff's δ": None if main is None else main["cliffs_delta"],
                     "p Wilcoxon": None if main is None else main["wilcoxon_p"], "p ajustado (Holm)": None if main is None else main.get("p_holm"),
                     "Direção": direction, "Evidência": None if main is None else main["evidence_level"],
                     "Nota": "não significativo ≠ igual: com poucos pares o teste tem pouco poder"})
    return pd.DataFrame(rows)


def ontology_table(df: pd.DataFrame, *, n_boot: int = 10000) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", sort=False):
        piv = g.pivot_table(index="seed", columns="arm", values="fidelity_to_oracle", aggfunc="first")
        e = g[g["arm"] == "reloaded_owl_full"].set_index("seed")
        cats = e["ontology_category"].value_counts().to_dict() if len(e) else {}
        n_eff = int(cats.get("ontology_effectively_used", 0))
        n_all = int(len(e))
        d_, e_, f_ = (piv[a] if a in piv else pd.Series(dtype=float) for a in ("reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled"))
        eff_seeds = e.index[e["ontology_category"] == "ontology_effectively_used"].tolist() if len(e) else []
        row = {"Dataset": ds, "D sem OWL": _fmt(d_.mean(), d_.std(ddof=1)), "E +OWL": _fmt(e_.mean(), e_.std(ddof=1)),
               "F permutada": _fmt(f_.mean(), f_.std(ddof=1)), "ontology_effectively_used": f"{n_eff}/{n_all} seeds",
               "categorias (E)": json.dumps(cats), "Δ(E−D) todas as seeds": (e_ - d_).mean(), "Δ(E−F) todas as seeds": (e_ - f_).mean()}
        if n_eff:
            sub = g[g["seed"].isin(eff_seeds)]
            r = an.run_contrasts(sub, scheme="holdout", contrasts=[c for c in CONTRASTS if c.name in ("incremental_ontology", "NEGATIVE_CONTROL_real_vs_shuffled")], n_boot=n_boot)
            r = r[(r["status"] == "ok") & (r["metric"] == "fidelity_to_oracle")].set_index("contrast") if len(r) else r
            row["Δ(E−D) só seeds efetivas"] = r.loc["incremental_ontology", "mean_diff"] if "incremental_ontology" in r.index else None
            row["Δ(E−F) só seeds efetivas"] = r.loc["NEGATIVE_CONTROL_real_vs_shuffled", "mean_diff"] if "NEGATIVE_CONTROL_real_vs_shuffled" in r.index else None
            row["n_seeds efetivas"] = n_eff
            row["leitura"] = "evidência (positiva ou negativa) apenas sobre as seeds com ontologia efetivamente usada"
        else:
            row["Δ(E−D) só seeds efetivas"] = row["Δ(E−F) só seeds efetivas"] = None
            row["n_seeds efetivas"] = 0
            row["leitura"] = "sem evidência de utilização semântica (NÃO equivale a «a ontologia não melhora»)"
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking agregado DESCRITIVO (média das médias por dataset); só depois das tabelas individuais e dos contrastes."""
    rows = []
    for metric in ("fidelity_to_oracle", "accuracy_real_labels"):
        per = df.groupby(["dataset", "arm"])[metric].mean().unstack("arm")
        arms = [a for a in ARM_ORDER if a in per and per[a].notna().all()]
        if len(arms) < 2:
            continue
        ranks = per[arms].rank(axis=1, ascending=False, method="average")
        for a in arms:
            rows.append({"metric": metric, "Arm": ARM_LABEL[a], "mean_rank": ranks[a].mean(), "mean_value": per[a].mean(), "n_datasets": int(len(per))})
    return pd.DataFrame(rows).sort_values(["metric", "mean_rank"]).reset_index(drop=True) if rows else pd.DataFrame()


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df.empty:
        return "_(sem dados)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(out) + "\n"


def invariants_markdown(report: Dict[str, Any]) -> str:
    out = []
    for unit, checks in report["units"].items():
        out.append(f"### {unit}\n")
        out.append(_md(pd.DataFrame([{"invariante": c["invariant"], "ok": "✔" if c["passed"] else "✘", "detalhe": c["detail"]} for c in checks])))
    m = report["metrics_recomputable_from_raw"]
    out.append(f"**Métricas recomputáveis a partir dos brutos:** {'✔' if m['ok'] else '✘'} ({m['checked_values']} valores, max|Δ|={m['max_abs_diff']:.2e})\n")
    out.append(f"**Todos os invariantes cumpridos:** {'SIM' if report['all_passed'] else 'NÃO'}\n")
    return "\n".join(out)


def raw_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "seed", "arm", "oracle_id", "accuracy_real_labels", "balanced_accuracy_real_labels", "macro_f1_real_labels", "fidelity_to_oracle",
            "disagreement_rate_to_oracle", "fidelity_train", "node_count", "leaf_count", "depth", "features_used", "membership_queries",
            "semantic_split_count", "ontology_effectively_used", "ontology_category", "mirror_applied"]
    t = df[[c for c in cols if c in df]].copy()
    t["_o"] = t["arm"].map({a: i for i, a in enumerate(ARM_ORDER)})
    t = t.sort_values(["dataset", "seed", "_o"]).drop(columns="_o")
    t["oracle_id"] = t["oracle_id"].fillna("").astype(str).str[:8]
    return t


def smoke_report(root: Path) -> Path:
    root = Path(root)
    rows, records, splits = load_raw(root)
    report = inv.check_all(records, splits, expected_arms=ARM_ORDER)
    (root / "invariants.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    meta = [json.loads(p.read_text()) for p in sorted((root / "raw").glob("*/seed*/RUN_META.txt"))]
    text = ["# Protocol smoke test (1 master seed por dataset)\n",
            "> O objetivo NÃO é interpretar quem venceu: só verificar invariantes do protocolo. Os números abaixo são resultados BRUTOS, sem leitura científica.\n",
            "## 1. Invariantes\n", invariants_markdown(report),
            "## 2. Resultados brutos por dataset × seed × braço\n", _md(raw_table(rows)),
            "## 3. Execução\n", _md(pd.DataFrame(meta))]
    path = root / "SMOKE_REPORT.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def main_report(root: Path, *, n_boot: int = 10000) -> Path:
    root = Path(root)
    rows, records, splits = load_raw(root)
    manifest = mf.load_manifest()
    main_ids = manifest["main_ranking_datasets"]
    main, controlled = rows[rows["dataset"].isin(main_ids)], rows[~rows["dataset"].isin(main_ids)]
    tables = root / "tables"; tables.mkdir(exist_ok=True)
    report = inv.check_all(records, splits, expected_arms=ARM_ORDER)
    (tables / "invariants.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    ind, con, ont = individual_table(main), contrast_table(main, n_boot=n_boot), ontology_table(main, n_boot=n_boot)
    rank = ranking_table(main)
    for name, t in (("individual", ind), ("contrasts", con), ("ontology", ont), ("ranking", rank)):
        t.to_csv(tables / f"{name}.csv", index=False)
    text = ["# BENCHMARK PRINCIPAL — datasets reais offline\n",
            f"Manifesto `{manifest['manifest_sha256'][:12]}`, master seeds {manifest['master_seeds']}, datasets {main_ids}. "
            "Resultados reconstruídos só dos brutos imutáveis.\n",
            "## 1. Invariantes do protocolo\n", invariants_markdown(report),
            "## 2. Tabela individual (média ± desvio entre master seeds)\n", _md(ind),
            "## 3. Contrastes causais pareados (a − b; sinal NÃO assumido)\n", _md(con.drop(columns=["contrast"], errors="ignore")),
            "## 4. Ontologia: D sem OWL · E +OWL · F permutada\n", _md(ont),
            "## 5. Ranking agregado (descritivo; só depois dos pontos anteriores)\n",
            "_Quatro datasets offline não sustentam, por si, uma conclusão geral de superioridade._\n", _md(rank)]
    if len(controlled):
        text += ["## 6. Validação controlada (sintéticos) — secção separada, fora do ranking\n", _md(individual_table(controlled))]
    path = root / "BENCHMARK_REPORT.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--mode", choices=["smoke", "main"], required=True)
    a = ap.parse_args()
    print(smoke_report(Path(a.root)) if a.mode == "smoke" else main_report(Path(a.root)))
