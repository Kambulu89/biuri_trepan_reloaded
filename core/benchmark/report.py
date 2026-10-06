"""Geração dos relatórios: summary por execução, SCIENTIFIC_VALIDATION_REPORT, controlo negativo e ablação."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from core.benchmark import analysis as an
from core.benchmark import stats as st
from core.benchmark.runner import NOT_APPLICABLE_ABLATIONS

PRIMARY_TABLE = ("accuracy_real_labels", "balanced_accuracy_real_labels", "macro_f1_real_labels", "fidelity_to_oracle",
                 "node_count", "depth", "m_of_n_count", "semantic_split_count", "membership_queries", "tree_training_time")


def _f(v, nd=4) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))


def _pm(row) -> str:
    return f"{_f(row['mean'])} ± {_f(row['std'])} [{_f(row['ci_low'])}, {_f(row['ci_high'])}]"


def aggregate_table(agg: pd.DataFrame, dataset: str, metrics: Sequence[str] = PRIMARY_TABLE) -> str:
    g = agg[agg["dataset"] == dataset]
    arms = list(dict.fromkeys(g["arm"]))
    head = "| arm | oracle | " + " | ".join(metrics) + " |\n|---|---|" + "---|" * len(metrics) + "\n"
    out = head
    for arm in arms:
        ga = g[g["arm"] == arm]
        oracle = ga["oracle_name"].iloc[0]
        cells = []
        for m in metrics:
            r = ga[ga["metric"] == m]
            cells.append(_pm(r.iloc[0]) if len(r) else "—")
        out += f"| `{arm}` | {oracle} | " + " | ".join(cells) + " |\n"
    return out


def contrast_table(res: pd.DataFrame, dataset: str, groups: Sequence[str] = None) -> str:
    g = res[(res["dataset"] == dataset)]
    if groups:
        g = g[g["group"].isin(groups)]
    out = "| contraste | métrica | A − B | IC95% (bootstrap pareado) | pares | Wilcoxon p | p Holm | rank-biserial | Cliff δ | evidência |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for _, r in g.iterrows():
        if r["status"] != "ok":
            out += f"| {r['contrast']} | {r.get('metric', '-')} | — | — | — | — | — | — | — | {r['status']}: {r.get('reason', '')} |\n"
            continue
        out += (f"| `{r['arm_a']}` − `{r['arm_b']}` | {r['metric']} | {_f(r['mean_diff'])} | [{_f(r['ci_low'])}, {_f(r['ci_high'])}] | "
                f"{int(r['n_pairs'])} | {_f(r['wilcoxon_p'])} | {_f(r.get('p_holm'))} | {_f(r['rank_biserial'], 2)} ({r['effect_label']}) | "
                f"{_f(r['cliffs_delta'], 2)} | **{r['evidence_level']}** |\n")
    return out


def null_results(result_entries: List[Dict[str, Any]]) -> List[str]:
    """Secção Negative / Null Results: tudo o que NÃO suporta a hipótese ou limita a conclusão."""
    items = []
    for e in result_entries:
        r, a = e["result"], e["analysis"]
        ds = r.dataset
        res = a["contrasts"]
        sems = r.semantic_report
        if sems and not any(s["semantic_available"] for s in sems):
            reasons = sorted({s["reason"] for s in sems})
            items.append(f"**{ds}**: semântica indisponível em todos os splits ({'; '.join(reasons)}). Os braços Reloaded+OWL não foram executados.")
        elif sems and not all(s["semantic_available"] for s in sems):
            items.append(f"**{ds}**: semântica indisponível em {sum(not s['semantic_available'] for s in sems)}/{len(sems)} splits.")
        rej = [s for s in sems if s["semantic_available"] and not s.get("mlp_enrichment_accepted")]
        if sems and rej:
            items.append(f"**{ds}**: o gate NÃO aceitou o MLP Ontológico como oráculo em {len(rej)}/{len(sems)} splits (ontology_valid=true, mlp_enrichment_accepted=false). "
                         "O pipeline end-to-end usou o MLP Original nesses splits.")
        if len(res):
            ok = res[(res["status"] == "ok") & (res["dataset"] == ds)]
            for _, c in ok.iterrows():
                if c["metric"] in an.PRIMARY_METRICS and c["group"] in ("main", "negative_control", "end_to_end", "baseline") and c["n_pairs"] >= 3:
                    lo, hi = c["ci_low"], c["ci_high"]
                    if lo <= 0 <= hi:
                        items.append(f"**{ds}** · `{c['arm_a']}` vs `{c['arm_b']}` ({c['metric']}): diferença {_f(c['mean_diff'])}, IC95% [{_f(lo)}, {_f(hi)}] inclui 0 → sem efeito detectável.")
                    elif hi < 0:
                        items.append(f"**{ds}** · `{c['arm_a']}` é PIOR que `{c['arm_b']}` ({c['metric']}): {_f(c['mean_diff'])}, IC95% [{_f(lo)}, {_f(hi)}].")
            nc = a.get("negative_control", [])
            for v in nc:
                if v["metric"] in an.PRIMARY_METRICS and v["verdict"] != "REAL_GT_SHUFFLED":
                    items.append(f"**{ds}** · controlo negativo ({v['metric']}): {v['verdict']} (Δ real−shuffled = {_f(v['mean_diff'])}, IC [{_f(v['ci_low'])}, {_f(v['ci_high'])}], {v['n_pairs']} pares) → o ganho NÃO pode ser atribuído à ontologia.")
        df = r.frame()
        if "query_budget_exhausted" in df and df["query_budget_exhausted"].fillna(False).astype(bool).any():
            frac = df[df["query_budget_exhausted"].notna()]["query_budget_exhausted"].astype(bool).mean()
            items.append(f"**{ds}**: o orçamento de membership queries esgotou em {100 * frac:.0f}% das árvores TREPAN (limita o crescimento; igual em todos os braços).")
        for s in r.skipped:
            if str(s.get("reason", "")).startswith("erro"):
                items.append(f"**{ds}** · braço `{s['arm']}` falhou em {s['split_id']}: {s['reason']}")
    return items or ["Nenhum resultado nulo/negativo detectado nesta execução (isto não significa ausência de limitações — ver secção Limitações)."]


def ablation_markdown(entries: List[Dict[str, Any]]) -> str:
    out = ["# ABLATION REPORT\n",
           "Todos os braços TREPAN partilham: split, seed, oráculo (MLP Original, salvo `reloaded_e2e`), orçamento de queries, max_depth, max_nodes, "
           "amostragem base e avaliação. A única variável por braço está na coluna «o que muda».\n",
           "| braço | o que muda |\n|---|---|"]
    for a in (entries[0]["result"].arms if entries else []):
        out.append(f"| `{a['arm_id']}` | {a['description']} |")
    out.append("\n## Efeito de cada componente (diferença A − B, mesma seed/split)\n")
    for e in entries:
        r, a = e["result"], e["analysis"]
        out.append(f"### {r.dataset}\n")
        out.append(contrast_table(a["contrasts"], r.dataset, groups=["ablation", "baseline", "main"]))
        out.append("\n**Decomposição da fidelidade (Δ face ao braço D salvo indicação):**\n")
        for att in a["attribution"]:
            out.append(f"- ganho arquitectural (D − C): {_f(att['architectural_gain'])}; features OWL: {_f(att['owl_features_effect'])}; "
                       f"score semântico: {_f(att['semantic_score_effect'])}; ontologia incremental (E − D): {_f(att['incremental_ontology'])}; "
                       f"total (E − C): {_f(att['total_vs_original'])}; real − shuffled: {_f(att['real_vs_shuffled'])}.")
        out.append("")
    out.append("## Ablações não aplicáveis neste runner (e porquê)\n")
    out += [f"- **{k}**: {v}" for k, v in NOT_APPLICABLE_ABLATIONS.items()]
    return "\n".join(out) + "\n"


def negative_control_markdown(entries: List[Dict[str, Any]]) -> str:
    out = ["# NEGATIVE CONTROL REPORT (OWL real vs OWL permutada)\n",
           "O controlo preserva dimensionalidade, nº de features derivadas, distribuição de pesos/grupos/relatedness e protocolo "
           "(mesmo oráculo, split, seed e orçamento), mas permuta a correspondência feature↔entidade ontológica (e calcula as features "
           "derivadas sobre colunas trocadas). Se `real ≈ shuffled`, o ganho não é atribuível à ontologia.\n"]
    for e in entries:
        r, a = e["result"], e["analysis"]
        out.append(f"## {r.dataset}\n")
        out.append(contrast_table(a["contrasts"], r.dataset, groups=["negative_control"]))
        for v in a["negative_control"]:
            out.append(f"- `{v['metric']}`: **{v['verdict']}** (Δ={_f(v['mean_diff'])}, IC95% [{_f(v['ci_low'])}, {_f(v['ci_high'])}], {v['n_pairs']} pares)")
        for att in a["attribution"]:
            out.append(f"- Conclusão ({att['metric']}): {att['attribution']}")
        sh = [s for s in r.semantic_report if s["semantic_available"]]
        if sh:
            diff = sum(s["real_structure_signature"] != s["shuffled_structure_signature"] for s in sh)
            out.append(f"- Verificação: a estrutura semântica permutada difere da real em {diff}/{len(sh)} splits.")
        out.append("")
    return "\n".join(out) + "\n"


def summary_markdown(result, analysis) -> str:
    cfg = result.config
    out = [f"# Resumo — {result.dataset}\n", f"- experiment_id: `{result.experiment_id}`", f"- amostras/features/classes: {result.n_samples}/{result.n_features}/{result.n_classes} {result.class_counts}",
           f"- esquema: {cfg['scheme']} · splits: {len(result.splits)} · seeds: {cfg['seeds']}",
           f"- dataset_hash: `{result.dataset_hash[:16]}…` · ontology_hash: `{str(result.ontology_hash)[:16]}…`\n",
           "## Métricas agregadas (média ± desvio [IC95% bootstrap])\n", aggregate_table(analysis["aggregate"], result.dataset),
           "\n## Contrastes pareados\n", contrast_table(analysis["contrasts"], result.dataset)]
    return "\n".join(out) + "\n"


def scientific_validation_markdown(entries: List[Dict[str, Any]], cross: Dict[str, Any] = None, policy: st.EvidencePolicy = st.EvidencePolicy(),
                                   verification: Dict[str, Any] = None) -> str:
    first = entries[0]["result"]
    cfg = first.config
    L = ["# SCIENTIFIC VALIDATION REPORT\n",
         "> Gerado automaticamente por `core.benchmark`. Nenhuma hierarquia de métodos é imposta: os números decidem; resultados nulos e negativos estão na secção própria.\n",
         "## 1. Protocolo\n",
         f"- Esquema de avaliação: **{cfg['scheme']}**; teste final bloqueado (usado uma vez por split, após todos os ajustes; `EvaluationProtocolGuard`). "
         "A selecção do oráculo (MLP Original vs Ontológico) usa SÓ CV interna no treino.",
         f"- Seeds (fixadas antes dos resultados): {cfg['seeds']} · test_size={cfg['test_size']} · splits estratificados e **pareados** (mesmos índices para todos os braços; `split_hash` guardado).",
         f"- Orçamento TREPAN idêntico para todos os braços: {cfg['tree']} (regra automática declarada: `min_sample=max(n_train, min(1000, max(120, 3·n_train)))`, `max_queries=max_nodes·min_sample`).",
         f"- Paridade de MLPs: mesmo procedimento e mesmo nº de trials para Original e Ontológico (`mlp_trials={cfg['mlp_trials']}`).",
         f"- Gate do oráculo: MLP Ontológico só é professor se o ganho de balanced accuracy na CV interna > {cfg['oracle_gate_margin']}.",
         "- **Accuracy** = vs rótulos reais (`*_real_labels`). **Fidelity** = vs predições do respectivo oráculo (`fidelity_to_oracle`), sempre com o oráculo identificado.",
         f"- IC95%: bootstrap percentil ({cfg['n_boot']} reamostragens); para diferenças, bootstrap dos pares. Testes: Wilcoxon signed-rank pareado (t pareado só informativo; validade exige normalidade e n≥8). "
         "Effect sizes: rank-biserial, Cliff's δ, Cohen's dz. Correcção de múltiplas comparações: Holm (família primária: fidelity e accuracy por dataset) e BH (suplementar).",
         f"- Níveis de evidência (limiares fixados a priori): STATISTICALLY_SUPPORTED exige ≥{policy.min_units_supported} pares, teste ≥{policy.min_test_samples} amostras, p Holm<{policy.alpha}, IC sem 0, |rank-biserial|≥{policy.min_effect} e, para alegações semânticas, controlo negativo ultrapassado; "
         "INDICATIVE caso contrário (≥3 pares); MECHANISM_ONLY com <3 pares.",
         "- Folds repetidos não são independentes: nesses esquemas o nível máximo é INDICATIVE e reporta-se também o t corrigido de Nadeau-Bengio.\n",
         "## 2. Datasets e execuções\n",
         "| dataset | amostras | features | classes | splits | n_test (min) | ontologia | semântica disponível | manifest |", "|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        r = e["result"]
        sem = sum(s["semantic_available"] for s in r.semantic_report)
        L.append(f"| {r.dataset} | {r.n_samples} | {r.n_features} | {r.n_classes} | {len(r.splits)} | {min(s['n_test'] for s in r.splits)} | "
                 f"{'sim' if r.ontology_hash != 'none' else 'não'} | {sem}/{len(r.splits)} | `{e.get('run_dir', r.experiment_id)}` |")
    L.append("")
    for e in entries:
        r, a = e["result"], e["analysis"]
        L += [f"## 3. Resultados — {r.dataset}\n", "### 3.1 Métricas (média ± desvio [IC95%])\n", aggregate_table(a["aggregate"], r.dataset),
              "\n### 3.2 Comparações pareadas (A − B)\n", contrast_table(a["contrasts"], r.dataset),
              "\n### 3.3 Complexidade × fidelity\n",
              aggregate_table(a["aggregate"], r.dataset, ("fidelity_to_oracle", "node_count", "leaf_count", "depth", "average_rule_length", "m_of_n_count")),
              "\nÁrvores maiores não são automaticamente melhores: avaliar fidelity **versus** complexidade.\n"]
        for att in a["attribution"]:
            L.append(f"**Atribuição ({att['metric']})**: {att['attribution']}.\n")
    L += ["## 4. Controlo negativo e ablação\n", "Ver `NEGATIVE_CONTROL_REPORT.md` e `ABLATION_REPORT.md` (mesmos dados, mesmo protocolo).\n"]
    if cross:
        L += ["## 5. Entre datasets\n", f"- datasets: {cross['n_datasets']}"]
        for c in cross["contrasts"]:
            L.append(f"- `{c['contrast']}`: {c['per_dataset']} · Wilcoxon entre datasets: {c['wilcoxon_p_across_datasets']} {c.get('note', '')}")
        fr = cross["friedman"]
        L.append(f"- Friedman: {fr if not fr.get('valid') else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fr.items() if k != 'posthoc'}}\n")
    L += ["## 6. Negative / Null Results\n"]
    L += [f"- {x}" for x in null_results(entries)]
    if verification:
        L += ["\n## 7. Verificação das métricas a partir das predições\n",
              f"- valores recomputados: {verification['checked_values']} · máx. |diferença|: {verification['max_abs_diff']:.2e} · {'OK' if verification['ok'] else 'FALHOU'}"]
    L += ["\n## 8. Limitações\n",
          "- Evidência limitada pelo nº de seeds/datasets e pelo tamanho do teste; qualquer nível abaixo de STATISTICALLY_SUPPORTED é exploratório.",
          "- TBoxes de benchmark são de domínio e podem ter sido construídas a partir das descrições das features; não são ontologias externas independentes.",
          "- O mirror do Reloaded (`semantic_mirror_applied`) pode devolver o Original sobre features enriquecidas quando não há efeito semântico mensurável; está registado por árvore.",
          "- A fidelity do `reloaded_e2e` pode usar um oráculo diferente (MLP Ontológico): só é comparável via accuracy vs rótulos reais; o runner marca esses contrastes como não comparáveis.",
          "- Fidelity avalia-se no conjunto de teste (não nas queries sintéticas).",
          "- Ablações sem reasoner / relacionais / agregados / constraints / poda semântica: ver `ABLATION_REPORT.md` (não expostas como switches).",
          "- Datasets pequenos: com ~50–150 amostras de teste a variância da fidelity é elevada; use repeated_cv e/ou mais datasets."]
    return "\n".join(L) + "\n"
