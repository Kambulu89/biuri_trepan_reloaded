"""Gera o relatório Markdown da experiência de tuning da estrutura do TREPAN a partir dos JSON (sem lógica por dataset).

Uso: python scripts/render_tuning_experiment.py SAIDA.md DATASET_A:seed1,seed2,... DATASET_B:... --dir DIRETORIA
(os ficheiros esperados são DIRETORIA/<dataset>_<seed>.json)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def fmt(v, nd=3):
    return f"{v:.{nd}f}"


def run_section(d: dict, full: bool) -> list[str]:
    t = d["tuning"]; plan = t["cv_plan"]
    lines = [f"#### {d['dataset']} — seed mestre {d['seed']} (treino {d['manifest']['train_rows']}, teste {d['manifest']['test_rows']})", "",
             f"CV: {plan['scheme']}; {plan['folds']} dobras × {plan['repeats']} repetições (seeds {plan['seeds']}) = {plan['n_splits']} partições; "
             f"teste usado na seleção: **{plan['test_set_used']}**. Regra: {t['selection_rule']}.", ""]
    oc = d.get("oracle_contract") or {}
    if oc:
        o = oc.get("oracle", {})
        lines += [f"Oráculo congelado: `oracle_id={o.get('oracle_id')}` ({o.get('kind')}, construtor `{o.get('builder')}`); Original e Reloaded "
                  f"consultaram o mesmo oráculo: **{oc.get('single_oracle_for_all_trees')}** (ids {oc.get('tree_oracle_ids')}); "
                  f"queries por escopo: { {k: v['queries'] for k, v in (oc.get('trees') or {}).items()} }; pesos inalterados no fim: {oc.get('unchanged_after')}.", ""]
    hist = t["structure_history"]
    bc = t.get("budget_check") or {}
    lines += [f"Política de orçamento: **{t.get('query_budget_policy')}** — orçamento comum a todos os candidatos: "
              f"{bc.get('common_budget')} (mesmo para todos: {bc.get('all_candidates_same_budget')}); consumo máximo observado: "
              f"{bc.get('max_queries_used')}; algum ajuste esgotou o orçamento: **{bc.get('any_budget_exhausted')}**.", ""]
    lines += ["| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | "
              "instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in hist:
        s = h["stats"]
        if not s:
            lines.append(f"| {h['label']} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | {h['verdict']['status']}: {h['verdict']['text']} |"); continue
        lines.append(f"| {h['label']} | {fmt(s['fidelity_mean'])} | {fmt(s['fidelity_std'])} | {fmt(s['fidelity_min'])} | "
                     f"{s['nodes_mean']:.1f} ± {s['nodes_std']:.1f} | {fmt(s['nodes_cv'], 2)} | {s['depth_mean']:.1f} ({fmt(s['depth_cv'], 2)}) | "
                     f"{s['leaves_mean']:.1f} ({fmt(s['leaves_cv'], 2)}) | {fmt(s['structural_instability'], 2)} | "
                     f"{s['fraction_at_node_cap']:.0%} ({s['node_cap_reached_count']}/{s['n_splits']}) | {s['max_nodes']} | "
                     f"{'⚠ sim' if s['structural_stability_censored'] else 'não'} | {s['query_budget']} | {s['queries_used_mean']:.0f} | "
                     f"{s['budget_exhausted_count']}/{s['n_splits']} | {s['budget_exhausted_fraction']:.0%} | {s['time_mean_s']:.1f} | "
                     f"**{h['verdict']['status']}** — {h['verdict']['text']} |")
    lines.append("")
    lines += ["Estabilidade da fidelity (entre dobras e entre seeds):", "",
              "| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |",
              "|---|---|---|---|---|---|"]
    for h in hist:
        s = h["stats"]
        if s:
            lines.append(f"| {h['label']} | {fmt(s['fidelity_mean'])} | {fmt(s['fidelity_std'])} | {fmt(s['within_seed_fold_std_mean'])} | "
                         f"{fmt(s['between_seed_std'])} | {', '.join(fmt(v) for v in s['fidelity_per_seed_mean'])} |")
    lines.append("")
    lines += ["Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?", "",
              "| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for h in hist:
        s = h["stats"]
        if not s:
            continue
        g = s["structure_diagnostics"]
        roots = ", ".join(f"{k} {v:.0%}" for k, v in list(g["root_feature_freq"].items())[:3]) or "—"
        used = ", ".join(f"{k} {v:.0%}" for k, v in list(g["feature_usage_freq"].items())[:4])
        sj = g["same_size_feature_jaccard_mean"]; ssj = g["same_size_split_jaccard_mean"]
        lines.append(f"| {h['label']} | {roots} | {g['root_feature_modal_share']:.0%} | {fmt(g['feature_set_jaccard_mean'], 2)} | "
                     f"{fmt(g['split_signature_jaccard_mean'], 2)} | {g['same_size_pairs']} | "
                     f"{'—' if sj is None else fmt(sj, 2)} | {'—' if ssj is None else fmt(ssj, 2)} | {g['stump_fraction']:.0%} | {used} |")
    lines.append("")
    if full:
        cols = [f"s{r['seed']}·f{r['fold']}" for r in hist[0]["per_split"]]
        lines += ["Fidelity por partição (repetição/seed · dobra):", "", "| configuração | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for h in hist:
            lines.append(f"| {h['label']} | " + " | ".join(fmt(r["fidelity"]) for r in h["per_split"]) + " |")
        lines += ["", "Nós por partição:", "", "| configuração | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for h in hist:
            lines.append(f"| {h['label']} | " + " | ".join(str(r["nodes"]) for r in h["per_split"]) + " |")
        lines.append("")
    sel = t["structure_selection"]; bs = sel["bootstrap"]
    ru = sel.get("bootstrap_runner_up"); ex = t.get("capacity_expansion") or {}
    mc = bs.get("selection_probability_mc_interval")
    lines += [f"**Configuração escolhida pela CV completa:** `{sel['selected_config_full_cv']}` — estado interno do tuning: "
              f"**{sel['status']}** ({sel.get('status_reason')}).", "",
              "Expansão de capacidade = SATURAÇÃO + GANHO DE VALIDAÇÃO (só CV do treino; mesmas partições; mesmo purity_epsilon):", "",
              f"- initial_node_grid: {ex.get('initial_node_grid')}; final_node_grid: {ex.get('final_node_grid')}; "
              f"capacity_expansion_rounds: {ex.get('capacity_expansion_rounds')}; expansion_triggered: {ex.get('expansion_triggered')}; "
              f"expansion_stop_reason: `{ex.get('expansion_stop_reason')}`; fraction_at_node_cap (escolhida): "
              f"{('%.0f%%' % (100 * ex['fraction_at_node_cap'])) if ex.get('fraction_at_node_cap') is not None else '—'}",
              f"- interpretação: {ex.get('interpretation') or '—'}", "",
              "| previous_max_nodes | candidate_max_nodes | previous_fidelity | candidate_fidelity | fidelity_delta | teste | p-value | IC95 do ganho | capacity_gain_supported | fraction_at_node_cap | decisão |",
              "|---|---|---|---|---|---|---|---|---|---|---|",
              *[(lambda st, te: f"| {st['previous_max_nodes']} | {st['candidate_max_nodes']} | "
                 f"{('%.4f' % st['previous_fidelity']) if st['previous_fidelity'] is not None else '—'} | "
                 f"{('%.4f' % st['candidate_fidelity']) if st['candidate_fidelity'] is not None else '—'} | "
                 f"{('%+.4f' % st['fidelity_delta']) if st['fidelity_delta'] is not None else '—'} | "
                 f"{'Nadeau–Bengio pareado, unilateral' if te.get('available') else '—'} | "
                 f"{('%.4f' % te['p_value']) if te.get('available') else '—'} | "
                 f"{('[%+.4f, %+.4f]' % tuple(te['ci95_delta'])) if te.get('available') and te.get('ci95_delta') else '—'} | "
                 f"{st['capacity_gain_supported']} | {('%.0f%%' % (100 * st['fraction_at_node_cap'])) if st['fraction_at_node_cap'] is not None else '—'} | "
                 f"{st.get('decision', '—')} |")(st, st.get('statistical_test') or {}) for st in ex.get('steps', [])], "",
              "Robustez interna da política (bootstrap de blocos das repetições; teste não usado):", "",
              f"- bootstrap_method: **{bs['method']}**; bootstrap_samples: {bs['bootstrap_samples']}; avaliações distintas: {bs['bootstrap_evaluations']}"
              f" (blocos = {bs['n_blocks']} repetições × {plan['folds']} dobras)",
              f"- selected_config_probability (escolhida pela CV completa): **{bs['selected_config_probability']:.1%}**"
              + (f" (intervalo Monte-Carlo 95%: {mc[0]:.1%}–{mc[1]:.1%})" if mc else " (exato: sem erro Monte-Carlo)"),
              f"- bootstrap_modal_config: `{bs['bootstrap_modal_config']}` com probabilidade {bs['bootstrap_modal_probability']:.1%}"
              f" — selecionada é a moda: **{bs['selected_is_modal']}**" + (" ⚠ **fragilidade FORTE da seleção full-CV**" if bs.get('full_cv_selection_fragile') else ""),
              f"- bootstrap_runner_up: {('`' + ru['label'] + '` ' + format(ru['probability'], '.1%')) if ru else '—'}; "
              f"top1_top2_margin: {format(bs['top1_top2_margin'], '.1%') if bs['top1_top2_margin'] is not None else '—'}",
              f"- IC bootstrap 95% da fidelity média da escolhida: {('[%.3f, %.3f]' % tuple(bs['winner_fidelity_ci'])) if bs['winner_fidelity_ci'] else '—'}",
              f"- conjunto equivalente (fidelity indistinguível da escolhida): equivalent_candidate_count={sel['equivalent_candidate_count']}; "
              f"equivalent_set_probability={('%.1f%%' % (100 * sel['equivalent_set_probability'])) if sel['equivalent_set_probability'] is not None else '—'}; "
              f"exact_selection_probability={sel['exact_selection_probability']:.1%}; bootstrap_modal_probability={bs['bootstrap_modal_probability']:.1%}; "
              f"ids: {sel['equivalent_candidate_ids']}",
              f"- comportamento da árvore escolhida: {sel['tree_behavior']}",
              "- distribuição das configurações escolhidas: " + (", ".join(f"`{k}` {v:.1%}" for k, v in bs["distribution"].items()) or "—"),
              f"- vencedora por repetição (apenas informativo; 1 repetição = {plan['folds']} dobras): {sel['per_repeat_winners']}",
              f"- saturação da escolhida: {sel['node_cap']['node_cap_reached_count']}/{plan['n_splits']} árvores no teto max_nodes={sel['node_cap']['max_nodes']} "
              f"(fraction_at_node_cap {sel['node_cap']['fraction_at_node_cap']:.0%}); structural_stability_censored: "
              f"**{sel['node_cap']['structural_stability_censored']}** (evidência estrutural: {sel['node_cap'].get('structural_stability_evidence')})", ""]
    tm = d.get("test_models") or {}
    o, r = tm.get("original") or {}, tm.get("reloaded") or {}
    lines += [f"Teste (usado uma só vez, depois da escolha): TREPAN Original — nós {o.get('nodes')}, folhas {o.get('leaves')}, "
              f"accuracy {o.get('accuracy')}, fidelity {o.get('oracle_fidelity')}; MLP accuracy {(tm.get('mlp_original') or {}).get('accuracy')}.", ""]
    return lines


def conclusion(name: str, runs: list[dict]) -> list[str]:
    """Conclusão calculada a partir dos dados (sem texto fixo por dataset)."""
    stump_cands, total = [], 0
    for d in runs:
        for h in d["tuning"]["structure_history"]:
            s = h.get("stats")
            if not s:
                continue
            g = s["structure_diagnostics"]
            if g["stump_fraction"] > 0:
                stump_cands.append((d["seed"], h["label"], g["stump_fraction"], s["fidelity_mean"], s["structural_instability"],
                                    h["verdict"]["reason_code"]))
            total += 1
    out = [f"### Conclusão — {name}", ""]
    if not stump_cands:
        out += ["Nenhuma configuração produziu árvores de ≤3 nós (stumps) nas partições de CV desta experiência.", ""]
        return out
    n_win = sum(1 for c in stump_cands if c[5] == "WINNER")
    partial = any(0.0 < c[2] < 1.0 for c in stump_cands)
    if partial:
        out += ["**A árvore de 3 nós não é suportada como uma estrutura robusta e consistentemente preferível; a sua fidelity pode "
                "ser equivalente, mas a sua estrutura apresenta forte dependência da amostragem.**", ""]
    else:
        out += ["Há configurações que produzem sempre stumps (fração 100%): nesses casos o tamanho não depende da amostragem, "
                "pelo que não se afirma dependência da amostragem.", ""]
    out += [
            f"Evidência (só treino, CV repetida): {len(stump_cands)} de {total} (seed mestre × configuração) produziram stumps em "
            f"parte das partições; destas, {n_win} foram selecionadas.", "",
            "| seed mestre | configuração | fração de stumps | fidelity média | instab. estrutural | resultado |", "|---|---|---|---|---|---|"]
    for seed, label, frac, fid, inst, code in stump_cands:
        out.append(f"| {seed} | {label} | {frac:.0%} | {fid:.3f} | {inst:.2f} | {code} |")
    out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("specs", nargs="+"); ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    md = ["# Tuning da estrutura do TREPAN — Repeated Stratified K-Fold, orçamento comum não limitante e seleção lexicográfica com estabilidade estrutural", "",
          "Gerado por `scripts/render_tuning_experiment.py` a partir de `scripts/run_trepan_tuning_experiment.py`. "
          "Nenhuma regra por dataset; o teste só é usado uma vez, depois da configuração escolhida.", ""]
    for spec in args.specs:
        name, seeds = spec.split(":")
        seeds = [int(s) for s in seeds.split(",")]
        runs = [json.loads((Path(args.dir) / f"{name}_{s}.json").read_text()) for s in seeds if (Path(args.dir) / f"{name}_{s}.json").exists()]
        if not runs:
            continue
        md += [f"## {runs[0]['dataset']} — oráculo `{runs[0].get('oracle_builder', 'factory')}`", "",
               "### 1. Estabilidade interna do tuning (block bootstrap, por seed mestre)", "",
               "| seed mestre | oracle_id | escolhida (CV completa) | P(escolhida) | moda do bootstrap | P(moda) | bootstrap_runner_up | top1−top2 | equivalentes (P família) | método (amostras) | grelha final (rondas; paragem) | estado |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        picks, n_stable = [], 0
        for d in runs:
            sel = d["tuning"]["structure_selection"]; bs = sel["bootstrap"]; picks.append(sel["selected_config_full_cv"]); n_stable += bool(sel["stable"])
            ru = sel.get("bootstrap_runner_up"); ex = d["tuning"].get("capacity_expansion") or {}
            md.append(f"| {d['seed']} | `{((d.get('oracle_contract') or {}).get('oracle') or {}).get('oracle_id')}` | `{sel['selected_config_full_cv']}` | "
                      f"{bs['selected_config_probability']:.1%} | `{bs['bootstrap_modal_config']}` | {bs['bootstrap_modal_probability']:.1%} | "
                      f"{('`' + ru['label'] + '` ' + format(ru['probability'], '.1%')) if ru else '—'} | "
                      f"{format(bs['top1_top2_margin'], '.1%') if bs['top1_top2_margin'] is not None else '—'} | "
                      f"{sel['equivalent_candidate_count']} ({sel['equivalent_set_probability']:.0%}) | {bs['method']} ({bs['bootstrap_samples']}) | "
                      f"{ex.get('final_node_grid')} ({ex.get('capacity_expansion_rounds')}; {ex.get('expansion_stop_reason')}) | {sel['status']} |")
        cnt = Counter(picks); modal, n_modal = cnt.most_common(1)[0]
        md += ["", "### 2. Robustez externa entre seeds mestre (evidência empírica; não define o estado do tuning)", "",
               "| seed mestre | selecionada | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |", "|---|---|---|---|---|---|"]
        for d in runs:
            sel = d["tuning"]["structure_selection"]; o = (d.get("test_models") or {}).get("original") or {}
            md.append(f"| {d['seed']} | `{sel['selected_config_full_cv']}` | {o.get('nodes')} | {o.get('accuracy')} | {o.get('oracle_fidelity')} | {d['wall_seconds']:.0f} |")
        consensus = n_modal / len(picks)
        md += ["", f"Configurações escolhidas: {dict(cnt)}. **Consenso de seleção = {consensus:.0%} ({n_modal}/{len(picks)}) nas master seeds avaliadas** "
               f"(moda `{modal}`); seeds mestre com estado estável (`stable_exact` ou `stable_equivalent_set`): {n_stable}/{len(picks)}.", ""]
        if consensus >= 0.6 and n_stable / len(picks) >= 0.6:
            md += ["Leitura: o consenso externo e a reamostragem interna apontam no mesmo sentido; só neste caso se usa o termo *robusto/estável* "
                   "(com a ressalva do número reduzido de master seeds).", ""]
        else:
            md += ["Leitura: a evidência de robustez **não** é suficiente (consenso externo e/ou reamostragem interna não a sustentam); "
                   "usar *incerto*, não *estável*.", ""]
        md += conclusion(runs[0]["dataset"], runs)
        for i, d in enumerate(runs):
            md += run_section(d, full=True)
    Path(args.out).write_text("\n".join(md))
    print("escrito", args.out)


if __name__ == "__main__":
    main()
