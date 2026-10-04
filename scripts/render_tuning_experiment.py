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
    hist = t["structure_history"]
    lines += ["| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in hist:
        s = h["stats"]
        if not s:
            lines.append(f"| {h['label']} | — | — | — | — | — | — | — | — | — | — | — | {h['verdict']['status']}: {h['verdict']['text']} |"); continue
        lines.append(f"| {h['label']} | {fmt(s['fidelity_mean'])} | {fmt(s['fidelity_std'])} | {fmt(s['fidelity_min'])} | "
                     f"{s['nodes_mean']:.1f} ± {s['nodes_std']:.1f} | {s['depth_mean']:.1f} | {fmt(s['between_seed_std'])} | "
                     f"{fmt(s['within_seed_fold_std_mean'])} | {fmt(s['nodes_cv'], 2)} | {s['time_mean_s']:.1f} | {s['query_budget']} | "
                     f"{s['budget_exhausted_fraction']:.0%} | **{h['verdict']['status']}** — {h['verdict']['text']} |")
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
    sel = t["structure_selection"]
    lines += [f"**Configuração selecionada:** `{sel['selected_label']}`. Vencedora por repetição (seed): "
              f"{list(zip(sel['seeds'], sel['per_seed_winners']))}; concordância {sel['agreement']:.0%} "
              f"(limiar {sel['threshold']:.0%}) → tuning **{'estável' if sel['stable'] else 'INSTÁVEL'}**.", ""]
    tm = d.get("test_models") or {}
    o, r = tm.get("original") or {}, tm.get("reloaded") or {}
    lines += [f"Teste (usado uma só vez, depois da escolha): TREPAN Original — nós {o.get('nodes')}, folhas {o.get('leaves')}, "
              f"accuracy {o.get('accuracy')}, fidelity {o.get('oracle_fidelity')}; MLP accuracy {(tm.get('mlp_original') or {}).get('accuracy')}.", ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("specs", nargs="+"); ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    md = ["# Tuning da estrutura do TREPAN — Repeated Stratified K-Fold e seleção lexicográfica", "",
          "Gerado por `scripts/render_tuning_experiment.py` a partir de `scripts/run_trepan_tuning_experiment.py`. "
          "Nenhuma regra por dataset; o teste só é usado uma vez, depois da configuração escolhida.", ""]
    for spec in args.specs:
        name, seeds = spec.split(":")
        seeds = [int(s) for s in seeds.split(",")]
        runs = [json.loads((Path(args.dir) / f"{name}_{s}.json").read_text()) for s in seeds if (Path(args.dir) / f"{name}_{s}.json").exists()]
        if not runs:
            continue
        md += [f"## {runs[0]['dataset']}", "", "### Estabilidade da configuração entre seeds mestre", "",
               "| seed mestre | selecionada | concordância interna entre repetições | tuning | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |",
               "|---|---|---|---|---|---|---|---|"]
        picks = []
        for d in runs:
            sel = d["tuning"]["structure_selection"]; picks.append(sel["selected_label"])
            o = (d.get("test_models") or {}).get("original") or {}
            md.append(f"| {d['seed']} | `{sel['selected_label']}` | {sel['agreement']:.0%} | {'estável' if sel['stable'] else '**INSTÁVEL**'} | "
                      f"{o.get('nodes')} | {o.get('accuracy')} | {o.get('oracle_fidelity')} | {d['wall_seconds']:.0f} |")
        cnt = Counter(picks); modal, n_modal = cnt.most_common(1)[0]
        md += ["", f"Configurações escolhidas entre seeds mestre: {dict(cnt)}. Moda `{modal}` em {n_modal}/{len(picks)} "
               f"({n_modal / len(picks):.0%}) → seleção entre seeds **{'estável' if n_modal / len(picks) >= 0.6 else 'INSTÁVEL'}** (limiar 60%).", ""]
        for i, d in enumerate(runs):
            md += run_section(d, full=True)
    Path(args.out).write_text("\n".join(md))
    print("escrito", args.out)


if __name__ == "__main__":
    main()
