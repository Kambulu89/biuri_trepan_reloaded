#!/usr/bin/env python3
"""Gera SEMANTIC_VALIDATION_REPORT.md a partir dos JSON de run_semantic_validation.py.

Todos os números vêm do JSON (nada é escrito à mão), para que o relatório seja
exatamente o que a execução produziu.

    python scripts/render_semantic_report.py out.md run1/semantic_validation.json [run2/...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def load(paths):
    merged, version = {}, None
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        version = data.get("semantic_pipeline_version", version)
        merged.update(data["datasets"])
    return version, merged


def main(argv):
    out, inputs = Path(argv[1]), argv[2:]
    version, runs = load(inputs)
    by_ds = defaultdict(list)
    for key, row in runs.items():
        by_ds[key.split("@")[0]].append((key.split("@seed")[1], row))

    L = ["# Relatório de validação semântica", "",
         f"Versão do pipeline semântico: `{version}`. Gerado por `scripts/render_semantic_report.py` "
         "a partir de `scripts/run_semantic_validation.py`; todos os números vêm do JSON da execução.", "",
         "**Protocolo:** só dados de desenvolvimento (nenhum teste externo); `OntologyProcessor` refeito em cada fold "
         "só com o treino do fold; MLP base e MLP+OWL otimizados em separado com a mesma lista de candidatos, "
         "folds internos e semente; utilidade ponderada e IC bootstrap emparelhado (95%) definidos antes de correr; "
         "limiares não foram ajustados depois de ver resultados.", "",
         "## 1. Cobertura OWL e estado da ontologia", "",
         "| Dataset | Estado | Mapeadas | Ambíguas | Colisões | Entropia | Profundidade | ABox | Reasoner | Riqueza |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, rows in sorted(by_ds.items()):
        r = rows[0][1]; q, rs = r["quality"], r["reasoner"]
        rich = (q.get("semantic_richness") or {})
        L.append(f"| {ds} | {q['status']} | {q['mapped_features']}/{q['total_features']} ({f(q['feature_coverage'], 2)}) | "
                 f"{q['ambiguous_matches']} | {q['entity_collisions']} | {f(q['mapping_entropy'], 2)} | "
                 f"{q['ontology_depth']} | {q['abox']} | "
                 f"{'consistente' if rs.get('consistent') else 'n/d'} ({f(rs.get('duration_seconds'), 2)} s, "
                 f"{rs.get('inferred_axioms_count')} inferidos) | {rich.get('level')} "
                 f"({', '.join(rich.get('knowledge_sources') or []) or 'só taxonomia'}) |")

    L += ["", "## 2. Features geradas, estáveis e selecionadas", "",
          "| Dataset | Semente | Geradas | Retidas (novidade) | Estáveis | Selecionadas | Decisão |", "|---|---|---|---|---|---|---|"]
    for ds, rows in sorted(by_ds.items()):
        for seed, r in sorted(rows):
            fu = r["full"]
            L.append(f"| {ds} | {seed} | {fu['generated']} | {fu['retained']} | {len(fu['stable_features'] or [])} | "
                     f"{len(fu['selected_features'] or [])} | `{fu['decision']}` |")

    L += ["", "## 3. MLP base vs MLP + OWL (OOF, desenvolvimento)", "",
          "| Dataset | Sem. | Balanced Acc. base | + OWL | Macro-F1 base | + OWL | Ganho de utilidade | IC 95% | Evidência | MLP aceite | TREPAN disponível |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, rows in sorted(by_ds.items()):
        for seed, r in sorted(rows):
            fu = r["full"]; b, w = fu.get("base"), fu.get("with_owl")
            if not b:
                L.append(f"| {ds} | {seed} | — | — | — | — | — | — | — | {fu['semantic_mlp_accepted']} | {fu['semantic_trepan_available']} |")
                continue
            ci = fu["utility_gain_ci"]
            strength = ("n/a" if not fu["semantic_mlp_accepted"] else "forte" if ci[0] > 0 else "fraca")
            L.append(f"| {ds} | {seed} | {f(b['balanced_accuracy'])} | {f(w['balanced_accuracy'])} | {f(b['macro_f1'])} | "
                     f"{f(w['macro_f1'])} | {f(fu['utility_gain'], 4)} | [{f(ci[0], 3)}; {f(ci[1], 3)}] | {strength} | "
                     f"{fu['semantic_mlp_accepted']} | {fu['semantic_trepan_available']} |")

    decisions = defaultdict(int)
    for rows in by_ds.values():
        for _, r in rows:
            decisions[r["full"]["decision"]] += 1
    L += ["", "Distribuição das decisões: " + ", ".join(f"`{k}`={v}" for k, v in sorted(decisions.items())) + ".", ""]

    L += ["## 4. Ablação por componente", "",
          "Cada linha restringe as features semânticas a um tipo. `NO_NOVEL_FEATURES` significa que a OWL "
          "desse dataset não produz features desse tipo (não é uma falha do pipeline).", "",
          "| Dataset | Sem. | Agregados | Relacionais | Restrições/categóricas | Inferidas pelo reasoner | Todas |", "|---|---|---|---|---|---|---|"]
    def cell(s):
        if s is None:
            return "—"
        g = s.get("utility_gain")
        return f"`{s['decision'].replace('REJECT_', 'R_').replace('ACCEPT_', 'A_')}` ({f(g, 3)})"
    for ds, rows in sorted(by_ds.items()):
        for seed, r in sorted(rows):
            ab = r.get("ablation") or {}
            L.append(f"| {ds} | {seed} | {cell(ab.get('B_aggregates_only'))} | {cell(ab.get('C_relational_only'))} | "
                     f"{cell(ab.get('D_constraints_only'))} | {cell(ab.get('E_reasoner_inferred_only'))} | {cell(r['full'])} |")

    L += ["", "## 5. Controlo negativo: semântica baralhada", "",
          "A mesma ontologia, mas com a atribuição feature→entidade baralhada. Se o ganho real fosse conhecimento "
          "semântico, os controlos deveriam ficar sistematicamente abaixo do real.", "",
          "| Dataset | Sem. | Ganho real | Ganhos dos controlos | Controlos ≥ real |", "|---|---|---|---|---|"]
    tot_ge, tot = 0, 0
    for ds, rows in sorted(by_ds.items()):
        for seed, r in sorted(rows):
            c = r["control_summary"]
            ge = c["controls_ge_real"]
            if ge is not None:
                tot_ge += ge; tot += len(c["control_gains"])
            L.append(f"| {ds} | {seed} | {f(c['real_utility_gain'], 4)} | "
                     f"{', '.join(f(g, 4) for g in c['control_gains'])} | {ge}/{len(c['control_gains'])} |")
    L += ["", f"No total, {tot_ge} de {tot} controlos com semântica baralhada tiveram ganho ≥ ao da ontologia real."]
    runs_all = [r for rows in by_ds.values() for _, r in rows]
    accepted = [r for r in runs_all if r["full"]["semantic_mlp_accepted"]]
    strong = [r for r in accepted if r["full"]["utility_gain_ci"][0] > 0]
    with_rel = sorted(ds for ds, rows in by_ds.items()
                      if (rows[0][1].get("ablation") or {}).get("C_relational_only", {}).get("decision") != "REJECT_NO_NOVEL_FEATURES")
    L += ["", "## 6. Leitura (calculada a partir dos resultados acima)", "",
          f"- Execuções: {len(runs_all)}; MLP+OWL aceite em {len(accepted)} ({len(strong)} com evidência forte, "
          f"i.e. IC da utilidade acima de zero; {len(accepted) - len(strong)} com evidência fraca).",
          f"- Controlos com semântica baralhada com ganho ≥ ao real: {tot_ge}/{tot}"
          + (" — mais de metade: os ganhos observados não se distinguem do ruído." if tot and tot_ge * 2 > tot
             else " — menos de metade."),
          f"- Datasets cuja OWL gera features relacionais (as únicas não lineares): {', '.join(with_rel) or 'nenhum'}.",
          "- A semântica continua disponível para o TREPAN em todas as execuções (`semantic_trepan_available`), "
          "mesmo quando o enriquecimento do MLP é rejeitado: os dois estados são independentes.",
          "- **Limites:** poucas sementes e amostras pequenas; um único protocolo de validação cruzada; ontologias de "
          "benchmark sem propriedades de objeto, restrições nem limites declarados (ver `docs/SEMANTIC_PIPELINE_AUDIT.md`); "
          "nos datasets em que o baralhamento preserva a estrutura de grupos (ex. iris, 2 grupos) o controlo pode coincidir "
          "com o real. Nenhum limiar foi ajustado depois de ver estes resultados. Este relatório **não** demonstra "
          "superioridade do MLP ontológico nem do TREPAN Reloaded."]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"escrito {out} ({len(L)} linhas)")


if __name__ == "__main__":
    main(sys.argv)
