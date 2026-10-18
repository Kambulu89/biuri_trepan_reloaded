"""B3-Q (Doc. 61 §2): análise de sensibilidade da porta G2s ao estimador genérico e ao limiar τ_s, a partir da evidência já existente (``generic_estimator_sensitivity.json``; sementes 9800, sintético, sem
dados reais, sem rótulos de teste, sem qualquer ontologia). Não escolhe limiar: tabula, por estimador e por τ, que células passariam a porta (taxa de acerto do método genérico ≤ τ) na regra pontual e na
regra conservadora (média + 1,645·EP ≤ τ). A escolha do estimador de referência e de τ é do responsável pela metodologia e deve ser feita ANTES de avaliar qualquer variante OWL."""
import json
import sys
from pathlib import Path

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)


def main(src: str, out: str):
    d = json.loads(Path(src).read_text())
    cells = list(d["results"])
    ests = list(next(iter(d["results"].values())))
    table = {}
    for e in ests:
        table[e] = {}
        for tau in TAUS:
            pt = [c for c in cells if d["results"][c][e]["mean"] <= tau]
            cons = [c for c in cells if d["results"][c][e]["mean"] + 1.645 * d["results"][c][e]["se"] <= tau]
            table[e][str(tau)] = {"pass_point": pt, "pass_conservative": cons}
    # concordância entre estimadores: quantas células mudam de decisão (passa/falha) em τ = 0,8 consoante o estimador
    tau = "0.8"
    decisions = {c: {e: (c in table[e][tau]["pass_point"]) for e in ests} for c in cells}
    disagree = [c for c, v in decisions.items() if len(set(v.values())) > 1]
    res = {"source": src, "reps": d.get("reps"), "seed_base": d.get("seed_base"), "taus": list(TAUS), "cells": cells, "estimators": ests, "table": table, "decisions_at_tau_0.8": decisions,
           "cells_where_estimator_choice_changes_the_decision_at_tau_0.8": disagree,
           "note": "nenhuma escolha feita; 'passa' = o método genérico NÃO resolve a tarefa (taxa ≤ τ); a escolha do estimador de referência altera a decisão nas células indicadas"}
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("células com decisão dependente do estimador (τ=0,8):", disagree)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
