"""B3-Q (Doc. 62 §8): VERIFICAÇÃO DOS PRESSUPOSTOS da regra de decisão da porta G2s ``média + 1,645·EP ≤ τ`` sobre os valores por réplica (``generic_estimator_sensitivity_raw.json``; sintético; sementes
9800; sem ontologia, sem dados reais, sem teste). Não escolhe limiares. Verifica: (i) discretização/limites dos valores por réplica (fração de g=6 indicadores ∈ {0,1/6,…,1}); (ii) nível real da regra
por simulação (taxa de falsa aprovação quando a verdade é τ e quando é τ+δ) para n=20; (iii) alternativas (t de Student, bootstrap percentil, Bonferroni sobre as 36 comparações célula×estimador);
(iv) dependência entre estimadores dentro de cada réplica (mesmos dados) e portanto de ``best_of_all``; (v) concordância das decisões entre regras. A regra NÃO é evidência confirmatória: aplica-se a um gerador sintético
com réplicas independentes por semente, mas cada célula×estimador é um teste separado (multiplicidade) e a incerteza do gerador não está coberta."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

G = 6


def decisions(vals, tau, rule, rng):
    v = np.asarray(vals, float)
    n, m, se = len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    if rule == "normal_1.645":
        return bool(m + 1.645 * se <= tau)
    if rule == "t_19_0.95":
        return bool(m + stats.t.ppf(0.95, n - 1) * se <= tau)
    if rule == "bonferroni_36":
        return bool(m + stats.norm.ppf(1 - 0.05 / 36) * se <= tau)
    if rule == "bootstrap_95":
        bm = np.array([rng.choice(v, n, replace=True).mean() for _ in range(2000)])
        return bool(np.quantile(bm, 0.95) <= tau)
    raise ValueError(rule)


def operating_characteristic(tau, delta, n=20, sims=20000, seed=1):
    """P(regra aprova) quando cada réplica é Binomial(G, p)/G com p = τ + δ (verdade fora da região de aprovação se δ ≥ 0)."""
    rng = np.random.default_rng(seed)
    p = min(0.999, tau + delta)
    x = rng.binomial(G, p, size=(sims, n)) / G
    m, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(n)
    return float(np.mean(m + 1.645 * se <= tau))


def main(src, out):
    d = json.loads(Path(src).read_text())
    cells, ests = list(d["results"]), list(next(iter(d["results"].values())))
    rng = np.random.default_rng(7)
    rules = ("normal_1.645", "t_19_0.95", "bonferroni_36", "bootstrap_95")
    taus = (0.5, 0.6, 0.7, 0.8, 0.9)
    dec = {r: {str(t): {c: {e: decisions(d["results"][c][e]["values"], t, r, rng) for e in ests} for c in cells} for t in taus} for r in rules}
    agree = {}
    for t in taus:
        diff = [(c, e) for c in cells for e in ests if len({dec[r][str(t)][c][e] for r in rules}) > 1]
        agree[str(t)] = {"n_tests": len(cells) * len(ests), "n_rule_disagreements": len(diff), "disagreements": [f"{c}|{e}" for c, e in diff]}
    # (iv) dependência entre estimadores: correlação entre estimadores nas réplicas da mesma célula (dados partilhados)
    corr = {}
    base = [e for e in ests if e != "best_of_all"]
    for c in cells:
        M = np.array([d["results"][c][e]["values"] for e in base])
        cm = np.corrcoef(M)
        corr[c] = round(float(np.nanmean(cm[np.triu_indices(len(base), 1)])), 3)
    res = {"source": src, "reps": d.get("reps"), "values_are_multiples_of": 1.0 / G, "all_values_in_grid": bool(all(abs(v * G - round(v * G)) < 1e-9 for c in cells for e in ests for v in d["results"][c][e]["values"])),
           "rules": list(rules), "decisions": dec, "agreement_between_rules": agree,
           "operating_characteristic_normal_rule_n20": {str(t): {str(dl): operating_characteristic(t, dl) for dl in (0.0, 0.05, 0.10, 0.15)} for t in taus},
           "mean_pairwise_correlation_between_estimators_within_cell": corr,
           "notes": ["a regra normal usa quantil z=1,645 com n=20 réplicas de valores discretos limitados a [0,1]: o nível real pode exceder 5 %",
                     "36 comparações (6 células × 6 estimadores) ⇒ multiplicidade; Bonferroni é conservadora",
                     "estimadores partilham os dados de cada réplica (correlação ≠ 0): best_of_all é um máximo de estimadores dependentes (viés otimista para o método genérico, ou seja, conservador para a porta)",
                     "nada aqui é evidência confirmatória nem escolhe limiares"]}
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("grelha discreta:", res["all_values_in_grid"], "| OC τ=0.8:", res["operating_characteristic_normal_rule_n20"]["0.8"], "| discordâncias τ=0.8:", agree["0.8"]["n_rule_disagreements"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
