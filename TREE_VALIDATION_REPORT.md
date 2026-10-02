# TREE VALIDATION REPORT (smoke end-to-end)

Dataset: sklearn breast_cancer (apenas smoke), split 70/30 estratificado, seed 42, MLP (32,) como oráculo
(accuracy teste = 0,9298). Protocolo único declarado a priori e idêntico para as duas árvores TREPAN:
`min_sample=1000, max_nodes=31, max_depth=8, max_n=3, beam_width=2, max_queries=10000`.
Reprodução: `python scripts/tree_validation_smoke.py` (dados completos em `TREE_VALIDATION_RESULTS.json`).
O teste só mede; não escolhe splits, poda nem parâmetros.

| | C4.5 | TREPAN Original | TREPAN Reloaded (semântica OFF, mesmo oráculo) |
|---|---|---|---|
| Treino | rótulos reais | oráculo MLP Original | oráculo MLP Original |
| Nós (bruto → final) | 29 → 17 | 3 → 3 | 3 → 3 |
| Folhas / profundidade | 9 / 6 | 2 / 1 | 2 / 1 |
| Queries usadas / budget | — | 2204 / 10000 (não esgotado) | 2204 / 10000 (não esgotado) |
| Candidatos simples avaliados | — | 540 | 540 |
| m-of-n avaliados / seleccionados | — | 36 / 1 (2-de-3 na raiz) | 36 / 1 |
| Best-first verificado | — | sim | sim |
| Poda | 29 → 17 (pessimista) | 0 nós | 0 nós |
| Motivos de paragem | PURE=8, PRUNED=1 | PURE=2 (Wilson, 98,6 %) | PURE=2 |
| Fidelity vs MLP Original | (concordância informativa 0,9415) | 0,9240 | 0,9240 |
| Accuracy vs y real | 0,9357 | 0,9298 | 0,9298 |
| Comprimento médio da regra | 4,0 | 1,0 (1 m-of-n) | 1,0 |
| Tempo de treino | 0,41 s | 0,17 s (queries 0,04 s, m-of-n 0,08 s) | 0,26 s |

## Sensibilidade ao orçamento de queries (TREPAN Original; budgets declarados a priori)

| budget | nós | queries | esgotado | paragens | fidelity | accuracy real |
|---|---|---|---|---|---|---|
| 2000 | 3 | 2000 | **sim** | PURE=1, **QUERY_BUDGET_EXHAUSTED=1** | 0,924 | 0,930 |
| 5000 | 3 | 2204 | não | PURE=2 | 0,924 | 0,930 |
| 10000 | 3 | 2204 | não | PURE=2 | 0,924 | 0,930 |
| 20000 | 3 | 2204 | não | PURE=2 | 0,924 | 0,930 |

## Leitura

- O stump de 3 nós de TREPAN é **cientificamente legítimo** neste protocolo (`legitimate_by_construction=True`): ambos os filhos
  são estatisticamente puros face ao oráculo. Com budget 2000 o mesmo stump aparece mas por esgotamento de orçamento —
  situação antes invisível, agora marcada `STOP_QUERY_BUDGET_EXHAUSTED`.
- O C4.5 é maior porque aprende os rótulos reais (ruído incluído) e não tem critério de pureza estatística face a um oráculo suave.
  Maior não é melhor: C4.5 accuracy 0,9357 vs 0,9298 com 17 vs 3 nós (1 teste m-of-n).
- Original e Reloaded (α=β=0, sem ontologia) são idênticos — o controlo "arquitectura sem semântica" funciona. O efeito da ontologia
  não foi medido aqui (sem OWL no smoke); resultado não conclusivo, não favorável nem desfavorável ao Reloaded.
- Um só dataset e uma só seed: não é evidência científica, apenas verificação de funcionamento.
