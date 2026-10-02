# TREE VALIDATION REPORT (smoke end-to-end, agnóstico a datasets)

O código de construção de árvores não conhece nenhum dataset. O smoke aceita **qualquer CSV numérico** (última coluna = alvo):
`python scripts/tree_validation_smoke.py --csv ficheiro.csv --out resultado.json`; sem `--csv` gera dados sintéticos
(`make_classification`, 600 amostras, 12 features, 3 % de ruído de rótulo). Os números abaixo são da execução sintética
(split 70/30 estratificado, seed 42, MLP (32,) como oráculo); dados completos em `TREE_VALIDATION_RESULTS.json`.
Protocolo único, igual para as duas árvores TREPAN, fixado antes de olhar para os resultados:
`min_sample=1000, max_nodes=31, max_depth=8, max_n=3, beam_width=2, max_queries=10000`. O teste só mede.

| | C4.5 | TREPAN Original | TREPAN Reloaded (semântica OFF, mesmo oráculo) |
|---|---|---|---|
| Treino | rótulos reais | oráculo MLP Original | oráculo MLP Original |
| Nós (bruto → final) | 77 → 53 | 23 → 19 | 23 → 19 |
| Folhas / profundidade | 27 / 10 | 10 / 5 | 10 / 5 |
| Queries / budget | — | 10000 / 10000 (**esgotado**) | 10000 / 10000 (**esgotado**) |
| Candidatos simples avaliados | — | 6042 | 6042 |
| m-of-n avaliados / seleccionados | — | 381 / 10 | 381 / 10 |
| Best-first verificado | — | sim | sim |
| Poda | 77 → 53 (pessimista) | 23 → 19 | 23 → 19 |
| Motivos de paragem | PURE=18, MIN_SAMPLES=2, PRUNED=7 | QUERY_BUDGET_EXHAUSTED=9, PRUNED=1 | idem |
| Fidelity vs MLP Original | (concordância informativa 0,856) | 0,850 | 0,850 |
| Accuracy vs y real | 0,844 | 0,828 | 0,828 |
| Tempo de treino | 0,51 s | 1,17 s (m-of-n 0,84 s) | 1,65 s |

## Sensibilidade ao orçamento (TREPAN Original; budgets declarados a priori)

| budget | nós | profundidade | queries | esgotado | paragens | fidelity | accuracy real |
|---|---|---|---|---|---|---|---|
| 2000 | 3 | 1 | 2000 | sim | BUDGET=1, PRUNED=1 | 0,750 | 0,750 |
| 5000 | 13 | 5 | 5000 | sim | BUDGET=7 | 0,828 | 0,817 |
| 10000 | 19 | 5 | 10000 | sim | BUDGET=9, PRUNED=1 | 0,850 | 0,828 |
| 20000 | 27 | 6 | 13230 | não | MAX_NODES=13, PRUNED=1 | 0,883 | 0,850 |

## Leitura
- Neste problema o factor que limita o tamanho do TREPAN é o **orçamento global de queries**, e agora é visível (`STOP_QUERY_BUDGET_EXHAUSTED`
  em 9 de 10 folhas). Isto confirma a recomendação: o budget deve ser fixado por protocolo prévio (ou passar a ser por nó), nunca ajustado
  olhando para o resultado. Com problemas fáceis (execução anterior, outro dataset) o mesmo código pára por pureza com 3 nós — ambos os
  comportamentos são legítimos e agora distinguíveis pelo relatório; é o dataset que decide, não o código.
- Original e Reloaded (α=β=0, sem ontologia) são idênticos: o controlo "arquitectura sem semântica" funciona.
  O efeito da ontologia não foi medido (sem OWL no smoke) — resultado inconclusivo.
- Uma execução, uma seed, dados sintéticos: verificação de funcionamento, não evidência científica.
