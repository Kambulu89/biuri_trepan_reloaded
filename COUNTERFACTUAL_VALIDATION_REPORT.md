# Validação do pipeline contrafactual

Script: `PYTHONPATH=. python scripts/run_cf_validation.py` (dados **sintéticos**, MLP sklearn, split 70/30 com seed 0, 12 instâncias
estratificadas do teste; constraints/densidade só do treino; `max_iterations=25`, `max_time=20s`). Saídas em `results/cf_validation/<caso>/`
(`counterfactuals.csv`, `counterfactual_report.json`, `summary.md`, `benchmark.json`, `example_text.txt`). Uma só seed/dataset: **evidência
exploratória, sem IC nem testes de significância**. Binário: 1 alvo/instância; multiclasse: 1 pedido por classe alvo.

| caso | método | sucesso | proximidade (Gower) | esparsidade | plausibilidade | validade semântica* | tempo (s) | falhas |
|---|---|---|---|---|---|---|---|---|
| binário, sem OWL | LORE-insp. | 1.00 | 0.045 | 1.00 | 0.48 | n/d | 0.58 | — |
| | CLEAR-insp. | 1.00 | 0.042 | 1.00 | 0.50 | n/d | 0.11 | — |
| | CoGS-insp. | 1.00 | 0.105 | 1.08 | 0.44 | n/d | 0.67 | — |
| binário, com OWL | LORE-insp. | 0.50 | 0.117 | 1.17 | 0.33 | 1.0 | 0.57 | 6 CONSTRAINT_INFEASIBLE |
| | CLEAR-insp. | 0.58 | 0.253 | 2.14 | 0.44 | 1.0 | 0.10 | 5 CONSTRAINT_INFEASIBLE |
| | CoGS-insp. | 1.00 | 0.125 | 1.75 | 0.45 | 1.0 | 0.70 | — |
| multiclasse, sem OWL | LORE-insp. | 1.00 | 0.033 | 1.00 | 0.60 | n/d | 0.60 | — |
| | CLEAR-insp. | 1.00 | 0.032 | 1.00 | 0.59 | n/d | 0.12 | — |
| | CoGS-insp. | 1.00 | 0.070 | 1.00 | 0.27 | n/d | 0.71 | — |
| multiclasse, com OWL | LORE-insp. | 0.58 | 0.158 | 1.07 | 0.63 | 1.0 | 0.59 | 10 CONSTRAINT_INFEASIBLE |
| | CLEAR-insp. | 0.83 | 0.209 | 2.00 | 0.74 | 1.0 | 0.11 | 4 CONSTRAINT_INFEASIBLE |
| | CoGS-insp. | 1.00 | 0.121 | 1.83 | 0.28 | 1.0 | 0.73 | — |

\* sobre os sucessos; por construção 1.0, pois um CF que viole constraints hard nunca é SUCCESS.

## Leitura (sem ranking oculto)
* Todos os CFs devolvidos passam no modelo e nas constraints hard (verificado por testes e pelo próprio pipeline).
* Com constraints OWL (dependência income→children, intervalos), LORE e CLEAR falham mais (`CONSTRAINT_INFEASIBLE`) porque as suas
  propostas violam a dependência; CoGS (GA tipado) encontra sempre uma alternativa mas com maior distância/esparsidade. É um resultado
  do desenho dos métodos *-inspired* neste dataset sintético, não prova de superioridade.
* A plausibilidade do CoGS é baixa em multiclasse (0.27–0.28): produz CFs válidos mas em zonas menos densas.
* Sem ontologia a validade semântica é `NOT_AVAILABLE`, nunca assumida.

## Limitações
Dataset sintético e uma seed; plausibilidade é proxy de densidade; métodos não canónicos; não há avaliação humana de
acionabilidade; sem causalidade. Falhas não foram escondidas (contadas por estado).
