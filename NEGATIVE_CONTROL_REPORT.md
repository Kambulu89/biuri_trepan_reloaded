# NEGATIVE CONTROL REPORT (OWL real vs OWL permutada)

O controlo preserva dimensionalidade, nº de features derivadas, distribuição de pesos/grupos/relatedness e protocolo (mesmo oráculo, split, seed e orçamento), mas permuta a correspondência feature↔entidade ontológica (e calcula as features derivadas sobre colunas trocadas). Se `real ≈ shuffled`, o ganho não é atribuível à ontologia.

## synthetic_binary

| contraste | métrica | A − B | IC95% (bootstrap pareado) | pares | Wilcoxon p | p Holm | rank-biserial | Cliff δ | evidência |
|---|---|---|---|---|---|---|---|---|---|
| `reloaded_owl_full` − `reloaded_owl_shuffled` | fidelity_to_oracle | -0.0231 | [-0.0564, 0.0051] | 6 | 0.3125 | 1.0000 | -0.60 (large) | -0.31 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | accuracy_real_labels | -0.0179 | [-0.0462, 0.0026] | 6 | 0.3750 | 1.0000 | -0.70 (large) | -0.22 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | node_count | 1.0000 | [-0.6667, 2.3333] | 6 | 0.5000 | — | 0.60 (large) | 0.39 | **INDICATIVE** |

- `fidelity_to_oracle`: **REAL_APPROX_SHUFFLED** (Δ=-0.0231, IC95% [-0.0564, 0.0051], 6 pares)
- `accuracy_real_labels`: **REAL_APPROX_SHUFFLED** (Δ=-0.0179, IC95% [-0.0462, 0.0026], 6 pares)
- `node_count`: **REAL_APPROX_SHUFFLED** (Δ=1.0000, IC95% [-0.6667, 2.3333], 6 pares)
- Conclusão (fidelity_to_oracle): NÃO atribuível à ontologia: real ≈ shuffled (efeito de arquitectura/features/ruído)
- Verificação: a estrutura semântica permutada difere da real em 6/6 splits.

## synthetic_3class

| contraste | métrica | A − B | IC95% (bootstrap pareado) | pares | Wilcoxon p | p Holm | rank-biserial | Cliff δ | evidência |
|---|---|---|---|---|---|---|---|---|---|
| `reloaded_owl_full` − `reloaded_owl_shuffled` | fidelity_to_oracle | 0.0256 | [-0.0282, 0.0795] | 6 | 0.5000 | 1.0000 | 0.47 (medium) | 0.03 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | accuracy_real_labels | 0.0128 | [-0.0564, 0.0744] | 6 | 0.7500 | 1.0000 | 0.19 (small) | -0.06 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | node_count | -0.6667 | [-1.3333, 0.0000] | 6 | 0.5000 | — | -1.00 (large) | -0.28 | **INDICATIVE** |

- `fidelity_to_oracle`: **REAL_APPROX_SHUFFLED** (Δ=0.0256, IC95% [-0.0282, 0.0795], 6 pares)
- `accuracy_real_labels`: **REAL_APPROX_SHUFFLED** (Δ=0.0128, IC95% [-0.0564, 0.0744], 6 pares)
- `node_count`: **REAL_APPROX_SHUFFLED** (Δ=-0.6667, IC95% [-1.3333, 0.0000], 6 pares)
- Conclusão (fidelity_to_oracle): NÃO atribuível à ontologia: real ≈ shuffled (efeito de arquitectura/features/ruído)
- Verificação: a estrutura semântica permutada difere da real em 6/6 splits.

