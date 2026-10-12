# BASELINE_VS_OPTIMIZED_EQUIVALENCE

Datasets: breast_cancer, iris, wine · campos comparados: 78 · FAIL: 6 · tolerância numérica: 0.0 (comparação exata)

**Resultado global: FAIL**

| campo | iris | wine | breast_cancer |
|---|---|---|---|
| conjunto de braços A–F | PASS | PASS | PASS |
| split_hash | PASS | PASS | PASS |
| preprocessing_id | PASS | PASS | PASS |
| oracle_id | PASS | PASS | PASS |
| tuning config selecionada (estrutural) | FAIL | FAIL | FAIL |
| purity_epsilon | PASS | PASS | PASS |
| max_nodes | PASS | PASS | PASS |
| query_budget | PASS | PASS | PASS |
| predictions MLP (A) | PASS | PASS | PASS |
| predictions C4.5 (B) | PASS | PASS | PASS |
| predictions TREPAN Original (C) | PASS | PASS | PASS |
| predictions Reloaded Core (D) | PASS | PASS | PASS |
| predictions Reloaded + OWL (E) | PASS | PASS | PASS |
| predictions Reloaded + OWL shuffled (F) | PASS | PASS | PASS |
| fidelity | PASS | PASS | PASS |
| accuracy | PASS | PASS | PASS |
| balanced accuracy | PASS | PASS | PASS |
| macro-F1 | PASS | PASS | PASS |
| nós | PASS | PASS | PASS |
| folhas | PASS | PASS | PASS |
| profundidade | PASS | PASS | PASS |
| quality gate | PASS | PASS | PASS |
| semantic audit | FAIL | FAIL | FAIL |
| ontology_effectively_used / categoria | PASS | PASS | PASS |
| estrutura das árvores (features/splits) | PASS | PASS | PASS |
| contrastes | PASS | PASS | PASS |

## Falhas

- iris / tuning config selecionada (estrutural): relatório do tuning estrutural difere
- iris / semantic audit: auditoria semântica difere
- wine / tuning config selecionada (estrutural): relatório do tuning estrutural difere
- wine / semantic audit: auditoria semântica difere
- breast_cancer / tuning config selecionada (estrutural): relatório do tuning estrutural difere
- breast_cancer / semantic audit: auditoria semântica difere

_Ignorado: tempos, contadores de trabalho (queries por escopo do tuning, número de ajustes) e identificadores de execução._
