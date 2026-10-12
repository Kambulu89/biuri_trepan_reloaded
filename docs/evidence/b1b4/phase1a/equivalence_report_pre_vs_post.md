# BASELINE_VS_OPTIMIZED_EQUIVALENCE

Datasets: breast_cancer, digits, iris, wine · campos comparados: 182 · FAIL: 0 · tolerância numérica: 0.0 (comparação exata)

**Resultado global: PASS**

| campo | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| conjunto de braços A–F | PASS | PASS | PASS | PASS |
| split_hash | PASS | PASS | PASS | PASS |
| preprocessing_id | PASS | PASS | PASS | PASS |
| oracle_id | PASS | PASS | PASS | PASS |
| tuning config selecionada (estrutural) | PASS | PASS | PASS | PASS |
| purity_epsilon | PASS | PASS | PASS | PASS |
| max_nodes | PASS | PASS | PASS | PASS |
| query_budget | PASS | PASS | PASS | PASS |
| predictions MLP (A) | PASS | PASS | PASS | PASS |
| predictions C4.5 (B) | PASS | PASS | PASS | PASS |
| predictions TREPAN Original (C) | PASS | PASS | PASS | PASS |
| predictions Reloaded Core (D) | PASS | PASS | PASS | PASS |
| predictions Reloaded + OWL (E) | PASS | PASS | PASS | PASS |
| predictions Reloaded + OWL shuffled (F) | PASS | PASS | PASS | PASS |
| fidelity | PASS | PASS | PASS | PASS |
| accuracy | PASS | PASS | PASS | PASS |
| balanced accuracy | PASS | PASS | PASS | PASS |
| macro-F1 | PASS | PASS | PASS | PASS |
| nós | PASS | PASS | PASS | PASS |
| folhas | PASS | PASS | PASS | PASS |
| profundidade | PASS | PASS | PASS | PASS |
| quality gate | PASS | PASS | PASS | PASS |
| semantic audit | PASS | PASS | PASS | PASS |
| ontology_effectively_used / categoria | PASS | PASS | PASS | PASS |
| estrutura das árvores (features/splits) | PASS | PASS | PASS | PASS |
| contrastes | PASS | PASS | PASS | PASS |

_Ignorado: tempos, contadores de trabalho (queries por escopo do tuning, número de ajustes) e identificadores de execução._
