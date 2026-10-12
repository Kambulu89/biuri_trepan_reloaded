# BASELINE_VS_OPTIMIZED_EQUIVALENCE

Datasets: breast_cancer, iris, wine · campos comparados: 78 · FAIL: 39 · tolerância numérica: 0.0 (comparação exata)

**Resultado global: FAIL**

| campo | iris | wine | breast_cancer |
|---|---|---|---|
| conjunto de braços A–F | PASS | PASS | PASS |
| split_hash | PASS | PASS | PASS |
| preprocessing_id | PASS | PASS | PASS |
| oracle_id | FAIL | FAIL | FAIL |
| tuning config selecionada (estrutural) | FAIL | FAIL | FAIL |
| purity_epsilon | FAIL | PASS | PASS |
| max_nodes | FAIL | PASS | PASS |
| query_budget | PASS | PASS | PASS |
| predictions MLP (A) | PASS | PASS | PASS |
| predictions C4.5 (B) | PASS | PASS | PASS |
| predictions TREPAN Original (C) | FAIL | PASS | FAIL |
| predictions Reloaded Core (D) | FAIL | PASS | FAIL |
| predictions Reloaded + OWL (E) | PASS | PASS | FAIL |
| predictions Reloaded + OWL shuffled (F) | FAIL | PASS | FAIL |
| fidelity | FAIL | PASS | FAIL |
| accuracy | FAIL | PASS | FAIL |
| balanced accuracy | FAIL | PASS | FAIL |
| macro-F1 | FAIL | PASS | FAIL |
| nós | FAIL | PASS | FAIL |
| folhas | FAIL | PASS | FAIL |
| profundidade | FAIL | PASS | FAIL |
| quality gate | FAIL | PASS | FAIL |
| semantic audit | FAIL | FAIL | FAIL |
| ontology_effectively_used / categoria | PASS | PASS | PASS |
| estrutura das árvores (features/splits) | FAIL | PASS | FAIL |
| contrastes | FAIL | FAIL | FAIL |

## Falhas

- iris / oracle_id: oracle_id difere
- iris / tuning config selecionada (estrutural): relatório do tuning estrutural difere
- iris / purity_epsilon: 0.01 != 0.05
- iris / max_nodes: 63 != 31
- iris / predictions TREPAN Original (C): predições diferentes
- iris / predictions Reloaded Core (D): predições diferentes
- iris / predictions Reloaded + OWL shuffled (F): predições diferentes
- iris / fidelity: difere em ['reloaded_core', 'reloaded_owl_shuffled', 'trepan_original']
- iris / accuracy: difere em ['reloaded_core', 'reloaded_owl_shuffled', 'trepan_original']
- iris / balanced accuracy: difere em ['reloaded_core', 'reloaded_owl_shuffled', 'trepan_original']
- iris / macro-F1: difere em ['reloaded_core', 'reloaded_owl_shuffled', 'trepan_original']
- iris / nós: difere em ['reloaded_core', 'reloaded_owl_shuffled', 'trepan_original']
- iris / folhas: difere em ['reloaded_core', 'reloaded_owl_shuffled', 'trepan_original']
- iris / profundidade: difere em ['reloaded_core', 'reloaded_owl_shuffled', 'trepan_original']
- iris / quality gate: campos do gate diferem
- iris / semantic audit: auditoria semântica difere
- iris / estrutura das árvores (features/splits): estrutura difere
- iris / contrastes: contrastes diferem
- wine / oracle_id: oracle_id difere
- wine / tuning config selecionada (estrutural): relatório do tuning estrutural difere
- wine / semantic audit: auditoria semântica difere
- wine / contrastes: contrastes diferem
- breast_cancer / oracle_id: oracle_id difere
- breast_cancer / tuning config selecionada (estrutural): relatório do tuning estrutural difere
- breast_cancer / predictions TREPAN Original (C): predições diferentes
- breast_cancer / predictions Reloaded Core (D): predições diferentes
- breast_cancer / predictions Reloaded + OWL (E): predições diferentes
- breast_cancer / predictions Reloaded + OWL shuffled (F): predições diferentes
- breast_cancer / fidelity: difere em ['reloaded_owl_shuffled']
- breast_cancer / accuracy: difere em ['reloaded_owl_full', 'reloaded_owl_shuffled']
- breast_cancer / balanced accuracy: difere em ['reloaded_core', 'reloaded_owl_full', 'reloaded_owl_shuffled', 'trepan_original']
- breast_cancer / macro-F1: difere em ['reloaded_core', 'reloaded_owl_full', 'reloaded_owl_shuffled', 'trepan_original']
- breast_cancer / nós: difere em ['reloaded_core', 'reloaded_owl_full', 'reloaded_owl_shuffled', 'trepan_original']
- breast_cancer / folhas: difere em ['reloaded_core', 'reloaded_owl_full', 'reloaded_owl_shuffled', 'trepan_original']
- breast_cancer / profundidade: difere em ['reloaded_core', 'reloaded_owl_full', 'reloaded_owl_shuffled', 'trepan_original']
- breast_cancer / quality gate: campos do gate diferem
- breast_cancer / semantic audit: auditoria semântica difere
- breast_cancer / estrutura das árvores (features/splits): estrutura difere
- breast_cancer / contrastes: contrastes diferem

_Ignorado: tempos, contadores de trabalho (queries por escopo do tuning, número de ajustes) e identificadores de execução._
