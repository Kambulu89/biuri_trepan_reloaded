# ABLATION REPORT

Todos os braços TREPAN partilham: split, seed, oráculo (MLP Original, salvo `reloaded_e2e`), orçamento de queries, max_depth, max_nodes, amostragem base e avaliação. A única variável por braço está na coluna «o que muda».

| braço | o que muda |
|---|---|
| `mlp_original` | MLP treinado no espaço original |
| `mlp_ontological` | MLP treinado no espaço enriquecido (OWL real) |
| `c45` | C4.5 supervisionado com rótulos reais |
| `trepan_original` | TREPAN Original -> MLP Original |
| `reloaded_lambda0` | Reloaded sem semântica (λ=0), mesma infraestrutura -> MLP Original |
| `reloaded_semantic_score` | Reloaded + score semântico (features originais) -> MLP Original |
| `reloaded_owl_features` | Reloaded + features OWL, sem score semântico (λ=0) -> MLP Original |
| `reloaded_owl_full` | Reloaded + OWL real completa -> MLP Original (mesmo oráculo) |
| `reloaded_owl_shuffled` | CONTROLO NEGATIVO: Reloaded + OWL permutada -> MLP Original |
| `reloaded_e2e` | Pipeline completo: Reloaded + OWL real -> oráculo seleccionado pelo gate |
| `trepan_original_no_mofn` | TREPAN Original sem m-of-n (max_n=1) |
| `reloaded_owl_full_no_mofn` | Reloaded OWL completa sem m-of-n |
| `reloaded_owl_full_no_active_queries` | Reloaded OWL completa sem active queries |
| `reloaded_owl_full_no_error_focus` | Reloaded OWL completa sem refinamento focado no erro |

## Efeito de cada componente (diferença A − B, mesma seed/split)

### synthetic_binary

| contraste | métrica | A − B | IC95% (bootstrap pareado) | pares | Wilcoxon p | p Holm | rank-biserial | Cliff δ | evidência |
|---|---|---|---|---|---|---|---|---|---|
| `mlp_ontological` − `mlp_original` | accuracy_real_labels | -0.0205 | [-0.0282, -0.0103] | 6 | 0.0625 | 1.0000 | -1.00 (large) | -0.56 | **INDICATIVE** |
| `mlp_ontological` − `mlp_original` | balanced_accuracy_real_labels | -0.0207 | [-0.0285, -0.0104] | 6 | 0.0625 | — | -1.00 (large) | -0.58 | **INDICATIVE** |
| `mlp_ontological` − `mlp_original` | macro_f1_real_labels | -0.0205 | [-0.0282, -0.0103] | 6 | 0.0625 | — | -1.00 (large) | -0.50 | **INDICATIVE** |
| `reloaded_lambda0` − `trepan_original` | fidelity_to_oracle | 0.0000 | [0.0000, 0.0000] | 6 | 1.0000 | 1.0000 | 0.00 (negligible) | 0.00 | **IDENTICAL_RESULTS** |
| `reloaded_lambda0` − `trepan_original` | accuracy_real_labels | 0.0000 | [0.0000, 0.0000] | 6 | 1.0000 | 1.0000 | 0.00 (negligible) | 0.00 | **IDENTICAL_RESULTS** |
| `reloaded_lambda0` − `trepan_original` | node_count | 0.0000 | [0.0000, 0.0000] | 6 | 1.0000 | — | 0.00 (negligible) | 0.00 | **IDENTICAL_RESULTS** |
| `reloaded_semantic_score` − `reloaded_lambda0` | fidelity_to_oracle | 0.0154 | [-0.0282, 0.0538] | 6 | 0.5938 | 1.0000 | 0.29 (small) | 0.47 | **INDICATIVE** |
| `reloaded_semantic_score` − `reloaded_lambda0` | accuracy_real_labels | 0.0000 | [-0.0410, 0.0385] | 6 | 0.7500 | 1.0000 | -0.20 (small) | 0.03 | **INDICATIVE** |
| `reloaded_semantic_score` − `reloaded_lambda0` | node_count | 0.0000 | [-2.0000, 1.6667] | 6 | 1.0000 | — | 0.00 (negligible) | 0.08 | **INDICATIVE** |
| `reloaded_owl_features` − `reloaded_lambda0` | fidelity_to_oracle | -0.0205 | [-0.0513, 0.0103] | 6 | 0.3750 | 1.0000 | -0.53 (large) | -0.36 | **INDICATIVE** |
| `reloaded_owl_features` − `reloaded_lambda0` | accuracy_real_labels | -0.0205 | [-0.0615, 0.0128] | 6 | 0.4375 | 1.0000 | -0.47 (medium) | -0.25 | **INDICATIVE** |
| `reloaded_owl_features` − `reloaded_lambda0` | node_count | -0.6667 | [-2.3333, 0.6667] | 6 | 0.7500 | — | -0.50 (large) | -0.17 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_lambda0` | fidelity_to_oracle | 0.0359 | [-0.0000, 0.0718] | 6 | 0.1250 | 1.0000 | 0.71 (large) | 0.61 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_lambda0` | accuracy_real_labels | 0.0308 | [-0.0077, 0.0795] | 6 | 0.2500 | 1.0000 | 0.57 (large) | 0.44 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_lambda0` | node_count | 0.3333 | [-0.6667, 1.3333] | 6 | 1.0000 | — | 0.33 (medium) | 0.17 | **INDICATIVE** |
| `reloaded_owl_full` − `trepan_original` | fidelity_to_oracle | 0.0359 | [-0.0000, 0.0718] | 6 | 0.1250 | 1.0000 | 0.71 (large) | 0.61 | **INDICATIVE** |
| `reloaded_owl_full` − `trepan_original` | accuracy_real_labels | 0.0308 | [-0.0077, 0.0795] | 6 | 0.2500 | 1.0000 | 0.57 (large) | 0.44 | **INDICATIVE** |
| `reloaded_owl_full` − `trepan_original` | node_count | 0.3333 | [-0.6667, 1.3333] | 6 | 1.0000 | — | 0.33 (medium) | 0.17 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | fidelity_to_oracle | -0.0410 | [-0.0846, -0.0077] | 6 | 0.2500 | 1.0000 | -1.00 (large) | -0.47 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | node_count | -1.0000 | [-3.3333, 0.6667] | 6 | 0.7500 | — | -0.50 (large) | -0.25 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | fidelity_to_oracle | -0.0821 | [-0.1308, -0.0385] | 6 | 0.0312 | 0.5938 | -1.00 (large) | -0.83 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | node_count | -2.3333 | [-4.6667, -0.3333] | 6 | 0.2500 | — | -1.00 (large) | -0.50 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | fidelity_to_oracle | -0.0103 | [-0.0333, 0.0103] | 6 | 0.6250 | 1.0000 | -0.40 (medium) | -0.17 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | node_count | -1.6667 | [-2.6667, -0.6667] | 6 | 0.1250 | — | -1.00 (large) | -0.50 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | fidelity_to_oracle | -0.0179 | [-0.0437, 0.0051] | 6 | 0.5000 | 1.0000 | -0.60 (large) | -0.31 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | node_count | -0.6667 | [-2.3333, 0.6667] | 6 | 0.7500 | — | -0.50 (large) | -0.22 | **INDICATIVE** |


**Decomposição da fidelidade (Δ face a λ=0 salvo indicação):**

- arquitectura (λ=0 vs Original): 0.0000; features OWL: -0.0205; score semântico: 0.0154; OWL completa vs λ=0: 0.0359; total vs Original: 0.0359; real − shuffled: -0.0231.

### synthetic_3class

| contraste | métrica | A − B | IC95% (bootstrap pareado) | pares | Wilcoxon p | p Holm | rank-biserial | Cliff δ | evidência |
|---|---|---|---|---|---|---|---|---|---|
| `mlp_ontological` − `mlp_original` | accuracy_real_labels | 0.0051 | [-0.0077, 0.0179] | 6 | 1.0000 | 1.0000 | 0.07 (negligible) | 0.11 | **INDICATIVE** |
| `mlp_ontological` − `mlp_original` | balanced_accuracy_real_labels | 0.0051 | [-0.0083, 0.0185] | 6 | 0.8750 | — | 0.13 (small) | 0.17 | **INDICATIVE** |
| `mlp_ontological` − `mlp_original` | macro_f1_real_labels | 0.0054 | [-0.0074, 0.0184] | 6 | 0.5625 | — | 0.33 (medium) | 0.11 | **INDICATIVE** |
| `reloaded_lambda0` − `trepan_original` | fidelity_to_oracle | 0.0000 | [0.0000, 0.0000] | 6 | 1.0000 | 1.0000 | 0.00 (negligible) | 0.00 | **IDENTICAL_RESULTS** |
| `reloaded_lambda0` − `trepan_original` | accuracy_real_labels | 0.0000 | [0.0000, 0.0000] | 6 | 1.0000 | 1.0000 | 0.00 (negligible) | 0.00 | **IDENTICAL_RESULTS** |
| `reloaded_lambda0` − `trepan_original` | node_count | 0.0000 | [0.0000, 0.0000] | 6 | 1.0000 | — | 0.00 (negligible) | 0.00 | **IDENTICAL_RESULTS** |
| `reloaded_semantic_score` − `reloaded_lambda0` | fidelity_to_oracle | 0.0154 | [-0.0179, 0.0615] | 6 | 1.0000 | 1.0000 | 0.07 (negligible) | 0.19 | **INDICATIVE** |
| `reloaded_semantic_score` − `reloaded_lambda0` | accuracy_real_labels | -0.0026 | [-0.0333, 0.0308] | 6 | 0.8125 | 1.0000 | -0.14 (small) | -0.08 | **INDICATIVE** |
| `reloaded_semantic_score` − `reloaded_lambda0` | node_count | -1.0000 | [-3.3333, 1.3333] | 6 | 0.6250 | — | -0.40 (medium) | -0.19 | **INDICATIVE** |
| `reloaded_owl_features` − `reloaded_lambda0` | fidelity_to_oracle | 0.0103 | [-0.0103, 0.0282] | 6 | 0.5000 | 1.0000 | 0.50 (large) | 0.03 | **INDICATIVE** |
| `reloaded_owl_features` − `reloaded_lambda0` | accuracy_real_labels | 0.0103 | [-0.0154, 0.0308] | 6 | 0.7500 | 1.0000 | 0.30 (medium) | 0.08 | **INDICATIVE** |
| `reloaded_owl_features` − `reloaded_lambda0` | node_count | 0.0000 | [0.0000, 0.0000] | 6 | 1.0000 | — | 0.00 (negligible) | 0.00 | **IDENTICAL_RESULTS** |
| `reloaded_owl_full` − `reloaded_lambda0` | fidelity_to_oracle | 0.0410 | [-0.0103, 0.1026] | 6 | 0.3125 | 1.0000 | 0.60 (large) | 0.31 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_lambda0` | accuracy_real_labels | 0.0128 | [-0.0308, 0.0564] | 6 | 0.6562 | 1.0000 | 0.29 (small) | -0.03 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_lambda0` | node_count | 0.0000 | [-2.0000, 2.0000] | 6 | 1.0000 | — | 0.00 (negligible) | -0.11 | **INDICATIVE** |
| `reloaded_owl_full` − `trepan_original` | fidelity_to_oracle | 0.0410 | [-0.0103, 0.1026] | 6 | 0.3125 | 1.0000 | 0.60 (large) | 0.31 | **INDICATIVE** |
| `reloaded_owl_full` − `trepan_original` | accuracy_real_labels | 0.0128 | [-0.0308, 0.0564] | 6 | 0.6562 | 1.0000 | 0.29 (small) | -0.03 | **INDICATIVE** |
| `reloaded_owl_full` − `trepan_original` | node_count | 0.0000 | [-2.0000, 2.0000] | 6 | 1.0000 | — | 0.00 (negligible) | -0.11 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | fidelity_to_oracle | -0.0128 | [-0.0538, 0.0205] | 6 | 0.8750 | 1.0000 | -0.20 (small) | -0.08 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | node_count | 1.0000 | [-0.6667, 3.0000] | 6 | 0.7500 | — | 0.50 (large) | 0.19 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | fidelity_to_oracle | -0.1026 | [-0.1333, -0.0641] | 6 | 0.0312 | 0.5938 | -1.00 (large) | -0.72 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | node_count | 0.3333 | [-0.6667, 1.3333] | 6 | 1.0000 | — | 0.33 (medium) | 0.14 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | fidelity_to_oracle | -0.0385 | [-0.0923, 0.0231] | 6 | 0.3125 | 1.0000 | -0.52 (large) | -0.31 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | node_count | -0.3333 | [-1.3333, 0.6667] | 6 | 1.0000 | — | -0.33 (medium) | -0.19 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | fidelity_to_oracle | -0.0846 | [-0.1436, -0.0231] | 6 | 0.0938 | 1.0000 | -0.86 (large) | -0.64 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | node_count | -0.6667 | [-1.6667, 0.6667] | 6 | 0.6250 | — | -0.50 (large) | -0.22 | **INDICATIVE** |


**Decomposição da fidelidade (Δ face a λ=0 salvo indicação):**

- arquitectura (λ=0 vs Original): 0.0000; features OWL: 0.0103; score semântico: 0.0154; OWL completa vs λ=0: 0.0410; total vs Original: 0.0410; real − shuffled: 0.0256.

## Ablações não aplicáveis neste runner (e porquê)

- **sem_reasoner**: o reasoner é pré-requisito do quality gate OWL; desligá-lo invalida a ontologia (não exposto como switch).
- **sem_relational_features**: o OntologyProcessor não expõe um switch independente para features relacionais.
- **sem_aggregates**: idem: agregados são gerados no mesmo passo que as demais features derivadas.
- **sem_constraints**: constraints de domínio só actuam na geração de queries do extractor da GUI (não neste runner).
- **sem_semantic_pruning**: o Reloaded actual não implementa poda semântica (só colapso de subárvores idênticas).
