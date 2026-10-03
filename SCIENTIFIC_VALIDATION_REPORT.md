# SCIENTIFIC VALIDATION REPORT

> Gerado automaticamente por `core.benchmark`. Nenhuma hierarquia de métodos é imposta: os números decidem; resultados nulos e negativos estão na secção própria.

## 1. Protocolo

- Esquema de avaliação: **holdout**; teste final bloqueado (usado uma vez por split, após todos os ajustes; `EvaluationProtocolGuard`). A selecção do oráculo (MLP Original vs Ontológico) usa SÓ CV interna no treino.
- Seeds (fixadas antes dos resultados): [11, 22, 33, 44, 55, 66] · test_size=0.25 · splits estratificados e **pareados** (mesmos índices para todos os braços; `split_hash` guardado).
- Orçamento TREPAN idêntico para todos os braços: {'min_sample': 400, 'max_queries': None, 'max_nodes': 15, 'max_depth': 8, 'max_n': 3, 'beam_width': 2, 'min_samples_leaf': 2, 'max_features_per_node': 12} (regra automática declarada: `min_sample=max(n_train, min(1000, max(120, 3·n_train)))`, `max_queries=max_nodes·min_sample`).
- Paridade de MLPs: mesmo procedimento e mesmo nº de trials para Original e Ontológico (`mlp_trials=0`).
- Gate do oráculo: MLP Ontológico só é professor se o ganho de balanced accuracy na CV interna > 0.01.
- **Accuracy** = vs rótulos reais (`*_real_labels`). **Fidelity** = vs predições do respectivo oráculo (`fidelity_to_oracle`), sempre com o oráculo identificado.
- IC95%: bootstrap percentil (2000 reamostragens); para diferenças, bootstrap dos pares. Testes: Wilcoxon signed-rank pareado (t pareado só informativo; validade exige normalidade e n≥8). Effect sizes: rank-biserial, Cliff's δ, Cohen's dz. Correcção de múltiplas comparações: Holm (família primária: fidelity e accuracy por dataset) e BH (suplementar).
- Níveis de evidência (limiares fixados a priori): STATISTICALLY_SUPPORTED exige ≥10 pares, teste ≥150 amostras, p Holm<0.05, IC sem 0, |rank-biserial|≥0.3 e, para alegações semânticas, controlo negativo ultrapassado; INDICATIVE caso contrário (≥3 pares); MECHANISM_ONLY com <3 pares.
- Folds repetidos não são independentes: nesses esquemas o nível máximo é INDICATIVE e reporta-se também o t corrigido de Nadeau-Bengio.

## 2. Datasets e execuções

| dataset | amostras | features | classes | splits | n_test (min) | ontologia | semântica disponível | manifest |
|---|---|---|---|---|---|---|---|---|
| synthetic_binary | 260 | 9 | 2 | 6 | 65 | sim | 6/6 | `results/benchmark_smoke/20261002T230514_synthetic_binary_732f4521` |
| synthetic_3class | 260 | 12 | 3 | 6 | 65 | sim | 6/6 | `results/benchmark_smoke/20261002T230913_synthetic_3class_732f4521` |

## 3. Resultados — synthetic_binary

### 3.1 Métricas (média ± desvio [IC95%])

| arm | oracle | accuracy_real_labels | balanced_accuracy_real_labels | macro_f1_real_labels | fidelity_to_oracle | node_count | depth | m_of_n_count | semantic_split_count | membership_queries | tree_training_time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlp_original` | n/a | 0.9615 ± 0.0161 [0.9487, 0.9718] | 0.9616 ± 0.0162 [0.9490, 0.9720] | 0.9615 ± 0.0161 [0.9487, 0.9718] | — | — | — | — | — | — | — |
| `mlp_ontological` | n/a | 0.9410 ± 0.0204 [0.9282, 0.9564] | 0.9410 ± 0.0205 [0.9280, 0.9564] | 0.9410 ± 0.0205 [0.9281, 0.9564] | — | — | — | — | — | — | — |
| `c45` | n/a | 0.8692 ± 0.0398 [0.8385, 0.8949] | 0.8692 ± 0.0395 [0.8390, 0.8950] | 0.8690 ± 0.0398 [0.8384, 0.8948] | — | 26.0000 ± 1.6733 [25.0000, 27.3333] | 8.1667 ± 1.3292 [7.3333, 9.1667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 ± 0.0000 [0.0000, 0.0000] | — | 0.1165 ± 0.0124 [0.1074, 0.1254] |
| `trepan_original` | MLP Original | 0.8795 ± 0.0429 [0.8436, 0.9077] | 0.8798 ± 0.0434 [0.8437, 0.9084] | 0.8793 ± 0.0429 [0.8435, 0.9074] | 0.8615 ± 0.0257 [0.8462, 0.8821] | 14.0000 ± 1.0954 [13.3333, 14.6667] | 4.8333 ± 0.9832 [4.1667, 5.5000] | 6.0000 ± 0.0000 [6.0000, 6.0000] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2256.0000 ± 33.6214 [2234.5000, 2283.0000] | 0.4108 ± 0.0269 [0.3918, 0.4300] |
| `reloaded_lambda0` | MLP Original | 0.8795 ± 0.0429 [0.8436, 0.9077] | 0.8798 ± 0.0434 [0.8437, 0.9084] | 0.8793 ± 0.0429 [0.8435, 0.9074] | 0.8615 ± 0.0257 [0.8462, 0.8821] | 14.0000 ± 1.0954 [13.3333, 14.6667] | 4.8333 ± 0.9832 [4.1667, 5.5000] | 6.0000 ± 0.0000 [6.0000, 6.0000] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2256.0000 ± 33.6214 [2234.5000, 2283.0000] | 0.6047 ± 0.0324 [0.5797, 0.6248] |
| `reloaded_semantic_score` | MLP Original | 0.8795 ± 0.0528 [0.8410, 0.9154] | 0.8795 ± 0.0532 [0.8403, 0.9155] | 0.8789 ± 0.0537 [0.8394, 0.9152] | 0.8769 ± 0.0515 [0.8359, 0.9077] | 14.0000 ± 1.6733 [12.6667, 15.0000] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 6.1667 ± 0.9832 [5.5000, 6.8333] | 6.5000 ± 0.8367 [5.8333, 7.0000] | 2284.0000 ± 38.0684 [2256.3333, 2312.5208] | 4.3393 ± 0.4131 [4.0571, 4.6641] |
| `reloaded_owl_features` | MLP Original | 0.8590 ± 0.0562 [0.8205, 0.9000] | 0.8589 ± 0.0561 [0.8205, 0.8999] | 0.8588 ± 0.0563 [0.8203, 0.9000] | 0.8410 ± 0.0453 [0.8077, 0.8718] | 13.3333 ± 1.9664 [12.0000, 14.6667] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 5.6667 ± 1.0328 [4.8333, 6.3333] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2252.5000 ± 36.8877 [2227.0000, 2280.8333] | 1.4324 ± 0.1303 [1.3434, 1.5253] |
| `reloaded_owl_full` | MLP Original | 0.9103 ± 0.0329 [0.8846, 0.9308] | 0.9100 ± 0.0329 [0.8843, 0.9307] | 0.9101 ± 0.0330 [0.8843, 0.9307] | 0.8974 ± 0.0318 [0.8718, 0.9179] | 14.3333 ± 1.0328 [13.6667, 15.0000] | 4.3333 ± 0.8165 [4.0000, 5.0000] | 6.3333 ± 0.5164 [6.0000, 6.6667] | 6.6667 ± 0.5164 [6.3333, 7.0000] | 2276.3333 ± 63.0196 [2230.0000, 2321.5000] | 4.9933 ± 0.2668 [4.7863, 5.1667] |
| `reloaded_owl_shuffled` | MLP Original | 0.9282 ± 0.0303 [0.9051, 0.9487] | 0.9282 ± 0.0304 [0.9050, 0.9487] | 0.9281 ± 0.0303 [0.9050, 0.9487] | 0.9205 ± 0.0545 [0.8769, 0.9564] | 13.3333 ± 1.5055 [12.3333, 14.3333] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 5.5000 ± 1.2247 [4.5000, 6.0000] | 6.1667 ± 0.7528 [5.6667, 6.6667] | 2250.0000 ± 41.7229 [2222.1667, 2283.0000] | 5.0451 ± 0.2963 [4.8476, 5.2743] |
| `reloaded_e2e` | MLP Ontológico | 0.9128 ± 0.0387 [0.8769, 0.9538] | 0.9129 ± 0.0393 [0.8764, 0.9545] | 0.9127 ± 0.0388 [0.8767, 0.9538] | 0.9231 ± 0.0615 [0.8615, 0.9846] | 14.3333 ± 1.1547 [13.0000, 15.0000] | 4.0000 ± 1.0000 [3.0000, 5.0000] | 6.0000 ± 1.0000 [5.0000, 7.0000] | 6.6667 ± 0.5774 [6.0000, 7.0000] | 2248.0000 ± 70.5479 [2200.0000, 2329.0000] | 4.9652 ± 0.1636 [4.8633, 5.1539] |
| `trepan_original_no_mofn` | MLP Original | 0.8282 ± 0.0641 [0.7846, 0.8769] | 0.8280 ± 0.0638 [0.7846, 0.8764] | 0.8279 ± 0.0640 [0.7844, 0.8766] | 0.8205 ± 0.0636 [0.7744, 0.8667] | 13.0000 ± 2.1909 [11.3333, 14.3333] | 4.5000 ± 1.0488 [3.6667, 5.1667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2228.1667 ± 26.4909 [2208.8333, 2246.6667] | 0.3324 ± 0.0380 [0.3080, 0.3614] |
| `reloaded_owl_full_no_mofn` | MLP Original | 0.8128 ± 0.0562 [0.7692, 0.8488] | 0.8128 ± 0.0563 [0.7695, 0.8493] | 0.8123 ± 0.0564 [0.7690, 0.8485] | 0.8154 ± 0.0524 [0.7795, 0.8513] | 12.0000 ± 3.0332 [9.6667, 14.0000] | 4.1667 ± 0.7528 [3.6667, 4.6667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 5.5000 ± 1.5166 [4.3333, 6.5000] | 2243.1667 ± 31.2948 [2220.0000, 2265.8458] | 1.9401 ± 0.1003 [1.8602, 2.0038] |
| `reloaded_owl_full_no_active_queries` | MLP Original | 0.9000 ± 0.0386 [0.8718, 0.9282] | 0.8999 ± 0.0393 [0.8712, 0.9285] | 0.8996 ± 0.0391 [0.8710, 0.9281] | 0.8872 ± 0.0332 [0.8641, 0.9103] | 12.6667 ± 1.9664 [11.3333, 14.0000] | 4.1667 ± 0.9832 [3.5000, 4.8333] | 5.8333 ± 0.9832 [5.1667, 6.5000] | 5.8333 ± 0.9832 [5.1667, 6.5000] | 2263.3333 ± 40.3022 [2231.0000, 2288.0000] | 4.6026 ± 0.2508 [4.4276, 4.7776] |
| `reloaded_owl_full_no_error_focus` | MLP Original | 0.8923 ± 0.0377 [0.8667, 0.9179] | 0.8920 ± 0.0383 [0.8659, 0.9182] | 0.8920 ± 0.0380 [0.8661, 0.9179] | 0.8795 ± 0.0394 [0.8513, 0.9077] | 13.6667 ± 1.6330 [12.3333, 14.6667] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 5.0000 ± 0.8944 [4.3333, 5.6667] | 6.3333 ± 0.8165 [5.6667, 6.8333] | 2252.8333 ± 30.4396 [2229.3333, 2270.5000] | 4.8248 ± 0.1524 [4.7152, 4.9292] |


### 3.2 Comparações pareadas (A − B)

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
| `reloaded_owl_full` − `reloaded_owl_shuffled` | fidelity_to_oracle | -0.0231 | [-0.0564, 0.0051] | 6 | 0.3125 | 1.0000 | -0.60 (large) | -0.31 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | accuracy_real_labels | -0.0179 | [-0.0462, 0.0026] | 6 | 0.3750 | 1.0000 | -0.70 (large) | -0.22 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | node_count | 1.0000 | [-0.6667, 2.3333] | 6 | 0.5000 | — | 0.60 (large) | 0.39 | **INDICATIVE** |
| `reloaded_e2e` − `trepan_original` | accuracy_real_labels | 0.0385 | [0.0026, 0.0821] | 6 | 0.1875 | 1.0000 | 0.80 (large) | 0.56 | **INDICATIVE** |
| end_to_end_vs_original | fidelity_to_oracle | — | — | — | — | — | — | — | not_comparable: oráculos diferentes (['MLP Ontológico', 'MLP Original'] vs ['MLP Original']): fidelity não comparável |
| `reloaded_e2e` − `trepan_original` | node_count | 0.3333 | [-0.6667, 1.3333] | 6 | 1.0000 | — | 0.33 (medium) | 0.17 | **INDICATIVE** |
| `c45` − `trepan_original` | accuracy_real_labels | -0.0103 | [-0.0333, 0.0128] | 6 | 0.6875 | 1.0000 | -0.33 (medium) | -0.14 | **INDICATIVE** |
| `c45` − `trepan_original` | balanced_accuracy_real_labels | -0.0106 | [-0.0339, 0.0122] | 6 | 0.5625 | — | -0.33 (medium) | -0.19 | **INDICATIVE** |
| `c45` − `trepan_original` | node_count | 12.0000 | [10.6667, 13.3333] | 6 | 0.0312 | — | 1.00 (large) | 1.00 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | fidelity_to_oracle | -0.0410 | [-0.0846, -0.0077] | 6 | 0.2500 | 1.0000 | -1.00 (large) | -0.47 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | node_count | -1.0000 | [-3.3333, 0.6667] | 6 | 0.7500 | — | -0.50 (large) | -0.25 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | fidelity_to_oracle | -0.0821 | [-0.1308, -0.0385] | 6 | 0.0312 | 0.5938 | -1.00 (large) | -0.83 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | node_count | -2.3333 | [-4.6667, -0.3333] | 6 | 0.2500 | — | -1.00 (large) | -0.50 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | fidelity_to_oracle | -0.0103 | [-0.0333, 0.0103] | 6 | 0.6250 | 1.0000 | -0.40 (medium) | -0.17 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | node_count | -1.6667 | [-2.6667, -0.6667] | 6 | 0.1250 | — | -1.00 (large) | -0.50 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | fidelity_to_oracle | -0.0179 | [-0.0437, 0.0051] | 6 | 0.5000 | 1.0000 | -0.60 (large) | -0.31 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | node_count | -0.6667 | [-2.3333, 0.6667] | 6 | 0.7500 | — | -0.50 (large) | -0.22 | **INDICATIVE** |


### 3.3 Complexidade × fidelity

| arm | oracle | fidelity_to_oracle | node_count | leaf_count | depth | average_rule_length | m_of_n_count |
|---|---|---|---|---|---|---|---|
| `mlp_original` | n/a | — | — | — | — | — | — |
| `mlp_ontological` | n/a | — | — | — | — | — | — |
| `c45` | n/a | — | 26.0000 ± 1.6733 [25.0000, 27.3333] | 13.5000 ± 0.8367 [13.0000, 14.1667] | 8.1667 ± 1.3292 [7.3333, 9.1667] | 5.0159 ± 0.5330 [4.6317, 5.4061] | 0.0000 ± 0.0000 [0.0000, 0.0000] |
| `trepan_original` | MLP Original | 0.8615 ± 0.0257 [0.8462, 0.8821] | 14.0000 ± 1.0954 [13.3333, 14.6667] | 7.5000 ± 0.5477 [7.1667, 7.8333] | 4.8333 ± 0.9832 [4.1667, 5.5000] | 3.4405 ± 0.3524 [3.1935, 3.6905] | 6.0000 ± 0.0000 [6.0000, 6.0000] |
| `reloaded_lambda0` | MLP Original | 0.8615 ± 0.0257 [0.8462, 0.8821] | 14.0000 ± 1.0954 [13.3333, 14.6667] | 7.5000 ± 0.5477 [7.1667, 7.8333] | 4.8333 ± 0.9832 [4.1667, 5.5000] | 3.4405 ± 0.3524 [3.1935, 3.6905] | 6.0000 ± 0.0000 [6.0000, 6.0000] |
| `reloaded_semantic_score` | MLP Original | 0.8769 ± 0.0515 [0.8359, 0.9077] | 14.0000 ± 1.6733 [12.6667, 15.0000] | 7.5000 ± 0.8367 [6.8333, 8.0000] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 3.2966 ± 0.1698 [3.1875, 3.4246] | 6.1667 ± 0.9832 [5.5000, 6.8333] |
| `reloaded_owl_features` | MLP Original | 0.8410 ± 0.0453 [0.8077, 0.8718] | 13.3333 ± 1.9664 [12.0000, 14.6667] | 7.1667 ± 0.9832 [6.5000, 7.8333] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 3.2431 ± 0.2464 [3.0764, 3.4375] | 5.6667 ± 1.0328 [4.8333, 6.3333] |
| `reloaded_owl_full` | MLP Original | 0.8974 ± 0.0318 [0.8718, 0.9179] | 14.3333 ± 1.0328 [13.6667, 15.0000] | 7.6667 ± 0.5164 [7.3333, 8.0000] | 4.3333 ± 0.8165 [4.0000, 5.0000] | 3.2500 ± 0.3260 [3.0625, 3.5208] | 6.3333 ± 0.5164 [6.0000, 6.6667] |
| `reloaded_owl_shuffled` | MLP Original | 0.9205 ± 0.0545 [0.8769, 0.9564] | 13.3333 ± 1.5055 [12.3333, 14.3333] | 7.1667 ± 0.7528 [6.6667, 7.6667] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 3.1925 ± 0.2236 [3.0198, 3.3452] | 5.5000 ± 1.2247 [4.5000, 6.0000] |
| `reloaded_e2e` | MLP Ontológico | 0.9231 ± 0.0615 [0.8615, 0.9846] | 14.3333 ± 1.1547 [13.0000, 15.0000] | 7.6667 ± 0.5774 [7.0000, 8.0000] | 4.0000 ± 1.0000 [3.0000, 5.0000] | 3.2024 ± 0.3241 [2.8571, 3.5000] | 6.0000 ± 1.0000 [5.0000, 7.0000] |
| `trepan_original_no_mofn` | MLP Original | 0.8205 ± 0.0636 [0.7744, 0.8667] | 13.0000 ± 2.1909 [11.3333, 14.3333] | 7.0000 ± 1.0954 [6.1667, 7.6667] | 4.5000 ± 1.0488 [3.6667, 5.1667] | 3.2214 ± 0.5146 [2.8425, 3.5863] | 0.0000 ± 0.0000 [0.0000, 0.0000] |
| `reloaded_owl_full_no_mofn` | MLP Original | 0.8154 ± 0.0524 [0.7795, 0.8513] | 12.0000 ± 3.0332 [9.6667, 14.0000] | 6.5000 ± 1.5166 [5.3333, 7.5000] | 4.1667 ± 0.7528 [3.6667, 4.6667] | 3.0556 ± 0.4399 [2.7083, 3.3333] | 0.0000 ± 0.0000 [0.0000, 0.0000] |
| `reloaded_owl_full_no_active_queries` | MLP Original | 0.8872 ± 0.0332 [0.8641, 0.9103] | 12.6667 ± 1.9664 [11.3333, 14.0000] | 6.8333 ± 0.9832 [6.1667, 7.5000] | 4.1667 ± 0.9832 [3.5000, 4.8333] | 3.0734 ± 0.3223 [2.8373, 3.2817] | 5.8333 ± 0.9832 [5.1667, 6.5000] |
| `reloaded_owl_full_no_error_focus` | MLP Original | 0.8795 ± 0.0394 [0.8513, 0.9077] | 13.6667 ± 1.6330 [12.3333, 14.6667] | 7.3333 ± 0.8165 [6.6667, 7.8333] | 4.5000 ± 0.5477 [4.1667, 4.8333] | 3.2788 ± 0.3058 [3.0565, 3.5060] | 5.0000 ± 0.8944 [4.3333, 5.6667] |


Árvores maiores não são automaticamente melhores: avaliar fidelity **versus** complexidade.

**Atribuição (fidelity_to_oracle)**: NÃO atribuível à ontologia: real ≈ shuffled (efeito de arquitectura/features/ruído).

## 3. Resultados — synthetic_3class

### 3.1 Métricas (média ± desvio [IC95%])

| arm | oracle | accuracy_real_labels | balanced_accuracy_real_labels | macro_f1_real_labels | fidelity_to_oracle | node_count | depth | m_of_n_count | semantic_split_count | membership_queries | tree_training_time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlp_original` | n/a | 0.8897 ± 0.0343 [0.8692, 0.9179] | 0.8896 ± 0.0346 [0.8687, 0.9181] | 0.8899 ± 0.0344 [0.8691, 0.9181] | — | — | — | — | — | — | — |
| `mlp_ontological` | n/a | 0.8949 ± 0.0369 [0.8769, 0.9256] | 0.8947 ± 0.0373 [0.8763, 0.9256] | 0.8953 ± 0.0366 [0.8774, 0.9257] | — | — | — | — | — | — | — |
| `c45` | n/a | 0.8231 ± 0.0693 [0.7821, 0.8795] | 0.8230 ± 0.0694 [0.7821, 0.8796] | 0.8221 ± 0.0689 [0.7819, 0.8787] | — | 31.0000 ± 2.1909 [29.6667, 32.6667] | 7.0000 ± 1.2649 [6.1667, 8.0000] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 ± 0.0000 [0.0000, 0.0000] | — | 0.1963 ± 0.0078 [0.1915, 0.2024] |
| `trepan_original` | MLP Original | 0.8000 ± 0.0487 [0.7590, 0.8282] | 0.8007 ± 0.0483 [0.7602, 0.8291] | 0.7970 ± 0.0516 [0.7539, 0.8276] | 0.8000 ± 0.0592 [0.7538, 0.8436] | 13.6667 ± 2.4221 [11.6667, 15.0000] | 4.3333 ± 0.8165 [3.6667, 4.8333] | 5.0000 ± 0.8944 [4.3333, 5.6667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2231.3333 ± 32.9889 [2209.6667, 2256.0000] | 0.6815 ± 0.0668 [0.6386, 0.7320] |
| `reloaded_lambda0` | MLP Original | 0.8000 ± 0.0487 [0.7590, 0.8282] | 0.8007 ± 0.0483 [0.7602, 0.8291] | 0.7970 ± 0.0516 [0.7539, 0.8276] | 0.8000 ± 0.0592 [0.7538, 0.8436] | 13.6667 ± 2.4221 [11.6667, 15.0000] | 4.3333 ± 0.8165 [3.6667, 4.8333] | 5.0000 ± 0.8944 [4.3333, 5.6667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2231.3333 ± 32.9889 [2209.6667, 2256.0000] | 0.9556 ± 0.1682 [0.8371, 1.0813] |
| `reloaded_semantic_score` | MLP Original | 0.7974 ± 0.0750 [0.7436, 0.8487] | 0.7975 ± 0.0751 [0.7431, 0.8491] | 0.7974 ± 0.0747 [0.7437, 0.8488] | 0.8154 ± 0.0533 [0.7769, 0.8538] | 12.6667 ± 2.6583 [10.6667, 14.3333] | 4.0000 ± 0.8944 [3.3333, 4.6667] | 5.8333 ± 1.3292 [4.8333, 6.6667] | 5.8333 ± 1.3292 [4.8333, 6.6667] | 2224.8333 ± 21.9218 [2210.1625, 2242.3333] | 5.3067 ± 0.2340 [5.1344, 5.4630] |
| `reloaded_owl_features` | MLP Original | 0.8103 ± 0.0409 [0.7795, 0.8410] | 0.8107 ± 0.0415 [0.7799, 0.8413] | 0.8081 ± 0.0411 [0.7776, 0.8377] | 0.8103 ± 0.0589 [0.7718, 0.8564] | 13.6667 ± 2.4221 [11.6667, 15.0000] | 4.1667 ± 0.7528 [3.6667, 4.6667] | 5.0000 ± 0.8944 [4.3333, 5.6667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2226.0000 ± 33.6927 [2204.4958, 2251.8333] | 1.6122 ± 0.1391 [1.5072, 1.7140] |
| `reloaded_owl_full` | MLP Original | 0.8128 ± 0.0626 [0.7718, 0.8641] | 0.8131 ± 0.0630 [0.7720, 0.8647] | 0.8126 ± 0.0619 [0.7722, 0.8627] | 0.8410 ± 0.0719 [0.7949, 0.9000] | 13.6667 ± 1.6330 [12.3333, 14.6667] | 4.1667 ± 0.4082 [4.0000, 4.5000] | 6.0000 ± 0.6325 [5.5000, 6.5000] | 6.3333 ± 0.8165 [5.6667, 6.8333] | 2213.1667 ± 29.1027 [2193.4875, 2233.8417] | 5.2206 ± 0.1723 [5.0983, 5.3485] |
| `reloaded_owl_shuffled` | MLP Original | 0.8000 ± 0.0708 [0.7461, 0.8487] | 0.8000 ± 0.0704 [0.7462, 0.8482] | 0.7998 ± 0.0717 [0.7445, 0.8476] | 0.8154 ± 0.0524 [0.7744, 0.8487] | 14.3333 ± 1.6330 [13.0000, 15.0000] | 4.1667 ± 0.4082 [4.0000, 4.5000] | 6.3333 ± 0.8165 [5.6667, 6.8333] | 6.6667 ± 0.8165 [6.0000, 7.0000] | 2232.1667 ± 15.6258 [2221.1625, 2243.0000] | 5.5303 ± 0.2449 [5.3754, 5.7253] |
| `reloaded_e2e` | MLP Ontológico | 0.7436 ± 0.0582 [0.6769, 0.7846] | 0.7422 ± 0.0562 [0.6782, 0.7835] | 0.7404 ± 0.0567 [0.6777, 0.7879] | 0.7641 ± 0.0540 [0.7077, 0.8154] | 14.3333 ± 1.1547 [13.0000, 15.0000] | 4.0000 ± 1.0000 [3.0000, 5.0000] | 6.6667 ± 0.5774 [6.0000, 7.0000] | 6.6667 ± 0.5774 [6.0000, 7.0000] | 2238.0000 ± 48.5077 [2209.0000, 2294.0000] | 5.3845 ± 0.2517 [5.2249, 5.6747] |
| `trepan_original_no_mofn` | MLP Original | 0.7692 ± 0.0741 [0.7179, 0.8205] | 0.7689 ± 0.0744 [0.7173, 0.8205] | 0.7683 ± 0.0738 [0.7172, 0.8194] | 0.7872 ± 0.0562 [0.7487, 0.8256] | 14.6667 ± 0.8165 [14.0000, 15.0000] | 4.3333 ± 0.5164 [4.0000, 4.6667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 2199.1667 ± 31.3523 [2177.1667, 2223.1708] | 0.4461 ± 0.0343 [0.4221, 0.4724] |
| `reloaded_owl_full_no_mofn` | MLP Original | 0.7256 ± 0.0711 [0.6769, 0.7769] | 0.7253 ± 0.0712 [0.6758, 0.7775] | 0.7224 ± 0.0748 [0.6699, 0.7775] | 0.7385 ± 0.0667 [0.6923, 0.7897] | 14.0000 ± 1.6733 [12.6667, 15.0000] | 4.3333 ± 0.5164 [4.0000, 4.6667] | 0.0000 ± 0.0000 [0.0000, 0.0000] | 6.5000 ± 0.8367 [5.8333, 7.0000] | 2220.3333 ± 19.4902 [2205.0000, 2232.3333] | 2.0561 ± 0.0860 [1.9957, 2.1231] |
| `reloaded_owl_full_no_active_queries` | MLP Original | 0.7872 ± 0.0787 [0.7308, 0.8410] | 0.7868 ± 0.0788 [0.7302, 0.8407] | 0.7862 ± 0.0794 [0.7296, 0.8407] | 0.8026 ± 0.0756 [0.7513, 0.8564] | 13.3333 ± 0.8165 [13.0000, 14.0000] | 4.5000 ± 0.8367 [4.0000, 5.1667] | 6.0000 ± 0.6325 [5.5000, 6.5000] | 6.1667 ± 0.4082 [6.0000, 6.5000] | 2227.8333 ± 18.5194 [2214.4958, 2240.5042] | 4.8584 ± 0.3894 [4.6585, 5.1826] |
| `reloaded_owl_full_no_error_focus` | MLP Original | 0.7564 ± 0.0500 [0.7204, 0.7923] | 0.7570 ± 0.0488 [0.7212, 0.7918] | 0.7527 ± 0.0556 [0.7092, 0.7927] | 0.7564 ± 0.0587 [0.7128, 0.8000] | 13.0000 ± 1.7889 [11.6667, 14.3333] | 4.1667 ± 0.9832 [3.5000, 5.0000] | 5.5000 ± 1.3784 [4.5000, 6.5000] | 6.0000 ± 0.8944 [5.3333, 6.6667] | 2211.3333 ± 58.6742 [2162.0000, 2244.3333] | 5.4034 ± 0.3377 [5.1969, 5.6806] |


### 3.2 Comparações pareadas (A − B)

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
| `reloaded_owl_full` − `reloaded_owl_shuffled` | fidelity_to_oracle | 0.0256 | [-0.0282, 0.0795] | 6 | 0.5000 | 1.0000 | 0.47 (medium) | 0.03 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | accuracy_real_labels | 0.0128 | [-0.0564, 0.0744] | 6 | 0.7500 | 1.0000 | 0.19 (small) | -0.06 | **INDICATIVE** |
| `reloaded_owl_full` − `reloaded_owl_shuffled` | node_count | -0.6667 | [-1.3333, 0.0000] | 6 | 0.5000 | — | -1.00 (large) | -0.28 | **INDICATIVE** |
| `reloaded_e2e` − `trepan_original` | accuracy_real_labels | 0.0000 | [-0.0410, 0.0462] | 6 | 0.9375 | 1.0000 | 0.05 (negligible) | -0.08 | **INDICATIVE** |
| end_to_end_vs_original | fidelity_to_oracle | — | — | — | — | — | — | — | not_comparable: oráculos diferentes (['MLP Ontológico', 'MLP Original'] vs ['MLP Original']): fidelity não comparável |
| `reloaded_e2e` − `trepan_original` | node_count | 0.0000 | [-2.0000, 2.0000] | 6 | 1.0000 | — | 0.00 (negligible) | -0.11 | **INDICATIVE** |
| `c45` − `trepan_original` | accuracy_real_labels | 0.0231 | [-0.0205, 0.0693] | 6 | 0.6250 | 1.0000 | 0.33 (medium) | 0.00 | **INDICATIVE** |
| `c45` − `trepan_original` | balanced_accuracy_real_labels | 0.0222 | [-0.0212, 0.0689] | 6 | 0.6875 | — | 0.24 (small) | -0.06 | **INDICATIVE** |
| `c45` − `trepan_original` | node_count | 17.3333 | [15.3333, 19.6667] | 6 | 0.0312 | — | 1.00 (large) | 1.00 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | fidelity_to_oracle | -0.0128 | [-0.0538, 0.0205] | 6 | 0.8750 | 1.0000 | -0.20 (small) | -0.08 | **INDICATIVE** |
| `trepan_original_no_mofn` − `trepan_original` | node_count | 1.0000 | [-0.6667, 3.0000] | 6 | 0.7500 | — | 0.50 (large) | 0.19 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | fidelity_to_oracle | -0.1026 | [-0.1333, -0.0641] | 6 | 0.0312 | 0.5938 | -1.00 (large) | -0.72 | **INDICATIVE** |
| `reloaded_owl_full_no_mofn` − `reloaded_owl_full` | node_count | 0.3333 | [-0.6667, 1.3333] | 6 | 1.0000 | — | 0.33 (medium) | 0.14 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | fidelity_to_oracle | -0.0385 | [-0.0923, 0.0231] | 6 | 0.3125 | 1.0000 | -0.52 (large) | -0.31 | **INDICATIVE** |
| `reloaded_owl_full_no_active_queries` − `reloaded_owl_full` | node_count | -0.3333 | [-1.3333, 0.6667] | 6 | 1.0000 | — | -0.33 (medium) | -0.19 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | fidelity_to_oracle | -0.0846 | [-0.1436, -0.0231] | 6 | 0.0938 | 1.0000 | -0.86 (large) | -0.64 | **INDICATIVE** |
| `reloaded_owl_full_no_error_focus` − `reloaded_owl_full` | node_count | -0.6667 | [-1.6667, 0.6667] | 6 | 0.6250 | — | -0.50 (large) | -0.22 | **INDICATIVE** |


### 3.3 Complexidade × fidelity

| arm | oracle | fidelity_to_oracle | node_count | leaf_count | depth | average_rule_length | m_of_n_count |
|---|---|---|---|---|---|---|---|
| `mlp_original` | n/a | — | — | — | — | — | — |
| `mlp_ontological` | n/a | — | — | — | — | — | — |
| `c45` | n/a | — | 31.0000 ± 2.1909 [29.6667, 32.6667] | 16.0000 ± 1.0954 [15.3333, 16.8333] | 7.0000 ± 1.2649 [6.1667, 8.0000] | 4.7053 ± 0.3090 [4.5116, 4.9583] | 0.0000 ± 0.0000 [0.0000, 0.0000] |
| `trepan_original` | MLP Original | 0.8000 ± 0.0592 [0.7538, 0.8436] | 13.6667 ± 2.4221 [11.6667, 15.0000] | 7.3333 ± 1.2111 [6.3333, 8.0000] | 4.3333 ± 0.8165 [3.6667, 4.8333] | 3.2310 ± 0.3490 [2.9726, 3.4583] | 5.0000 ± 0.8944 [4.3333, 5.6667] |
| `reloaded_lambda0` | MLP Original | 0.8000 ± 0.0592 [0.7538, 0.8436] | 13.6667 ± 2.4221 [11.6667, 15.0000] | 7.3333 ± 1.2111 [6.3333, 8.0000] | 4.3333 ± 0.8165 [3.6667, 4.8333] | 3.2310 ± 0.3490 [2.9726, 3.4583] | 5.0000 ± 0.8944 [4.3333, 5.6667] |
| `reloaded_semantic_score` | MLP Original | 0.8154 ± 0.0533 [0.7769, 0.8538] | 12.6667 ± 2.6583 [10.6667, 14.3333] | 6.8333 ± 1.3292 [5.8333, 7.6667] | 4.0000 ± 0.8944 [3.3333, 4.6667] | 3.0042 ± 0.4363 [2.6778, 3.3056] | 5.8333 ± 1.3292 [4.8333, 6.6667] |
| `reloaded_owl_features` | MLP Original | 0.8103 ± 0.0589 [0.7718, 0.8564] | 13.6667 ± 2.4221 [11.6667, 15.0000] | 7.3333 ± 1.2111 [6.3333, 8.0000] | 4.1667 ± 0.7528 [3.6667, 4.6667] | 3.1893 ± 0.3433 [2.9429, 3.4167] | 5.0000 ± 0.8944 [4.3333, 5.6667] |
| `reloaded_owl_full` | MLP Original | 0.8410 ± 0.0719 [0.7949, 0.9000] | 13.6667 ± 1.6330 [12.3333, 14.6667] | 7.3333 ± 0.8165 [6.6667, 7.8333] | 4.1667 ± 0.4082 [4.0000, 4.5000] | 3.0972 ± 0.2246 [2.9583, 3.2708] | 6.0000 ± 0.6325 [5.5000, 6.5000] |
| `reloaded_owl_shuffled` | MLP Original | 0.8154 ± 0.0524 [0.7744, 0.8487] | 14.3333 ± 1.6330 [13.0000, 15.0000] | 7.6667 ± 0.8165 [7.0000, 8.0000] | 4.1667 ± 0.4082 [4.0000, 4.5000] | 3.1597 ± 0.1852 [3.0208, 3.2708] | 6.3333 ± 0.8165 [5.6667, 6.8333] |
| `reloaded_e2e` | MLP Ontológico | 0.7641 ± 0.0540 [0.7077, 0.8154] | 14.3333 ± 1.1547 [13.0000, 15.0000] | 7.6667 ± 0.5774 [7.0000, 8.0000] | 4.0000 ± 1.0000 [3.0000, 5.0000] | 3.1607 ± 0.2702 [2.8571, 3.3750] | 6.6667 ± 0.5774 [6.0000, 7.0000] |
| `trepan_original_no_mofn` | MLP Original | 0.7872 ± 0.0562 [0.7487, 0.8256] | 14.6667 ± 0.8165 [14.0000, 15.0000] | 7.8333 ± 0.4082 [7.5000, 8.0000] | 4.3333 ± 0.5164 [4.0000, 4.6667] | 3.2500 ± 0.2236 [3.1042, 3.4167] | 0.0000 ± 0.0000 [0.0000, 0.0000] |
| `reloaded_owl_full_no_mofn` | MLP Original | 0.7385 ± 0.0667 [0.6923, 0.7897] | 14.0000 ± 1.6733 [12.6667, 15.0000] | 7.5000 ± 0.8367 [6.8333, 8.0000] | 4.3333 ± 0.5164 [4.0000, 4.6667] | 3.2014 ± 0.2806 [3.0069, 3.4167] | 0.0000 ± 0.0000 [0.0000, 0.0000] |
| `reloaded_owl_full_no_active_queries` | MLP Original | 0.8026 ± 0.0756 [0.7513, 0.8564] | 13.3333 ± 0.8165 [13.0000, 14.0000] | 7.1667 ± 0.4082 [7.0000, 7.5000] | 4.5000 ± 0.8367 [4.0000, 5.1667] | 3.2351 ± 0.3227 [3.0446, 3.5000] | 6.0000 ± 0.6325 [5.5000, 6.5000] |
| `reloaded_owl_full_no_error_focus` | MLP Original | 0.7564 ± 0.0587 [0.7128, 0.8000] | 13.0000 ± 1.7889 [11.6667, 14.3333] | 7.0000 ± 0.8944 [6.3333, 7.6667] | 4.1667 ± 0.9832 [3.5000, 5.0000] | 3.1042 ± 0.4668 [2.8333, 3.5000] | 5.5000 ± 1.3784 [4.5000, 6.5000] |


Árvores maiores não são automaticamente melhores: avaliar fidelity **versus** complexidade.

**Atribuição (fidelity_to_oracle)**: NÃO atribuível à ontologia: real ≈ shuffled (efeito de arquitectura/features/ruído).

## 4. Controlo negativo e ablação

Ver `NEGATIVE_CONTROL_REPORT.md` e `ABLATION_REPORT.md` (mesmos dados, mesmo protocolo).

## 5. Entre datasets

- datasets: 2
- `NEGATIVE_CONTROL_real_vs_shuffled`: {'synthetic_binary': -0.02308, 'synthetic_3class': 0.02564} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `ablation_no_active_queries`: {'synthetic_binary': -0.01026, 'synthetic_3class': -0.03846} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `ablation_no_error_focus`: {'synthetic_binary': -0.01795, 'synthetic_3class': -0.08462} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `ablation_no_mofn`: {'synthetic_binary': -0.08205, 'synthetic_3class': -0.10256} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `ablation_original_no_mofn`: {'synthetic_binary': -0.04103, 'synthetic_3class': -0.01282} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `owl_features_effect`: {'synthetic_binary': -0.02051, 'synthetic_3class': 0.01026} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `owl_full_vs_architecture`: {'synthetic_binary': 0.0359, 'synthetic_3class': 0.04103} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `reloaded_architecture_sanity`: {'synthetic_binary': 0.0, 'synthetic_3class': 0.0} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `reloaded_owl_vs_original_same_oracle`: {'synthetic_binary': 0.0359, 'synthetic_3class': 0.04103} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- `semantic_score_effect`: {'synthetic_binary': 0.01538, 'synthetic_3class': 0.01538} · Wilcoxon entre datasets: None menos de 5 datasets: sem teste entre datasets (apenas descritivo)
- Friedman: {'valid': False, 'reason': 'Friedman exige >=3 datasets completos'}

## 6. Negative / Null Results

- **synthetic_binary**: o gate NÃO aceitou o MLP Ontológico como oráculo em 3/6 splits (ontology_valid=true, mlp_enrichment_accepted=false). O pipeline end-to-end usou o MLP Original nesses splits.
- **synthetic_binary** · `mlp_ontological` é PIOR que `mlp_original` (accuracy_real_labels): -0.0205, IC95% [-0.0282, -0.0103].
- **synthetic_binary** · `reloaded_semantic_score` vs `reloaded_lambda0` (fidelity_to_oracle): diferença 0.0154, IC95% [-0.0282, 0.0538] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_semantic_score` vs `reloaded_lambda0` (accuracy_real_labels): diferença 0.0000, IC95% [-0.0410, 0.0385] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_features` vs `reloaded_lambda0` (fidelity_to_oracle): diferença -0.0205, IC95% [-0.0513, 0.0103] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_features` vs `reloaded_lambda0` (accuracy_real_labels): diferença -0.0205, IC95% [-0.0615, 0.0128] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_full` vs `reloaded_lambda0` (fidelity_to_oracle): diferença 0.0359, IC95% [-0.0000, 0.0718] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_full` vs `reloaded_lambda0` (accuracy_real_labels): diferença 0.0308, IC95% [-0.0077, 0.0795] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_full` vs `trepan_original` (fidelity_to_oracle): diferença 0.0359, IC95% [-0.0000, 0.0718] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_full` vs `trepan_original` (accuracy_real_labels): diferença 0.0308, IC95% [-0.0077, 0.0795] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_full` vs `reloaded_owl_shuffled` (fidelity_to_oracle): diferença -0.0231, IC95% [-0.0564, 0.0051] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `reloaded_owl_full` vs `reloaded_owl_shuffled` (accuracy_real_labels): diferença -0.0179, IC95% [-0.0462, 0.0026] inclui 0 → sem efeito detectável.
- **synthetic_binary** · `c45` vs `trepan_original` (accuracy_real_labels): diferença -0.0103, IC95% [-0.0333, 0.0128] inclui 0 → sem efeito detectável.
- **synthetic_binary** · controlo negativo (fidelity_to_oracle): REAL_APPROX_SHUFFLED (Δ real−shuffled = -0.0231, IC [-0.0564, 0.0051], 6 pares) → o ganho NÃO pode ser atribuído à ontologia.
- **synthetic_binary** · controlo negativo (accuracy_real_labels): REAL_APPROX_SHUFFLED (Δ real−shuffled = -0.0179, IC [-0.0462, 0.0026], 6 pares) → o ganho NÃO pode ser atribuído à ontologia.
- **synthetic_3class**: o gate NÃO aceitou o MLP Ontológico como oráculo em 3/6 splits (ontology_valid=true, mlp_enrichment_accepted=false). O pipeline end-to-end usou o MLP Original nesses splits.
- **synthetic_3class** · `mlp_ontological` vs `mlp_original` (accuracy_real_labels): diferença 0.0051, IC95% [-0.0077, 0.0179] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_semantic_score` vs `reloaded_lambda0` (fidelity_to_oracle): diferença 0.0154, IC95% [-0.0179, 0.0615] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_semantic_score` vs `reloaded_lambda0` (accuracy_real_labels): diferença -0.0026, IC95% [-0.0333, 0.0308] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_features` vs `reloaded_lambda0` (fidelity_to_oracle): diferença 0.0103, IC95% [-0.0103, 0.0282] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_features` vs `reloaded_lambda0` (accuracy_real_labels): diferença 0.0103, IC95% [-0.0154, 0.0308] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_full` vs `reloaded_lambda0` (fidelity_to_oracle): diferença 0.0410, IC95% [-0.0103, 0.1026] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_full` vs `reloaded_lambda0` (accuracy_real_labels): diferença 0.0128, IC95% [-0.0308, 0.0564] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_full` vs `trepan_original` (fidelity_to_oracle): diferença 0.0410, IC95% [-0.0103, 0.1026] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_full` vs `trepan_original` (accuracy_real_labels): diferença 0.0128, IC95% [-0.0308, 0.0564] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_full` vs `reloaded_owl_shuffled` (fidelity_to_oracle): diferença 0.0256, IC95% [-0.0282, 0.0795] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_owl_full` vs `reloaded_owl_shuffled` (accuracy_real_labels): diferença 0.0128, IC95% [-0.0564, 0.0744] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `reloaded_e2e` vs `trepan_original` (accuracy_real_labels): diferença 0.0000, IC95% [-0.0410, 0.0462] inclui 0 → sem efeito detectável.
- **synthetic_3class** · `c45` vs `trepan_original` (accuracy_real_labels): diferença 0.0231, IC95% [-0.0205, 0.0693] inclui 0 → sem efeito detectável.
- **synthetic_3class** · controlo negativo (fidelity_to_oracle): REAL_APPROX_SHUFFLED (Δ real−shuffled = 0.0256, IC [-0.0282, 0.0795], 6 pares) → o ganho NÃO pode ser atribuído à ontologia.
- **synthetic_3class** · controlo negativo (accuracy_real_labels): REAL_APPROX_SHUFFLED (Δ real−shuffled = 0.0128, IC [-0.0564, 0.0744], 6 pares) → o ganho NÃO pode ser atribuído à ontologia.

## 7. Verificação das métricas a partir das predições

- valores recomputados: 1308 · máx. |diferença|: 1.11e-16 · OK

## 8. Limitações

- Evidência limitada pelo nº de seeds/datasets e pelo tamanho do teste; qualquer nível abaixo de STATISTICALLY_SUPPORTED é exploratório.
- TBoxes de benchmark são de domínio e podem ter sido construídas a partir das descrições das features; não são ontologias externas independentes.
- O mirror do Reloaded (`semantic_mirror_applied`) pode devolver o Original sobre features enriquecidas quando não há efeito semântico mensurável; está registado por árvore.
- A fidelity do `reloaded_e2e` pode usar um oráculo diferente (MLP Ontológico): só é comparável via accuracy vs rótulos reais; o runner marca esses contrastes como não comparáveis.
- Fidelity avalia-se no conjunto de teste (não nas queries sintéticas).
- Ablações sem reasoner / relacionais / agregados / constraints / poda semântica: ver `ABLATION_REPORT.md` (não expostas como switches).
- Datasets pequenos: com ~50–150 amostras de teste a variância da fidelity é elevada; use repeated_cv e/ou mais datasets.
