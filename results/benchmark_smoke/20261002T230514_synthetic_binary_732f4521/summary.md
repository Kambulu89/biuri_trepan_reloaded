# Resumo — synthetic_binary

- experiment_id: `20261002T230514_synthetic_binary_732f4521`
- amostras/features/classes: 260/9/2 {'0': 128, '1': 132}
- esquema: holdout · splits: 6 · seeds: [11, 22, 33, 44, 55, 66]
- dataset_hash: `1cc0664bf7bcd69d…` · ontology_hash: `43e2650bc6701e97…`

## Métricas agregadas (média ± desvio [IC95% bootstrap])

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


## Contrastes pareados

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

