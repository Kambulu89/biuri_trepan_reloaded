# Resumo — synthetic_3class

- experiment_id: `20261002T230913_synthetic_3class_732f4521`
- amostras/features/classes: 260/12/3 {'0': 85, '1': 87, '2': 88}
- esquema: holdout · splits: 6 · seeds: [11, 22, 33, 44, 55, 66]
- dataset_hash: `1bd8cd6293df7f0e…` · ontology_hash: `ca94db2aa39361ea…`

## Métricas agregadas (média ± desvio [IC95% bootstrap])

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


## Contrastes pareados

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

