# METRICS AUDIT

216 chamadas de métricas sklearn em 28 ficheiros (excluindo `core/benchmark/`, o módulo central).

| ficheiro | chamadas |
|---|---|
| `core/metrics_comparator.py` | 39 |
| `core/mlp_optimizer.py` | 20 |
| `core/trepan_reloaded_extractor.py` | 15 |
| `gui/biuri_app_complete.py` | 15 |
| `core/c45_j48_tree.py` | 12 |
| `core/classification_metrics.py` | 12 |
| `core/mlp_diagnostic.py` | 12 |
| `core/ablation_study.py` | 10 |
| `counterfactuals/surrogate_improvement.py` | 9 |
| `core/trepan_original.py` | 8 |
| `counterfactuals/analisis/consolidar_resultados.py` | 7 |
| `core/active_query_engine.py` | 6 |
| `core/multiobjective_tree_selector.py` | 6 |
| `core/oracle_optimization.py` | 6 |
| `core/biomedical_validation.py` | 5 |

## Linhas de fidelity (31)

| ficheiro:linha | 1.º argumento | veredicto |
|---|---|---|
| `core/ablation_study.py:76` | `oracle_pred` | ok (oráculo) |
| `core/ablation_study.py:77` | `original_pred` | ok (oráculo) |
| `core/ablation_study.py:260` | `mlp_pred` | ok (oráculo) |
| `core/ablation_study.py:261` | `mlp_pred` | ok (oráculo) |
| `core/active_query_engine.py:123` | `val_labels` | ok (verificado manualmente: val_labels = _oracle_probabilities(oracle, ref_val)) |
| `core/active_query_engine.py:245` | `val_labels` | ok (verificado manualmente: val_labels = _oracle_probabilities(oracle, ref_val)) |
| `core/c45_j48_tree.py:894` | `y_mlp` | ok (oráculo) |
| `core/c45_j48_tree.py:895` | `y_mlp` | ok (oráculo) |
| `core/controlled_trepan_experiment.py:395` | `oracle_pred` | ok (oráculo) |
| `core/metrics_comparator.py:1384` | `y_mlp_pred` | ok (oráculo) |
| `core/metrics_comparator.py:1389` | `y_mlp_pred` | ok (oráculo) |
| `core/metrics_comparator.py:1390` | `y_mlp_pred` | ok (oráculo) |
| `core/metrics_comparator.py:1464` | `y_mlp_ref` | ok (oráculo) |
| `core/metrics_comparator.py:1465` | `y_mlp_pred` | ok (oráculo) |
| `core/metrics_comparator.py:1470` | `y_mlp_ref` | ok (oráculo) |
| `core/metrics_comparator.py:1471` | `y_mlp_ref` | ok (oráculo) |
| `core/metrics_comparator.py:1522` | `y_mlp_pred` | ok (oráculo) |
| `core/metrics_comparator.py:1527` | `y_mlp_pred` | ok (oráculo) |
| `core/metrics_comparator.py:1528` | `y_mlp_pred` | ok (oráculo) |
| `core/multiobjective_tree_selector.py:71` | `oracle_y` | ok (oráculo) |
| `core/soft_global_tree.py:62` | `oracle_y` | ok (oráculo) |
| `core/surrogate_acceptance.py:140` | `active_pred` | ok (oráculo) |
| `core/trepan_original.py:1147` | `y_oracle_train` | ok (oráculo) |
| `core/trepan_original.py:1160` | `y_oracle_eval` | ok (oráculo) |
| `core/trepan_reloaded_extractor.py:1133` | `y_oracle_eval` | ok (oráculo) |
| `core/trepan_reloaded_extractor.py:3599` | `y_mlp` | ok (oráculo) |
| `core/trepan_scientific_tuning.py:43` | `teacher` | ok (oráculo) |
| `counterfactuals/analisis/consolidar_resultados.py:94` | `y_mlp` | ok (oráculo) |
| `counterfactuals/analisis/consolidar_resultados.py:96` | `y_mlp` | ok (oráculo) |
| `counterfactuals/pipelines/pipeline_improve.py:169` | `mlp.predict(X_val)` | ok (oráculo) |
| `counterfactuals/surrogate_improvement.py:472` | `oracle_predictions` | ok (oráculo) |

**Resumo:** ok=31
