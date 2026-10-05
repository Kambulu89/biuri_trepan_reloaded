# Contrato de experimento científico (oráculo único e congelado)

Para que as diferenças entre TREPAN Original, TREPAN Reloaded e variantes possam ser atribuídas **só ao TREPAN**, o modo
científico/benchmark segue sempre este fluxo (`core/scientific_experiment_contract.py`):

```
dataset → split estratificado → treino/calibração do MLP (só treino) → CONGELAR o oráculo
        → TREPAN Original, TREPAN Reloaded e todas as variantes consultam exatamente esse oráculo
```

## O que é garantido e registado

| Elemento | Como |
|---|---|
| Oráculo imutável | `FrozenOracle`: expõe `predict`/`predict_proba`; `fit`/`partial_fit` levantam `OracleContractViolation`. |
| Identidade | `oracle_id` = hash dos hiperparâmetros + arrays ajustados (pesos, escaladores, Pipelines) + decisões num conjunto-sonda (o treino). |
| Prova por árvore | `scope(label)` regista, por árvore/etapa, o `oracle_id` consultado, nº de chamadas e de queries. |
| Original vs Reloaded | `production_training` verifica `oracle_id_of(pair.base_oracle) == oracle_id_of(pair.reloaded_oracle_for_audit) == oracle_id` (desembrulha `OriginalOracleProjection`); senão levanta `OracleContractViolation`. |
| Oráculo não mudou | `verify_unchanged()` recalcula o hash dos pesos no fim (`unchanged_after`). |
| Relatório | `evaluation["oracle_contract"]` (identidade, escopos `tuning` / `trepan_pair`, `tree_oracle_ids`, `single_oracle_for_all_trees`). |

## Construtores de oráculo (configuráveis, sem nomes de datasets)

`ORACLE_BUILDERS` regista construtores `(Z_treino, y_treino, seed) -> modelo`:

- `factory` — o MLP do caminho de produção (`build_mlp_for_data` + `fit_with_convergence`);
- `robust` — o treino robusto de MLP (`train_robust_mlp_original`), o mesmo mecanismo de seleção usado pela GUI.

`train_production_dataframe(..., oracle_builder="factory"|"robust")` escolhe o construtor; o MLP é treinado só no
treino e, a partir daí, é o mesmo objeto congelado para o tuning e para todas as árvores. A GUI mantém as suas
funcionalidades próprias; o benchmark científico usa este pipeline único e reproduzível.

Uso direto (várias variantes sob o mesmo oráculo):

```python
oracle = build_frozen_oracle(Z_train, y_train, seed=42, builder="factory")
report = run_benchmark_with_frozen_oracle(oracle, {
    "trepan_original": lambda o: TrepanOriginalClassifier(**kw).fit(Z_train, oracle=o, feature_names=names),
    "trepan_reloaded": lambda o: TrepanReloadedClassifier(**kw).fit(Z_train, oracle=o, feature_names=names),
})
# report["oracle"]["oracle_id"], report["trees"][label]["queries"], report["single_oracle"], report["unchanged_after"]
```
