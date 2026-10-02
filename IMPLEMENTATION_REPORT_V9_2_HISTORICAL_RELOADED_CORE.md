# BIURI / TREPAN Reloaded V9.2 — Integração do núcleo histórico no Reloaded

Data: 29/09/2026

## Objectivo desta ronda

Eliminar a dependência estrutural de CART/soft-tree do caminho principal do **TREPAN Reloaded com OWL**, de modo que TREPAN Original e TREPAN Reloaded partilhem a mesma família algorítmica histórica. CART permanece apenas em helpers/baselines legados explicitamente identificados.

## Alterações implementadas

### 1. Novo `TrepanReloadedClassifier`

Criado `core/trepan_reloaded_historical.py`.

A classe herda directamente de `TrepanOriginalClassifier` e conserva:

- crescimento best-first;
- membership queries por nó;
- `min_sample`;
- restrições raiz→nó;
- KDE/frequências marginais;
- testes `m-of-n`;
- teste de pureza;
- pruning histórico.

A extensão Reloaded entra apenas por hooks explícitos:

1. pesos semânticos das features na prioridade/score de splits;
2. projecção semântica das membership queries antes de consultar o oráculo.

Com pesos iguais a 1 e sem projector, o Reloaded produz exactamente a mesma árvore, previsões e quantidade de queries que o Original para a mesma seed/configuração.

### 2. Hooks no núcleo Original

`core/trepan_original.py` ganhou hooks neutros:

- `_feature_priority_scores`;
- `_split_selection_score`;
- `_split_audit_metadata`;
- `_draw_membership_queries`.

No `TrepanOriginalClassifier` estes hooks mantêm o comportamento histórico original. O Reloaded apenas os especializa.

O audit de cada split passa a distinguir:

- `information_gain`: ganho de informação bruto;
- `selection_score`: score realmente usado na selecção.

### 3. Membership queries semanticamente coerentes

No caminho OWL, o Reloaded projecta as queries geradas pelo modelo histórico:

- preferencialmente através do `OntologyProcessor.transform_matrix`, ajustado apenas no treino;
- quando o utilizador fornece uma matriz já enriquecida sem transformer disponível, usa uma projecção empírica baseada exclusivamente em linhas do treino, nunca no teste externo.

Depois da projecção, as restrições do caminho são novamente verificadas antes de a query ser enviada ao oráculo.

### 4. Caminho principal do Reloaded simplificado

`TrepanReloadedExtractor.extract_tree_with_ontology()` já não chama no caminho principal:

- `_extract_faithful_reloaded_tree`;
- `refine_with_active_queries`;
- `SoftDecisionTreeClassifier`;
- `select_global_surrogate`;
- `redistill_to_crisp_tree`;
- `_fit_tree_with_semantic_tuning`;
- `_calibrate_onto_feature_bias_weight_internal`.

Depois do fit/transform ontológico, o método delega directamente para `_extract_historical_reloaded_core()`.

### 5. CART removido da calibração principal

O peso semântico deixa de ser calibrado por uma árvore CART proxy no caminho principal. O valor configurado é tratado como prior semântico explícito e auditável.

`_calibrate_onto_feature_bias_weight_internal` foi preservado apenas para compatibilidade histórica e não é chamado pela extracção principal V9.2.

### 6. CART/soft-tree demovidos a baseline legado

Os imports de active-query, multiobjective, plausible-CF e soft-tree foram retirados do módulo principal porque já não eram usados pela extracção primária.

`DecisionTreeClassifier` deixou de ser importado globalmente. Os caminhos legados que ainda o usam passam por `_legacy_cart_baseline()`, com import lazy e nome explícito.

Isto não remove funcionalidade antiga, mas impede que CART seja confundido com o modelo final TREPAN Reloaded.

### 7. Métrica semântica sem CART

`_calculate_semantic_group_fidelity()` deixou de treinar árvores CART temporárias. A fidelidade por grupo é agora calculada directamente entre o TREPAN histórico e o oráculo nas regiões onde as features do grupo têm maior activação/desvio.

### 8. Artefacto autónomo

Ao terminar `TrepanOriginalClassifier.fit()`, a referência ao oráculo é descartada (`oracle_ = None`). A árvore final não precisa do MLP para `predict`/`predict_proba` e deixa de serializar indirectamente extractor, GUI ou callbacks de treino.

O teste de serialização do Reloaded histórico passou.

## Testes adicionados

Novo ficheiro:

- `tests/test_reloaded_historical_core_v92.py`

Cobre:

- igualdade exacta Reloaded neutro vs Original;
- viés semântico aplicado à procura de splits sem CART;
- membership queries através do projector semântico;
- ausência dos refinadores CART/soft-tree no método principal OWL;
- modelo final nativo `TrepanReloadedClassifier`;
- auditoria `cart_used_for_final=false` / `soft_tree_used_for_final=false`;
- serialização sem closure/oráculo de treino.

`tests/test_true_trepan_integration_v92.py` foi actualizado para verificar `TrepanReloadedClassifier` como núcleo final do Reloaded.

## Comandos e resultados reais

### Testes novos (antes da implementação)

```text
pytest -q tests/test_reloaded_historical_core_v92.py
4 failed
```

Falhas esperadas: módulo ainda inexistente e método principal ainda chamava o refinador legado.

### Depois da implementação

```text
pytest -q tests/test_reloaded_historical_core_v92.py
5 passed
```

### Núcleo histórico + integração

```text
pytest -q tests/test_trepan_original_historical_v92.py tests/test_true_trepan_integration_v92.py
12 passed
```

### Validação crítica alargada

```text
pytest -q \
  tests/test_reloaded_historical_core_v92.py \
  tests/test_trepan_original_historical_v92.py \
  tests/test_true_trepan_integration_v92.py \
  tests/test_dataset_agnostic_matrix_v92.py \
  tests/test_artifacts_v92.py \
  tests/test_trepan_reloaded_context.py \
  tests/test_trepan_reloaded_oracle.py \
  tests/test_no_ontology_mirror_original.py::test_mirror_original_tree_same_predictions

30 passed, 8 warnings in 28.79s
```

Os 8 warnings são `ConvergenceWarning` de MLPs de teste e não falhas do TREPAN.

### Teste pesado não concluído

`tests/test_no_ontology_mirror_original.py::test_no_ontology_metrics_identical_to_original` excedeu 120 s neste runtime. Não é declarado aprovado nem reprovado.

## Ambiente executado

- Python: 3.13.5
- NumPy: 2.3.5
- pandas: 2.2.3
- scikit-learn: 1.8.0

Este runtime continua fora da matriz-alvo declarada 3.11/3.12; por isso compatibilidade nesses intérpretes requer CI real.

## Integridade do protocolo confirmatório

`results/confirmatory_v7/` foi comparado contra a candidate anterior:

- 5 ficheiros antes;
- 5 ficheiros depois;
- todos os SHA-256 idênticos.

`run_confirmatory_locked.py --execute` não foi executado.

## Estado científico após esta ronda

O modelo final segue agora:

```text
TREPAN Original histórico
        │
        ├── best-first
        ├── membership queries por nó
        ├── min_sample
        ├── constraints
        ├── KDE/frequências
        ├── Information Gain
        ├── m-of-n
        └── pruning

TREPAN Reloaded
        │
        └── MESMO núcleo histórico
             + features OWL
             + prior semântico no split
             + membership queries projectadas semanticamente
```

Assim, CART/soft-tree deixam de constituir o modelo final ou um passo obrigatório da extracção Reloaded. Permanecem somente em caminhos legados/baselines auxiliares preservados por compatibilidade.

## O que ainda NÃO foi validado nesta ronda

- suite completa num único `pytest`;
- Python 3.11 e 3.12;
- PyQt6 real;
- HermiT/Java e ontologia OWL real ponta-a-ponta;
- Windows;
- TensorFlow/CLEAR;
- o teste pesado de métricas de espelhamento que excedeu o timeout.

Nada desses pontos é declarado concluído.

### Repetição agregada final

Uma repetição posterior do mesmo conjunto, executada juntamente com `compileall`, ultrapassou o limite de 120 s após apresentar 20 testes concluídos. A execução anterior de 30 testes terminou integralmente e permanece o resultado válido reportado. O `compileall` foi depois repetido isoladamente e terminou com `COMPILEALL_OK`.
