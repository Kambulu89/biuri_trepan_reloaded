# BIURI / TREPAN Reloaded V9.2 — Semantic Audit & Metrics Fix

Data: 2026-09-29

## Objectivo

Corrigir os problemas observados na GUI após a separação dos gates ontológicos, mantendo o sistema agnóstico a datasets.

## Correcções implementadas

1. **MLP Ontológico no gráfico**
   - O comparador só cria a série `mlp_ontological` quando o oráculo activo é realmente ontológico e distinto do MLP Original.
   - Wrappers `ORACLE_TYPE=projected_original` são tratados como MLP Original e não aparecem como um segundo professor.

2. **Fidelidade Original vs Reloaded**
   - `overall_fidelity` e `fidelity_to_mlp_original` passam a usar a concordância exacta `accuracy_score` sobre as mesmas predições.
   - Bootstrap permanece apenas como intervalo de confiança e guarda `point_estimate`.
   - Isto elimina diferenças artificiais como 90,7% vs 90,6% causadas por comparar média bootstrap de um lado com estimativa pontual do outro.

3. **Auditoria semântica por nó**
   - Para cada split do Reloaded é calculado também o vencedor `data-only` usando o mesmo motor histórico, dados, candidatos e limites, mas neutralizando a semântica.
   - Regista: `information_gain`, `selection_score`, `semantic_bonus`, `data_only_test`, `semantic_test`, `decision_changed`, `decision_reinforced` e `ontology_influenced`.
   - Métricas agregadas: `Ontology Usage Rate`, `Semantic Decision Impact`, número de splits alterados/reforçados e bónus semântico médio.

4. **Semântica quando todas as features têm score de matching igual**
   - A normalização de pesos podia transformar 30 features igualmente mapeadas em pesos todos iguais a 1.0, anulando o efeito semântico.
   - Foi acrescentada coerência de grupos ontológicos para regras m-of-n. A regra pode ser reforçada quando os seus literais pertencem ao mesmo grupo OWL, sem nomes de datasets, thresholds hard-coded ou aliases específicos.
   - Grupos são derivados genericamente de metadados OWL (`subPropertyOf`/pais da propriedade, grupo semântico ou conceito mapeado).

5. **Extracção de propriedades/relacionamentos OWL**
   - O extractor recolhe a união de `properties`, `data_properties`, `object_properties` e `annotation_properties` quando disponíveis.
   - Relações `subPropertyOf` entre propriedades passam a entrar na hierarquia e no relatório.
   - Isto evita reportar zero relacionamentos em ontologias que são maioritariamente DatatypeProperties.

6. **Relatório honesto**
   - Fidelidade alta já não implica automaticamente a frase “com conhecimento de domínio”.
   - O relatório distingue fidelidade preditiva de impacto semântico observado.
   - Quando a OWL está válida/mapeada mas não altera/reforça splits, isso é reportado explicitamente.
   - O relatório inclui a auditoria por nó.

## Agnosticismo ao dataset

Nenhuma regra específica para Breast Cancer, Iris, Adult ou outro dataset foi adicionada. Os testes usam dados sintéticos e ontologias dummy. Os ficheiros ARFF/OWL enviados pelo utilizador foram usados apenas para diagnóstico manual.

## Testes adicionados

`tests/test_semantic_audit_metrics_v92.py`

Valida:
- mesma árvore/predições => mesma fidelidade exacta Original/Reloaded;
- professor rejeitado/fallback não aparece como MLP Ontológico;
- `projected_original` não é rotulado como professor ontológico;
- auditoria semântica contém vencedor data-only vs Reloaded;
- `Ontology Usage Rate` e `Semantic Decision Impact` ficam no intervalo válido;
- propriedades OWL são recolhidas mesmo quando a ontologia expõe apenas `data_properties()`;
- pesos individuais iguais ainda podem usar coerência de grupo OWL em regras m-of-n.

## Comandos executados e resultados reais

```text
pytest -q tests/test_semantic_audit_metrics_v92.py
6 passed
```

```text
pytest -q tests/test_reloaded_historical_core_v92.py \
  tests/test_semantic_audit_metrics_v92.py \
  tests/test_ontology_gate_separation_v92.py \
  tests/test_controlled_trepan_ablation_v92.py \
  tests/test_trepan_original_historical_v92.py \
  tests/test_trepan_original_fidelity.py \
  tests/test_gui_explanation_regressions.py
37 passed, 3 warnings
```

```text
pytest -q tests/test_dataset_agnostic_matrix_v92.py \
  tests/test_artifacts_v92.py tests/test_preprocessing_v92.py \
  tests/test_data_contract_v92.py tests/test_evaluation_protocol_v92.py \
  tests/test_mlp_factory_v92.py tests/test_model_cache_schema.py \
  tests/test_reloaded_historical_core_v92.py tests/test_true_trepan_integration_v92.py \
  tests/test_trepan_reloaded_context.py tests/test_trepan_reloaded_oracle.py \
  tests/test_semantic_audit_metrics_v92.py tests/test_ontology_gate_separation_v92.py
63 passed, 7 warnings
```

```text
python -m compileall -q core gui counterfactuals tests scripts
COMPILEALL_OK
```

A suite completa foi iniciada, não apresentou falha antes do limite do runtime, mas excedeu 120 s e foi interrompida aproximadamente após 26%+. Portanto **não é declarada como concluída**.

## Confirmatório congelado

`results/confirmatory_v7/` foi comparado com a candidate de entrada:

```text
5 ficheiros antes
5 ficheiros depois
SHA-256 idênticos: True
```

`run_confirmatory_locked.py --execute` não foi executado.

## Não validado neste ambiente

- PyQt6 GUI real;
- owlready2/HermiT;
- Windows;
- Python 3.11/3.12;
- TensorFlow/CLEAR;
- suite pytest integral até ao fim.
