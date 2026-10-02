# BIURI / TREPAN Reloaded V9.2 — Auditoria experimental controlada

Data: 29/09/2026

## Problema corrigido

A ablação anterior ainda tinha um confundidor: o braço sem OWL usava o MLP Original como oráculo, enquanto o braço com OWL construía um `ResidualOntologicalOracle`. Logo, uma diferença de desempenho poderia resultar da troca de professor e não da ontologia.

## Implementação

Foi criado `core/controlled_trepan_experiment.py` com:

- `ControlledTrepanConfig`;
- `OriginalOracleProjection`;
- `fit_controlled_trepan_pair`;
- `evaluate_controlled_trepan_pair`;
- `oracle_health_gate`.

O protocolo não aceita `X_test`/`y_test` durante o fit. O teste bloqueado só entra em `evaluate_controlled_trepan_pair`, depois de ambos os braços estarem congelados.

### Mesma identidade do oráculo

O TREPAN Original consulta o MLP Original directamente.

O Reloaded enriquecido consulta `OriginalOracleProjection`, que apenas selecciona as features originais do vector enriquecido e chama o **mesmo objecto MLP**. Nenhum MLP residual/ontológico é treinado nesta ablação.

### Mesmo orçamento

O protocolo verifica programaticamente:

- mesma seed;
- mesmo `max_nodes`;
- mesma profundidade;
- mesmo `min_sample`;
- mesmo `max_queries`;
- mesmo `max_n`;
- mesmo beam width;
- mesmo `min_samples_leaf`.

Divergência nesses itens aborta a experiência.

### Gate do oráculo

Criado gate exclusivamente por CV do treino. O MLP é válido apenas quando:

1. supera o DummyClassifier pela margem configurada;
2. não fica abaixo do C4.5-Nativo por mais que a margem configurada.

Nenhum teste/holdout é consultado pelo gate.

### Estatística OWL

`paired_owl_analysis()` foi corrigido para usar dataset como unidade estatística. As repetições são agregadas primeiro dentro de cada dataset. Datasets com `oracle_valid=false` são excluídos do efeito agregado e listados explicitamente.

## Testes adicionados

`tests/test_controlled_trepan_ablation_v92.py` cobre:

- igualdade exacta Original vs Reloaded com semântica neutra;
- mesmo MLP através de `OriginalOracleProjection` no espaço aumentado;
- teste final ausente do fit e avaliação final pareada única;
- ausência de `ResidualOntologicalOracle` na ablação OWL;
- estatística agregada por dataset;
- gate MLP vs Dummy e C4.5 somente no treino;
- exclusão de dataset com oráculo inválido.

Os 7 testes passaram.

## Ablação executada neste ambiente

Comando:

```text
python scripts/run_ablation_v9_2.py
```

Resultado real:

```json
{
  "protocol": "controlled_same_oracle_same_seed_same_budget_v9_2",
  "datasets": 5,
  "without_owl_ok": 5,
  "without_owl_invalid_oracle": 0,
  "without_owl_failed": 0,
  "with_owl_executed": 0,
  "owlready2_available": false,
  "note": "Nenhum efeito OWL foi calculado porque o braço OWL não foi executado."
}
```

Nos cinco datasets executados, o controlo neutro produziu:

- `delta_accuracy = 0.0`;
- `delta_balanced_accuracy = 0.0`;
- `delta_oracle_fidelity = 0.0`.

Isto confirma, nesta execução, que o Reloaded sem extensão semântica espelha exactamente o Original sob o protocolo pareado.

## Validação de regressão

Execuções concluídas nesta ronda, em grupos disjuntos:

- 28 passed, 1 skipped, 1 warning — protocolo controlado + núcleo histórico + integração + pipeline científico;
- 6 passed — matriz dataset-agnostic + artefactos;
- 7 passed, 8 warnings — contexto/oráculo Reloaded + espelhamento sem OWL.

Total dos grupos concluídos: **41 testes aprovados, 1 saltado, 0 falhas**.

O teste pesado `test_no_ontology_metrics_identical_to_original` continua fora da declaração de sucesso porque excede o limite do runtime quando executado no fluxo pesado de `MetricsComparator`.

## Não validado neste ambiente

- braço OWL real com `owlready2` + HermiT/Java;
- Python 3.11 e 3.12;
- PyQt6;
- TensorFlow/CLEAR;
- Windows;
- suite completa numa única execução.

Nenhum ganho OWL é reivindicado nesta candidate.
