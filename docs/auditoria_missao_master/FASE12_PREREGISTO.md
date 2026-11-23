# Fase 12 — pré-registo: controlos de atribuição no pipeline de produção (P03)

Escrito antes das execuções de medição.

## 1. Problema

P03: no pipeline de produção (`core/production_training.py`, `scripts/biuri_cli.py train-production`) não há controlos por omissão (`semantic_attribution=None`). A diferença Reloaded − Original mistura mecanismos semânticos com outros, mas o relatório não o dizia. O gate de atribuição já existe (`core/semantic_attribution.py`): compara, só no treino, a semântica real com controlos que usam a mesma maquinaria sem significado (semântica baralhada e atributos aleatórios). A CLI não o expunha.

## 2. Alterações (só relato e exposição; nenhum modelo muda)

1. `semantic_attribution_guard` no relatório (`evaluation.semantic_attribution_guard`) e no manifesto (`semantic_attribution_status`), sempre presente:

   | Estado | Quando |
   |---|---|
   | `NO_ONTOLOGY` | sem ontologia aceite |
   | `NOT_EVALUATED_AGAINST_CONTROLS` | com ontologia e sem gate (é o caso por omissão) |
   | `ATTRIBUTED` / `NOT_ATTRIBUTED` | decisão do gate |

   A alegação só é `ontology_attributable_gain_supported_on_training_controls` com `ATTRIBUTED`; em todos os outros casos é `do_not_attribute_reloaded_gain_to_ontology`.
2. `biuri_cli train-production --semantic-attribution` liga o gate com `AttributionConfig()` (3 controlos, 3 dobras internas, ganho mínimo 0, vitória em ≥ 60 % das comparações).
3. O valor por omissão **não muda**. Ligar o gate por omissão mudaria o comportamento da produção: quando a semântica não é atribuível, o Reloaded fica idêntico ao Original. Essa decisão é do responsável e fica para depois desta medição.

## 3. Medição

Breast Cancer de produção (ARFF e OWL enriquecida, sementes 7, 42 e 123), com o código desta fase:

- **A, sem gate:** tem de reproduzir as execuções recalculadas em `PRODUCAO_BC_RECALCULO.md` (código `8375c6c`): mesmas instâncias de teste, mesmo oráculo, mesmas previsões. A guarda tem de dizer `NOT_EVALUATED_AGAINST_CONTROLS`.
- **B, com gate** (`--semantic-attribution`).

Reporto por semente:
- o estado do gate e o modo escolhido para o Reloaded (`augmented`, `original_space` ou `neutral`);
- a accuracy e a fidelidade do Reloaded e do Original, em A e em B;
- o número de nós;
- o ganho no teste face aos controlos (`test_attribution`, só relatado);
- o tempo de execução de B face a A.

Nenhum critério decide o valor por omissão. A medição serve para o responsável decidir.

## 4. Testes

`tests/test_production_attribution_guard.py` (6 testes): lógica da guarda; execução por omissão com ontologia (`NOT_EVALUATED_AGAINST_CONTROLS`, também no `manifest.json`); gate com ontologia não informativa (não atribuído); gate com ontologia informativa (o estado da guarda é o do gate); sem ontologia (`NO_ONTOLOGY`); a opção da CLI passa `AttributionConfig` e a ausência dela passa `None`.

## 5. Fora do âmbito

A GUI usa outro pipeline (P08) e não é alterada nesta fase.
