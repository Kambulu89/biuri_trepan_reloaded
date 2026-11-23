# Fase 12 — resultados: controlos de atribuição no pipeline de produção (P03)

Análise segundo `FASE12_PREREGISTO.md` (commit `03471b9`, anterior às execuções). Dados em `evidencia/fase12/`.

## 1. Alterações

- `core/production_training.py`: nova função `semantic_attribution_guard`. O relatório (`evaluation.semantic_attribution_guard`) e o manifesto (`semantic_attribution_status`) declaram sempre se a diferença Reloaded − Original foi avaliada contra controlos. Só relata: não muda nenhum modelo.
- `scripts/biuri_cli.py train-production --semantic-attribution`: liga o gate de atribuição que já existia (`AttributionConfig()`: 3 controlos, 3 dobras internas, só treino). A saída da CLI passa a mostrar `semantic_attribution_status`.
- O valor por omissão **não mudou**.
- Testes: `tests/test_production_attribution_guard.py`, 6 aprovados.

## 2. Execução

6 execuções do Breast Cancer de produção (sementes 7, 42 e 123, sem e com gate), todas com código de saída 0, com o código de `03471b9`. Em todas, a accuracy e a fidelidade recalculadas a partir dos modelos guardados coincidem com o relatório.

- **A (sem gate) reproduz exatamente o recálculo anterior** (`PRODUCAO_BC_RECALCULO.md`, código `8375c6c`): as mesmas instâncias de teste, as mesmas previsões de todos os modelos, as mesmas probabilidades do MLP e o mesmo número de nós, nas três sementes. A guarda não mudou nada.
- **B usa as mesmas instâncias de teste e o mesmo MLP do que A**, nas três sementes.

## 3. Resultados

| Semente | Estado da guarda (A, sem gate) | Estado do gate (B) | Ganho médio de fidelidade sobre os controlos (validação interna, só treino) | Vitórias / derrotas / empates (9 comparações) | Modo escolhido para o Reloaded (B) |
|---|---|---|---|---|---|
| 7 | `NOT_EVALUATED_AGAINST_CONTROLS` | `NOT_ATTRIBUTED` | +0,0047 | 5 / 1 / 3 (56 %, abaixo dos 60 % exigidos) | `neutral` |
| 42 | `NOT_EVALUATED_AGAINST_CONTROLS` | `NOT_ATTRIBUTED` | 0,0000 | 1 / 3 / 5 | `neutral` |
| 123 | `NOT_EVALUATED_AGAINST_CONTROLS` | `NOT_ATTRIBUTED` | 0,0000 | 1 / 3 / 5 | `neutral` |

| Semente | Execução | Acc. Original | Acc. Reloaded | Fid. Original | Fid. Reloaded | Nós Original / Reloaded | Estado da contribuição semântica | Tempo (s) |
|---|---|---|---|---|---|---|---|---|
| 7 | A | 0,916 | 0,951 | 0,930 | 0,965 | 57 / 47 | `SEMANTIC_CONTRIBUTION_SUPPORTED` | 1162 |
| 7 | B | 0,916 | 0,916 | 0,930 | 0,930 | 57 / 57 | `NO_OBSERVED_SEMANTIC_CONTRIBUTION` | 862 |
| 42 | A | 0,944 | 0,937 | 0,958 | 0,937 | 27 / 29 | `SEMANTIC_EFFECT_WITHOUT_FIDELITY_GAIN` | 843 |
| 42 | B | 0,944 | 0,944 | 0,958 | 0,958 | 27 / 27 | `NO_OBSERVED_SEMANTIC_CONTRIBUTION` | 1138 |
| 123 | A | 0,937 | 0,951 | 0,909 | 0,923 | 31 / 29 | `SEMANTIC_CONTRIBUTION_SUPPORTED` | 688 |
| 123 | B | 0,937 | 0,937 | 0,909 | 0,909 | 31 / 31 | `NO_OBSERVED_SEMANTIC_CONTRIBUTION` | 1499 |

Os tempos não são comparáveis entre execuções: correram 3 em paralelo e as de B correram antes das de A, em lotes diferentes. A razão B/A por semente vai de 0,74 a 2,18.

## 4. Leitura

1. **Sem gate, o relatório declarava contribuição semântica** (`SEMANTIC_CONTRIBUTION_SUPPORTED`) nas sementes 7 e 123, com +3,5 e +1,4 pp de fidelidade do Reloaded sobre o Original. **Com os controlos, essa diferença não é atribuível à ontologia em nenhuma das três sementes**: a semântica real não bate de forma sistemática os controlos com a mesma maquinaria e sem significado. Com o gate, o Reloaded volta a ser idêntico ao Original (modo `neutral`).
2. Isto confirma o P03 com dados: a diferença Reloaded − Original da produção, sem controlos, não é evidência de ganho semântico. A guarda nova impede que o relatório a apresente como tal (`do_not_attribute_reloaded_gain_to_ontology`).
3. A guarda não alterou nenhum resultado (A ≡ recálculo anterior).

## 5. Decisão pendente (do responsável)

Ligar o gate por omissão mudaria o comportamento da produção: no Breast Cancer, o Reloaded passaria a ser igual ao Original nas três sementes, e cada execução faria mais 3 ajustes de controlos por dobra interna. A alteração desta fase é segura nos dois cenários: o relatório diz sempre se houve controlos.

## 6. Limitações

- Um dataset e três sementes. O gate depende do limiar de vitórias (60 %); a semente 7 ficou em 56 %.
- O gate mede fidelidade em validação interna; a accuracy não entra na decisão.
- A GUI usa outro pipeline (P08) e não foi alterada.
