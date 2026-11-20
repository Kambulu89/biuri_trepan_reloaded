# Correções implementadas

Ramo local `auditoria/missao-master`. Nenhum push foi feito. Nenhum baseline, dataset, resultado histórico ou limiar científico foi alterado.

| Commit | Problema | Ficheiros | Reprodução antes | Verificação depois |
|---|---|---|---|---|
| `af02d5d` | P09 — coluna numérica com marcadores de ausente inferida como categórica | `core/data_contract.py` (`_coerce_numeric_text`), `tests/test_audit_numeric_with_missing_tokens.py` | 5 dos 7 testes novos falhavam com o código original | 7 aprovados |
| `16e6be8` | P01 — gate declarava contribuição semântica sem decisão alterada | `core/semantic_contribution_gate.py`, `tests/test_audit_semantic_contribution_gate.py` | Execução real: Breast Cancer, semente 7, estado «SUPPORTED» com impacto 0 | 3 testes novos + `tests/test_production_semantic_v92.py` aprovados |
| `010619f` | P20 — regressão de `af02d5d`: atributo ARFF nominal `{01,1}` descartado; P21 — ordem declarada do alvo ignorada na produção, `?` no alvo | `core/data_loading.py`, `core/data_contract.py`, `core/evaluation_protocol.py`, `core/pipeline_v92.py`, `core/production_training.py`, `core/benchmark/runner.py`, `gui/biuri_app_complete.py`, `tests/test_arff_declared_schema.py`, `tests/test_identity_metamorphic.py` | 11 de 13 testes ARFF falhavam com o código de `a0ec5dc` (`evidencia/arff_identidade/testes_arff_antes_da_correcao_a0ec5dc.log`) | 13 testes ARFF e 6 de identidade aprovados; ver «Regressão» |

## Detalhe

**P09.** Depois de substituir os marcadores de ausente por `NaN`, as colunas de texto cujos valores presentes são todos números passam a numéricas. O alvo nunca é convertido. Colunas com mistura de texto e números continuam categóricas. Princípio: o tipo inferido não pode depender de existirem ausentes.

**P01.** O gate passa a devolver `NO_SEMANTIC_DECISION_CHANGED` (não aceite) quando o impacto semântico é 0 e nenhuma intervenção EFSR foi aceite. O relatório inclui três campos novos: `evidence_level = "observed_single_split"`, `controls_evaluated = False`, `attributable_to_ontology = False`. O estado `SEMANTIC_CONTRIBUTION_SUPPORTED` mantém-se para os restantes casos, mas agora declara que é um efeito observado numa partição e não uma atribuição.

**P20 e P21 (`010619f`).** O carregador passa o esquema declarado no ARFF (`df.attrs['arff_schema']`). O contrato não converte atributos declarados nominais: `01`, `1` e o ausente ficam três categorias distintas e o atributo chega ao modelo. Com uma ordem de classes declarada (`{1,01}`), `pipeline_v92` e `production_training` treinam com códigos 0..k-1 nessa ordem; previsões, probabilidades e matriz de confusão seguem-na e `?` no alvo nunca é classe. A GUI remove as linhas com alvo ausente antes do treino. O benchmark numérico recusa explicitamente atributos nominais e alvo ausente. Um CSV sem esquema mantém a inferência anterior: sem tipo declarado, `01` e `1` continuam indistinguíveis.

Os testes de identidade (`tests/test_identity_metamorphic.py`: só o nome do ficheiro, só o `dataset_id`, a mesma OWL, previsões do oráculo e de cada árvore comparadas instância a instância, no benchmark, em `production_training` e em `pipeline_v92`) também passam com o código de `a0ec5dc` (`evidencia/arff_identidade/testes_identidade_em_a0ec5dc.log`). A identidade já era preservada; faltava cobertura. Conseguem falhar: uma semente diferente e uma única previsão alterada são detetadas.

### Impacto nas experiências anteriores

| Experiência | Código usado | Passou pelo código afetado? | Categorias suscetíveis de fusão? | Ordem declarada ≠ ordem lexicográfica? | Recalcular? |
|---|---|---|---|---|---|
| Benchmark das fases 1 a 9 (`baseline_main`, `fase2_*` … `fase9`) | Registo de datasets em `validation/benchmark` (`Dataset` em memória; German e Adult com one-hot próprio) | Não: não usa `build_data_contract`, `pipeline_v92` nem `production_training`; `Dataset.from_file` não é usado | — | — | Não |
| Breast Cancer com ontologia enriquecida (`bc_enriched`, `scripts/run_benchmark.py`) | `Dataset.from_file` (ARFF) | Sim, o carregador do benchmark | Não: 30 atributos numéricos | `{M,B}`: a ordem declarada já era usada pelo benchmark; sem `?` no alvo, as novas recusas não se aplicam | Não |
| Produção, Breast Cancer, sementes 42, 7, 123 (`producao_bc/`, `scripts/biuri_cli.py train-production`, 2026-11-11 23:51, depois de `af02d5d`) | `load_tabular` → `build_data_contract` → `production_training` | Sim | Não: 30 atributos numéricos (o scipy já os entrega como números; a conversão não os altera) | **Sim:** declarado `{M,B}`, os modelos treinaram com `['B','M']` ordenado | **Sim, se os números forem usados.** Com a correção, os modelos treinam com `M=0, B=1`; a partição estratificada, o MLP e as árvores podem mudar. Afeta os números citados em `AUDITORIA_ALGORITMOS.md` (achado 3: 23 contra 27 nós, fidelidade +0,007), a reprodução de P01 e a linha da produção em `VALIDACAO_FUNCIONAL.md`. A correção de P01 em si é lógica e está coberta por testes unitários |
| Scripts exploratórios da fase 1 (`evidencia/exp_*.py`, `c45_oracle_variant.py`) | `sklearn.datasets` em memória | Não | — | — | Não |

Nenhuma experiência anterior tinha atributos nominais com valores que parecem números a passar pelo código afetado: **não houve fusão em nenhum resultado registado.** A única consequência para resultados anteriores é a ordem das classes nas três execuções de produção do Breast Cancer. Verificação mínima: voltar a correr uma semente e comparar `split.test_indices`, as métricas e o número de nós com `producao_bc/production_report_seed7.json`.

## Regressão

- Suite completa antes das correções: 1732 aprovados, 40 ignorados (`evidencia/pytest_full_baseline.log`).
- Suite completa depois das correções: 1742 aprovados, 40 ignorados (`evidencia/pytest_full_after_fixes.log`). Os 10 testes a mais são os novos.
- Suite completa depois de P20/P21 (código de `010619f`, executada em `7a4ccf7`): 1773 aprovados, 40 ignorados, 0 falhados, código de saída 0 (`evidencia/arff_identidade/pytest_completo_7a4ccf7.log`). Inclui os 19 testes dirigidos novos e os testes acrescentados nas fases 2 a 9.

## O que não corrigi, e porquê

| Problema | Razão |
|---|---|
| P04 (pré-seleção por variância) | Altera o TREPAN Original, que o `CLAUDE.md` do projeto congela. Não precisa de código: basta `max_features_per_node = nº de atributos`. Proponho-o para um manifest v2. |
| P02, P06, P07 | Alteram o Reloaded ou o protocolo de comparação; exigem um manifest novo e pré-registo. |
| P03, P08 | Mudanças de arquitetura; descritas em `PLANO_CORRECCAO.md`. |
| P05 | Regenerar o manifest é uma decisão do responsável do projeto: fixa uma nova referência. |
| P12, P16 | A missão pede para não eliminar código sem confirmar impacto; há testes que o referenciam. |
