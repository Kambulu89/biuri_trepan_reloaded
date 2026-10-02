# BIURI / TREPAN Reloaded V9.1 - Implementação do prompt de correção

## Base de trabalho

Esta versão parte de `biuri_trepan_reloaded_v9_1_auditfix.zip` e aplica os requisitos do documento `Prompt_para_implementação_e_correção_do_BIURI_TREPAN_Reloaded.pdf`.

## Alteração solicitada pelo utilizador: Precisão na comparação

A comparação visual principal passou a usar **Precision Macro / Precisão Macro**, calculada com `precision_score(y_true, y_pred, average="macro", zero_division=0)`. `accuracy_score` continua calculado apenas como métrica adicional de auditoria e não é usado para preencher barras/campos chamados Precisão.

A interface compara, quando disponíveis: MLP Original, MLP Ontológico, TREPAN Original, C4.5-Nativo e TREPAN Reloaded. A fidelidade continua separada da qualidade contra rótulos reais.

## Requisitos implementados

1. **Métricas canónicas** (`core/classification_metrics.py`): Accuracy, Precision Macro, Precision Weighted, Recall Macro, Macro-F1, Balanced Accuracy, matriz de confusão, TP/FP/FN/TN e previsões positivas por classe, taxa de degenerescência e semântica explícita de cada métrica.
2. **Gate do MLP Ontológico** (`core/ontology_acceptance.py`): conjunção obrigatória de Precision Macro, Recall Macro, Macro-F1, Balanced Accuracy e, no protocolo atual, Accuracy. Se qualquer condição falha, `accepted=False`, `fallback_to_original=True` e o oráculo ativo é o MLP Original.
3. **Espaços de features** (`core/feature_space_contract.py`): `original_space`, `ontological_space` e `residual_space`; lista nominal, número de features, hash do schema, transformador e origem das features; falha explícita por ordem/nome/dimensão incompatíveis.
4. **ModelBundle auditável** (`core/model_bundle.py`): schema fingerprint, transformer id e proveniência opcional no bundle; logs de schema e transformador.
5. **Features OWL** (`core/onto_feature_selector.py`): auditoria de constantes/quase constantes, duplicadas, missing, cardinalidade e proveniência; filtro que rejeita features sem proveniência, duplicadas ou quase constantes.
6. **Protocolo sem fuga** (`core/evaluation_protocol.py`): treino, validação, holdout de aceitação e teste externo bloqueado; uso do teste em seleção gera erro; avaliação final é permitida uma única vez e o relatório expõe flags de leakage.
7. **Contrafactuais** (`core/counterfactual_contract.py`): metadados obrigatórios por lote (professor, feature space, schema, ontologia, dataset, seed, método), validação de mudança da previsão do professor, confiança mínima, semântica, densidade, duplicação, teste e schema; reutilização entre professores/espaços/ontologias é bloqueada.
8. **Contextos de CF na GUI**: TREPAN Original recebe contexto do MLP Original/original_space; TREPAN Reloaded recebe MLP Ontológico/ontological_space quando aceito, ou fallback explícito ao MLP Original. Os contextos carregam hashes e ids.
9. **Gate do TREPAN Reloaded** (`core/surrogate_acceptance.py`): Precision Macro, Recall Macro, Macro-F1, Balanced Accuracy e Accuracy não podem degradar além da tolerância; fidelidade mínima e limite de complexidade continuam obrigatórios.
10. **MLP/convergência**: parâmetros-base atualizados para 2000 iterações, `learning_rate_init=5e-4`, `validation_fraction=0.15`, `n_iter_no_change=30`, `tol=1e-4`, `alpha=1e-4`; `core/mlp_factory.py` fornece fábrica central e avaliação multi-seed; auditoria existente de convergência é preservada.
11. **Comparação e relatórios** (`core/metrics_comparator.py`, `gui/pyqt_metrics_visualizer.py`, `gui/biuri_app_complete.py`): Precision Macro é a métrica principal apresentada; radar e diferenças também usam métricas macro/balanced; fidelidade ao MLP Original e ao oráculo ativo permanece separada.
12. **MLP Ontológico no comparador**: quando existe matriz/oráculo enriquecido, o comparador expõe `mlp_ontological` separadamente do `mlp` original.
13. **Testes novos** (`tests/test_prompt_implementation_v9_1.py`): 14 testes específicos do novo contrato.

## Testes executados neste ambiente

### Compilação

`python -m compileall -q core gui counterfactuals tests` -> **PASS**.

### Regressão científica selecionada

Foram executados 10 ficheiros de testes críticos (contratos do prompt, remediação anterior, schema, protocolo, oráculo residual, MLP degenerado, otimizador, rótulos reais, protocolo científico e melhoria contrafactual):

- **77 passed**
- **0 failed**
- **0 skipped** nesse conjunto
- **20 warnings**, todos `ConvergenceWarning` de fixtures/testes legados que usam `max_iter` baixo; os warnings não foram ocultados.

### Smoke test end-to-end do comparador

Foi executado um smoke test com Iris, MLP, TREPAN Original, C4.5 e TREPAN Reloaded. O comparador produziu **Precision Macro distinta de Accuracy**, provando que a UI pode apresentar Precisão sem reutilizar `accuracy_score`, e o `protocol_audit` confirmou `test_used_for_selection=False`.

## Limitações do ambiente atual

A suíte integral não pôde ser executada aqui porque este runtime não contém `PyQt6`, `owlready2` nem `dtreeviz`. A tentativa de instalar `owlready2` falhou por indisponibilidade de rede/DNS. Java 21 está disponível. Portanto, os testes GUI e os testes que constroem ontologias Owlready/HermiT devem ser executados no ambiente-alvo Windows/Python 3.11 definido pelo projeto antes de uma conclusão confirmatória final.

Nenhum resultado do teste externo foi usado para selecionar hiperparâmetros, features, contrafactuais ou árvores.

## Ficheiros principais alterados/adicionados

- `core/classification_metrics.py` (novo)
- `core/feature_space_contract.py` (novo)
- `core/counterfactual_contract.py` (novo)
- `core/mlp_factory.py` (novo)
- `core/ontology_acceptance.py`
- `core/surrogate_acceptance.py`
- `core/evaluation_protocol.py`
- `core/onto_feature_selector.py`
- `core/model_bundle.py`
- `core/mlp_optimizer.py`
- `core/oracle_optimization.py`
- `core/metrics_comparator.py`
- `gui/biuri_app_complete.py`
- `gui/pyqt_metrics_visualizer.py`
- `tests/test_prompt_implementation_v9_1.py` (novo)

## Estado científico

A versão não fabrica superioridade. O MLP Ontológico e o TREPAN Reloaded só podem ser marcados como aceites quando os gates explícitos passam. Se falharem, o resultado inferior continua reportado e o fallback é acionado; as métricas não são substituídas nem ajustadas artificialmente.
