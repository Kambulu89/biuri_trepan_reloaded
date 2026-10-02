# IMPLEMENTATION REPORT — BIURI / TREPAN RELOADED V9.2

## 1. Estado da entrega

Esta entrega é uma **implementação candidata/parcial da V9.2**, não uma declaração de conclusão integral das 13 fases. Foi implementado e executado um caminho novo, agnóstico ao dataset, cobrindo contrato de dados, preprocessing sem leakage, MLP adaptativo, avaliação, artefactos/inferência, TREPAN canónico inicial e ablação sem OWL.

Os pontos que não puderam ser executados ou que ainda mantêm caminhos legados são listados explicitamente neste relatório.

**Nunca foi executado** `scripts/run_confirmatory_locked.py --execute`.

`results/confirmatory_v7/` foi preservado. Os SHA-256 foram calculados antes e depois e a comparação teve **0 diferenças**.

## 2. Ambiente realmente utilizado

Execução local desta sessão:

- Sistema: Linux (container de execução)
- Python: **3.13.5**
- scikit-learn: **1.8.0**
- NumPy: **2.3.5**
- pandas: **2.2.3**
- PyQt6: **não instalado**
- owlready2/HermiT: **não instalado / não validado**
- TensorFlow/CLEAR: **não instalado / não validado**
- Python 3.11: **não disponível neste runtime**
- Python 3.12: **não disponível neste runtime**

O `pyproject.toml` da V9.2 declara Python `>=3.11,<3.13`, e o CI Windows foi configurado para Python 3.11 e 3.12. Como essas versões não estavam instaladas neste runtime, **R5 não foi validada localmente na combinação-alvo**. O facto de os testes seleccionados passarem em Python 3.13.5 é evidência adicional, mas não substitui a matriz requerida 3.11/3.12.

O pacote extraído não continha `.git`. Por isso não foi possível produzir honestamente “um commit por fase” preservando o histórico original. Nenhum commit fictício foi criado.

## 3. Linha de base e protecção do confirmatório congelado

Antes das alterações foi criado um inventário SHA-256 de todos os ficheiros em `results/confirmatory_v7/`. No fim foi repetido o processo. `diff` devolveu ficheiro vazio: **0 linhas de diferença**.

A primeira tentativa da suite completa no estado base encontrou dependências opcionais ausentes (`owlready2`/PyQt6) durante collection. Os testes opcionais foram posteriormente preparados para skip explícito quando a dependência não estiver disponível.

## 4. Fase 10.2 + 10.3 — Hepatitis e bootstrap

### Implementado

- `counterfactuals/datasets/preparar_datasets.py`
  - classe do Hepatitis corrigida para a **primeira coluna**;
  - removida imputação global anterior ao split;
  - `NaN` é preservado para o pipeline de preprocessing.
- `counterfactuals/_bootstrap.py`
  - removido `os.chdir()` do código de biblioteca;
  - caminhos deixam de depender do directório actual do processo.
- `counterfactuals/experimentos/hepatitis/STALE.md`
  - resultados históricos dependentes do processamento antigo marcados como obsoletos; não foram reescritos.
- `tests/test_v92_phase10_fixes.py`.

### Resultado executado

Os 2 testes específicos desta correcção passaram.

### Estado

**Validado no runtime disponível.**

## 5. Fase 1 — Contrato de dados

### Implementado

Novo `core/data_contract.py` com:

- `DataContract` / `ColumnContract` serializáveis;
- alvo por nome ou índice;
- sugestão automática sem confirmação (`alvo_nao_confirmado`);
- marcadores de falta configuráveis (`NaN`, `None`, `""`, `?`, `NA`, `N/A`, `null`);
- validações de alvo com uma só classe, classes demasiado raras, alvo com missing e aparência de regressão;
- tipos inferidos: numérico contínuo/discreto, categórico, binário, texto livre, data/hora, ID-like e constante;
- avisos para elevada taxa de falta, desbalanceamento, n pequeno, matriz larga, duplicados, colunas constantes/ID-like e provável leakage;
- heurística de stump/CV e comparação directa para nomear features potencialmente vazadas.

Novas excepções em `core/scientific_errors.py`, incluindo `DataContractError`, `PreprocessingError`, `TrainingError`, `NonConvergenceError`, `InsufficientSignalError` e `ArtifactCompatibilityError`.

### Testes

`tests/test_data_contract_v92.py` cobre missing, alta cardinalidade, ID, constante, fuga de alvo, labels numéricos/texto, nomes problemáticos e datas.

### Estado

**Implementado e validado pelos testes direccionados.**

## 6. Fase 2 — Pré-processamento dentro do pipeline

### Implementado

Novo `core/preprocessing.py`:

- `DataPreprocessor` sklearn-compatible;
- `ColumnTransformer` construído a partir do contrato;
- imputação mediana para numéricas;
- indicadores de missing quando aplicável;
- one-hot com `handle_unknown="ignore"`;
- frequency encoding para alta cardinalidade;
- descarte de constantes/ID-like/texto conforme o contrato;
- `feature_names_out` e `feature_origins`;
- DataFrame com colunas fora de ordem;
- categorias novas, `None`, `NaN`, `""` e `?` sem falha;
- serialização/round-trip.

Foi também corrigida incompatibilidade observada no scikit-learn 1.8 quando colunas binárias numéricas eram tratadas como categóricas: as colunas categóricas/frequency são normalizadas antes do imputador/encoder.

### Limites ainda existentes

- O fluxo principal V9.2 usa o novo preprocessor, mas nem todos os fluxos históricos foram migrados.
- `counterfactuals/cf_mlp_trainer.py` ainda possui caminho legado próprio.
- O wiring OWL completo com as duas vistas de features não pôde ser executado sem `owlready2/HermiT`.

### Estado

**Implementado no novo pipeline; integração legada parcial.**

## 7. Fase 3 — Fábrica única/adaptativa de MLP

### Implementado

`core/mlp_factory.py` foi refeito para possuir uma fábrica adaptativa baseada em:

- número de amostras;
- número de features;
- número de classes;
- geometria larga (`features >> amostras`).

Perfis incluem:

- amostras pequenas: `lbfgs`, sem early stopping;
- conjuntos maiores: `adam` e early stopping;
- regularização mais forte para datasets largos;
- largura das camadas com piso/teto;
- `signal_capacity_audit()` contra `DummyClassifier`.

`core/mlp_optimizer.py` delega a construção à fábrica e usa grelhas condicionais por solver em vez de combinações semanticamente inúteis.

`core/ablation_study.py` e `core/oracle_optimization.py` também foram apontados para a fábrica adaptativa nos caminhos tocados.

### Pendência

`counterfactuals/cf_mlp_trainer.py` ainda não foi totalmente unificado com a fábrica; portanto a exigência “TODOS os caminhos” não está concluída.

### Estado

**Parcial: núcleo e ablação migrados; contrafactuais legados ainda pendentes.**

## 8. Fase 4 — Convergência e métricas do modelo entregue

### Implementado

`core/mlp_convergence.py` passou a:

- capturar `ConvergenceWarning` durante fit;
- distinguir `max_iter_reached`, `stopped_early_unverified` e `converged`;
- guardar `stopped_by`, loss curve, validation scores e melhor score quando disponível;
- percorrer estimadores de `CalibratedClassifierCV` e agregar estados.

Calibração em `core/oracle_optimization.py`:

- decisão de aceitar/rejeitar calibração usa **somente validação interna do treino**;
- compara balanced accuracy e macro-F1;
- rejeita calibração degradante além da tolerância;
- modelo final é depois ajustado usando o conjunto de treino disponível.

`core/mlp_trainer.py` foi alterado para que a selecção/optimização não consulte `X_test`. Só depois de seleccionado/calibrado o modelo final recebe a avaliação no holdout.

### Estado

**Implementado e testado nos caminhos tocados.**

## 9. Fase 5 — Avaliação e gates

### Implementado

`core/evaluation_protocol.py` ganhou:

- escolha de estratégia por tamanho (`EvaluationStrategy`);
- CV estratificada repetida para n pequeno;
- descriptor de holdout + bootstrap para conjuntos maiores;
- métrica primária dependente de desbalanceamento;
- resolução estatística mínima;
- decisão de não-inferioridade com estado `inconclusive`;
- conjunto de métricas de classificação;
- gate de capacidade preditiva contra Dummy.

### Pendência

A migração de **todos** os gates históricos (oracle/surrogate/C4.5/semânticos) para o novo mecanismo não foi auditada linha a linha em todos os módulos legados.

### Estado

**Infraestrutura implementada; adopção global parcial.**

## 10. Fase 6 — Timeouts, cancelamento, erros e logging

### Implementado

- `core/runtime_control.py`: deadline/cancelamento cooperativo.
- GridSearch do optimizador foi substituído no caminho novo por iteração de candidatos/folds com checks entre unidades de trabalho.
- Optuna recebe callback de stop/deadline nos caminhos tocados.
- novas excepções tipadas.
- `core/logging_config.py`: logging central, JSON opcional, RotatingFileHandler.
- removidos todos os `except:` nus encontrados.

### Auditoria estática final

- `except:` nu: **0**
- `except Exception`: **156**
- chamadas `print()` em `core/`, `gui/`, `counterfactuals/`: **477**

### Conclusão

A exigência de rever todos os `except Exception` e migrar todos os `print()` não foi concluída.

### Estado

**Parcial.**

## 11. Fase 7 — TREPAN Original verdadeiro / nomenclatura

### Implementado

Novo `core/trepan_original.py`:

- `TrepanOriginalClassifier` baseado em `CanonicalTrepanClassifier`;
- expansão best-first/m-of-n herdada do caminho canónico;
- limites de nós/profundidade/queries;
- suporte a consultas ao oráculo e sampling empírico.

Em `core/trepan_extractor.py` foi adicionado o alias honesto:

`DistilledCARTExtractor = TREPANExtractor`

`core/algorithm_identity.py` e `docs/TREPAN_IDENTITY_V9_2.md` documentam a distinção.

### Testes

Os testes sintéticos verificaram node cap, fidelidade elevada num oráculo conhecido e presença de split m-of-n no caso preparado.

### Limitação metodológica

A implementação nova ainda usa amostragem empírica/marginal e **não implementa completamente a amostragem condicionada por nó/KDE que seria necessária para uma reprodução literal do TREPAN original de Craven & Shavlik**.

Além disso, muitos caminhos históricos da GUI/contrafactuais ainda chamam `TREPANExtractor`, isto é, o CART destilado legado.

### Estado

**Parcial e explicitamente não apresentado como reprodução literal completa.**

## 12. Fase 8 — Artefactos, cache e inferência

### Implementado

Novo `core/artifacts.py`:

- directório de artefacto;
- `manifest.json` + `model.joblib`;
- versão do formato;
- versões Python/numpy/pandas/sklearn;
- schema/contrato/preprocessor;
- hiperparâmetros, seeds, métricas, convergência/calibração;
- classes e hashes;
- validação de compatibilidade com `ArtifactCompatibilityError`.

Novo `core/inference.py`:

- `load_artifact`;
- `Predictor.predict`;
- `predict_proba`;
- `explain`;
- `validate`.

`core/model_cache.py`:

- hash passa a abranger todo o DataFrame usando `hash_pandas_object`;
- teste confirma que alterar uma única linha muda a chave.

Novo `scripts/migrate_legacy_pickles_v92.py`.

### Auditoria executada dos pickles

- total encontrado: **64**
- carregáveis: **52**
- incompatíveis/falharam: **12**

Relatório: `results/legacy_pickle_audit_v9_2.json`.

Ocorreram avisos reais de incompatibilidade de versões do scikit-learn durante essa auditoria; não foram ocultados.

### Pendência

O backend histórico do `model_cache` ainda não foi migrado integralmente para armazenar somente o formato de artefacto V9.2; a nova chave está implementada, mas permanecem caminhos pickle legados.

### Estado

**Parcial: formato/API novos validados; cache histórico ainda em transição.**

## 13. Fase 9 — Entrada genérica CLI + GUI

### Implementado

Novo `core/data_loading.py`:

- CSV/TSV;
- ARFF;
- Parquet;
- Excel;
- mensagens accionáveis quando extras opcionais estão ausentes.

Novo `scripts/biuri_cli.py` com comandos headless para treino, avaliação, predição e explicação usando o caminho V9.2.

### Não implementado/validado

A GUI PyQt6 pedida (preview, selector de alvo, overrides, painel de warnings, tradução integral PT-PT) **não foi concluída**. PyQt6 também não está instalado neste runtime, logo testes GUI não foram executados.

Também não foi criado ainda um pacote instalável `python -m biuri`; a CLI disponível é `scripts/biuri_cli.py`.

### Estado

**CLI/loading implementados; GUI pendente.**

## 14. Fase 10 — Remoção de dependências específicas por dataset

### Implementado

- correcções Hepatitis/bootstrap;
- import absoluto no pipeline de treino de contrafactuais;
- teste estático impede nomes de datasets em `core/`/`gui/` fora dos módulos permitidos;
- alteração de identificador importado de owlready para não gerar falso positivo no guard;
- TensorFlow passou a extra opcional no packaging.

### Ainda pendente

- `counterfactuals/dataset_config.py` ainda existe e continua ligado a caminhos legados;
- `counterfactuals/cf_mlp_trainer.py` ainda não foi totalmente substituído pelo preprocessor/factory V9.2;
- a totalidade dos pipelines COGS/CLEAR não foi refactorada para receber apenas DataFrame+contract+artifact;
- comportamento de falha estatística do CLEAR não foi auditado ponta a ponta sem TensorFlow.

### Estado

**Parcial.**

## 15. Fase 11 — Ontologia opcional, ablação e oracle health

### Implementado

`core/ablation_study.py` passou a construir o MLP pela fábrica adaptativa e a registar auditoria de capacidade preditiva.

Novo `scripts/run_ablation_v9_2.py` escreve apenas em `results/ablation_v9_2/`.

### Execução real nesta sessão

Braço **sem OWL**:

- datasets executados: **5/5**
- linhas marcadas com oráculo inválido: **0**
- brazo OWL executado: **0**

Deltas de saúde do oráculo impressos pelo runner:

- iris: 0.58618
- wine: 0.64198
- breast_cancer: 0.42858
- diabetes_progression: 0.14337
- digits: 0.88274

Estes nomes aparecem apenas no benchmark/ablação permitido pela especificação; não são presets do pipeline genérico.

OWL foi marcado `not_executed` porque `owlready2/HermiT` não está disponível. **Não foi calculado nem inventado qualquer efeito OWL V9.2.**

A primeira tentativa da ablação detectou duas falhas reais no preprocessing com colunas categóricas numéricas em sklearn 1.8; as falhas foram corrigidas e a execução foi repetida com sucesso no braço sem OWL.

### Estado

**Sem-OWL executado; OWL não validado.**

## 16. Fase 12 — Manutenibilidade, testes e CI

### Implementado

- `tests/test_dataset_agnostic_matrix_v92.py` cobre pipeline completo em cenários sintéticos, incluindo mixed categorical/missing/ID/constant, wide e imbalance/multiclass;
- testes específicos para contrato, preprocessing, MLP, convergência, avaliação, runtime, artefactos, carregamento, TREPAN e guardas estáticas;
- CI Windows com Python 3.11 e 3.12;
- extras opcionais em `pyproject.toml`;
- documentação nova:
  - `docs/DATASET_AGNOSTICO.md`
  - `docs/MIGRACAO_V9_2.md`
  - `docs/TREPAN_IDENTITY_V9_2.md`
  - `CHANGELOG_V9_2.md`
  - este `IMPLEMENTATION_REPORT_V9_2.md`.

### Não concluído

Os monólitos `core/trepan_reloaded_extractor.py`, `core/metrics_comparator.py` e `gui/biuri_app_complete.py` não foram completamente decompostos nesta sessão.

## 17. Testes executados

### Suite crítica final direccionada

Comando equivalente:

```bash
pytest -q \
  tests/test_v92_phase10_fixes.py \
  tests/test_data_contract_v92.py \
  tests/test_preprocessing_v92.py \
  tests/test_mlp_factory_v92.py \
  tests/test_convergence_v92.py \
  tests/test_evaluation_protocol_v92.py \
  tests/test_runtime_control_v92.py \
  tests/test_artifacts_v92.py \
  tests/test_data_loading_v92.py \
  tests/test_trepan_original_v92.py \
  tests/test_dataset_agnostic_matrix_v92.py \
  tests/test_v92_static_guards.py \
  tests/test_mlp_optimizer.py \
  tests/test_audit_remediation_v9_1.py \
  tests/test_prompt_implementation_v9_1.py
```

Resultado real observado anteriormente nesta sessão:

**70 passed in 23.81s**

Vários outros ficheiros legados foram executados individualmente e passaram, incluindo testes de scientific protocol, C4.5, bundle/cache, feature alignment, residual oracle, TREPAN, surrogate improvement e treino.

### Suite completa

`pytest -q` foi tentado mais de uma vez, mas o conjunto completo excedeu o limite de execução desta ferramenta (uma tentativa ultrapassou 240 s e não terminou).

Por isso:

**NÃO se declara a suite completa como passada.**

## 18. Critérios de aceitação A1–A8

| Critério | Estado nesta entrega | Evidência / motivo |
|---|---|---|
| A1 sintéticos ponta-a-ponta | **Parcialmente validado** | matriz V9.2 + testes específicos passam, mas não todos os cenários adicionais possíveis em todos os caminhos legados |
| A2 nenhum uso do test na selecção | **Validado no caminho MLP V9.2 tocado** | `run_full_mlp_optimization` selecciona sem test; final avaliado depois. Não foi provado formalmente para cada caminho legado do repositório |
| A3 métricas do modelo final | **Validado no caminho MLP V9.2** | calibração decidida internamente e avaliação final depois |
| A4 nenhum nome de dataset em core/gui | **Validado por teste estático**, com excepções permitidas | teste V9.2 passa |
| A5 inferência aceita novas categorias/missing | **Validado** | testes de preprocessing/artefacto/inferência passam |
| A6 artefactos independentes do CWD e incompatibilidade clara | **Validado no formato V9.2** | round-trip e erro de versão testados |
| A7 confirmatory_v7 inalterado | **Validado** | SHA-256 antes/depois: 0 diferenças |
| A8 suite completa | **NÃO VALIDADO** | execução integral não terminou no limite; opcionais PyQt/OWL/TF indisponíveis |

## 19. Alterações físicas em relação ao ZIP V9.1

Comparação de conteúdo excluindo caches/bytecode:

- ficheiros existentes modificados: **26**
- ficheiros novos: **29** antes da criação desta documentação final
- ficheiros removidos: **0**

Não foi removida funcionalidade física do pacote.

## 20. Pendências prioritárias para declarar V9.2 concluída

1. Executar a suite integral numa máquina/CI com Python 3.11 e 3.12 e obter `pytest` verde.
2. Instalar/testar PyQt6 e finalizar a GUI dataset-agnostic (preview, alvo, overrides, PT-PT, warnings/gates).
3. Refactorar `counterfactuals/cf_mlp_trainer.py` e pipelines COGS/CLEAR para o único DataPreprocessor/MLP factory/artifact.
4. Remover dependência operacional de `counterfactuals/dataset_config.py` do caminho principal.
5. Migrar totalmente o storage do cache para artefactos V9.2 e eliminar dependências de pickle solto onde possível.
6. Validar em CI/GUI real a integração já feita do núcleo histórico nos caminhos principais Original/Reloaded e remover apenas os helpers CART legados que deixarem de ser necessários.
7. Comparar a implementação histórica com o código TREPAN original preservado pela Univ. Wisconsin, se o toolchain histórico puder ser executado.
8. Completar a revisão dos **156 `except Exception`** e **477 `print()`** legados.
9. Validar OWL/HermiT e executar a ablação pareada V9.2 em `results/ablation_v9_2/`, sem alterar resultados congelados.
10. Validar TensorFlow/CLEAR e o reporte explícito de zero contrafactuais quando houver falha.
11. Decompor os monólitos tocados, mantendo a suite verde.
12. Executar a matriz CI Windows e Linux real antes de etiquetar uma release final.

## 21. Conclusão

A implementação feita nesta sessão já estabelece um **núcleo V9.2 genuinamente agnóstico ao dataset**, com contrato explícito, preprocessing train-only, tolerância a categorias novas/missing, MLP adaptativo, separação da selecção e teste, artefactos versionados e inferência headless. Também corrige os defeitos concretos do Hepatitis, `os.chdir`, cache parcial e ablação com MLP fixo.

Contudo, **não é correcto declarar toda a V9.2 concluída**: GUI, contrafactuais legados, OWL/HermiT, CLEAR/TensorFlow, migração integral de cache, adopção global do TREPAN canónico, limpeza total de erros/logging e suite completa em Python 3.11/3.12 ainda precisam de validação/integração.

## Incremento: TREPAN Original histórico (29/09/2026)

### Implementado

- `core/trepan_original.py` deixou de ser um wrapper de `CanonicalTrepanClassifier` com queries globais.
- Implementados membership queries **por nó**, restrições raiz->nó, modelo de distribuições com frequências/KDE, `DrawInstance` condicionado, `min_sample`, best-first `reach * (1-fidelity)`, Information Gain, beam search m-of-n, teste de alteração de partição, pureza estatística e pruning.
- Adicionados presets `nips_1995` e `thesis_1996` via `TrepanOriginalClassifier.from_preset(...)`.
- `core/pipeline_v92.py` continua dataset-agnostic e usa o novo núcleo histórico; a configuração normal é identificada como adaptativa e distinta dos presets de reprodução histórica.
- O antigo `core/trepan_extractor.py` foi preservado por compatibilidade porque GUI e contrafactuais legados ainda dependem da interface de árvore sklearn. Ele deve ser tratado como CART destilado, não como implementação histórica.

### Testes adicionados

- `tests/test_trepan_original_historical_v92.py`

### Comandos executados

```text
pytest -q tests/test_trepan_original_historical_v92.py tests/test_trepan_original_v92.py
6 passed

pytest -q tests/test_trepan_original_historical_v92.py tests/test_trepan_original_v92.py tests/test_dataset_agnostic_matrix_v92.py::test_small_mixed_missing_id_constant_roundtrip
7 passed
```

Tentativas de executar em conjunto alguns testes legados de GUI/extractor ultrapassaram o limite temporal do runtime; não foram marcadas como aprovadas.

## Incremento: integração global do núcleo TREPAN histórico (29/09/2026)

O ponto pendente “integrar `TrepanOriginalClassifier` em todos os caminhos que apresentam CART como TREPAN Original” foi executado nos caminhos principais desta candidate.

### Implementado

- criado `TrepanOriginalExtractor` compatível com a API legada, mas que entrega exclusivamente `TrepanOriginalClassifier`;
- GUI principal e visualizador suportam nós `m-of-n` nativos;
- pipelines de contrafactuais e regras globais preservam `m-of-n`;
- `TrepanReloadedExtractor` usa o mesmo núcleo histórico tanto no modo sem OWL como no modelo final do modo OWL aceite;
- `Predictor.explain(row)` devolve o caminho histórico realmente percorrido;
- `TREPANExtractor` antigo ficou explicitamente como CART destilado legado/deprecated;
- identidade central separa `TREPAN Original`, `TREPAN Reloaded` e `CART Destilado (legado)`.

Relatório detalhado: `IMPLEMENTATION_REPORT_V9_2_TREPAN_INTEGRATION.md`.

### Resultados reais desta ronda

- núcleo + integração histórica: **13 passed**;
- artefactos + preprocessing: **9 passed**;
- matriz dataset-agnostic: **3 passed**;
- contrafactuais + fidelidade: **12 passed**;
- routing Reloaded/oráculo: **6 passed**;
- identidade/protocolo: **5 passed**;
- espelhamento directo sem ontologia: **1 passed**;
- `compileall`: **exit 0**;
- `results/confirmatory_v7/`: **5/5 hashes SHA-256 idênticos**.

Esses grupos foram executados isoladamente. Uma tentativa agregada ultrapassou 120 s após 23 testes apresentados como aprovados; portanto a suite completa continua **não validada como um único run**.

PyQt6, owlready2/HermiT e TensorFlow não estavam instalados e não são declarados validados.

## Incremento: núcleo histórico também no TREPAN Reloaded (29/09/2026)

O caminho principal com OWL foi refactorado para entregar sempre `TrepanReloadedClassifier`, uma extensão directa do `TrepanOriginalClassifier`. CART, soft-tree e refinadores antigos deixaram de participar da construção do modelo final; foram preservados apenas como caminhos/baselines legados.

O Reloaded usa o mesmo best-first, membership queries por nó, `min_sample`, constraints, KDE/frequências, Information Gain, `m-of-n` e pruning do Original. A contribuição Reloaded passa a entrar directamente como prior semântico dos splits e projecção coerente das membership queries.

Validação crítica executada nesta ronda: **30 passed, 8 warnings in 28.79s**. O teste pesado `test_no_ontology_metrics_identical_to_original` excedeu 120 s e não é declarado validado.

Detalhes: `IMPLEMENTATION_REPORT_V9_2_HISTORICAL_RELOADED_CORE.md`.


## Incremento — protocolo controlado Original vs Reloaded

Foi criada uma camada experimental pareada (`core/controlled_trepan_experiment.py`) para garantir o mesmo MLP Original, seed e orçamento TREPAN nos dois braços. O braço OWL usa `OriginalOracleProjection`, não um oráculo residual. O gate do oráculo usa somente CV do treino contra Dummy e C4.5. A análise pareada usa dataset como unidade estatística. Nesta ronda, 41 testes adicionais/dirigidos passaram, 1 foi saltado por dependência opcional e o braço OWL não foi executado por ausência de owlready2/HermiT. Ver `IMPLEMENTATION_REPORT_V9_2_CONTROLLED_ABLATION.md`.


## Incremento — separação dos gates ontológicos (2026-09-29)

Foi implementada a separação entre validade estrutural/matching da OWL, utilidade OOF das features derivadas e aceitação do professor ontológico. A GUI passa a usar o quality report para o mapping, métricas ausentes são N/A e o TREPAN Reloaded pode continuar a usar uma OWL estruturalmente válida mesmo quando o MLP ontológico não é aceite. Ver `IMPLEMENTATION_REPORT_V9_2_ONTOLOGY_GATE_SEPARATION.md`.

## Semantic Real-Gain Contract

A etapa posterior ao EFSR fecha quatro defeitos observados em execução real: relações
OWL mapeadas mas não propagadas, zero regiões EFSR elegíveis por threshold rígido,
aceitação baseada em queries sintéticas sem confirmação nas linhas reais internas e
divergência Reloaded/Original apesar de impacto semântico igual a zero.

Detalhes e validação: `IMPLEMENTATION_REPORT_V9_2_SEMANTIC_REAL_GAIN.md`.
