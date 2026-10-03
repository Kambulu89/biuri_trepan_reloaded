# Relatório de implementação — interface, observabilidade e diagnóstico

Branch `claude/repository-architecture-analysis-s1vl5a` (PR #1). Cada alteração é um commit pequeno e
testado. A auditoria feita **antes** de alterar a GUI está em `docs/GUI_AUDIT.md`; capturas offscreen
dos estados testados em `docs/interface_audit/`.

**Regra cumprida:** a camada de interface não altera nem recalcula resultados científicos. A única
alteração a código científico é **aditiva** (o TREPAN passou a *registar* porque parou) e foi
verificada com impressões digitais sha256 de regras, previsões, queries e nº de nós antes/depois:
idênticas.

## 1. Arquitetura anterior

`BiuriApp` (`gui/biuri_app_complete.py`, ~5300 linhas) guardava o estado científico em ~25 atributos e
apresentava-o como **um bloco de texto** (`results_text`) montado por concatenação dentro de
`_execute_training_pipeline`, mais gráficos matplotlib (`MetricsVisualizer`). Não havia modelo de dados
de resultado, estado explícito, indicação de cache/desatualização, nem erros estruturados.
Existiam dois pipelines paralelos: o legado/GUI e o headless (`core/production_training.py`).

## 2. Problemas encontrados

G1–G14 em `docs/GUI_AUDIT.md`. Os mais graves: valores em falta mostrados como `0` (e "Fidelity 0.0%"
para MLP/C4.5, que não têm Oracle); C4.5 apresentado como se tivesse fidelity ao Oracle; botões nunca
desativados; resultados antigos apresentados como atuais após mudar OWL/peso/dataset; rejeição do
enriquecimento só como código `REJECT_*`; motivos de paragem das árvores invisíveis (e nem registados no
backend); percentagens de progresso inventadas; erros sem contexto e traceback perdido; "Exportar
resultados" a gravar também a imagem da árvore; terminologia e idioma misturados.

## 3. Estado da GUI depois

Novo separador **🔍 Auditoria** (Resumo · Ontologia · Modelos · Árvores · Semântica · Experiência ·
Registo) alimentado por um único `ExperimentResult`. Os botões refletem o estado real
(`NO_DATA → DATA_LOADED → MODEL_TRAINED → SEMANTIC_VALIDATED → TREES_BUILT → RESULTS_READY`, mais `ERROR`),
com tooltip a explicar porque estão desativados. Banners de **RESULTADO EM CACHE** e **RESULTADOS
DESATUALIZADOS**. Exportação dividida em *Exportar resultados (dados)* e *Exportar árvore (imagem)*.

## 4. Alterações (commits, por ordem)

| Commit | Alteração | Partes |
|---|---|---|
| `51c8e3d` | TREPAN regista motivo de paragem por nó e global (`stop_reasons_`, `stop_summary_`) — só observabilidade | 23, 24 |
| `2cf114d` | `ExperimentResult` (+ `Measure`/`Reason`), `build_info`, builders a partir do pipeline headless | 2, 4, 9, 16 |
| `ed07b10` | `gui/strings.py`: i18n pt/en + terminologia fixa | 32, 33 |
| `0d383f6` | `experiment_state` (máquina de estados, stale) e `messages` (4 níveis, erros estruturados) | 3, 18, 21, 22, 39 |
| `14e6199` | `result_presenter` (N/A com significado, grupos de métricas, oracle, tabelas, tooltips) | 5–16, 23–31, 34 |
| `7b57608` | `export_results` (results/manifest/config/metrics.csv/semantic_*.csv/tree_diagnostics/messages/report.md) | 35, 36 |
| `954ece5` | `result_builder`: adaptador `BiuriApp → ExperimentResult` sem recálculo | 2 |
| `c17f400` | `audit_panel` (Qt, camada fina) | 5–31, 34, 38 |
| `2451d46` | `audit_controller`: estado↔botões, stale, cache, erros, progresso sem percentagens | 3, 17–22, 37, 39 |
| `d66628b` | workers emitem traceback (`failed_detail`); visualizador sem "0.0%" de fidelity; export de árvore PNG/SVG/PDF | 9, 12, 21, 36 |
| `b5a7f32` | integração no `BiuriApp` (separador, botões, stale, cache, erro estruturado, export separado) | 3, 17–21, 35, 36 |
| seguintes | score semântico total/stop reason/tuning/CF; painel com scroll e tabelas ajustadas; controlo negativo no builder; testes de não-recálculo; screenshots | 13, 14, 29, 40 |

## 5. Componentes

| Módulo | Papel | Qt? |
|---|---|---|
| `core/experiment_result.py` | **Fonte única de verdade**: `ExperimentResult`, `Measure(value, reason)`, `Reason`, proveniência | não |
| `core/experiment_builders.py` | Constrói o resultado a partir dos relatórios do pipeline (headless) | não |
| `core/build_info.py` | versão, commit, `+alterações locais`, `SEMANTIC_PIPELINE_VERSION` | não |
| `gui/strings.py` | Idioma principal **pt-PT** (+ `en`), `TERMS` fixos, `tr()` nunca lança | não |
| `gui/experiment_state.py` | `ExperimentStateMachine`, `StaleTracker`/`Fingerprint` | não |
| `gui/messages.py` | `Level`, `MessageLog`, `format_error`, `derive_messages` | não |
| `gui/result_presenter.py` | Formata; **não calcula** | não |
| `gui/result_builder.py` | Lê atributos do `BiuriApp`/`MetricsComparator` (duck-typing) | não |
| `gui/export_results.py` | Exportação estruturada | não |
| `gui/audit_controller.py` | Liga a app ao estado/stale/mensagens/painel | mínimo |
| `gui/audit_panel.py` | Widgets | sim |

Camadas: dados → builders → presenter/estado/mensagens/export (testáveis sem Qt) → widgets finos.

## 6. Como as perguntas do objetivo são respondidas

| Pergunta | Onde |
|---|---|
| Dataset ativo, seed, build | Resumo; Experiência (ID, seed, hashes, build, timestamp, cache) |
| OWL carregada; ontologia validada? | Resumo; Ontologia (estrutural, reasoner, mapeamento, cobertura, ambíguos, TBox, ABox) |
| Enriquecimento aceite e porquê | Ontologia: 3 eixos separados + explicação humana com utilidade base/OWL/Δ |
| Oracle de cada TREPAN | Resumo; cartões; tabela de Fidelity (coluna "Oracle usado") |
| Nós, porque parou, queries | Resumo; Árvores (diagnóstico completo) |
| Accuracy / fidelity e contra quem | Modelos: Desempenho preditivo (vs rótulos) · Surrogate Fidelity (vs Oracle) · Complexidade |
| Features OWL selecionadas / splits semânticos | Semântica (filtro "só selecionadas") |
| Cache, seed/configuração | Banner + Experiência |

## 7. Estados testados

Capturas em `docs/interface_audit/`: `00` sem resultados · `01` resumo com enriquecimento rejeitado e
cache · `02` ontologia (3 eixos) · `03` métricas em 3 grupos (C4.5 "Não aplicável — este modelo não
tem Oracle"; MLP Ontológico "Não calculado — professor semântico rejeitado") · `04/05` árvores
(stump → "Árvore pequena: diagnóstico disponível", nunca "erro") · `06` semântica com banner
DESATUALIZADO e controlo negativo · `07` modo científico.
Estados do `BiuriApp` real (offscreen) verificados por teste: `NO_DATA` (todos os botões desativados
com motivo) → `DATA_LOADED` → treino (ocupado) → erro estruturado (volta ao último estado válido) →
`TREES_BUILT/RESULTS_READY` → stale por mudança de peso OWL/dataset.

**Execução real do `BiuriApp` (offscreen, sem ontologia, 240 linhas × 3 features, treino científico
completo, 898 s):** estado `Sem dados` (botões desativados com motivo) → `Dados carregados` →
`Árvores construídas`; o Resumo mostrou `Ontologia validada?: Indisponível — ontologia não carregada`,
`Oracle do TREPAN Original/Reloaded: MLP Original`, `Nós 15 / 15`, `Queries 4000 / 4000` (orçamento
esgotado: Sim), `Fidelity (vs Oracle) 0.944 / 0.944`, `Splits semânticos: Indisponível — nenhuma feature
semântica gerada`, `Cache: Calculado neste treino` e um único aviso `SCIENTIFIC_WARNING single_seed`.
Esta corrida expôs ainda um defeito de apresentação (código técnico `no_ontology` em vez de texto humano),
corrigido e coberto por teste. Não foi possível exercitar a interface com ontologia real nesta corrida.

## 8. Testes

Novos ficheiros (todos a passar): `test_tree_stop_diagnostics`, `test_experiment_result`,
`test_experiment_builders_production`, `test_gui_strings`, `test_gui_state_and_messages`,
`test_result_presenter`, `test_export_results`, `test_result_builder_app`, `test_audit_panel_qt`,
`test_audit_controller`, `test_gui_audit_integration`, `test_gui_no_scientific_recalculation`.

Cobrem: transições de estado, significado dos N/A, rótulos das métricas, oracle por árvore, resultado
stale, indicador de cache, estado semântico, diagnóstico/stump, tratamento de erros (estrutura, ID,
traceback no log), troca de árvores, **ausência de recálculo na UI** (verificação estática por AST — sem
sklearn/predict/fit/métricas — e teste dinâmico com modelos "armadilha"), exportação, terminologia
proibida e botões/estado no `BiuriApp` real.

**Suite completa:** 631 passaram, 3 falharam (12 min 42 s). As 3 falhas são **pré-existentes e não
relacionadas** com esta tarefa (as duas de `test_scientific_superiority_pipeline_v7` já existiam antes
de qualquer alteração à GUI; a de `test_counterfactual_gui` só ficou visível por o PyQt6 estar agora
instalado no ambiente):

* `tests/test_counterfactual_gui.py::test_counterfactual_panel_options_and_results`
* `tests/test_scientific_superiority_pipeline_v7.py::test_soft_global_tree_selects_without_external_test`
* `tests/test_scientific_superiority_pipeline_v7.py::test_fidelity_hierarchy_c45_original_reloaded`

## 9. Limitações (honestas)

* **Dois pipelines.** O `ExperimentResult` é construído a partir do pipeline headless
  (`from_production_report`) e do estado do `BiuriApp` (`build_experiment_result`). No pipeline legado,
  a "utilidade base/OWL/Δ" vem de `balanced_accuracy_original/onto/gain` da aceitação, não do
  `evaluate_semantic_enrichment` do headless; está rotulada como utilidade mas não é a mesma medida.
* **Nós desenhados** (`rendered_nodes`) aparecem sempre como "Não executado — árvore ainda não
  desenhada": o widget de visualização não reporta o que desenhou.
* **Controlo negativo:** o gate de atribuição usa *features derivadas aleatórias* como controlo; o braço
  "OWL baralhada" existe na interface mas só é preenchido se um relatório o fornecer. Ablação e
  benchmark só aparecem quando há dados (não são gerados pela GUI).
* **Seed** da GUI é a constante `42` já existente (`random_state=42`); a interface passou a mostrá-la,
  não a torná-la configurável.
* **Cache:** só o MLP Original tem cache na GUI; o indicador reflete exatamente isso.
* **Idioma:** os textos *novos* passam por `gui/strings.py`; mensagens legadas em espanhol/misto dentro de
  `_execute_training_pipeline` e diálogos antigos **não** foram migradas (alterá-las arriscava regressões
  fora do âmbito). O bloco de texto do separador "Resultados" mantém-se como registo legado.
* **Progresso:** o diálogo passou a indeterminado com texto por etapa; não há percentagem real a mostrar.
* **Falhas pré-existentes não relacionadas** (3, listadas na secção 8): não foram tocadas.
* **Log de consola legado:** `log_mlp_optimization` imprime `Test accuracy: 0.0000` quando o holdout está
  bloqueado (`test_metrics=None`). É só consola; na interface o valor aparece como "Não executado —
  comparação de métricas ainda não executada", nunca como 0.
* Os screenshots são offscreen (renderização Qt "offscreen"), não capturas de um ecrã real.
