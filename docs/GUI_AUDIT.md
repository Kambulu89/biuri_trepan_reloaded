# Auditoria da GUI (estado anterior às alterações)

Feita **antes** de qualquer alteração à interface, lendo `gui/biuri_app_complete.py`,
`gui/training_worker.py`, `gui/metrics_worker.py`, `gui/ontology_status_presenter.py` e
`gui/pyqt_metrics_visualizer.py`. Serve de base ao `IMPLEMENTATION_REPORT_INTERFACE_DIAGNOSTICS.md`.

## O que a GUI mostrava

| Área | Como era mostrado |
|---|---|
| Resultado do treino | Um único bloco de texto (`results_text`, `QTextEdit`) montado por concatenação de strings dentro de `_execute_training_pipeline` (~800 linhas) |
| Métricas | `MetricsComparisonWidget` → `MetricsVisualizer` (gráficos matplotlib + etiquetas) |
| Ontologia | Parágrafo de texto (`ontology_status_presenter`) |
| Estado | Inexistente: atributos soltos em `BiuriApp` (`mlp_model`, `trepan_*_tree`, `ontology_acceptance`…) |
| Erros | `QMessageBox.critical(str(exc))`; traceback perdido |

## Problemas encontrados

| # | Problema | Evidência | Efeito |
|---|---|---|---|
| G1 | Sem fonte única de verdade: o estado científico vive em ~25 atributos e em texto | `BiuriApp.__init__` | a GUI reconstrói/duplica informação; não há objeto auditável nem exportável |
| G2 | Valores em falta mostrados como `0` | `MetricsComparisonWidget.update_comparison_results` (`fid_val = 0.0`, `or 0`) | "Fidelity 0.0%" para MLP/C4.5 que não têm Oracle |
| G3 | C4.5 com fidelity como se tivesse Oracle | mesmo widget (`fid_key='c45_j48'`) | confunde concordância com o MLP com fidelity |
| G4 | Sem máquina de estados; botões nunca desativados | `setup_connections`; guards com `QMessageBox` | o utilizador só descobre o erro depois de clicar |
| G5 | Sem indicação de cache nem de resultado desatualizado | `perf.used_cache` só no texto | resultados antigos apresentados como atuais após mudar OWL/peso/dataset |
| G6 | Rejeição do enriquecimento só como código (`REJECT_*`) | `ontology_acceptance['reason']` | sem utilidade base/OWL/delta nem explicação humana |
| G7 | Três perguntas distintas misturadas num parágrafo | `build_ontology_status_text` | ontologia válida ≠ enriquecimento do MLP aceite ≠ semântica no TREPAN |
| G8 | Motivos de paragem das árvores invisíveis (e nem registados no backend) | `TrepanOriginalClassifier.fit` | "árvore com 1 nó" lido como erro |
| G9 | Barra de progresso com percentagens fixas | `progress("mlp", 8, …)` | percentagens inventadas (8, 35, 72, 85…) |
| G10 | Erros sem contexto: sem onde/ação/ID, traceback perdido | `_on_training_failed` | impossível relacionar mensagem ↔ experiência |
| G11 | Exportar resultados também gravava a imagem da árvore | `export_results` | duas ações misturadas; formato único (PNG) |
| G12 | Texto misturado ES/PT, termos variáveis (Trepan-Reloaded, Fidelidade/Fidelidad, Oráculo) | grep | terminologia inconsistente |
| G13 | Sem modo básico/científico, sem tooltips de métrica, sem proveniência | — | números sem contexto |
| G14 | Seed fixa (42) e hashes de dataset/OWL apenas internos | `random_state=42` espalhado | reprodutibilidade invisível |

Nada disto exigiu alterar algoritmos científicos; apenas a camada de apresentação e observabilidade
(a única alteração a código científico é aditiva: o TREPAN passou a **registar** porque parou).
