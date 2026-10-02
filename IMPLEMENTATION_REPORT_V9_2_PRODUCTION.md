# IMPLEMENTATION REPORT — BIURI / TREPAN Reloaded V9.2 Production Candidate

## Implementado nesta etapa

- `core/ontology_semantic_graph.py`: grafo semântico agnóstico ao dataset (classes/propriedades, subsunção, domínio/range, equivalência, grupos, relatedness e matriz de relações).
- `core/semantic_contribution_gate.py`: gate separado para contribuição efetivamente observada no TREPAN Reloaded.
- `core/trepan_reloaded_historical.py`: suporte a matriz de relatedness semântica em regras m-of-n.
- `core/controlled_trepan_experiment.py`: propaga grupos/matriz e devolve auditoria semântica na avaliação final.
- `core/production_training.py`: treino de produção headless com contrato, preprocessing, MLP, C4.5, TREPAN Original/Reloaded pareados, quality gate, reasoner, semantic graph, claim guard, artefacto e manifesto.
- `core/production_inference.py`: carregamento e inferência do bundle de produção sem GUI.
- `scripts/run_reloaded_ablation_v92.py`: ablação multi-seed genérica, sem nomes de datasets.
- `scripts/biuri_cli.py`: `train-production`, `predict-production`, `explain-production`.
- requirements separados por capacidade opcional.
- preflight V9.2 aceita Python 3.11/3.12 e torna OWL/CLEAR opcionais por flags.
- compatibilidade do helper legado de synthetic sampling preservada usando KDE/frequências do TREPAN histórico, sem voltar a CART.

## Validação neste ambiente

Ambiente observado: Python 3.13.5/Linux. Este ambiente NÃO é o alvo declarado de produção.

Resultados executados:
- `compileall core gui counterfactuals tests scripts`: OK.
- testes novos de produção/semântica: 4 passed.
- críticos V9.2 (semantic audit, ontology gate, controlled ablation, historical TREPAN, dataset agnostic): 32 passed.
- static/runtime/artifact/production: 11 passed.
- batch inicial: 73 passed, 2 skipped.
- batch de dados/evaluation/alignment/GUI regressions: 17 passed.
- MLP degenerate/factory/ontology bias: 19 passed.
- MLP optimizer: 6 passed.
- model bundle/cache/matching: 5 passed, 1 skipped.
- mirror sem OWL: 2 passed.
- batch científico/intermédio: 79 passed, 2 skipped.
- batch final de regressões: 57 passed, 2 skipped.

A execução monolítica de `pytest -q` excedeu o limite temporal deste runtime sem mostrar falhas antes do timeout. Os mesmos ficheiros foram executados por grupos e passaram conforme acima.

## Não validado aqui

- Python 3.11 real.
- Python 3.12 real.
- GUI PyQt6 real.
- Windows.
- Owlready2 + Java/HermiT/Pellet real.
- TensorFlow/CLEAR real.

Esses itens permanecem bloqueadores para chamar esta candidate de release final de produção.

## Dívida legada ainda existente

A auditoria estática global encontra 0 `except:` nus, mas ainda existem `except Exception` e `print()` em módulos históricos. O núcleo novo usa erros tipados/relatórios estruturados, porém a migração total dos módulos legados ainda não foi concluída.

## Protocolo congelado

`results/confirmatory_v7/` não foi executado nem editado nesta etapa.
