# IMPLEMENTATION REPORT — V9.2 Error-Focused Semantic Refinement

## Escopo

Implementação do Error-Focused Semantic Refinement (EFSR) sobre a candidate
`biuri_trepan_reloaded_v9_2_adaptive_semantic_production.zip`.

## Implementado

- `core/error_focused_semantic_refinement.py`
  - entropia normalizada do oráculo em linhas já conhecidas;
  - perfil robusto de features associadas à discordância;
  - suporte semântico estrutural agnóstico ao dataset;
  - fidelidade local de splits;
  - proximidade a anchors de erro.
- `core/trepan_reloaded_historical.py`
  - prioridade best-first focada em regiões de erro;
  - perfis de erro por nó;
  - candidatos `m-of-n` orientados pelo erro e pela OWL;
  - ambas as orientações de cada literal (`>` e `<=`) consideradas;
  - membership queries dirigidas a anchors de erro sem aumentar o budget;
  - gate obrigatório de ganho de fidelidade local;
  - fallback exacto para o split data-only quando a proposta semântica não melhora;
  - auditoria EFSR por nó e agregada;
  - diagonal da matriz de relatedness deixa de activar semântica artificialmente.
- `core/controlled_trepan_experiment.py`
  - parâmetros EFSR ficam apenas no braço Reloaded;
  - auditoria preserva o mesmo orçamento Original/Reloaded;
  - relatório final exporta `semantic_split_audit` e `error_region_audit`.
- `core/trepan_scientific_tuning.py`
  - tuning train-only inclui força/threshold/gate EFSR;
  - capacidade comum continua seleccionada antes da semântica.
- `core/training_config.py`, GUI e extractor
  - EFSR activado por defeito no modo Científico;
  - parâmetros propagados de forma consistente.
- `core/semantic_contribution_gate.py`
  - reporta intervenções EFSR tentadas/aceites/rejeitadas.
- artefacto de produção
  - formato: `biuri-v9.2-production-4-error-focused-semantic`.
- `scripts/production_readiness_v92.py`
  - novo check `error_focused_semantic_refinement`.

## Testes adicionados

`tests/test_error_focused_semantic_refinement_v92.py` cobre:

1. detecção de região de erro e oportunidade semântica;
2. rejeição de candidato sem ganho de fidelidade;
3. aceitação de candidato com ganho real;
4. query budget inalterado e auditável;
5. EFSR excluído da configuração comum Original/Reloaded;
6. regiões abaixo do threshold não sofrem substituição semântica;
7. cenário sintético integrado em que a semântica corrige um distractor.

## Resultados executados neste ambiente

Baterias direcionadas concluídas sem falhas (execuções separadas):

- `test_error_focused_semantic_refinement_v92.py`: 7 passed
- `test_production_semantic_v92.py`: 4 passed
- `test_production_no_cart_v92.py`: 5 passed
- `test_reloaded_scientific_optimization_v92.py`: 6 passed
- `test_semantic_audit_metrics_v92.py`: 6 passed
- `test_controlled_trepan_ablation_v92.py`: 7 passed
- `test_ontology_gate_separation_v92.py`: 11 passed
- `test_true_trepan_integration_v92.py`: 7 passed
- `test_scientific_protocol_unittest.py`: 6 passed (5 ConvergenceWarning)
- `test_scientific_mode_lock_v92.py`: 6 passed
- `test_audit_remediation_v9_1.py`: 13 passed
- `test_reloaded_historical_core_v92.py`: 5 passed
- `test_trepan_original_historical_v92.py`: 5 passed

Total das baterias direcionadas listadas: **88 testes aprovados, 0 falhas**.

`python -m compileall -q core gui counterfactuals tests scripts`: aprovado.

A suite monolítica `pytest -q` foi iniciada com 300 s e excedeu o timeout por
volta de 23%, sem falha apresentada antes da interrupção. Não é declarada como
suite integral aprovada.

## Validação sintética de mecanismo

Em cenário sintético controlado (não benchmark):

- TREPAN Original fidelity: 0.913333...
- TREPAN Reloaded + EFSR fidelity: 1.000000
- Delta: +0.086666...
- Original escolheu um distractor univariado;
- Reloaded escolheu uma regra semântica `2-of-3` e o gate local confirmou ganho.

Este resultado apenas prova a capacidade do mecanismo; não é uma claim sobre os
datasets da tese.

## Confirmatório congelado

Os 5 SHA-256 em `results/confirmatory_v7/` permanecem byte-a-byte iguais aos da
candidate de entrada. `run_confirmatory_locked.py --execute` não foi executado.

## Limitações do ambiente

`production_readiness` permanece `ready=false` porque este runtime usa Python
3.13.5 e não possui toda a stack-alvo (por exemplo, `dtreeviz`; PyQt6/Owlready2
não foram solicitados no preflight desta execução). O alvo de produção continua
Python 3.11/3.12 no Windows com as dependências opcionais necessárias.
