# Correção C4.5 / TREPAN — BIURI/TREPAN Reloaded V9.1

Data: 2026-09-27

## Objetivo

Corrigir a comparação entre C4.5, TREPAN Original e TREPAN Reloaded sem fabricar resultados. O C4.5 continua um baseline supervisionado pelos rótulos reais; nunca é usado como oráculo do TREPAN.

A política implementada é:

- a comparação principal usa **Precision Macro**;
- o TREPAN Reloaded só é marcado como tendo alcançado o objetivo C4.5 quando as métricas exigidas são não-inferiores ao C4.5 no mesmo conjunto de avaliação;
- se o C4.5 estiver acima, o sistema mostra `ABAIXO DO BASELINE` e mantém os valores observados;
- C4.5 não participa de gates de fidelidade como se fosse um professor;
- fidelidade dos TREPAN continua sendo reportada com referência explícita ao MLP correspondente, e o painel de controlo usa MLP Original como referência comum.

## Ficheiros alterados

- `core/c45_baseline_gate.py` — novo gate auditável contra C4.5.
- `core/trepan_reloaded_extractor.py` — Precision Macro como métrica primária nos refinamentos e remoção de C4.5 dos critérios de fidelidade.
- `core/surrogate_acceptance.py` — estado explícito `c45_baseline_gate_accepted` e `c45_is_oracle=False`.
- `core/metrics_comparator.py` — gate C4.5 no resultado comparativo e no relatório; locked-test continua audit-only.
- `core/metrics_view_model.py` — `precision_macro` e `precision_weighted` expostas separadamente.
- `gui/biuri_app_complete.py` — métricas C4.5 macro coerentes com a interface/gates.
- `gui/pyqt_metrics_visualizer.py` — linha de baseline C4.5, estados por TREPAN e deltas em pontos percentuais.
- `tests/test_c45_baseline_policy_v9_1.py` — novos testes de regressão.
- `CHANGELOG_V9_1.md` — registo da alteração.

## Comportamento esperado para os valores mostrados pelo utilizador

Com Precision Macro:

- TREPAN Original: 86,3%
- C4.5-Nativo: 92,3%
- TREPAN Reloaded: 89,5%

O sistema deve declarar:

- TREPAN Original vs C4.5: **ABAIXO DO BASELINE**, delta = -6,0 pp.
- TREPAN Reloaded vs C4.5: **ABAIXO DO BASELINE**, delta = -2,8 pp.
- TREPAN Reloaded vs TREPAN Original em Precision Macro: +3,2 pp.

Esses valores não são corrigidos artificialmente. O gate apenas impede que o resultado seja apresentado como tendo alcançado o baseline C4.5.

## Validação executada

### Sintaxe

`python -m compileall -q core gui tests` → aprovado.

### Regressão focada

42 testes aprovados, 0 falhados. Inclui gates científicos, C4.5, dominância, prompt V9.1 e auditoria V9.1.

### Regressão ampliada sem dependências OWL/GUI indisponíveis

97 testes aprovados, 0 falhados, 36 warnings. Os warnings são `ConvergenceWarning` de fixtures com `max_iter` reduzido.

### Limitação do ambiente

A suíte OWL integral não pôde ser executada porque `owlready2` não está instalado neste runtime. A tentativa ampliada chegou a 98 testes aprovados antes de falhas/erros exclusivamente associados à ausência dessa dependência. A GUI PyQt6 também não é executável neste ambiente, mas todos os ficheiros modificados passaram `compileall`.

### Smoke test do comparador

Foi executado um smoke test independente com Iris. O `c45_baseline_gate` foi produzido corretamente, `c45_is_oracle=False`, `require_c45_noninferiority=True` no audit do substituto, e os valores de `precision_macro` foram calculados separadamente para TREPAN Original, C4.5 e TREPAN Reloaded.

## Regra científica preservada

O projeto não garante por construção que o C4.5 terá uma métrica menor. Em vez disso, torna **alcançar/superar o C4.5 um critério explícito de sucesso**. Se o dado real não satisfizer esse critério, o sistema mostra a falha. Isso evita manipulação de métricas e mantém a comparação reproduzível e auditável.
