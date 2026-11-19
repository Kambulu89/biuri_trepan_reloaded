# Correções implementadas

Ramo local `auditoria/missao-master`. Nenhum push foi feito. Nenhum baseline, dataset, resultado histórico ou limiar científico foi alterado.

| Commit | Problema | Ficheiros | Reprodução antes | Verificação depois |
|---|---|---|---|---|
| `af02d5d` | P09 — coluna numérica com marcadores de ausente inferida como categórica | `core/data_contract.py` (`_coerce_numeric_text`), `tests/test_audit_numeric_with_missing_tokens.py` | 5 dos 7 testes novos falhavam com o código original | 7 aprovados |
| `16e6be8` | P01 — gate declarava contribuição semântica sem decisão alterada | `core/semantic_contribution_gate.py`, `tests/test_audit_semantic_contribution_gate.py` | Execução real: Breast Cancer, semente 7, estado «SUPPORTED» com impacto 0 | 3 testes novos + `tests/test_production_semantic_v92.py` aprovados |

## Detalhe

**P09.** Depois de substituir os marcadores de ausente por `NaN`, as colunas de texto cujos valores presentes são todos números passam a numéricas. O alvo nunca é convertido. Colunas com mistura de texto e números continuam categóricas. Princípio: o tipo inferido não pode depender de existirem ausentes.

**P01.** O gate passa a devolver `NO_SEMANTIC_DECISION_CHANGED` (não aceite) quando o impacto semântico é 0 e nenhuma intervenção EFSR foi aceite. O relatório inclui três campos novos: `evidence_level = "observed_single_split"`, `controls_evaluated = False`, `attributable_to_ontology = False`. O estado `SEMANTIC_CONTRIBUTION_SUPPORTED` mantém-se para os restantes casos, mas agora declara que é um efeito observado numa partição e não uma atribuição.

## Regressão

- Suite completa antes das correções: 1732 aprovados, 40 ignorados (`evidencia/pytest_full_baseline.log`).
- Suite completa depois das correções: 1742 aprovados, 40 ignorados (`evidencia/pytest_full_after_fixes.log`). Os 10 testes a mais são os novos.

## O que não corrigi, e porquê

| Problema | Razão |
|---|---|
| P04 (pré-seleção por variância) | Altera o TREPAN Original, que o `CLAUDE.md` do projeto congela. Não precisa de código: basta `max_features_per_node = nº de atributos`. Proponho-o para um manifest v2. |
| P02, P06, P07 | Alteram o Reloaded ou o protocolo de comparação; exigem um manifest novo e pré-registo. |
| P03, P08 | Mudanças de arquitetura; descritas em `PLANO_CORRECCAO.md`. |
| P05 | Regenerar o manifest é uma decisão do responsável do projeto: fixa uma nova referência. |
| P12, P16 | A missão pede para não eliminar código sem confirmar impacto; há testes que o referenciam. |
