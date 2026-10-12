# Documento 13 — Relatório da Fase 1a (conformidade EQ, correção da V1)

Commits: âncora `50f97b2` (antes de qualquer alteração a core/), correção `4f7d91f`. Base `main@1789c17`. B1–B4 **não** implementados; sem avaliação confirmatória; sem merge.

## Ficheiros alterados
`core/trepan_reloaded_historical.py` (único ficheiro de núcleo); novos: `tests/eq_support.py`, `tests/test_eq_reloaded_original.py`, `tests/golden/{make_effective_anchors.py,golden_effective.json}`, `docs/evidence/b1b4/equivalence_probe_result_post1a.jsonl`.

## Diagnóstico
V1: `_decision_sample` e `_draw_membership_queries_unfiltered` chamavam `oracle.predict_proba` (EFSR) em cada nó, mesmo sem ontologia; `_split_audit_metadata` acrescentava campos ao `split_audit_`. Correção: predicado único `ontology_effective_` (= `reload_extension_active_` OU domínio OWL ativo — este último incluído porque filtra queries mesmo sem projeção) e despacho para o Original nos ganhos sobrepostos.
V2 (`mirror_when_no_semantic_effect`): **não alterada** (D-16, caso 4 sem efeito mensurável: árvore ≡ Original, consultas duplicadas). Registada com `xfail` estrito.

## Resultados
* Âncora (252 entradas sintéticas, determinística): 171 com ontologia efetiva **idênticas bit-a-bit** após a correção; 81 sem ontologia efetiva **≡ Original** em regras, previsões, nós, membership queries, `split_audit_`, ordem de expansão e **fluxo de entradas ao oráculo** (hash, nº chamadas/amostras, sequência).
* Testes novos: 159 passed + 1 xfail (V2); espião que falha em `predict_proba`; tabela de verdade do predicado; propriedade (6 seeds, orçamentos/formas aleatórios); metamórfico (permutação de colunas); ontologia inativa/rejeitada; `test_allowlist_is_empty`; lista separada de campos de relato (D-15).
* DEV (iris, wine, breast_cancer × seeds 42, 7; braço D e defaults): regras, previsões, membership queries, nós e fluxo ao oráculo **iguais**; amostras ao oráculo iguais (p.ex. iris/42 652=652; antes 1712).
* Suite completa em 4 shards: sem falhas. Agnosticismo estático: OK.
* Tempo (sintético, mediana de 5, alternado): razão Reloaded(∅)/Original 1,07 (p=8) e **1,14 (p=30)**. O alvo ≤ 1,1× **não está confirmado** no segundo caso (0,04 s absolutos; custo fixo do pós-processamento de auditoria e ruído); medir no protocolo DEV real antes de declarar conformidade.

## Não conformidades / pendentes
V2 aberta; tempo ≤ 1,1× por confirmar; CI (Windows 3.11/3.12 + agnosticism-guard) no commit exato por verificar; Digits e D-13 (re-congelamento) por fazer; não foram corridos o agnosticismo dinâmico/metamórfico dedicado nem o relatório `equivalence_report.py`.
