# Documento 13 — Relatório da Fase 1a (conformidade EQ, correção da V1)

## Veredito
| Item | Estado |
|---|---|
| **V1** (consultas extra ao oráculo sem ontologia efetiva) | **FIXED / VERIFIED** |
| **V2** (duplicação de consultas no `mirror`) | **OPEN / PHASE_1B** (`xfail` estrito mantido) |
| **GLOBAL_EQ_CONTRACT** | **NOT_YET_FULLY_COMPLIANT** (violação conhecida V2) |
| Digits, agnosticismo, E1–E9, desempenho | verificados (abaixo) |
| CI Windows 3.11/3.12 no commit final | **ver §7 — não declarada aprovada sem execução verde** |

Base `main@1789c17`. Âncora `50f97b2` (antes de qualquer alteração a `core/`), correção `4f7d91f`. B1–B4 **não** implementados; sem avaliação confirmatória; baselines intocados; sem merge nem PR.

## 1. Alterações
Único ficheiro de núcleo: `core/trepan_reloaded_historical.py` (+25 linhas vs `main`; `git diff 1789c17..HEAD -- core gui counterfactuals` só lista este ficheiro). `core/trepan_original.py` **inalterado** (blob `2e80cd6`, o mesmo da âncora).
Predicado único `ontology_effective_` = `reload_extension_active_` OU domínio OWL ativo (este último incluído porque filtra as membership queries mesmo com a projeção desligada; sem isto o despacho desligaria um mecanismo ontológico legítimo). Despacho para o Original (`super()`) em `_priority`, `_decision_sample`, `_feature_priority_scores`, `_split_selection_score`, `_best_mofn`, `_draw_membership_queries`, `_split_audit_metadata`.

## 2. Diagnóstico da V1
`_decision_sample` e `_draw_membership_queries_unfiltered` chamavam `oracle.predict_proba` (perfil EFSR) em cada nó mesmo sem ontologia; `_split_audit_metadata` acrescentava campos ao `split_audit_`. Árvore e previsões eram iguais, o consumo do oráculo não (×2,05–2,33).

## 3. Equivalência E1–E9 (evidência)
* **Protocolo DEV real, pré (`e2a58ef`) vs pós-correção**, `equivalence_report.py`: iris/42, iris/7, wine/42, wine/7, breast_cancer/42, breast_cancer/7, **digits/42** → **182 campos, FAIL 0, comparação exata** (`oracle_id`, `split_hash`, tuning estrutural, previsões dos 6 braços, fidelidade, accuracy, nós, folhas, profundidade, quality gate, auditoria semântica, estrutura das árvores, contrastes). `docs/evidence/b1b4/phase1a/equivalence_report_pre_vs_post.{md,json}`.
* **Braço C (Original) vs D (Reloaded sem ontologia), mesma execução** (`arm_C_vs_D_within_run.md`): pré-correção `oracle_queries` D/C = 2,05–2,33 (Digits 164 124 vs 344 832); **pós-correção = 1,000 nas 7 unidades**, previsões e nós iguais.
* **Fluxo de entradas ao oráculo (hash, nº de chamadas/amostras, sequência)**: igual em 81/81 configurações sintéticas sem ontologia efetiva e nas 18 unidades da medição emparelhada (iris, wine, breast_cancer × 5 seeds, Digits × 3 seeds), além de 3 datasets DEV × 2 seeds na sonda. Regras, previsões, nós, `membership_queries_`, `split_audit_`, ordem de expansão: iguais.
* **Com ontologia efetiva**: 171/171 entradas da âncora **idênticas bit-a-bit** após a correção (Linux) e idênticas na âncora portável (todas as plataformas). As âncoras **não foram alteradas**; foi acrescentado um suplemento portável (§7) gerado com o código pré-correção (blob `0053b3d`).
* **Ontologia rejeitada / inativa / sem estrutura** (`depths_zero`, `groups_unique`, grafo `is_active=False`): ≡ Original com espião que falha em `predict_proba`.

## 4. Desempenho (medição emparelhada, isolada, mesmo split/MLP/hiperparâmetros por omissão, ambiente numérico fixado, aquecimento + 7 repetições alternadas; Digits 5)
| dataset | unidades | razão R/O (mediana das seeds) | intervalo entre seeds | pares: mediana [Q1, Q3] | Original (mediana) |
|---|---|---|---|---|---|
| iris | 5 | 1,049 | 0,897–1,097 | 1,030 [0,931; 1,098] | 0,27 s |
| wine | 5 | 0,997 | 0,967–1,050 | 1,009 [0,960; 1,078] | 0,63 s |
| breast_cancer | 5 | 1,010 | 0,951–1,056 | 1,004 [0,951; 1,045] | 1,11 s |
| digits | 3 | 1,006 | 0,992–1,066 | 1,006 [0,921; 1,075] | 1,93 s |
Global (120 pares): mediana 1,016. **Objetivo ≤ 1,1× cumprido** em todas as medianas por seed; a dispersão por par (até ~1,4× em fits de 0,3 s) é ruído de curta duração. CPU: mesma conclusão. O critério temporal nunca substitui a igualdade do fluxo, que é exata. (A primeira medição sintética curta, 1,14×, era ruído em 0,3 s.)

## 5. Testes e agnosticismo
`tests/test_eq_reloaded_original.py`: **159 passed + 1 xfailed (V2)**: espião que falha em `predict_proba`; tabela de verdade do predicado; propriedade (6 seeds, formas/orçamentos aleatórios, binário/multiclasse, discretas/constantes); metamórfico (permutação de colunas); ontologia inativa; âncora; `test_allowlist_is_empty`; lista separada de campos de relato (D-15). Agnosticismo: estático OK; `test_agnosticism_contract.py`, `test_dataset_agnosticism.py`, `test_v92_static_guards.py` (dinâmico, metamórfico, dataset desconhecido) + EQ: 245 passed, 1 xfailed. Suite completa local (4 shards): sem falhas.

## 6. Âncoras e hashes
| artefacto | SHA-256 / blob |
|---|---|
| `core/trepan_reloaded_historical.py` (pós) | `42f6abbd…efc8` / blob `efec6d10a9aeb699f78bd92863d121af53148cdd` (pré: blob `0053b3dc…`) |
| `core/trepan_original.py` | `c3a49657…d5f` / blob `2e80cd6ac1e185010ff15b50b400ad081fd81c1b` (inalterado) |
| `tests/golden/golden_effective.json` | `acc0b499…0d` |
| `tests/golden/golden_effective_portable.json` | `8cca1fea…4a1d` |
| `tests/eq_support.py` / `tests/test_eq_reloaded_original.py` | `a987ae8c…4d` / `1c017d16…617` |
| `perf_paired.jsonl` | `2708245a…db` |
| `equivalence_report_pre_vs_post.json` | `6ebc9386…fa1` |
| `arm_C_vs_D_within_run.md` | `9dc725bc…83e` |
(valores completos: `sha256sum` dos ficheiros no commit referido no §7.)

## 7. CI e incidente de portabilidade (registado, não escondido)
A CI **Windows (3.11 e 3.12) falhou** nos commits `4f7d91f`…`1c7bf7f` com **6 falhas**, todas no teste da âncora com ontologia efetiva: em Windows os floats das auditorias e os **bytes das amostras sintéticas** enviadas ao oráculo diferem no último bit (libm), enquanto regras, previsões, contagens e sequência de chamadas coincidem. Não é falha de equivalência (os testes EQ Original vs Reloaded, na mesma plataforma, passaram em Windows); é fragilidade do **hash exato** da âncora. Correção sem tocar nas âncoras: suplemento portável (floats e entradas ao oráculo com 8 algarismos significativos), gerado com o código pré-correção; Linux continua a comparar bit-a-bit. Commit `c91ff4b`. O workflow `agnosticism-guard` passou em todos os commits da fase. **Resultado da CI Windows em `c91ff4b`: ver atualização no final deste documento** (não declarado aprovado até haver execução verde).

## 8. Possíveis regressões / riscos residuais
Atributos de relato (`semantic_split_audit_`, `error_region_audit_`, …) mantêm-se presentes; sem ontologia efetiva `error_region_audit_` fica vazio e `semantic_split_audit_` só contém linhas derivadas de `split_audit_` (sem influência). Consumidores GUI/benchmark tolerantes (suite e protocolo sem falhas). O predicado inclui o domínio OWL: se no futuro surgir outro mecanismo ontológico, tem de entrar no predicado (teste de tabela de verdade).

## 9. Conclusão
Científica: sem ontologia efetiva o Reloaded é, comprovadamente em DEV, o Original (árvore, regras, previsões e consumo do oráculo); com ontologia efetiva o comportamento é o de `main`. Operacional: custo ≈ 1,0×. **V1 resolvida; o contrato global continua incumprido enquanto a V2 estiver aberta** → Fase 1b (`14_PLANO_FASE1B_V2.md`, só planeamento). Nenhum resultado desta fase prova a hierarquia de acurácia/fidelidade — esta continua por demonstrar.
