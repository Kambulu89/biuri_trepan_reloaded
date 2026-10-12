# Documento 11 — Diagnóstico e plano de correção da V1 (Fase 1a) + critérios de entrada e conclusão

Estado: **plano**; nenhum código alterado. V1 confirmada por ti como não-conformidade de E4 (não é «diferença justificada»). A correção é obrigatória antes de qualquer B1–B4.

## 1. Diagnóstico
Evidência (`docs/evidence/b1b4/equivalence_probe_result.jsonl`, 3 datasets × 2 seeds): sem informação ontológica, Reloaded e Original dão regras, previsões, nós e `membership_queries_` **iguais**, mas o Reloaded envia ao oráculo
**2,6–3,1× mais amostras** e **2,6–2,8× mais chamadas** (p.ex. iris/42: 652→1712 amostras, 5→13 chamadas), e o **fluxo de entradas** (hash) é diferente.
Custo observado no protocolo congelado v1 (iris, seed 42, DEV): braço Original 0,14 s vs Reloaded Core 0,38 s (≈ 2,7×) com a mesma árvore ⇒ trabalho extra sem efeito.

Causas no código (`core/trepan_reloaded_historical.py`):
1. `_decision_sample` (l. 544): em **todos** os nós chama `_build_error_region_profile(..., include_uncertainty=True)` → `_known_row_uncertainty` (l. 391) → `oracle.predict_proba(X)` (consulta extra ao oráculo) e escreve `_current_error_profile_`, `error_region_audit_`.
2. `_priority` (l. 526): só toca no perfil se a estrutura semântica estiver ativa (já inerte sem Ω; ainda assim calcula `_semantic_structure_active()`).
3. `_best_mofn` (l. 1010): quando «não neutro» corre uma segunda pesquisa data-only; quando neutro reutiliza — sem Ω é neutro, mas passa por `_ontology_guided_candidates` e por `_evaluate_error_focused_intervention`.
4. `fit` (l. 142): constrói pesos/grupos/relatedness/projetor e já calcula `self.reload_extension_active_` — **o predicado de «ontologia efetiva» já existe** mas não é usado para despachar.
5. `mirror_when_no_semantic_effect=True` (por omissão): **observação V2 (separada):** com Ω efetiva mas sem efeito mensurável, o Reloaded reexecuta o Original (equivalência por construção, mas **duplica** consultas ao oráculo no mesmo `FrozenOracle`). Não é V1; avaliar em Fase 1a se o contrato EQ de contabilidade se aplica (o contrato E4 é para Ω=∅).

## 2. Princípios da correção (os teus 7 requisitos)
1. **Preservar o Original canónico:** nenhuma edição a `core/trepan_original.py`, `core/c45_j48_tree.py`, `core/trepan_scientific_tuning.py`, `core/scientific_experiment_contract.py`. Alteração **só** em `core/trepan_reloaded_historical.py` (+ testes).
2. **Sem `predict_proba` extra sem enriquecimento:** o ramo `Ω=∅` não chama o oráculo para além do que o Original chama.
3. **Preservar sequência, lotes, argumentos e respostas** das consultas: o fluxo de entradas ao oráculo (cada chamada: método, forma, bytes) do Reloaded(Ω=∅) é **idêntico** ao do Original; o teste usa o oráculo-espião com hash do fluxo.
4. **E1–E9** preservadas (Doc. 7).
5. **Testes unitários, de integração e de propriedade** (§4).
6. **Datasets binários e multiclasse, tipos de atributos variados, múltiplas seeds** (§4).
7. **Verificador de agnosticismo + suite de regressão** completos.
Adicional: **o comportamento com Ω efetiva não pode mudar** (âncora dourada, §3 passo 0).

## 3. Desenho da correção (sequência de implementação)
0. **Âncora dourada (antes de tocar no código):** em `main@1789c17`, gerar com provider de grupos sintético e ontologia sintética (sem datasets conhecidos) o registo `golden_effective.json`: hash de regras, previsões, `membership_queries_`, fluxo de entradas ao oráculo (inclui `predict_proba`), auditoria EFSR, para N configurações. Será comparado **bit-a-bit** depois da correção: prova que Ω≠∅ não muda.
1. `TrepanReloadedClassifier._ontology_effective()` := `self.reload_extension_active_` (calculado no início do `fit`; documentado como predicado único: projetor não-nulo ∨ (α>0 ∧ ∃ profundidade>0) ∨ pesos≠1 ∨ grupos repetidos ∨ relatedness> limiar).
2. **Despacho no topo** em cada gancho sobreposto (`_decision_sample`, `_priority`, `_best_mofn`, `_split_selection_score`, `_feature_priority_scores`, `_draw_membership_queries`, geração de candidatos): `if not self._ontology_effective(): return super().<método>(...)` (Original puro).
3. **`fit` sem Ω:** os atributos de estado/auditoria semântica são inicializados (vazios) **sem** executar nenhuma lógica que consulte o oráculo; o resto do `fit` é o `TrepanOriginalClassifier.fit`.
4. **Campos de relatório** (`semantic_split_audit_`, `error_region_audit_`, contadores a zero): presença mantida para compatibilidade da GUI/benchmark; **não fazem parte de E1–E9**. **Ponto a confirmar por ti:** estes atributos de *reporting* (vazios sem Ω) contam como diferença não semântica? Recomendo **não** os incluir na allowlist (que fica vazia) porque não são saídas de E1–E9; só os trato como tal se confirmares.
5. Nenhuma alteração de valores por omissão (já verificado: parâmetros partilhados têm os mesmos defaults).
6. Remover o custo extra: sem `predict_proba` por nó, o tempo do braço R∅ aproxima-se do Original (alvo medido: ≤ 1,1× no protocolo v1 em iris/bc DEV).
Resultado esperado: `R∅ ≡ O` em E1–E9 incluindo o fluxo de entradas; `R_OWL` inalterado (âncora).

## 4. Testes (Fase 1a)
| Nível | Teste | Verifica |
|---|---|---|
| unitário | `test_eq_dispatch_no_predict_proba`: oráculo-espião que **falha** se `predict_proba` for chamado; `fit` com Ω=∅ | requisito 2 |
| unitário | `test_ontology_effective_predicate`: tabela de verdade (cada condição isolada) | predicado único |
| unitário | `test_eq_oracle_input_stream_identical` (hash de fluxo, nº de chamadas, formas) | requisitos 2–3, E4 |
| integração | `test_eq_runner_arm_C_vs_B`: `BenchmarkRunner` com os braços Original e Reloaded Core numa unidade sintética: linhas idênticas em regras/previsões/nós/`oracle_queries`/`membership_queries` (excluindo tempos) | E1–E9 no runner |
| integração | `test_eq_ontology_rejected` / `neutral`: ontologia presente mas rejeitada/neutra | requisito 1 |
| integração | `test_effective_behaviour_unchanged_golden`: compara com a âncora dourada | Ω≠∅ intacto |
| propriedade | geração aleatória (seeded) de datasets sintéticos: n ∈ [60, 600], p ∈ [3, 40], 2–6 classes, numéricos/binários/discretos/escalas muito diferentes/constantes/correlacionados/NaN onde o Original os aceite, desequilibrados; orçamentos aleatórios (`max_nodes`, `max_depth`, `min_samples_leaf`, `max_n`, `beam_width`, `purity_epsilon`); ≥ 5 seeds; Reloaded(Ω=∅, **todas** as flags EFSR/atributos ligados) ≡ Original em E1–E9 | requisito 6 |
| metamórfico | permutar colunas/linhas, renomear colunas, transformação monótona de features: equivalência preservada | agnosticismo |
| contrato | `test_allowlist_is_empty` | sem exceções |
| regressão | verificador estático + `tests/test_agnosticism_contract.py` + suite completa + CI Windows 3.11/3.12 | requisito 7 |
Datasets de teste: sintéticos gerados por seed (nunca os datasets CONFIRM; DEV apenas como sanidade).

## 5. Critérios de entrada na Fase 1a
1. Esta Fase 0.5 revista e a lista D-7 / margens D-8 fechadas por ti (§Doc. 12 «decisões pendentes»).
2. CI verde em `main` (já: `1789c17`); manifest v2 DRAFT revisto por ti (hashes de código a re-congelar na entrada).
3. **Âncora dourada** gerada e versionada **antes** da primeira linha de código.
4. Decisão sobre o drift de hashes do Original (Doc. 12, D-13).
5. Tu confirmas o ponto 4 de §3 (atributos de reporting) e a observação V2 (mirror).

## 6. Critérios de conclusão da Fase 1a (tudo obrigatório)
1. Todos os testes da §4 passam; a suite completa e o agnosticismo (estático, dinâmico, metamórfico) verdes localmente e na CI (Windows 3.11/3.12 + `agnosticism-guard`) **no commit exato** a integrar.
2. Relatório de conformidade: `R∅ ≡ O` em E1–E9 em ≥ 6 configurações sintéticas × ≥ 5 seeds + 3 datasets DEV × 2 seeds; fluxo de entradas ao oráculo idêntico; allowlist vazia.
3. Âncora dourada igual (Ω≠∅ inalterado).
4. Custo: `reloaded_core_time / original_time ≤ 1,1` em iris e breast_cancer (DEV, protocolo v1), reportado.
5. Nenhuma alteração a ficheiros congelados; B1–B4 **não** implementados; PR criado só com o teu pedido; merge só com autorização.
6. Documento de fecho com SHAs, contagens de testes e a decisão sobre V2.

## 7. Riscos
Divergências escondidas no `fit` (RNG, estado herdado) · gancho esquecido que consulte o oráculo · o espião de oráculo não cobre `predict_proba` indireto (mitigado: espião falha na chamada) · custo da âncora dourada · interação com o `mirror`.
