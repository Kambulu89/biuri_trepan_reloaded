# Doc. 55 — Contrato H2: literais m-of-n derivados de axiomas (mecanismos T1/T2)

Estado: **protótipo de validação, só `validation/`** (`h2_axiom_mechanisms.py`, `h2_trace.py`, `h2_contract.py`, `h2_german_proposal.py`, `h2_inference_audit.py`); **nada em `core/`**, nada altera o TREPAN Original, o C4.5 ou o MLP; `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`. Complementa o contrato geral por tipo de axioma (Doc. 51). Nenhuma campanha foi executada; os testes verificam **correção e rastreabilidade**, não desempenho (nenhuma accuracy ou fidelidade foi calculada ou consultada).

## 1. Escopo e mecanismos
| Mecanismo | O que faz | Natureza | Implementado |
|---|---|---|---|
| **M-R** (representação) | acrescenta ao X uma coluna binária por conceito definido (OR das colunas da sua extensão) | enriquecimento de representação | `augment`, `derived_union_literals` |
| **M-P** (poda lógica) | descarta candidatos m-of-n logicamente redundantes (colapso por subsunção, m=1 ou m=n) ou insatisfazíveis (AND de níveis disjuntos); decisão independente de dados e rótulos | restrição lógica sobre candidatos | `DerivedLiteralTrepan._redundancy` |
| M-G (reserva de *pool*) | reservar vagas no *pool* de literais para os derivados | orientação | **só desenho**: com *pool* suficiente reduz-se a M-R; só se distingue com *pool* limitado |
Cadeia obrigatória: `OWL → Reasoner → Fact Registry → Matching → Candidato m-of-n → Decisão de split → Árvore`. **Um literal derivado é um teste `1-de-n` (ou o AND dos literais negados do complemento) que o Original já pode formar** (Doc. 54 §5): os mecanismos orientam a pesquisa, não alargam o espaço de hipóteses.

## 2. Estados de utilização efetiva (por relação) e como se medem
| Estado | Definição operacional | Evidência |
|---|---|---|
| `AVAILABLE` | o axioma está no ficheiro OWL | censo de axiomas |
| `VALIDATED` | ontologia consistente e satisfatível (HermiT) | `h2_german_reasoner_check.json` |
| `MAPPED` | as entidades do axioma casam com atributos ARFF sem rejeição P3 | registo: `feature_entity` |
| `COMPILED` | origina um facto utilizável (`ATTRIBUTE_APPLIES_TO_CONCEPT`, `permitted_use ≠ NONE`) e um literal derivado | `derived_union_literals` |
| `CANDIDATE_GENERATED` | o literal derivado foi **avaliado** como candidato (simples ou dentro de um m-of-n) durante o ajuste | `semantic_trace_[literal].candidate_evaluations > 0` |
| `DECISION_INFLUENCED` | o teste escolhido num nó contém o literal **e** a ablação contrafactual desse axioma altera a escolha nesse nó; ou a poda M-P eliminou um candidato do beam | `decision_influence()` (chosen/changed) e `pruned_*` |
| `BENEFIT_OBSERVED` | ganho conjunto accuracy **e** fidelidade atribuído por contrastes emparelhados e controlos válidos | benchmark (não executado) |
O nome do braço **nunca** é prova de utilização semântica; `DECISION_INFLUENCED` exige o contrafactual. Influência ≠ melhoria.

## 3. Mapeamento axioma → literal → candidato (exemplo completo)
`AX-GC-ORD-savings_account-A62`: `EquivalentClasses(AtLeast_savings_account_A62 ObjectUnionOf(Lvl_savings_account_A62 Lvl_savings_account_A63 Lvl_savings_account_A64))` (**hipótese de IA, fonte não verificada**) →
HermiT infere `Lvl_A62 ⊑ AtLeast_A62`, `Lvl_A63 ⊑ AtLeast_A62`, `Lvl_A64 ⊑ AtLeast_A62` e a cadeia `AtLeast_A63 ⊑ AtLeast_A62` → registo: `ATTRIBUTE_APPLIES_TO_CONCEPT(has_savings_account__A6x, AtLeast_A62)` (derivação `INFERRED_BY_REASONER`) → extensão `{A62, A63, A64}` → coluna binária `onto_any[AtLeast_savings_account_A62] = max(A62, A63, A64)` → candidato simples `onto_any[…] > 0,5` e termo possível de m-of-n → (se escolhido) split na árvore. Equivalente do Original: `1-de-3` sobre as colunas A62, A63, A64 ou `2-de-2` sobre `A61 ≤ 0,5` e `A65 ≤ 0,5`.

## 4. Semântica preservada e mundo aberto
* A coluna derivada é função **determinística** das colunas base da mesma linha (treino **e consultas**): o adaptador recalcula-a nas consultas sintéticas (`_draw_membership_queries`) e o oráculo só vê as colunas base (`AugmentedOracle`) ⇒ **mesmo oráculo** do Original; sem isto as colunas derivadas amostradas de forma independente seriam incoerentes.
* **Categorias desconhecidas / não ordinais:** linha sem nível ativo ⇒ derivadas 0 (nunca se atribui a um nível); níveis não ordinais (`A65` desconhecido, `A14` sem conta) ficam **fora** das extensões ordinais.
* **Mundo aberto:** disjunções ficam `permitted_use = NONE` (nunca validam nem removem linhas); a poda por disjunção só elimina candidatos insatisfazíveis e está rotulada `DISJOINT_UNSATISFIABLE` = **redundante com a codificação one-hot** (não é conhecimento externo). Subsunção usada apenas como colapso lógico, nunca como preferência estatística.

## 5. Testes (todos a passar; ver `tests/`)
| Teste | O que demonstra |
|---|---|
| `test_h2_trace::test_disabled_semantics_is_identical_to_the_original_including_oracle_stream` | **equivalência quando a semântica está desativada**: mesmas regras, mesmo n.º de consultas, mesmas linhas consultadas |
| `…::test_derived_columns_are_coherent_in_training_and_queries_and_oracle_sees_only_base_columns` | mapeamento correto (OR), coerência nas consultas, mesmo oráculo (só colunas base) |
| `…::test_unknown_category_never_maps_to_a_level` | tratamento de categorias desconhecidas e do nível não ordinal |
| `…::test_trace_links_axiom_literal_candidates_and_decisions` | cadeia Axioma → Literal → Candidato → Decisão; proveniência (`fact_ids`, `needs_reasoner`); controlo de complexidade (`max_nodes`) |
| `…::test_redundancy_pruning_is_logical_and_data_independent` | M-P: colapso OR/AND, disjunção, polaridade; independente de rótulos |
| `…::test_counterfactual_ablation_removes_exactly_one_literal_from_the_trace` | ablação por axioma remove exatamente um literal do rastreio |
| `…::test_counterfactual_decision_influence_structure_and_consistency` | atribuição contrafactual por literal (nós escolhidos vs alterados), reprodutível |
| `…::test_no_leakage_construction_is_label_free` | a construção dos literais e `fit` não recebem rótulos/teste |
| `test_h2_contract_and_mechanisms` (7) | extensões exatas e aninhadas; dependência do reasoner; O3/H/I; mundo aberto (disjunção não remove linhas); fichas (IA não valida; sem dados; aprovações só com nome/data/assinatura); O2 válida, 61/61, 10 literais |
| `test_h2_audit` | auditoria das 55 inferências e dos 10 literais coerente com as fichas e o OWL |
**Controlo de complexidade:** nº de colunas acrescentadas = nº de literais (10 em O2); `max_nodes`, `max_depth`, `max_n` iguais; candidatos avaliados reportados por literal.

## 6. Rastreabilidade (`semantic_trace_`)
Por literal: conceito, atributos, `needs_reasoner`, `fact_ids` (ligação ao facto/axioma), `candidate_evaluations`, podas por razão, `chosen_in_nodes`; global: `pruned_total`, `n_internal_nodes`. Procedimento de atribuição: `decision_influence(X, nomes, literais, oráculo, params)` ajusta com todos os literais e, para cada axioma, sem ele (mesma semente, mesmo oráculo, mesmas instâncias) e devolve nós escolhidos/alterados.

## 7. Limitações e pendências
O adaptador usa a subclasse do Original em `validation/`; **não está ligado ao Reloaded do núcleo** (proibido). Os testes usam dados sintéticos de forma fechada; **não** correu nenhum ajuste de desempenho no German Credit. M-G e os tipos T3–T9 permanecem desenho. O perfil OWL 2 DL formal continua por verificar. As fontes e a verdade dos axiomas dependem de revisão humana (Doc. 54).
