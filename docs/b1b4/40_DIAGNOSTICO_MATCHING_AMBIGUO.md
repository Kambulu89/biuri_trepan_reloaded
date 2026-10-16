# Doc. 40 — Diagnóstico do matching ambíguo (quality gate)

Estado: **DIAGNÓSTICO E PROPOSTA; `core/` NÃO alterado.** Código: `validation/benchmark/b32_matching.py`; testes: `tests/test_b32_matching.py` (6); evidência: `docs/evidence/b1b4/b32/matching_diagnosis.json`. Não se autoriza alteração produtiva sem este impacto e sem aprovação.

## 1. Defeito e localização
`OntologyQualityGate.match_features` — `core/ontology_quality.py`:
- **L280–284:** os candidatos (classes, depois propriedades de dados, depois de objetos, pela ordem de `owlready2`) são ordenados por score com `sorted(..., reverse=True)` — ordenação **estável**: em empate vence o 1.º da lista.
- **L285–291:** se `ranked[0] − ranked[1] < min_ambiguity_margin` e os tipos são (classe, propriedade de dados), troca-se (preferência declarada: «atributos são medições, não conceitos») e regista-se `tie_break`.
- **L293–298:** `accepted = best ∧ best_score ≥ 0,72 ∧ (best_score ≥ 0,98 ∨ margin ≥ 0,08 ∨ tie_break)`. A cláusula `best_score ≥ 0,98` **dispensa a margem**.
`_candidate_score` (L224–258) devolve apenas 1,0 (igualdade normalizada), 0,97 (mesmos tokens), 0,86 (subcadeia ≥ 4) ou ≤ 0,92 (similaridade): portanto «≥ 0,98» ⇔ **correspondência exata**. Logo qualquer **empate de correspondências exatas** (duas entidades com score 1,0, margem 0) é aceite e a escolha é a do **acaso da ordem de declaração** — contra a regra de margem mínima positiva do próprio gate.

## 2. Reprodução (testada)
| caso | resultado atual |
|---|---|
| duas propriedades com o mesmo rótulo `thermal_mean`, ordem A,B | aceita `thermal_mean_a` (score 1,0; runner-up 1,0; **margem 0,0**) |
| a **mesma** ontologia com a ordem de declaração B,A | aceita `thermal_mean_b` — **entidade diferente** (escolha arbitrária; testado) |
| exato 1,0 vs. quase (0,86) | aceita (margem 0.14) — correto |
| classe e propriedade empatadas (1,0/1,0) | aceita `hasThermalMean` por `tie_break=datatype_property_over_class` — determinístico, política declarada |
| **empate de 3 vias** (classe + 2 propriedades) | aceita `hasThermalMeanA`: a preferência de tipo **mascara** o empate entre as 2 propriedades (`tie_sets`: 3 entidades no topo, 2 propriedades) |
O `_resolve_entity_collisions` (L321–345) trata **atributos** diferentes na mesma entidade, não **entidades** diferentes para o mesmo atributo.

## 3. Causa
Dois atalhos acumulados: (i) o limiar `≥ 0,98` foi pensado para «correspondência exata inequívoca», mas não verifica unicidade; (ii) a preferência de tipo resolve empates classe/propriedade sem verificar se sobra empate dentro do tipo preferido. A margem mínima só é efetiva abaixo de 0,98.

## 4. Contrato esperado
Um atributo só mapeia para uma entidade se existir **uma única** entidade elegível no topo: (a) exatamente uma entidade no melhor score, ou (b) a preferência de tipo (propriedade de dados sobre classe) resolve o empate com **exatamente uma** propriedade de dados no topo. Qualquer outro empate ⇒ `REJECTED_AMBIGUOUS` (determinístico, independente da ordem de declaração). Nunca se escolhe uma entidade arbitrariamente.

## 5. Políticas candidatas (funções puras já testadas em `validation/`)
| política | regra | cobre |
|---|---|---|
| **P1** | rejeitar `margin ≤ 1e-9` sem `tie_break` | empates 2 a 2 de entidades do mesmo tipo |
| **P2** | P1 + margem mínima para 0,98 ≤ score < 1,0 | (equivalente a P1: não existem scores em [0,98; 1,0)) |
| **P3 (recomendada)** | rejeitar se ≥ 2 entidades empatam no topo, exceto se a preferência de tipo deixa **exatamente uma** propriedade de dados no topo | também empates de 3 vias |
Proposta de alteração (NÃO aplicada), em `match_features`, após calcular `ranked`: calcular `top = [e for s,e in ranked if s >= best_score − 1e-9]` e, se `len(top) > 1`, só aceitar quando `sum(kind(e)=="datatype_property" for e in top) == 1`; manter o resto; registar `reason="tie_between_entities"`, `status="REJECTED_AMBIGUOUS"`.

## 6. Impacto nos resultados históricos e nas ontologias congeladas
Aplicando P1, P2 e P3 às decisões do gate nas 7 ontologias registadas:
| ontologia | atributos | aceites (atual) | mudam P1 | mudam P2 | mudam P3 | aceites com empate exato | margem mínima (aceites) | aceites com margem < 0,08 |
|---|---|---|---|---|---|---|---|---|
| iris | 4 | 4 | 0 | 0 | 0 | 0 | 0.238 | 0 |
| wine | 13 | 13 | 0 | 0 | 0 | 0 | 0.379 | 0 |
| breast_cancer | 30 | 30 | 0 | 0 | 0 | 0 | 0.269 | 0 |
| digits | 64 | 64 | 0 | 0 | 0 | 0 | 0.030 | 56 |
| synthetic_binary | 9 | 9 | 0 | 0 | 0 | 0 | 0.280 | 0 |
| synthetic_3class | 12 | 12 | 0 | 0 | 0 | 0 | 0.140 | 0 |
| german_credit | 61 | 61 | 0 | 0 | 0 | 0 | 0.136 | 0 |

**Nenhuma decisão muda** em nenhuma ontologia congelada (iris, wine, breast_cancer, digits, sintéticas, german_credit). Os 56 casos de `digits` com margem < 0,08 são correspondências **exatas únicas** (score 1,0 de `pixel_r_c`→`hasPixelRC` vs. 0,97 da entidade transposta `hasPixelCR`, que tem os mesmos tokens em ordem diferente) e mantêm-se. Logo: **resultados M29, âncoras, goldens e todas as campanhas históricas permanecem válidos** (o matching é determinístico e não muda).

## 7. Retrocompatibilidade e risco para o contrato EQ v2.0
- **S0/S1/S2 (sem ontologia, rejeitada, ou sem mecanismo executado): sem efeito.** O gate só corre com ontologia; `Reloaded(None) ≡ Original` não depende dele.
- **S3:** uma ontologia com empates deixaria de mapear o(s) atributo(s) em causa → menos features derivadas/estrutura; pode passar de S3 a S2 (sem estrutura ativa) ou ser rejeitada por cobertura (< 0,35). A porção Original da árvore continua ≡ Original (equivalência E1–E9 inalterada); a **tentativa semântica auditada** muda de tamanho (não há consultas escondidas).
- **Efeito em controlos:** os controlos permutado/aleatório constroem-se por nomes; um empate nunca é criado por eles (testado nos 7 casos).
- **Risco principal:** ontologias futuras com rótulos duplicados passariam a ter atributos não mapeados (comportamento desejado, mas visível); mitigação: relatório explícito `REJECTED_AMBIGUOUS` com as entidades empatadas.
- **Reprodutibilidade:** hoje o resultado depende da ordem de declaração (reprodução §2); P3 elimina essa dependência.

## 8. Testes
`test_current_gate_accepts_perfect_tie_and_choice_depends_on_declaration_order`, `test_policy_p1_rejects_ties_deterministically_but_keeps_documented_type_preference`, `test_policy_p1_is_order_invariant`, `test_candidate_policies_do_not_change_any_frozen_ontology_matching`, `test_three_way_tie_is_masked_by_type_preference_only_p3_rejects_it`, `test_p3_changes_nothing_in_frozen_ontologies`. No registo (`b31_facts`) o empate perfeito já é tratado como `PERFECT_TIE_AMBIGUOUS` (Doc. 34).

## 9. Decisão pedida
Aprovar (ou não) a alteração P3 em `core/ontology_quality.py` como correção isolada, com: teste de regressão de reprodução invertido (empate ⇒ `REJECTED_AMBIGUOUS`), prova de 0 mudanças nas 7 ontologias, e execução do verificador de agnosticismo e dos testes EQ. **Nesta fase não foi aplicada.**
