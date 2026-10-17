# Doc. 34 — Registo de factos semânticos (B3.1)

Estado: **IMPLEMENTADO EM `validation/` (experimental); `core/` NÃO alterado.** Código: `validation/benchmark/b31_facts.py`, `b31_propagation.py`; testes: `tests/test_b31_facts_relatedness.py`, `tests/test_b31_protocol.py`. Evidência: `docs/evidence/b1b4/b31/` (`dev_registries_summary.json`, `integrity_b30.json`, `freeze_B30.json`). Base: Docs. 31–33; autorização B3.1 (apenas `validation/`).

## 0. Encerramento da B3.0 (ordem 1–2)
| verificação | resultado |
|---|---|
| CI do commit `9bc7cbb` (API pública, jobs individuais) | **Windows 3.11 success · Windows 3.12 success · agnosticism-guard success** (todos os fluxos concluídos) |
| Integridade dos artefactos M29 (`validation/benchmark/b31_integrity.py`, reproduzível) | **302/302** hashes (20 diretórios de unidade) coincidem |
| `split.json` repostos | **20/20** presentes e com o hash registado **na altura da execução** em cada `UNIT_SHA256.json` (a proveniência é o registo por unidade escrito pelo executor, independente da cópia; os ficheiros foram copiados sem alteração da diretoria de saída da campanha) |
| Relatório M29 regenerado só dos brutos versionados | **JSON idêntico** (igualdade exata); F29≡A29 18/18; comparabilidade estrita 18/18 sem problemas |
| Estado congelado | `docs/evidence/b1b4/b31/freeze_B30.json` (hashes de árvore git por caminho, `core/` incluído); etiqueta local `b30-freeze` em `9bc7cbb` (o *push* da etiqueta foi recusado pelo remoto: o congelamento fica provado pelos hashes de árvore e pelo commit) |
| `core/` | idêntico entre `9bc7cbb` e o estado atual (`git diff --stat 9bc7cbb HEAD -- core` vazio) |
| Testes de regressão/agnosticismo | **428 passed, 40 skipped** + verificador estático OK (§7) |
A campanha M29 não foi repetida.

## 1. Papel do registo
Camada **tipada** entre a ontologia (OWL, reasoner, matching) e qualquer consumidor experimental (relatedness, orientação de candidatos). Não converte relações OWL em regras tabulares: produz *factos* com estatuto epistémico e **uso permitido** explícito; o consumidor tem de respeitar `permitted_use`. Não gera axiomas de domínio.

## 2. Contrato de `SemanticFact`
| campo | conteúdo |
|---|---|
| `id` | `F-<tipo>-<sha256[:12]>` determinístico sobre (tipo, entidades, axiomas de origem, derivação) |
| `kind` | `CONCEPT_SUBSUMPTION`, `CONCEPT_EQUIVALENCE`, `CONCEPT_DISJOINTNESS`, `PROPERTY_SUBSUMPTION`, `PROPERTY_EQUIVALENCE`, `ATTRIBUTE_APPLIES_TO_CONCEPT`, `VALUE_BOUND`, `ORDER_RELATION`, `SAME_MEASUREMENT_FAMILY`, `EMPIRICAL_ORDER_OBSERVATION`, `MATCH_FAILURE`, `UNSUPPORTED_AXIOM` |
| `entities` / `attributes` | entidades OWL envolvidas / atributos tabulares mapeados |
| `source_axioms` | axiomas de origem em texto (`A SubClassOf Root`, `p Domain d`, `C SubClassOf A [inferred]`, `p SubPropertyOf* q`); ou a citação documental |
| `provenance` | `ontology_sha256`, ficheiro, compilador, resultado do reasoner, fonte documental |
| `epistemic` | `LOGICALLY_ENTAILED` · `DOMAIN_DOCUMENTED` · `EMPIRICALLY_OBSERVED` · `HYPOTHESIS` · `UNKNOWN` |
| `derivation` | `ASSERTED` (explícito) · `DERIVED_CLOSURE` (fecho de explícitos, incl. herança de domínio por sub-propriedade e equivalência) · `INFERRED_BY_REASONER` · `DOCUMENT` · `DATA` |
| `reasoner_result` | `{engine, executed, consistent}` do reasoner no momento da compilação |
| `matching_confidence` | score do matching atributo↔entidade (gate de qualidade existente) |
| `scope` | `level` ∈ {`ontology_schema`, `tabular_attribute`, `individual_values`, `row_values`, `training_rows`} + parâmetros |
| `application_restrictions` | restrições de aplicação (ex.: `requires_row_as_individual_contract`) |
| `permitted_use` | `NONE` · `PREFERENCE` · `HARD_CONSTRAINT` (política única `decide_permitted_use`) |
| `use_state` | `NOT_USED` · `AVAILABLE` · `USED_AS_PREFERENCE` · `USED_AS_CONSTRAINT` · `REJECTED` |
| `rejection_reason` | motivo (ex.: `REJECTED_AMBIGUOUS`, `REJECTED_LOW_SCORE`, `ontology_status=INCONSISTENT`, `unsupported_axiom_type_no_tabular_contract`) |
| `relation_class`, `validity_conditions`, `review` | só para relações numéricas documentadas (§5) |

## 3. Estatuto epistémico e uso permitido
| estatuto | significado | uso máximo |
|---|---|---|
| `LOGICALLY_ENTAILED` | consequência da ontologia (explícita, fecho ou inferência do reasoner) **sobre o esquema/propriedades**; a aplicação a dados tabulares depende do contrato do tipo de facto | `PREFERENCE` para `ATTRIBUTE_APPLIES_TO_CONCEPT`, subsunções, equivalências; **`NONE`** para disjunção e `VALUE_BOUND` |
| `DOMAIN_DOCUMENTED` | fonte verificável + condições de validade declaradas | `PREFERENCE`; `HARD_CONSTRAINT` **só** se identidade/restrição formal, 5 condições declaradas e revisões **humanas** `APPROVED` |
| `EMPIRICALLY_OBSERVED` | regularidade nos dados (mesmo 100 %) | `NONE` (diagnóstico; nunca axioma universal) |
| `HYPOTHESIS` | fonte/condições em falta ou declarada hipótese | `NONE` |
| `UNKNOWN` | matching falhado/axioma não suportado | `NONE` |
O predicado `is_eliminable(fact)` é o **único** ponto onde se autoriza eliminar candidatos/consultas: exige `HARD_CONSTRAINT`, não rejeitado, e estatuto logicamente implicado ou documentado. Para uma ontologia compilada sem sidecar humano, **nenhum** facto é eliminável (testado).

## 4. Semântica OWL e mundo aberto
- **Axiomas reconhecidos** (extraídos *antes* de o reasoner materializar `is_a`): `SubClassOf` (nomeado), `EquivalentClasses`, `DisjointClasses/disjointWith`, `rdfs:domain` de propriedades de dados, `SubPropertyOf`, `EquivalentProperties`, facetas numéricas de `rdfs:range`.
- **Reconhecidos e não interpretados** (`UNSUPPORTED_AXIOM`, rejeitado com razão): `someValuesFrom`, `allValuesFrom`, cardinalidades, cadeias de propriedades — não há contrato tabular; são contados, nunca traduzidos.
- **Ausência ≠ negação:** não se criam factos negativos. Só uma disjunção *declarada* gera `CONCEPT_DISJOINTNESS`, com `permitted_use=NONE` e a restrição «não autoriza eliminar combinações de atributos» (conceitos disjuntos não implicam que atributos não coexistam numa linha).
- **`rdfs:domain`/`rdfs:range`/cardinalidades não são validadores fechados de linhas.** O domínio gera `ATTRIBUTE_APPLIES_TO_CONCEPT` («propriedade aplicável a indivíduos do conceito»), com a restrição explícita «não é validador fechado de linhas». Facetas geram `VALUE_BOUND` com `requires_row_as_individual_contract` e `permitted_use=NONE`.
- **Fecho:** subsunção/equivalência de conceitos, herança do domínio por super-propriedades e propriedades equivalentes; equivalência de classes funde nós (corrige a perda medida na B3.0).
- **Reasoner:** inconsistência/insatisfazibilidade ⇒ `ontology_status=INCONSISTENT` e **todos** os factos `REJECTED`. Subsunções/equivalências inferidas (`inferred_subclass_relations`/`inferred_equivalences`) viram factos `INFERRED_BY_REASONER` e propagam-se ao fecho dos atributos (testado com HermiT: `C ⊑ A` inferido chega a `temperature_peak`).
- **Matching:** o do gate existente (`OntologyQualityGate.match_features`); falhas viram `MATCH_FAILURE`/`UNKNOWN`, sem factos de atributo. Atributo casado com um **conceito** (não com propriedade de dados) é rejeitado (sem contrato de interpretação).

## 5. Relações numéricas (identidade, restrição formal, regularidade, hipótese)
`ORDER_RELATION` documentada exige: `source.citation` e as 5 condições (`same_quantity`, `same_observation_set`, `compatible_statistical_definitions`, `preprocessing_preserves_relation`, `missing_values_handled`) **todas presentes**; caso contrário o estatuto cai para `HYPOTHESIS`. `relation_class ∈ {MATHEMATICAL_IDENTITY, FORMAL_DOMAIN_CONSTRAINT, EMPIRICAL_REGULARITY, HYPOTHESIS}`; só as duas primeiras podem chegar a `HARD_CONSTRAINT`, e apenas com `review.domain_expert_status` **e** `review.methodology_status = APPROVED` fornecidos por humanos. O compilador **nunca** escreve `APPROVED`: por omissão `DOMAIN_REVIEW_PENDING`/`REVIEW_PENDING`, e acrescenta a restrição «sem revisão de domínio aprovada não é restrição obrigatória». `observe_order_relation` produz `EMPIRICALLY_OBSERVED` (taxa de violação, n) com `permitted_use=NONE` **mesmo com 100 %** (testado; uma relação que falha em 36 % das linhas é registada como tal e nunca promovida).

## 6. Resultados nas ontologias DEV (compilação real)
| ontologia | factos | rec. explícitos | descartados (sem contrato) | inferências novas | estado |
|---|---|---|---|---|---|
| iris | 11 | 10 | 1 (disjunção) | 0 → `NO_NEW_INFERENCES` | VALID |
| wine | 32 | 29 | 3 | 0 | VALID |
| breast_cancer | 66 | 63 | 3 | 0 | VALID |
| digits | 134 | 131 | 3 | 0 | VALID |
«Zero inferências novas numa ontologia consistente» é classificado `NO_NEW_INFERENCES`, **não** falha do reasoner nem da propagação (distinção implementada em `propagation_report`: `PROPAGATED` / `NO_NEW_INFERENCES` / `PROPAGATION_FAILURE_OR_IRRELEVANT`). Destinos de propagação por facto: `RECOGNIZED_EXPLICIT`, `RECOGNIZED_INFERRED`, `DISCARDED` (+razão), `BLOCKED_BY_MATCHING`, `IRRELEVANT_TO_FEATURES`, `REACHED_CANDIDATE`.

## 7. Testes e CI
Testes B3.1 (todos em `tests/`, só `validation/`): `test_b31_facts_relatedness.py` (26+2: campos/proveniência/ids determinísticos, separação dos estatutos, nenhum facto eliminável a partir da ontologia, domínio/disjunção/facetas não são validadores fechados, mundo aberto, inferência HermiT chega aos atributos, relatório de propagação, ontologia inconsistente sem factos úteis, matching bloqueia, relações documentadas e empíricas, política de uso permitido, relatedness: propriedades, ordem, sub-propriedade, equivalência, ancestral genérico, renomeação, empate perfeito, Wu–Palmer), `test_b31_steering_identifiability.py` (EQ sem unidades, rastreabilidade, congelamento v1/v2, estrutura das 12 variantes, abstenção) e `test_b31_protocol.py` (revisor não humano rejeitado, validação independente só com 3 revisões humanas + passos automáticos).
**Regressão desta fase** (execução local, Python 3.11): `tests/test_eq_reloaded_original.py`, agnosticismo (`test_agnosticism_contract`, `test_dataset_agnosticism`, `test_v92_static_guards`), contrato de hierarquia conjunta, M29 (relatório e executor), B3.0, B3.1, amostrador C, controlos ontológicos e B3-a (contrato): **428 passed, 40 skipped, 0 failed** (os 40 *skips* são pré-existentes: cenários dependentes de plataforma/GUI). Verificador estático de agnosticismo: OK.
**Nota de compatibilidade:** `test_validation_is_allowed_to_know_the_core…` faz `ast.parse` de todos os ficheiros de `validation/`; uma *f-string* aninhada (válida só em Python ≥ 3.12) no relatório da identificabilidade falhou em 3.11 e foi corrigida; todos os `b3*.py` e `test_b3*.py` são agora sintaticamente válidos em 3.11.
**CI (API pública do GitHub):** commit `9bc7cbb` (B3.0): `windows-tests (3.11)` **success**, `windows-tests (3.12)` **success**, `agnosticism-guard` **success**. Commits seguintes da B3.1: o *guard* passa em todos; o estado do Windows do commit final está no relatório de entrega (só declarado verde com todos os fluxos concluídos).

## 8. Limitações
1. A dedução de fecho de atributos (domínio por super-propriedade) é calculada pelo compilador (RDFS/OWL), não consultada ao HermiT; o reasoner valida consistência e fornece as inferências de classes.
2. Perfil OWL 2 DL: **não verificado formalmente** (sem *profile checker* no ambiente); o protocolo (Doc. 37) usa um proxy explícito (sem *punning* + classificação HermiT sem erro).
3. O matching é lexical (herdado): ver Doc. 32 §5; um empate perfeito (score ≥ 0,98) entre duas entidades é aceite pelo gate do núcleo — achado da B3.1 (Doc. 36 §6), não corrigido no núcleo; o registo trata-o como `PERFECT_TIE_AMBIGUOUS`.
4. Versões anteriores do compilador deixavam `MATCH_FAILURE` sem axiomas de origem e aceitavam empates perfeitos de matching; corrigido e registado como desvio pós-congelamento (Doc. 36 §3 e §8; 300 comparações sem diferenças).
