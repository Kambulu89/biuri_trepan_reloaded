# Doc. 31 — Auditoria da utilização real do conhecimento OWL (B3.0)

Estado: **AUDITORIA CONCLUÍDA; nada do núcleo alterado.** Só leitura de código, instrumentação observacional em `validation/benchmark/b30_*.py` e testes sintéticos (`tests/test_b30_semantic_audit.py`). Sem nova campanha, sem datasets confirmatórios, sem merge; B3-a/b/c/d e B4 não implementados. Evidência: `docs/evidence/b1b4/b30_audit/` (README com os comandos). Documentos irmãos: Doc. 32 (relatedness e axiomas) e Doc. 33 (proposta B3 revista).

**Resultado central.** Com as ontologias DEV e o caminho atual, a informação OWL chega ao Reloaded reduzida a: (i) uma **partição** das features em grupos (via `rdfs:domain`), (ii) **agregados** (médias padronizadas por grupo) como features derivadas, e (iii) quatro vetores/matrizes (pesos, grupos, relatedness, profundidades) em que **a relatedness é constante (0,7124) e a profundidade constante (1,0)**. Consequências medidas: (a) o gerador de candidatos m-of-n une **todas** as features originais num só componente semântico, qualquer que seja a ontologia (verdadeira ou permutada); (b) o termo `α·OntoDepth` soma a mesma constante a todos os candidatos; (c) a única diferença entre ontologia verdadeira e permutada que chega à pontuação é a composição dos agregados e um bónus de coesão de grupo de ≈ 4 % (1,15 vs 1,107). Isto explica, sem recorrer a ruído, por que a M29 não distingue real de permutada/aleatória (Doc. 30). **Não** é prova de que conhecimento verdadeiro e mais expressivo não ajudaria; mostra que o caminho atual **não o consegue transportar**.

## 1. Encerramento da M29 (ordem obrigatória 1–2)
### 1.1 CI do commit `b8d2428` (API pública do GitHub, jobs individuais)
| fluxo | job | resultado |
|---|---|---|
| Windows scientific regression | `windows-tests (3.11)` | **success** |
| Windows scientific regression | `windows-tests (3.12)` | **success** |
| Dataset agnosticism guard | `agnosticism-guard` | **success** |

Todos os passos de cada job com `success` (nenhum falhado ou ignorado). Os testes de equivalência Original≡Reloaded sem OWL fazem parte do fluxo e foram reexecutados localmente nesta fase (§1.2).

### 1.2 Integridade e consistência
| verificação | resultado |
|---|---|
| Hashes dos brutos (`UNIT_SHA256.json`, 20 diretórios de unidade, 302 ficheiros) | **302/302 coincidem** (na origem: 288/288 + 18 `split.json`) |
| **Achado:** `split.json` das 18 unidades está nos hashes mas **faltava** na cópia de evidência `docs/evidence/b1b4/m29_campaign/main/raw/` | **corrigido** nesta fase: copiados os 18 ficheiros de origem (inalterados, hash conferido). Nenhum resultado, classificação ou relatório foi alterado |
| Relatório regenerado só dos brutos versionados (`m29_report --sc …/sampler_c/dev_numeric --m29 …/main …/g29_pilot_patch`) vs `m29_campaign/report/m29_report.json` | **JSON idêntico** (igualdade exata) |
| F29 ≡ A29 (Reloaded sem OWL ≡ Original equivalente sob o amostrador C) | 18/18 |
| Comparabilidade estrita (11 propriedades), em execução e no relatório | 18/18; G29 isenta só `query_budget` (registado) |
| Estados do Doc. 30 | reproduzidos: Nível 1 `NO_GAIN` 10, `TRADEOFF` 4, `ONTOLOGICAL_JOINT_GAIN` 2, `EQUIVALENT` 2; atribuição `NOT_ATTRIBUTED` 16 / `NOT_ASSESSABLE` 2; Nível 2 `NOT_SATISFIED` 7, `HIERARCHY_INFEASIBLE` 9, `TRADEOFF` 1, `JOINT_HIERARCHY_SATISFIED` 1 |

Nenhum erro que invalide os resultados foi encontrado; **a campanha M29 não é repetida.** Há três esclarecimentos de leitura (errata do Doc. 30, §3.4) que não alteram números nem classificações.

## 2. Limitações dos controlos (ordem 3): auditoria
### 2.1 E29 — engenharia convencional
- **Dimensionalidade:** E29 tem ≤ `p//2` agregados de ≥ 2 colunas (`ConventionalFeatureProvider`, `core/benchmark/semantic.py:83-127`). Dimensão derivada média (E29 vs B29): breast_cancer **5,6 vs 53,0**; digits 2,67 vs 3,0; iris 1,0 vs 2,0; wine 2,2 vs 3,0. Em breast_cancer 50/53 features de B29 são **relacionais** (diferenças/rácios entre colunas da mesma família, lidas das anotações `measurementFamily`/`statisticRole`), sem análogo em E29.
- **Candidatos:** E29 tem estrutura semântica neutra (pesos 1, sem grupos, relatedness identidade, profundidades 0) ⇒ `_semantic_structure_active()` é falso ⇒ `_ontology_guided_candidates` devolve `[]` (`trepan_reloaded_historical.py:1422-1430`): **zero** candidatos semânticos, enquanto B29 avalia até 24 por nó. E29 é, na prática, o **Original sobre uma representação enriquecida convencional**.
- **Orçamento de consultas:** E29/B29 = 0,95 (breast_cancer), 0,98 (digits), 0,99 (iris), 1,09 (wine): comparável. O fator confundidor em E29 não é o orçamento; são **a dimensão e o gerador de candidatos**.
- **Conclusão:** E29 permanece `confundido` em 16/18 unidades (regra de dimensão >25 %). **Não** se usa E29 para afirmar superioridade causal da engenharia convencional sobre a ontologia, nem o contrário. **Controlo em falta (identificado):** *Original sobre a representação enriquecida OWL com estrutura semântica neutra* — isola representação vs mecanismo semântico (Doc. 33 §6).

### 2.2 G29 — Original com orçamento igualado
Fonte: `docs/evidence/b1b4/b30_audit/g29_audit.json` (por réplica) e quadro por unidade abaixo. «Pedido» = consultas de B29 (`sc_D_r{k}`, Doc. 26) da mesma unidade e réplica; «executado» = consultas efetivas de G29; Δ = executado − pedido; rácio = executado/pedido (média da unidade; entre parênteses mín–máx das réplicas). **Distribuição:** G29 obtém o orçamento extra **aumentando `min_sample` uniformemente** (fator ≈ 2,1–2,4×, amostragem por nó), enquanto o excesso de B29 vem de consultas ativas semânticas e da avaliação de candidatos adicionais; a *quantidade* pode igualar-se mas a *distribuição por nó* difere (colunas «q/nó» e «nós G/B»; `control_meta.note`: «NÃO equivalência total»).

| unidade | pedido (B29) | executado (G29) | Δ | rácio (min–max) | draws válidos | q/nó G | q/nó B | nós G/B | teto de nós G/B | contraste válido |
|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer:11 | 32146 | 33531 | +1385 | 1.04 (1.03–1.06) | 3/3 | 1106 | 1170 | 30/28 | 3/3 | sim |
| breast_cancer:123 | 34316 | 37224 | +2908 | 1.09 (1.06–1.12) | 3/3 | 1201 | 1506 | 31/24 | 3/3 | sim |
| breast_cancer:2024 | 65094 | 73359 | +8264 | 1.13 (1.05–1.24) | 3/3 | 1271 | 1103 | 58/59 | 3/3 | sim |
| breast_cancer:42 | 34716 | 37844 | +3128 | 1.09 (0.97–1.18) | 3/3 | 1288 | 1204 | 30/29 | 3/3 | sim |
| breast_cancer:7 | 34104 | 37028 | +2924 | 1.09 (1.07–1.10) | 3/3 | 1455 | 1156 | 26/30 | 3/3 | sim |
| digits:123 | 518358 | 526080 | +7722 | 1.01 (1.01–1.02) | 3/3 | 2085 | 2054 | 252/252 | 3/3 | sim |
| digits:42 | 349531 | 357075 | +7543 | 1.02 (1.02–1.03) | 3/3 | 1404 | 1382 | 254/253 | 3/3 | sim |
| digits:7 | 349725 | 358175 | +8450 | 1.02 (1.02–1.02) | 3/3 | 1405 | 1379 | 255/254 | 3/3 | sim |
| iris:11 | 22892 | 24128 | +1236 | 1.05 (0.94–1.11) | 3/3 | 901 | 756 | 27/30 | 3/3 | sim |
| iris:123 | 22791 | 22773 | -18 | 1.00 (0.94–1.04) | 3/3 | 374 | 374 | 61/61 | 3/3 | sim |
| iris:2024 | 23002 | 25205 | +2204 | 1.09 (1.00–1.18) | 3/3 | 869 | 838 | 29/28 | 3/3 | sim |
| iris:42 | 21258 | 20556 | -702 | 0.97 (0.88–1.08) | 3/3 | 677 | 735 | 30/29 | 3/3 | sim |
| iris:7 | 25719 | 25162 | -556 | 0.97 (0.85–1.09) | 3/3 | 913 | 849 | 28/30 | 3/3 | sim |
| wine:11 | 17106 | 16612 | -494 | 0.97 (0.68–1.23) | 2/3 | 736 | 564 | 24/30 | 1/3 | NÃO |
| wine:123 | 12542 | 17799 | +5258 | 1.41 (0.89–2.02) | 1/3 | 1266 | 759 | 14/18 | 0/1 | NÃO |
| wine:2024 | 40892 | 67798 | +26906 | 1.94 (1.03–3.69) | 2/3 | 1379 | 1353 | 54/32 | 2/1 | NÃO |
| wine:42 | 16708 | 18302 | +1594 | 1.17 (0.69–1.46) | 0/3 | 530 | 686 | 38/24 | 0/0 | NÃO |
| wine:7 | 15167 | 14622 | -545 | 0.96 (0.85–1.05) | 3/3 | 484 | 577 | 30/27 | 3/2 | sim |

Leitura: nas unidades de iris, breast_cancer e digits **todas** as réplicas ficam em [0,8; 1,25]. Em **wine só 8/15 réplicas** (53 %) e em 4/5 unidades de wine pelo menos uma réplica fica fora (wine:42: 0/3 dentro, apesar de a média da unidade ser 1,17). A regra pré-registada (Doc. 29, A3) é **assimétrica e ao nível da média da unidade**: só `confunde` se B29 perde e o controlo gastou > 1,25× (ou B29 ganha e gastou < 0,8×). Resultado: G29 `confundido` em 1/18 (wine:123, ρ = 1,41). **Validade do contraste:** informativa nas 13 unidades fora de wine e em wine:7; **frágil** nas outras 4 de wine (ρ por réplica entre 0,68 e 3,69). O contraste B29−G29 não é usado para atribuir ganho; serve só para excluir «mais orçamento» como explicação — e G29 ≈ A29 (accuracy 0,9355 vs 0,9349).

### 2.3 Breast Cancer — por que falharam as permutações
Auditoria dos 30 controlos (15 permutados + 15 de grupos aleatórios; ≤ 4 tentativas cada; `docs/evidence/b1b4/m29_campaign/main/audits/`): **29/30 esgotaram as 4 tentativas sem controlo completo** (15/15 permutados, 14/15 aleatórios); 1 (aleatório) obteve controlo completo à 2.ª tentativa. Razões nas tentativas falhadas: *dimensão derivada* (107 ocorrências: 48–50 ≠ 53) e *escala* (104 ocorrências; até 4,4×10¹¹ vs ≤ 248 permitido, i.e. 100× a da OWL real). Causa de raiz (estrutural, não acaso):
1. 50/53 derivadas são **relacionais** (família × papel): `*_worst_minus_mean`, `*_relative_worst_delta`, `*_family_contrast`, `*_normalized_error`, `*_error_ratio` (`ontology_processor.py:484-577`). A informação está em **quais colunas se combinam**.
2. Permutar **famílias inteiras** (blocos coerentes) deixa as derivadas *idênticas em distribuição* (apenas renomeadas): **não destrói** a relação, logo não é controlo negativo.
3. Uma permutação que **quebra** os blocos combina colunas de famílias e escalas incomparáveis (área ~10³ vs dimensão fractal ~10⁻²) ⇒ razões/diferenças com escala explosiva e, quando os papéis ficam incompletos, menos derivadas (dimensão ≠ 53).
Logo **não existe permutação simultaneamente destrutiva, de escala comparável e de dimensão igual** para features relacionais sem alterar o seu cálculo. Um controlo válido exige **definição própria** (ex.: derivadas calculadas sobre colunas padronizadas tanto no real como no controlo, o que muda a representação real e só é admissível numa versão nova e versionada; Doc. 33 §6).

### 2.4 Reconciliação de NOT_ATTRIBUTED / NOT_ASSESSABLE com as limitações efetivas
`attribute_to_ontology` (`core/hierarchy_contract.py:71-91`) devolve `NOT_ATTRIBUTED` **sempre que não há ganho conjunto estrito sobre o Original equivalente** («nada a atribuir»), *antes* de considerar a validade dos controlos; só com ganho direto e controlo exigido ausente/confundido devolve `NOT_ASSESSABLE`. Assim, **os 16 `NOT_ATTRIBUTED` não são evidência desfavorável derivada de controlos**: são unidades sem ganho direto (10 `NO_GAIN`, 4 `TRADEOFF`, 2 `EQUIVALENT`). Um controlo inválido (fallback de breast_cancer, E29 confundido) **nunca** foi tratado como evidência contra a ontologia. Os 2 `NOT_ASSESSABLE` são as 2 únicas unidades com ganho direto (digits:42: E29 confundido; wine:42: C29 e E29 confundidos). Classificações **não modificadas** (sem prova documental que o justifique).

| unidade | direto (vs Original equiv.) | atribuição | razão | C29 | D29 | E29 | G29 | hierarquia global |
|---|---|---|---|---|---|---|---|---|
| breast_cancer:11 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | fallback | fallback | confundido | válido | NOT_SATISFIED |
| breast_cancer:123 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | fallback | parcial | confundido | válido | HIERARCHY_INFEASIBLE |
| breast_cancer:2024 | TRADEOFF | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | fallback | fallback | confundido | válido | TRADEOFF |
| breast_cancer:42 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | fallback | fallback | confundido | válido | HIERARCHY_INFEASIBLE |
| breast_cancer:7 | TRADEOFF | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | fallback | fallback | confundido | válido | HIERARCHY_INFEASIBLE |
| digits:123 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | válido | válido | NOT_SATISFIED |
| digits:42 | ONTOLOGICAL_JOINT_GAIN | NOT_ASSESSABLE | ganho direto mas controlo(s) exigido(s) confundido(s)/ausente(s): conventional_features | válido | válido | confundido | válido | JOINT_HIERARCHY_SATISFIED |
| digits:7 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | válido | válido | NOT_SATISFIED |
| iris:11 | EQUIVALENT | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | válido | HIERARCHY_INFEASIBLE |
| iris:123 | TRADEOFF | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | válido | HIERARCHY_INFEASIBLE |
| iris:2024 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | válido | HIERARCHY_INFEASIBLE |
| iris:42 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | válido | NOT_SATISFIED |
| iris:7 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | válido | HIERARCHY_INFEASIBLE |
| wine:11 | TRADEOFF | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | válido | HIERARCHY_INFEASIBLE |
| wine:123 | EQUIVALENT | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | confundido | NOT_SATISFIED |
| wine:2024 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | confundido | confundido | válido | NOT_SATISFIED |
| wine:42 | ONTOLOGICAL_JOINT_GAIN | NOT_ASSESSABLE | ganho direto mas controlo(s) exigido(s) confundido(s)/ausente(s): permuted_ontology,conventional_features | confundido | válido | confundido | válido | HIERARCHY_INFEASIBLE |
| wine:7 | NO_GAIN | NOT_ATTRIBUTED | sem ganho conjunto direto (nada a atribuir) | válido | válido | confundido | válido | NOT_SATISFIED |

Recomendação de relato (sem alterar classificações): acrescentar sempre a coluna «razão» e separar `NOT_ATTRIBUTED(sem ganho direto)` de `NOT_ATTRIBUTED(controlos não confirmam)`; esta segunda categoria teve **0** unidades na M29.

## 3. Mapa de execução real do percurso OWL (P0)
Legenda: **[impl]** capacidade implementada; **[exec]** efetivamente executada na M29. Linhas = `ficheiro:linha` do commit `b8d2428`.

### 3.1 Cadeia
```
Ficheiro OWL (RDF/XML)
 → load_ontology_isolated               core/owl_runtime.py:78                 owlready2, mundo isolado por chamada
 → run_owl_reasoner (HermiT/Java)       core/ontology_reasoner.py:212          consistência, insatisfazíveis, subclasses inferidas
 → OntologyQualityGate.evaluate         core/ontology_quality.py:424           ABox seguro, genericidade, cobertura, reasoner exigido
   └ match_features (léxico)            core/ontology_quality.py:260           feature ↔ entidade, score, margem, colisões
 → OntologyProcessor.fit / transform    core/ontology_processor.py:74 / 236    features derivadas (hierárquicas, relacionais, restrição, categóricas)
 → OntologySemanticGraph.from_ontology  core/ontology_semantic_graph.py:170    nós, arestas por tipo, pais, domínios, grupos, profundidade, relatedness
 → SemanticContext (OwlSemanticProvider.build)  core/benchmark/semantic.py:192-262   pesos, grupos, relatedness, profundidades, projector
 → TrepanReloadedClassifier.fit(semantic_*)     core/trepan_reloaded_historical.py:322   normaliza pesos; guarda grupos/relatedness/profundidades
 → _best_mofn → _ontology_guided_candidates → _split_selection_score → gate EFSR   :1542 / :1422 / :1274 / :1006
 → árvore Reloaded (+ auditoria por split `semantic_split_audit_`)
```
No benchmark, `OwlSemanticProvider.build` repete carga + reasoner + gate + processador **por chamada/unidade** e ajusta **só sobre o treino**. (Nota: a ordem real é reasoner → gate/matching → processador → grafo; o parser e a «validação» DL coincidem com a carga e com o reasoner.)

### 3.2 Etapa a etapa (as 10 perguntas: recebe / produz / interpreta / ignora / rejeita / features / candidatos / pontuação / splits / efeito)
**E1 Carga** (`owl_runtime.py:78`). Recebe o caminho; produz o objeto `owlready2` num mundo isolado. Interpreta o que `owlready2` materializa (classes, propriedades, `subClassOf`, domain/range, restrições como objetos em `is_a`). Nada é filtrado; falha de parsing ⇒ «OWL inválida» (`semantic.py:215`). Sem features/candidatos/efeito próprios.

**E2 Validação / quality gate** (`ontology_quality.py:424-550`). Recebe nomes de features, ontologia e relatório do reasoner; produz `accepted`, `status` (`VALID_DOMAIN_ONTOLOGY`, `GENERIC_ONTOLOGY`, `CONTAMINATED_ONTOLOGY`, `LOGICALLY_INCONSISTENT`, `REASONER_NOT_VALIDATED`, `INSUFFICIENT_SCHEMA_COVERAGE`), métricas e avisos (a ontologia «semanticamente pobre» gera o aviso «limita-se a agregados de grupo»). Valida TBox utilizável, ausência de ABox com registos/identificadores de teste, genericidade/placeholder (`PLACEHOLDER_RE`), cobertura ≥ 0,35, ≥ 2 entidades distintas, consistência e classes satisfazíveis. **Não verifica o perfil OWL 2 DL em si** (sintaxe/perfil DL): só consistência lógica. A riqueza é apenas *relatada* (`semantic_richness`, `:130`). Rejeita as condições acima (testado: nomes `f1/Group1` ⇒ rejeitada). Não gera features nem candidatos; decide se o enriquecimento ocorre.

**E3 Reasoner** (`ontology_reasoner.py:212`). Recebe a ontologia; produz `consistent`, `unsatisfiable_classes`, `inferred_subclass_relations`, `inferred_axioms_count`, equivalências e tipos inferidos, e materializa a hierarquia. Interpreta toda a lógica DL do HermiT; nada filtrado; inconsistência ⇒ gate rejeita. **[exec]** nas 4 ontologias DEV: `consistent=True`, **0 subclasses/equivalências/axiomas inferidos** (TBoxes planas): o reasoner **valida mas não produz inferências aplicáveis**; ≈ 0,6 s por chamada.

**E4 Matching** (`ontology_quality.py:224-345`). Recebe nomes de features e entidades (classes, propriedades de dados e de objetos) com rótulos; produz `MatchDecision` (entidade, score, margem, vice, estado). Usa **só nomes/rótulos** (igualdade, mesmos tokens, subcadeia ≥ 4, `SequenceMatcher`/Jaccard); **não usa axiomas** nem tipos. Rejeita: score < 0,72; ambíguo (margem < 0,08, salvo score ≥ 0,98 ou desempate «propriedade de dados sobre classe»); colisões de entidade. **DEV:** cobertura 100 %, score mínimo 1,0 em 4/4; margem mínima 0,03 (digits, aceite por score ≥ 0,98). **Sintético:** «temp_mean», «temperature_average», «lumens» **não** casam (sinónimos/abreviaturas); «hasTemperature» é ambíguo e rejeitado — o matching é lexical, não semântico.

**E5 Grafo semântico** (`ontology_semantic_graph.py:170-264`). Recebe ontologia + matches aceites; produz nós, `edges` simétricas com `edge_types`, `parents`, `domains`, `feature_to_entity`, `feature_groups`, limites numéricos. **Interpreta:** `is_a` (subsunção; em propriedades, super-propriedade e a classe built-in `DatatypeProperty`), `equivalent_to`, `domain`, `range`, `ancestors()` do reasoner (`inferred_subsumption`) e facetas numéricas de `rdfs:range`. **Ignora:** `owl:disjointWith`/`AllDisjointClasses`, `someValuesFrom`/`allValuesFrom`/cardinalidades, cadeias de propriedades, características de propriedade e, na prática, `inverseOf` (o código lista `inverse_property`, mas a sonda não observou aresta `inverse`; testado, Doc. 32 §4). Nada é rejeitado; entidades built-in (`Thing`, `DatatypeProperty`) vão para `builtin_nodes` mas **permanecem como nós/arestas do grafo**. Os grupos (`_derive_feature_groups`, `:418`) usam só vizinhos *diretos* da propriedade com relação `subsumption/domain/equivalent`.

**E6 Features derivadas** (`ontology_processor.py:74-170`, `:401-700`). Recebe treino + matches; produz colunas **hierárquicas** (média padronizada das colunas cujo `domain`/`is_a` direto é a mesma classe; `:401`), **relacionais** (família×papel das *anotações* `measurementFamily`/`statisticRole`; `:484`), de **restrição** (facetas `xsd:min/maxInclusive` explícitas; quantis do treino só com `allow_train_calibrated_bounds`; `:611`) e **categóricas** (`:658`), com deduplicação e filtro de redundância linear. Axiomas que as originam: `domain`/`is_a`, **anotações não lógicas**, facetas de intervalo. Ignora equivalência entre classes e subsunção inferida. **DEV:** iris 2, wine 3, digits 3 (hierárquicas); breast_cancer **53** (3 + 50 relacionais); restrição 0; categóricas 0.

**E7 Contexto semântico** (`semantic.py:233-262`). Produz `weights` (1,0 originais; **1,35** derivadas), `groups` (`gl[-1]` do `feature_groups`; agregados = nome da classe), `relatedness` (**0 quando a feature não tem entidade — todos os agregados**), `depths` (derivadas = 1,0), `query_projector`. **Medido (4 ontologias DEV):** relatedness fora da diagonal ∈ {0; 0,7124}; profundidade ∈ {1,0}; pesos normalizados: originais 0,82–0,98 / derivadas 1,10–1,33. É a etapa onde a informação OWL **colapsa**.

**E8 Enriquecimento do MLP/oráculo.** **[impl]** o `OntologyProcessor` pode alimentar um MLP ontológico (`runner.py`, `use_enr`). **[exec]** *Não* na M29: o oráculo é o MLP **original congelado** (só colunas originais); as derivadas só entram na árvore e as consultas recalculam-nas por `query_projector`. «Enriquecimento do MLP» = **não utilizado** (por contrato do oráculo).

**E9 Candidatos m-of-n** (`trepan_reloaded_historical.py:1422-1530`). Recebe o nó (X, y de consultas) e os literais candidatos; devolve até `semantic_candidate_budget = 24` testes. Usa **só** grupos e relatedness do contexto. Ativação: `_semantic_structure_active()` (`:1407`; grupo repetido OU relatedness > 0,35 OU pesos ≠ 1 OU `α>0` com profundidade > 0) — **verdadeira em todas as ontologias DEV**; e elegibilidade EFSR. Seleciona ≤ 8 features (foco de erro, depois índice) e forma **componentes** por «mesmo grupo OU relatedness ≥ 0,35». **[exec, medido]** com relatedness 0,7124 entre **todas** as originais (iris 6/6 pares; wine 78/78; breast_cancer 435/435; digits 2016/2211 pares ≥ 0,35, todos entre originais), os componentes colapsam num só: **o conjunto de candidatos não é restringido pelos grupos** (teste sintético: partição verdadeira e permutada ⇒ um único componente).

**E10 Pontuação e aceitação** (`:1171-1345`; gate EFSR `:1006`). Pontuação = ganho normalizado × `w^k` + `α·OntoDepth` + `β·ErrorCoverage` (histórico) ou variantes B1. `w` = média geométrica dos pesos dos literais × **fator de coesão** `1 + 0,15·max(coesão_de_grupo, relatedness_média)`. **Medido:** `OntoDepth` = 1,0 para toda a feature ⇒ `α·1,0 = +0,35` **constante** em todos os candidatos (não altera a ordem; conta como «bónus semântico ≠ 0»); `relatedness_média` = 0,7124 para qualquer par de originais ⇒ fator ≥ 1,107 mesmo entre grupos diferentes, 1,15 no mesmo grupo; pares com agregado ⇒ relatedness 0 ⇒ só coesão de grupo. O gate EFSR aceita a proposta semântica só se a fidelidade local melhorar ≥ mínimo (região elegível). **Canais ontológicos efetivos até à pontuação:** pesos (agregados ×1,35), fator de coesão (≈ 4 % real vs permutada) e composição dos agregados.

**E11 Splits e árvore.** **[exec]** B29: ≈ 33,9 splits/árvore marcados «influenciados» (**96,8 %** — pela constante e pelos pesos ≠ 1, não por conteúdo) e **≈ 10,1 decisões efetivamente alteradas** (≈ 30 %); C29 (permutada) 9,7; D29 (aleatória) 9,6 (`m29_report.json::mechanism_aggregates`). A influência operacional é real e do mesmo tamanho nos controlos: **não identifica a ontologia**.

### 3.3 Capacidade implementada vs execução efetiva
| mecanismo | implementado | executado na M29 (DEV) | efeito específico da ontologia verdadeira |
|---|---|---|---|
| Reasoner (consistência) | sim | sim (4/4) | validação apenas |
| Reasoner (inferências) | sim | **0 inferências** | nenhum |
| Matching lexical | sim | cobertura 100 % | necessário, não discriminativo |
| Features hierárquicas (agregados) | sim | sim (2–3) | único canal com conteúdo ontológico (composição) |
| Features relacionais (anotações) | sim | só breast_cancer (50) | sem controlo válido; sem grupos/relatedness |
| Features de restrição (facetas) | sim | **0** (sem facetas nas TBoxes) | nenhum |
| Relatedness por tipo de relação | sim | constante 0,7124 | nenhum (Doc. 32) |
| Profundidade (`OntoDepth`) | sim | constante 1,0 | nenhum (offset) |
| Pesos de feature | sim | agregados ×1,35 | estrutural (independente do conteúdo) |
| Coesão de grupo | sim | sim | ≈ 4 % real vs permutada |
| Candidatos m-of-n guiados | sim | sim, ≤ 24/nó | não restringidos pelos grupos |
| Enriquecimento do MLP | sim | **não** (oráculo congelado) | — |
| Limites de domínio nas consultas | sim | **0 limites** declarados | nenhum |
| Disjunção, restrições existenciais/universais, cardinalidades, cadeias | não | — | ignorados |

### 3.4 Errata ao Doc. 30 (leitura; números inalterados)
1. §8 afirma «Atribuição ontológica em Breast Cancer: `NOT_ASSESSABLE`». No relatório, as 5 unidades de breast_cancer têm **`NOT_ATTRIBUTED` (sem ganho direto)**; seriam `NOT_ASSESSABLE` *se tivessem ganho direto* (permutado só em fallback; E29 confundido). Ver §2.4.
2. §6 diz que o rácio médio de wine (1,29) «⇒ essas unidades são confundidas». A regra pré-registada é assimétrica e por unidade: só 1 unidade (wine:123) fica `confundida` para G29 (T9); o nível de réplica está em §2.2.
3. A evidência `m29_campaign/main/raw/` não incluía os 18 `split.json` (agora incluídos).

## 4. Onde a informação semântica se perde (síntese)
1. **Matching → grafo:** só entidades casadas por nome; axiomas sobre entidades não casadas nem entram.
2. **Grafo → grupos:** só vizinhos diretos da propriedade; sem fecho transitivo, sem equivalência entre classes, sem subsunção inferida (testado).
3. **Grafo → relatedness:** a aresta `subsumption` para o nó genérico `DatatypeProperty` cria um *hub* que domina o caminho máx-produto (Doc. 32).
4. **Features → contexto:** agregados sem entidade ⇒ relatedness 0; relacionais sem grupo.
5. **Contexto → candidatos:** relatedness constante colapsa os componentes.
6. **Candidatos → pontuação:** `OntoDepth` constante; fator de coesão quase constante; pesos independentes do conteúdo.
7. **Pontuação → decisão:** ≈ 30 % das decisões mudam, igual nos controlos.
8. **Decisão → generalização:** M29: nenhum efeito diferencial.

## 5. Rastreabilidade (P4, estado atual)
A auditoria por split (`semantic_split_audit_`, `_annotate_semantic_split_row`, `:1132`) regista teste escolhido, entidades, grupos, `semantic_bonus`, `decision_changed`, razão textual e relatedness máxima. Cadeia exigida: `Axioma OWL → Relação extraída → Candidato m-of-n → Decisão → Efeito`.
| elo | disponível hoje | lacuna |
|---|---|---|
| Axioma → relação | `relationship_records()` (`:516`) lista arestas por tipo, **sem IRI do axioma** nem ligação à feature | falta identificador do axioma e do tipo (lógico vs anotação) |
| Relação → candidato | `semantic_groups`, relatedness máxima | não regista a **origem** do candidato (beam estatístico vs gerador semântico) |
| Candidato → decisão | `data_only_test`, `semantic_test`, `selected_test_after_error_gate`, gate EFSR | ok |
| Decisão → efeito | `decision_changed`, ganho de fidelidade local, bónus | sem efeito **na árvore final** por axioma |
A rastreabilidade demonstra influência operacional, não generalização. O Doc. 33 §5 propõe `semantic_trace_` (só relato).

## 6. Qualidade das ontologias DEV (P3)
Estados (definições operacionais; **não equivalentes**): `ONTOLOGY_VALID` = gate aceita; `ONTOLOGY_MAPPED` = cobertura ≥ 0,35 e sem ambiguidade nas aceites; `ONTOLOGY_EXPRESSIVE` = conhecimento relacional sobre atributos **além de uma partição plana** (hierarquia ≥ 3 níveis, restrições, facetas numéricas, objeto-propriedades, relações família×papel); `ONTOLOGY_EFFECTIVELY_USED` = influência operacional **dependente do conteúdo** na árvore; `ONTOLOGY_CAUSAL_GAIN_OBSERVED` = ganho conjunto atribuído com controlos válidos (Docs. 28/30).

| ontologia | classes | prop. dados | prop. objeto | hierarquia | disjunções | restrições / facetas | mapeadas (conf. mín.) | derivadas | grupos | riqueza (gate) | VALID | MAPPED | EXPRESSIVE | EFFECTIVELY_USED | CAUSAL_GAIN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iris | 3 | 4 | 0 | 2 níveis | 1 (`disjointWith`) | 0 / 0 | 4/4 (1,0) | 2 | 2 | poor | sim | sim | **não** | operacional, sem especificidade | **não** |
| wine | 4 | 13 | 0 | 2 | 1 (`AllDisjointClasses`) | 0 / 0 | 13/13 (1,0) | 3 | 3 | poor | sim | sim | **não** | idem | **não** |
| breast_cancer | 4 | 30 | 0 | 2 | 1 | 0 / 0; 60 usos de anotações `measurementFamily`/`statisticRole` | 30/30 (1,0) | 53 (3+50) | 3 | limited | sim | sim | **parcial** (anotações, não axiomas lógicos) | idem | **não avaliável** (sem controlo permutado válido) |
| digits | 4 | 64 | 0 | 2 | 1 | 0 / 0 | 64/64 (1,0; margem mín. 0,03) | 3 | 3 | poor | sim | sim | **não** | idem | **não** |

Axiomas «sem efeito observável»: `disjointWith`/`AllDisjointClasses` (nunca lidos), `rdfs:range` de tipos de dados (só serve para detetar inteiros/facetas: 0 aqui) e subclasses de topo para `owl:Thing`. Relações inferidas: **0**. Nenhum axioma foi acrescentado; qualquer enriquecimento futuro tem de representar conhecimento **verdadeiro, verificável e independente dos rótulos de teste** (Doc. 33 §7).

## 7. Conclusões da auditoria
1. As ontologias DEV são **válidas e bem mapeadas mas pobres** (partições planas): insuficientes para produzir ganho semântico, não por falha de execução.
2. O código **reconhece** menos do que «OWL» sugere (Doc. 32 §4): domínio, subclasse, equivalência (aresta), sub-propriedade, facetas; **ignora** disjunção, restrições e cardinalidades.
3. **Defeito medido de relatedness** (hub `DatatypeProperty`) torna a medida e os componentes de candidatos independentes da ontologia (Doc. 32).
4. A influência operacional (≈ 30 % de decisões alteradas) é real mas **não discrimina** verdadeiro de permutado; as métricas de «splits influenciados» estão inflacionadas por constantes.
5. O núcleo não foi alterado; as decisões ficam para o Doc. 33 e para aprovação.
