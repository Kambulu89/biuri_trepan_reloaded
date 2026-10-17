# Doc. 51 — Contrato axioma OWL → mecanismo de construção de candidatos m-of-n

Estado: **contrato e protótipo de validação** (`validation/benchmark/h2_contract.py`, `h2_axiom_mechanisms.py`, `h2_german_proposal.py`; testes `tests/test_h2_contract_and_mechanisms.py`). **Nenhum mecanismo foi integrado em `core/`**; `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`. Nada aqui é benchmark de desempenho: os testes verificam **correção semântica** (extensões exatas, aninhamento, dependência do reasoner, controlos, contrafactual, mundo aberto).

## 1. Princípios do contrato
1. **Relação OWL ≠ correlação estatística.** Um axioma exprime uma afirmação lógica sobre conceitos/propriedades; só se torna facto utilizável depois de (a) o reasoner a validar/inferir, (b) o matching mapear entidades a atributos ARFF, (c) o registo a tipar (`SemanticFact`), (d) um mecanismo declarado a consumir.
2. **Mundo aberto.** Ausência de afirmação ≠ falsidade. `disjointWith`, domínios, *ranges* e cardinalidades **não** se convertem em validações fechadas de linhas (nem em filtros de treino); disjunções só podam candidatos logicamente insatisfazíveis dada a semântica da coluna.
3. **Restrição lógica vs preferência.** `HARD_CONSTRAINT` só para axiomas `UNIVERSAL` com as três revisões humanas aprovadas e só na **geração de candidatos/consultas**, nunca sobre linhas de treino. Tudo o resto é `PREFERENCE` ou enriquecimento de representação.
4. **Regularidade empírica ≠ axioma.** Relações observadas nos dados (ex.: razões de WDBC, `flavanoids ≤ total_phenols`) não entram como axiomas universais sem justificação independente (Doc. 39).
5. **Os literais derivados não alargam o espaço de hipóteses do Original.** `OR(colunas)` é exatamente um teste `1-de-n` do m-of-n; o mecanismo **orienta e prioriza** a pesquisa (candidatos pré-montados), não acrescenta expressividade ao classificador. Por isso as comparações exigem **candidatos equiparados** (Doc. 52).

## 2. Cadeia obrigatória e estados
`OWL → Reasoner → Fact Registry → Matching → Candidato m-of-n → Decisão de split → Árvore`, com os estados `AVAILABLE → VALIDATED → MAPPED → COMPILED → USED → DECISION_INFLUENCED → BENEFIT_OBSERVED` (Doc. 45 Q6), nunca inferidos do nome do braço. Evidência observacional por estado:
* `COMPILED`: facto com `permitted_use ≠ NONE` no registo; `USED`: o literal derivado **entrou** no conjunto de candidatos gerados (registado em `semantic_trace_`);
* `DECISION_INFLUENCED`: o literal derivado foi o escolhido num split **e** o contrafactual (ablação do axioma = classe passa a primitiva, mesma ontologia) altera a escolha nesse nó — o contrafactual atribui a mudança a **um** axioma específico; contrastes com controlos permutado/aleatório separam «efeito do mecanismo» de «efeito do conhecimento»;
* `BENEFIT_OBSERVED`: só com ganho conjunto atribuído (Doc. 52).
A influência **não implica** melhoria de accuracy ou fidelidade.

## 3. Contrato por tipo de axioma (10 pontos + regra de mundo aberto + estado)
Os 10 pontos: (1) significado de domínio; (2) representação OWL; (3) o que o reasoner infere; (4) como chega aos atributos ARFF; (5) facto semântico tipado; (6) influência nos candidatos m-of-n; (7) natureza (restrição lógica / preferência / enriquecimento); (8) medida de influência efetiva; (9) controlo que neutraliza; (10) riscos (leakage, erro lógico, custo).

### T1_DEFINED_CLASS_UNION — estado: `IMPLEMENTED_PROTOTYPE_VALIDATION`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | uma categoria definida como união de níveis (p. ex. «poupança ≥ 500») ou um agrupamento taxonómico (p. ex. «veículo» = carro novo ∪ usado) |
| 2 | Representação OWL | classe definida `C ≡ L1 ⊔ … ⊔ Lk`, níveis `Li ⊑ Variável` |
| 3 | O que o reasoner infere | subsunção `Li ⊑ C` e, entre classes definidas, `C_k ⊑ C_j` quando o conjunto de níveis de C_k ⊆ C_j (inferência de HermiT, não declarada) |
| 4 | Caminho até aos atributos ARFF | atributo ARFF (coluna 0/1 de um nível) → entidade (propriedade de dados ou classe de nível) pelo matching P3 → conceitos que a subsumem (`ATTRIBUTE_APPLIES_TO_CONCEPT`, derivação ASSERTED ou INFERRED_BY_REASONER) |
| 5 | Facto semântico tipado | ATTRIBUTE_APPLIES_TO_CONCEPT (+ CONCEPT_SUBSUMPTION); extensão do conceito = conjunto de atributos |
| 6 | Influência nos candidatos m-of-n | gera um LITERAL derivado binário `OR(colunas da extensão)` que passa a competir como candidato simples e como literal de m-of-n |
| 7 | Natureza | REPRESENTATION_ENRICHMENT |
| 8 | Medida da influência efetiva | (a) o literal derivado entra no conjunto de candidatos gerados; (b) é escolhido num split; (c) contrafactual: ablação do axioma (a classe passa a primitiva) remove exatamente esse literal e regista a mudança de decisão |
| 9 | Controlo que neutraliza | O3 neutralizada (classe primitiva, mesmo nome, sem definição); H permutada (ordem trocada); I união aleatória de tamanho igual; J limiares aprendidos dos dados |
| 10 | Riscos (leakage / erro lógico / custo) | literal redundante com colunas existentes quando a extensão tem 1 elemento; tautologia quando cobre todos os níveis; custo de candidatos; enviesamento se a taxonomia refletir o rótulo |
| — | Regra de mundo aberto | mundo aberto: ausência de nível ≠ falso; o literal só se cria para colunas binárias mapeadas |

### T2_ORDINAL_ORDER — estado: `IMPLEMENTED_PROTOTYPE_VALIDATION`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | ordem entre níveis de uma variável ordinal documentada (poupança <100 < 100–500 < 500–1000 < ≥1000) |
| 2 | Representação OWL | cadeia de classes cumulativas `AtLeast_k ≡ L_k ⊔ … ⊔ L_m` (OWL 2 DL; sem propriedades transitivas sobre indivíduos) |
| 3 | O que o reasoner infere | `AtLeast_{k+1} ⊑ AtLeast_k` e `Li ⊑ AtLeast_k` para i ≥ k, todos INFERIDOS |
| 4 | Caminho até aos atributos ARFF | idêntico a T1 (as classes cumulativas são classes definidas) |
| 5 | Facto semântico tipado | ATTRIBUTE_APPLIES_TO_CONCEPT + CONCEPT_SUBSUMPTION inferidas |
| 6 | Influência nos candidatos m-of-n | literais de limiar cumulativo `nível ≥ k` (aninhados); permite regras m-of-n monótonas sobre variáveis ordinais one-hot |
| 7 | Natureza | REPRESENTATION_ENRICHMENT |
| 8 | Medida da influência efetiva | como T1 e ainda: a ordem do axioma deve ser a ordem do literal escolhido |
| 9 | Controlo que neutraliza | H = ordem permutada (OWL válido, semântica errada); O3 = classes cumulativas primitivas; J = ordenação dos níveis pela taxa de classe no TREINO (engenharia convencional) |
| 10 | Riscos (leakage / erro lógico / custo) | níveis não ordinais (desconhecido, «sem conta») têm de ficar de fora (erro lógico se incluídos); ordem pode ser convenção da codificação, não do domínio; o rótulo pode correlacionar com a ordem (fuga se a ordem se escolher pelos dados) |
| — | Regra de mundo aberto | ordem declarada só vale para níveis listados |

### T3_DISJOINTNESS — estado: `DESIGN_ONLY`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | níveis de uma variável categórica são mutuamente exclusivos |
| 2 | Representação OWL | `AllDisjointClasses(L1…Lk)` |
| 3 | O que o reasoner infere | consistência/insatisfazibilidade; instâncias em duas classes tornam a ontologia inconsistente |
| 4 | Caminho até aos atributos ARFF | mapeamento coluna→classe de nível; só válido se a coluna for o indicador de pertença |
| 5 | Facto semântico tipado | CONCEPT_DISJOINTNESS (permitted_use NONE hoje) |
| 6 | Influência nos candidatos m-of-n | poda de m-of-n com m ≥ 2 sobre literais mutuamente disjuntos (insatisfazível) |
| 7 | Natureza | LOGICAL_CONSTRAINT |
| 8 | Medida da influência efetiva | n.º de candidatos podados; contrafactual: sem o axioma, o candidato existe |
| 9 | Controlo que neutraliza | O3 sem disjunção |
| 10 | Riscos (leakage / erro lógico / custo) | redundante com a codificação one-hot (RECOVERABLE_FROM_DATA/encoding): não é conhecimento externo; nunca usar para validar ou descartar linhas |
| — | Regra de mundo aberto | NÃO converter em validação fechada de linhas; só poda candidatos logicamente insatisfazíveis dada a semântica da coluna |

### T4_VALUE_BOUND — estado: `DESIGN_ONLY`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | limite de domínio de um atributo (p. ex. idade ≥ 18 na população do estudo; contagens 0–16) |
| 2 | Representação OWL | restrição de tipo de dados com facetas (`xsd:minInclusive`/`maxInclusive`) numa `DatatypeProperty` |
| 3 | O que o reasoner infere | nenhuma inferência nova |
| 4 | Caminho até aos atributos ARFF | propriedade de dados mapeada ao atributo |
| 5 | Facto semântico tipado | VALUE_BOUND |
| 6 | Influência nos candidatos m-of-n | restringe limiares candidatos e a plausibilidade das consultas ao oráculo (amostragem) ao domínio válido |
| 7 | Natureza | LOGICAL_CONSTRAINT (só em geração de consultas/limiares, nunca em linhas de treino) |
| 8 | Medida da influência efetiva | fração de consultas ou limiares fora do domínio antes/depois; contrafactual: sem o limite |
| 9 | Controlo que neutraliza | O3 sem facetas; H limites trocados entre atributos (falsos: só como controlo negativo explícito, não como restrição dura) |
| 10 | Riscos (leakage / erro lógico / custo) | limite como convenção do dataset, não do domínio; risco de recorte de valores legítimos; mundo aberto |
| — | Regra de mundo aberto | limite vale no âmbito declarado; não se aplica a linhas observadas |

### T5_ORDER_BETWEEN_ATTRIBUTES — estado: `DESIGN_ONLY`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | relação de ordem definicional entre medidas (p. ex. «pior» ≥ «média» por definição) |
| 2 | Representação OWL | axioma de domínio sobre propriedades de dados; em OWL 2 DL só como anotação formal ou regra externa (SWRL não-DL) — ver limitação |
| 3 | O que o reasoner infere | nenhuma em OWL 2 DL puro |
| 4 | Caminho até aos atributos ARFF | pares de atributos mapeados |
| 5 | Facto semântico tipado | ORDER_RELATION (REGULARITY_ONLY até revisão) |
| 6 | Influência nos candidatos m-of-n | filtra consultas ao oráculo que violam a ordem e pode gerar literais de diferença |
| 7 | Natureza | LOGICAL_CONSTRAINT (consultas) / HEURISTIC_PREFERENCE |
| 8 | Medida da influência efetiva | violações de consultas antes/depois; contrafactual: sem o filtro |
| 9 | Controlo que neutraliza | O3 sem a relação; H ordem entre pares errados só como controlo negativo declarado |
| 10 | Riscos (leakage / erro lógico / custo) | regularidade empírica ≠ axioma universal (Doc. 39: razões não valem para agregados); OWL 2 DL não exprime ordem entre valores de propriedades de dados de forma nativa |
| — | Regra de mundo aberto | só com fonte e revisão; nunca derivada da correlação |

### T6_SUBPROPERTY_EQUIVALENCE — estado: `DESIGN_ONLY`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | atributos que medem a mesma grandeza (unidades diferentes) ou especialização de uma medida |
| 2 | Representação OWL | `SubDataPropertyOf`, `EquivalentDataProperties` |
| 3 | O que o reasoner infere | propriedades equivalentes colapsam; subpropriedade herda domínio/âmbito |
| 4 | Caminho até aos atributos ARFF | atributos mapeados a propriedades equivalentes |
| 5 | Facto semântico tipado | PROPERTY_EQUIVALENCE / PROPERTY_SUBSUMPTION |
| 6 | Influência nos candidatos m-of-n | de-duplicação de literais redundantes (poupa candidatos) e agrupamento de literais intermutáveis |
| 7 | Natureza | HEURISTIC_PREFERENCE |
| 8 | Medida da influência efetiva | candidatos removidos por redundância; contrafactual |
| 9 | Controlo que neutraliza | O3 sem equivalência; H equivalências trocadas |
| 10 | Riscos (leakage / erro lógico / custo) | colapso errado se a equivalência for aproximada; perda de informação |
| — | Regra de mundo aberto | equivalência só se declarada |

### T7_OBJECT_PROPERTY_ROLE — estado: `DESIGN_ONLY`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | papéis entre conceitos (mede, parte-de, causa) |
| 2 | Representação OWL | `ObjectProperty` (+ domínio/âmbito, caraterísticas) |
| 3 | O que o reasoner infere | propagação por propriedades transitivas/cadeias |
| 4 | Caminho até aos atributos ARFF | indireto: atributo→conceito→relação |
| 5 | Facto semântico tipado | (sem facto tipado ainda) |
| 6 | Influência nos candidatos m-of-n | ponderação de relatedness e expansão de unidades de pesquisa |
| 7 | Natureza | HEURISTIC_PREFERENCE |
| 8 | Medida da influência efetiva | alteração da ordenação das unidades; contrafactual |
| 9 | Controlo que neutraliza | O3 sem a propriedade; I relações aleatórias de grau igual |
| 10 | Riscos (leakage / erro lógico / custo) | relatedness constante hoje (hub `DatatypeProperty`); risco de confundir relação OWL com correlação |
| — | Regra de mundo aberto | mundo aberto |

### T8_CARDINALITY — estado: `DESIGN_ONLY`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | contagens admissíveis (exatamente um nível por variável) |
| 2 | Representação OWL | `ExactCardinality`/`FunctionalDataProperty` |
| 3 | O que o reasoner infere | inconsistência em violação |
| 4 | Caminho até aos atributos ARFF | indireto |
| 5 | Facto semântico tipado | (não suportado) |
| 6 | Influência nos candidatos m-of-n | limite para `n` de m-of-n sobre literais da mesma variável |
| 7 | Natureza | LOGICAL_CONSTRAINT |
| 8 | Medida da influência efetiva | candidatos com n excessivo podados |
| 9 | Controlo que neutraliza | O3 sem cardinalidade |
| 10 | Riscos (leakage / erro lógico / custo) | redundante com one-hot; semântica OWL de cardinalidade ≠ validação de linhas |
| — | Regra de mundo aberto | NÃO converter em validação de linhas |

### T9_SPATIAL_ADJACENCY — estado: `DESIGN_ONLY`
| # | Ponto | Conteúdo |
|---|---|---|
| 1 | Significado de domínio | vizinhança na grelha de pixels (tipo de facto novo) |
| 2 | Representação OWL | relação de objetos simétrica entre atributos-como-entidades |
| 3 | O que o reasoner infere | fecho de vizinhança |
| 4 | Caminho até aos atributos ARFF | atributo→entidade-pixel→vizinhos |
| 5 | Facto semântico tipado | SPATIAL_ADJACENCY (inexistente) |
| 6 | Influência nos candidatos m-of-n | unidades de pesquisa locais (m-of-n sobre vizinhança) |
| 7 | Natureza | HEURISTIC_PREFERENCE |
| 8 | Medida da influência efetiva | unidades geradas/escolhidas; contrafactual |
| 9 | Controlo que neutraliza | H adjacência permutada; I grafo aleatório de grau igual |
| 10 | Riscos (leakage / erro lógico / custo) | adjacência recuperável por correlação com n suficiente |
| — | Regra de mundo aberto | mundo aberto |

## 4. Protótipo T1/T2 e verificação de correção semântica (sem benchmark)
`derived_union_literals(registo, X, nomes)`: para cada conceito cuja **extensão** (atributos que a ele se aplicam, por asserção, fecho ou **inferência de HermiT**) tem ≥ 2 atributos binários da **mesma variável**, cria o literal `OR(colunas)`; descarta extensões constantes no treino e duplicados. `apply_literals` acrescenta as colunas sem remover linhas. Controlos: `control_neutralize` (ablação por axioma), `control_random_unions` (tamanhos iguais, mesma variável) e, ao nível do OWL, O3/H/I.
**Verificado em testes (7, todos a passar):** extensão exata e aninhada das classes cumulativas; **sem reasoner não há literais** (dependência do reasoner demonstrada); classes primitivas (O3) não geram nada; ordem permutada (H) dá o mesmo perfil de tamanhos com conteúdo diferente; o contrafactual remove exatamente o literal do axioma ablacionado; a disjunção fica `permitted_use = NONE` e **nenhuma linha é removida**; fichas de axioma bloqueiam validação por IA, derivação a partir de dados, aprovações sem nome/data/assinatura e `HARD_CONSTRAINT` sem as três revisões.
**Proposta German Credit (`proposals/german_credit/v0_draft`, rascunho de IA, DEV exploratório):** O1 plana (VALID, 61/61 mapeados, 0 inferências); O2 expressiva (VALID, 61/61 mapeados, **55 factos inferidos pelo HermiT**, **10 literais derivados**, todos dependentes do reasoner: tamanhos [2, 2, 2, 2, 2, 3, 3, 3, 3, 4]); O3 (VALID, 0 inferências, 0 literais); H (55 inferências, 10 literais com conteúdo diferente); I (58 inferências). Evidência: `docs/evidence/b1b4/b3q/h2_german_proposal_check.json`.

## 5. O que falta (dependências)
* Ligar `derived_union_literals` ao arnês de pesquisa (`SemanticSteeredTrepan`/`CountSteered`) em `validation/` com rastreio `semantic_trace_` por literal e ablação por axioma — **não feito**.
* T3–T9 só têm contrato; T5 (ordem entre atributos) esbarra na expressividade de OWL 2 DL (não exprime ordem entre valores de propriedades de dados de forma nativa) e T4 exige um facto `VALUE_BOUND` utilizável pela amostragem de consultas.
* Revisão humana dos axiomas (todos `*_REVIEW_PENDING`) e verificação das fontes (citadas de memória; sem acesso de rede).
* Verificador formal OWL 2 DL (hoje só proxy).
