# Doc. 32 — Análise de relatedness e de axiomas (B3.0)

Estado: **ANÁLISE CONCLUÍDA; cálculo do núcleo NÃO alterado.** Testes e instrumentação em `tests/test_b30_semantic_audit.py` (15 testes), `validation/benchmark/b30_axiom_probe.py`, `b30_relatedness_alternatives.py`, `b30_identifiability.py`, `b30_ontology_audit.py`. Ontologias sintéticas OWL 2 (RDF/XML) controladas, sem nomes de datasets conhecidos. Evidência: `docs/evidence/b1b4/b30_audit/`.

## 1. Resumo
1. A relatedness constante **0,7124 = 0,88 · 0,92 · 0,88** (dois saltos `subsumption` com decaimento 0,92) é o valor do caminho entre **quaisquer** duas propriedades de dados **através do nó genérico `DatatypeProperty`** (classe built-in a que todas pertencem por `is_a`). O caminho via `rdfs:domain` (0,68 · 0,92 · 0,68 ≈ 0,425) nunca vence o máximo-produto.
2. A medida **não distingue** pares do mesmo conceito de pares de conceitos distintos, **nem responde** a alterar a partição, a retirar todos os `domain`, nem à profundidade da hierarquia (testes). Só reage a ligações **diretas** entre propriedades mapeadas (`subPropertyOf` → 0,88; `equivalentProperty` → 1,0).
3. Alternativas justificadas (Wu–Palmer, Lin) são discriminativas nos mesmos casos (implementadas só em `validation/`, não no núcleo).
4. Dos tipos de axioma OWL examinados, o sistema **usa** `domain`, `subClassOf`, `equivalentClass` (aresta), `subPropertyOf`, facetas de intervalo e (não lógicas) anotações família×papel; **ignora** `disjointWith`, restrições existenciais/universais, cardinalidades, inversas e cadeias de propriedades.
5. Num problema sintético em que a relação verdadeira é um m-of-n dentro de um grupo, o Original já a recupera (12/12): **o teste não consegue mostrar benefício da ontologia**, mas mostra com clareza onde a diferença verdadeiro×permutado se perde (§6).

## 2. P2 — Diagnóstico da relatedness (as 10 perguntas)
| # | Pergunta | Resposta (evidência) |
|---|---|---|
| 1 | Onde é calculado? | `OntologySemanticGraph.relatedness` (`core/ontology_semantic_graph.py:462-492`); matriz montada em `OwlSemanticProvider.build` (`core/benchmark/semantic.py:245-249`) só quando **ambas** as features têm entidade (`entity_for_feature`). |
| 2 | Que axiomas contribuem? | Arestas simétricas criadas em `from_ontology` (`:210-251`): `is_a`→`subsumption`, `equivalent_to`→`equivalent`, `domain`, `range`, `inverse_property`→`inverse`, e `ancestors()` do reasoner→`inferred_subsumption`. Pesos `RELATION_WEIGHTS` (`:93`): equivalent 1,00; subsumption 0,88; inverse 0,82; inferred 0,78; domain 0,68; range 0,68. |
| 3 | Há valor de fallback? | Sim, três: aresta sem tipo ⇒ 0,50 (`_edge_strength`, `:456-460`); sem caminho em ≤ 6 saltos ⇒ 0,0; feature sem entidade (todos os agregados) ⇒ 0,0. **0,7124 não é fallback**: é o valor do caminho-hub. |
| 4 | Saturação / normalização excessiva? | Não há normalização; o valor está limitado a [0,1] e decai 0,92 por salto após o primeiro. Não satura: **colapsa** para uma constante porque o caminho de maior produto é sempre o do hub. |
| 5 | Usa distâncias entre conceitos? | Usa «força de caminho» máx-produto no grafo não dirigido (inclui nós built-in). Não usa profundidade, antepassado comum mais específico (LCA), nem conteúdo de informação. |
| 6 | Usa hierarquias verdadeiras? | `parents` (dirigido, sem built-in) só serve a profundidade e a grupos; a relatedness usa `edges` (não dirigido, **com** `DatatypeProperty`/`Thing`/`Property`). A hierarquia de classes só entra se a feature estiver ligada a ela por caminho mais forte que o hub — nunca no caso de propriedades de dados. |
| 7 | Distingue relacionados de não relacionados? | **Não** para propriedades de dados: mesmo valor com `domain` comum, `domain` diferente ou **sem `domain`** (`no_domain`: 0,7124 para os 3 pares). |
| 8 | Responde a perturbações ontológicas? | Não: remover `domain`, trocar a partição, aprofundar a hierarquia ⇒ matriz idêntica (`test_current_relatedness_same_for_different_partitions`, `…_is_partition_blind`). |
| 9 | É sensível a mudanças semanticamente relevantes? | Só às que criam aresta **direta** entre as duas propriedades: `subPropertyOf` ⇒ 0,88; `equivalentProperty` ⇒ 1,0 (sonda). A partição por domínio, as restrições, a disjunção e as hierarquias de classes são invisíveis. |
| 10 | Valores diferentes quando a ontologia contém relações realmente diferentes? | Apenas nos casos de (9). Nas 4 ontologias DEV: valores fora da diagonal ∈ {0; 0,7124} (o 0 é sempre «feature sem entidade», i.e. os agregados). |

### 2.1 Decomposição matemática
Para duas propriedades `p, q` com `p ⊑ DatatypeProperty` e `q ⊑ DatatypeProperty` (aresta `subsumption`, peso 0,88): `strength(p→DatatypeProperty) = 0,88` (1.º salto sem decaimento); `→ q`: `0,88 · 0,92 · 0,88 = 0,712448`. Via domínio: `0,68 · 0,92 · 0,68 = 0,425408`. Via domínio + superclasse comum: `0,68·0,92·0,88·0,92·0,68 < 0,35`. O máximo é sempre o hub. Como o limiar de relação em `_ontology_guided_candidates` é 0,35 (`semantic_relation_threshold`), **todos** os pares de originais ficam relacionados.

### 2.2 Consequências no Reloaded (medidas, Doc. 31 §3)
- `_ontology_guided_candidates`: todos os pares de originais ≥ 0,35 ⇒ **um único componente** ⇒ candidatos não restringidos pelos grupos.
- `_semantic_group_coherence_factor`: `max(coesão_de_grupo, relatedness_média)` = 0,7124 para qualquer par de originais ⇒ bónus 1 + 0,15·0,7124 = **1,107** entre grupos diferentes; 1,15 no mesmo grupo. Diferença verdadeira×permutada ≈ **4 %** do peso.
- `_semantic_support_vector` / `_compute_semantic_structure_active`: a estrutura fica sempre «ativa» (`related` verdadeiro) mesmo para uma ontologia sem `domain`.

## 3. Alternativas matematicamente justificadas (diagnóstico; não implementadas no núcleo)
Só usam a taxonomia declarada (is-a e `domain`) **excluindo nós genéricos** (`Thing`, `DatatypeProperty`, `Property`):
- **Wu–Palmer:** `wup(a,b) = 2·d(lca)/(d(a)+d(b))`, `d(raiz)=1`. Simétrica, reflexiva, em [0,1], monótona na especificidade do antepassado comum.
- **Lin (1998):** `lin(a,b) = 2·IC(lca)/(IC(a)+IC(b))`, `IC(c) = −log p(c)`, `p(c)` = fração das features mapeadas sob `c`. Usa só a ontologia + o mapeamento (sem rótulos nem dados); um conceito que subsume tudo tem IC = 0 ⇒ relação 0 (sem informação).
- Relatedness entre features = máximo sobre os conceitos (domínio/superclasse direta) das duas features.

**Resultados (testes):** (i) mesmo conceito = 1,0 > conceitos irmãos (Wu–Palmer 0,5; Lin 0,0); (ii) partições distintas ⇒ matrizes distintas; (iii) hierarquia de 3 níveis: `wup` = 1,0 > 0,667 > 0,333 (a atual: constante); (iv) conceitos sem antepassado comum não genérico: 0. **DEV (4 ontologias):** pares do mesmo grupo 1,0; de grupos distintos 0,5 (Wu–Palmer) ou 0,0 (Lin) — mas como as TBoxes DEV são **partições planas**, qualquer medida discriminativa reduz-se ao indicador «mesmo grupo»: **não acrescenta conhecimento**, apenas remove o hub. O valor das alternativas só aparece com taxonomias mais profundas (Doc. 33).

## 4. P1 — Expressividade: suporte real por tipo de axioma
Sondas controladas (`axiom_cases()`; 1 axioma por vez; `docs/evidence/b1b4/b30_audit/axiom_probe.json`). «Candidatos» = alimenta o gerador m-of-n; «splits» = pode alterar a escolha. Mundo aberto: nenhuma ausência de afirmação é lida como negação (verificado: o código não implementa negação por falha em nenhum dos caminhos acima).

| axioma | reconhece (grafo) | reasoner usa | mapeia para features | produz candidatos | influencia splits | limitações |
|---|---|---|---|---|---|---|
| `rdfs:subClassOf` entre classes | sim (`parents`, `subsumption`) | sim (hierarquia materializada) | sim: pai direto de `domain`/`is_a` ⇒ agregado | só via grupos | via pesos/agregados | só pai **direto**; sem fecho transitivo |
| `owl:equivalentClass` | aresta `equivalent` (1,0) | sim | **não**: `f1∈A`, `f2∈A2` com `A≡A2` ficam em grupos distintos; sem agregado | não | não | equivalência entre classes não funde grupos |
| `owl:disjointWith` / `AllDisjointClasses` | **não** (idêntico ao caso sem axioma) | consistência/insatisfazibilidade | não | não | não | nunca lido; poderia dar incompatibilidades |
| `rdfs:domain` | sim (aresta `domain` 0,68; `domains`) | — | **sim** (único sinal de partição; agregados e grupos) | via grupos | via agregados/pesos | sem `domain` ⇒ grupos singulares e nenhuma derivada |
| `rdfs:range` | sim (built-in; deteta inteiros e facetas) | — | facetas ⇒ features de restrição | não | não | tipos de dados: sem relações |
| `owl:someValuesFrom` | **não** (idêntico ao caso base) | sim (HermiT classifica) | não | não | não | só através da inferência de subclasse (que não propaga) |
| `owl:allValuesFrom` | **não** | idem | não | não | não | idem |
| cardinalidades OWL | **não** | idem | não | não | não | idem |
| propriedades de objetos / inversas | nós `object_property`; `inverseOf` **não produziu aresta `inverse`** na sonda (grafo idêntico ao caso base), apesar de `inverse_property` constar da lista de atributos lidos (`ontology_semantic_graph.py:210-216`) | sim | não (atributos tabulares são propriedades de dados) | não | não | sem tradução para colunas; leitura de `inverse_property` não observada |
| cadeias de propriedades | **não** | sim | não | não | não | — |
| propriedades de dados (`DatatypeProperty`) | sim | — | sim (matching) | — | — | só nomes/rótulos |
| `rdfs:subPropertyOf` (dados) | sim (`parents` da propriedade; aresta `subsumption` direta ⇒ rel. 0,88) | sim | sim: acrescenta o grupo `fAll` à lista; o contexto usa `gl[-1]` (ordem alfabética) | via grupo | via grupo | escolha do grupo «último» é arbitrária quando há vários |
| `owl:equivalentProperty` | sim (aresta `equivalent` ⇒ rel. 1,0) | sim | não funde features | não | pouco | — |
| facetas `xsd:min/maxInclusive` (ConstrainedDatatype em `range`) | sim (`entity_bounds`) | — | **sim** (`onto_f_within_min/max`) e máscara de validade das consultas | não | indireto | só com facetas explícitas; DEV: **0** |
| hierarquia de conceitos (≥ 3 níveis) | sim (`parents`, profundidade) | sim | agregados só do pai **direto** | não | `OntoDepth` | relatedness cega à profundidade (§2) |
| subsunção **inferida** pelo reasoner | arestas `inferred_subsumption` (0,78) | **sim** (testado: HermiT infere `C ⊑ A`) | **não**: `f1∈A` e `f2∈C` continuam em grupos distintos | não | não | informação inferida perde-se no passo de grupos |
| anotações `measurementFamily`/`statisticRole` | não são axiomas (lidas pelo processador) | — | **sim** (50 relacionais em breast_cancer) | não (sem grupo) | via pesos das derivadas | conhecimento não lógico; sem semântica OWL formal |
| restrições semânticas aplicáveis a atributos tabulares | **só** facetas numéricas e partição por domínio | — | — | — | — | restantes (incompatibilidades, definições) não existem no sistema |

**Princípio respeitado:** nenhuma tradução OWL→regra tabular foi acrescentada. Os axiomas ignorados **não** foram interpretados nem por negação por ausência nem por inferência de regras de domínio.

## 5. Matching e reasoner (testes)
- **Matching** (`test_matching_is_lexical_not_semantic`): igualdade/rótulo/tokens ⇒ aceite (score 1,0); «hasTemperature» ⇒ `REJECTED_AMBIGUOUS`; «temp_mean», «temperature_average», «lumens», «x1» ⇒ não mapeados. Segurança: nenhuma aceitação falsa nos casos de teste; **risco de falsos negativos** (sinónimos/abreviaturas) e de falsos positivos por subcadeia (score 0,86 aceitável com margem ≥ 0,08, p. ex. «brightness» → `hasBrightnessLevel`). `test_matching_rejects_placeholder_ontology`: nomes `f1/Group1` ⇒ ontologia genérica rejeitada.
- **Reasoner** (`test_reasoner_infers_subclass_but_groups_ignore_it`, Java/HermiT): `A ≡ ∃hasPart.B`, `C ⊑ ∃hasPart.B` ⇒ o reasoner infere `C ⊑ A` (não declarado); as features `f1∈A`, `f2∈C` **continuam** em grupos distintos e sem agregado — **a inferência verificada não chega ao Reloaded**.

## 6. P4 — Identificabilidade (síntese sintética, `b30_identifiability.py`)
Problema: 8 atributos (3 grupos: `Thermal*`, `Optical*`, `Acoustic*`); oráculo de forma fechada = **2-de-3 sobre os atributos do grupo 1**; 12 sementes; mesmos dados, oráculo e orçamento (`min_sample` 600, `max_nodes` 15, `max_n` 3). Braços: Original; `OWL_true`/`OWL_wrong` (caminho OWL **completo** com partição verdadeira/permutada); `GRP_true`/`GRP_wrong` (grupos explícitos e relatedness **discriminativa** 1/0 — contrafactual de diagnóstico).

| braço | n | fidelidade média ao oráculo | regra verdadeira recuperada | raiz dentro do grupo verdadeiro | decisões alteradas (média) | candidatos semânticos (média) | nós internos (média) |
|---|---|---|---|---|---|---|---|
| ORIG | 12 | 0.9817 | 12/12 | 12/12 | 0.00 | 0 | 1.08 |
| OWL_true | 12 | 0.9877 | 12/12 | 12/12 | 0.67 | 58 | 2.33 |
| OWL_wrong | 12 | 0.9889 | 12/12 | 12/12 | 0.33 | 48 | 1.42 |
| GRP_true | 12 | 0.9879 | 12/12 | 12/12 | 0.75 | 58 | 2.25 |
| GRP_wrong | 12 | 0.9909 | 9/12 | 11/12 | 0.50 | 64 | 2.75 |

Divergência verdadeiro×permutado por semente: OWL — árvores idênticas 0/12, mesma raiz 12/12; GRP — árvores idênticas 0/12, mesma raiz 9/12.

**Estrutura (semente 0):** componentes semânticos sobre as 8 originais — `OWL_true`: [0..7] (1 componente); `OWL_wrong`: [0..7] (1 componente); `GRP_true`: {0,1,2},{3,4,5},{6,7}; `GRP_wrong`: {0,3,6},{1,4,7},{2,5}. Relatedness entre originais: OWL ⇒ {0,7124}; GRP ⇒ {0; 1}.

**Leitura honesta:**
1. O Original recupera a regra em 12/12 e a fidelidade é ≈ 0,98: o problema é **fácil demais** para mostrar benefício da ontologia (não se afirma ganho nem ausência de ganho).
2. Onde a diferença verdadeiro×permutado **se perde** no caminho OWL: (a) *estrutura:* os **grupos** diferem (assinaturas distintas) — ok; (b) *relatedness:* **idêntica** (0,7124) — perde-se; (c) *componentes de candidatos:* **idênticos** (um só) — perde-se; (d) *pontuação:* só pesos/coesão (≈ 4 %) distinguem; (e) *árvores:* diferem em 12/12 sementes (`identical_trees_true_vs_wrong = 0/12`) mas por efeitos de composição dos agregados e dos pesos, com raiz igual em 12/12.
3. Com relatedness discriminativa (GRP), a restrição de candidatos aos grupos **segue a ontologia**: com a partição **errada** a regra verdadeira é recuperada só em 9/12 (vs 12/12 com a verdadeira e com o Original) — **uma restrição baseada em relações falsas prejudica**; é o risco que o Doc. 33 trata (restrição lógica vs preferência).
4. A rastreabilidade (Doc. 31 §5) liga split → grupos/entidades/bónus, mas não ao axioma de origem.

## 7. O que NÃO foi feito
Não se alterou `relatedness`, `RELATION_WEIGHTS`, grupos, candidatos, pesos nem o gate; não se acrescentaram axiomas a nenhuma ontologia; não se executou campanha DEV; não se usou teste externo. As alternativas de §3 são diagnósticos em `validation/`.
