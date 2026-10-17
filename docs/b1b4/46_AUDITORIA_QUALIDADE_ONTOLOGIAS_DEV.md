# Doc. 46 — Auditoria da qualidade das ontologias DEV (iris, wine, breast_cancer, digits)

Aplicação inicial do protocolo B3-Q (Doc. 45). **Só leitura**: OWL congelados (inalterados), resultados M29/`sampler_c` já existentes e dados de **treino** (holdout seed 42 reconstruído por `make_splits` e verificado contra o hash gravado em `split.json` das quatro ontologias: `split_hash_ok = true`). Nenhuma execução de benchmark, nenhum teste usado, nenhum axioma acrescentado, nenhuma ontologia ou resultado alterado.
Código e evidência: `validation/benchmark/b3q_audit.py` → `docs/evidence/b1b4/b3q/audit_dev_ontologies.json`; `b3q_partition_recoverability.py` → `partition_recoverability.json` (Doc. 48). Complementa a auditoria B3.0 (Docs. 31–33).

## 1. Resposta resumida às dez perguntas

| # | Pergunta | iris | wine | breast_cancer | digits |
|---|---|---|---|---|---|
| — | Asserções/axiomas no ficheiro (todas as funções) | 28 | 67 | 197 | 271 |
| 1 | Axiomas **de domínio substantivos** (equivalência, subpropriedade, restrição, cardinalidade, objeto, cadeia, inversa, facetas de valor) | **0** | **0** | **0** | **0** |
| 2 | Predominantemente **organizacionais** (declarações, anotações, tipagem `xsd:decimal`, raiz `owl:Thing`, taxonomia de 2.º nível, disjunção) | 24 | 54 | 167 | 207 |
|   | — e **agrupamento atributo→conceito** (`rdfs:domain`; única informação a nível de atributo) | 4 | 13 | 30 | 64 |
|   | — anotações **personalizadas** (invisíveis ao reasoner e ao núcleo) | 0 | 0 | **60** | 0 |
| 3 | **Inferidos** pelo reasoner (HermiT: pares de subclasse inferidos / consistente) | 0 / sim | 0 / sim | 0 / sim | 0 / sim |
| 4 | Relações **discriminativas** (sinergia por pares dentro vs fora do grupo, treino; valor-p de permutação) | p = 1,00 (só 2 pares intra-grupo; sem poder) | p = 0,85 | p = 0,85 | p = 0,23 |
| 5 | Chegam à construção m-of-n (núcleo, ontologia verdadeira: agregados semânticos usados / gerados) | 2 / 2 | 2,9 / 3 | 6,7 / 53 | 0 / 3 |
| 6 | **Alteram decisões** (n.º de decisões alteradas por árvore; ontologia verdadeira vs controlos) | 10,5 vs 10,4–10,7 | 4,2 vs 4,1–5,5 | 5,4 vs 5,0–5,5 | 26,8 vs 22,0–25,2 |
| 7–10 | Lacunas | ver §4–§6 | | | |
Factos compilados (registo B3.1): 11 / 32 / 66 / 134, utilizáveis 10 / 29 / 63 / 131; **todos** `LOGICALLY_ENTAILED`, **0** `HARD_CONSTRAINT`, todos `PREFERENCE` (ou `NONE` para disjunções). A diferença entre as 4 asserções de agrupamento de iris e os 8 factos `ATTRIBUTE_APPLIES_TO_CONCEPT` deve-se à herança pelo conceito-pai (`DomainMeasurement`).

## 2. Censo por função (reproduz-se com `b3q_audit.census`)
| Função | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| Anotação (`rdfs:label`/`comment`) | 9 | 19 | 36 | 70 |
| Declaração (classes, propriedades) | 7 | 17 | 36 | 68 |
| Agrupamento atributo→conceito (`rdfs:domain`) | 4 | 13 | 30 | 64 |
| Tipagem (`rdfs:range`, todos `xsd:decimal`, **sem facetas**) | 4 | 13 | 30 | 64 |
| Raiz (`subClassOf owl:Thing`) | 1 | 1 | 1 | 1 |
| Taxonomia (conceito ⊑ conceito) | 2 | 3 | 3 | 3 |
| Disjunção | 1 | 1 | 1 | 1 |
| Anotação personalizada (`measurementFamily`, `statisticRole`) | 0 | 0 | 60 | 0 |
| **Substantivo-relacional** | **0** | **0** | **0** | **0** |
Grupos: iris {Sepal 2, Petal 2}; wine {Acidity 3, Phenolic 5, PhysicalChemical 5}; breast_cancer {Mean 10, Error 10, Worst 10}; digits {Upper 24, Middle 24, Lower 16}.
**Nenhuma** das quatro tem object properties, restrições, cardinalidades, equivalências, subpropriedades, cadeias, inversas, limites de valor nem tipos inteiros.

## 3. O que cada partição significa (Q2/Q3, para revisão humana)
* **iris** — sépala vs pétala: distinção anatómica genuína (nível orgão), mas só dois grupos de dois atributos: quase nada para orientar uma pesquisa (6 pares, 2 intra-grupo).
* **wine** — `AcidityProfile`, `PhenolicProfile`, `PhysicalChemicalProfile`: agrupamento funcional por «tipo químico» construído a partir da descrição das *features*. Os critérios de pertença não estão documentados na ontologia; a pertença de alguns atributos (p. ex. `ash`, `alcalinity_of_ash` em «Acidity»; `magnesium`, `proline`, `hue`, `color_intensity`, `alcohol` em «PhysicalChemical») **exige revisão de um especialista de domínio**. Não se afirma erro; afirma-se ausência de justificação verificável.
* **breast_cancer** — agrupa por **estatística** (mean / error / worst), uma propriedade da construção das *features* do WDBC, não por conceito físico. O conhecimento de domínio mais útil (qual a **grandeza medida**: raio, textura, perímetro…, e a **função estatística**) está nas 60 anotações personalizadas `measurementFamily`/`statisticRole`, que **nem o reasoner nem o núcleo leem** (§5).
* **digits** — três faixas de linhas da grelha 8×8 (Upper/Middle/Lower): partição **geométrica arbitrária**, sem significado de domínio que a distinga de qualquer outra partição em faixas; sem a topologia espacial (vizinhança) nem os limites 0–16.

## 4. Utilização efetiva (Q6) e o problema dos controlos
Estados máximos atingidos por tipo de relação (agrupamento atributo→conceito) na pipeline atual:
`AVAILABLE ✔ → VALIDATED ✔ (HermiT consistente, gate válido) → MAPPED ✔ (cobertura 100 %, 0 ambiguidades, 0 colisões) → COMPILED ✔ (factos utilizáveis acima) → USED ✔ em iris, wine e breast_cancer (agregados semânticos consumidos); em digits 0 agregados usados, embora os splits contem como «influenciados» por outra via (pontuação semântica) — ambiguidade do indicador → DECISION_INFLUENCED ✘ (não demonstrável) → BENEFIT_OBSERVED ✘`.
`DECISION_INFLUENCED` **não é demonstrável com os indicadores atuais**: a tabela mostra, por conjunto de dados, médias por árvore com a ontologia verdadeira (`sampler_c` `sc_D`) e com os controlos aleatório e permutado (M29 `D`, `P`):

| Conjunto | Braço | Splits influenciados | Nós internos | Decisões alteradas | Agregados usados |
|---|---|---|---|---|---|
| iris | verdadeira / permutada / aleatória | 17,6 / 17,0 / 17,6 | 17,3 / 17,1 / 16,8 | 10,5 / 10,7 / 10,4 | 2,0 / 2,0 / 2,0 |
| wine | verdadeira / permutada / aleatória | 9,7 / 11,1 / 11,7 | 12,7 / 15,8 / 17,2 | 4,2 / 4,1 / 5,5 | 2,9 / 2,4 / 2,6 |
| breast_cancer | verdadeira / permutada / aleatória | 15,5 / 15,1 / 14,9 | 16,4 / 16,9 / 16,9 | 5,4 / 5,0 / 5,5 | 6,7 / 7,3 / 7,0 |
| digits | verdadeira / permutada / aleatória | 125,7 / 125,7 / 125,7 | 126,0 / 126,1 / 125,7 | 26,8 / 25,2 / 22,0 | 0 / 0 / 0 |
Duas ressalvas: os dados vêm de campanhas diferentes (leitura descritiva, sem teste); `ontology_influenced_splits` pode exceder o n.º de nós internos (contador não limitado por nós; definição a esclarecer). **Achado:** o rótulo `ontology_effectively_used` / `ontology_available_but_not_effective` é atribuído pelo **papel do braço** (`core/benchmark/ontology_gate.py`: `… and not control`), e os mesmos indicadores calculam-se «igual» nos controlos. Logo, a categoria **não** distingue conhecimento de estrutura arbitrária e não deve ser citada como evidência de uso do conhecimento. `BENEFIT_OBSERVED` não se verifica: M29 não mostrou ganho direto atribuível (Doc. 30).

## 5. Lacunas: o que depende da ontologia e o que depende da implementação

**Dependem da ontologia (conteúdo ausente):**
1. Nenhuma relação substantiva entre atributos ou entre medidas (equivalência, ordem, definição, parte-de, restrição de valor, cardinalidade).
2. Sem limites/tipos de valor (todos `xsd:decimal` sem facetas; digits são inteiros 0–16).
3. Sem relações de objeto nem papéis (grandeza medida, estatística, instrumento).
4. Partições organizacionais sem critério de pertença documentado (wine) ou sem sentido de domínio (digits); em breast_cancer, o conhecimento de grandeza/estatística está só em anotações.
5. Sem fontes citadas em nenhum axioma (Q2 `UNSOURCED` ao nível de axioma; os ficheiros indicam apenas «estrutura definida a partir da descrição das *features*»).

**Dependem da implementação (conteúdo existente mas inutilizado — B3.0, Docs. 31–33):**
1. `disjointWith`/`AllDisjointClasses`, restrições, cardinalidades, cadeias e inversas são ignoradas pelo grafo semântico do núcleo.
2. A relatedness é constante (0,712448 = 0,88·0,92·0,88) por via do *hub* genérico `DatatypeProperty`; a profundidade (OntoDepth) é constante 1,0; todas as originais ficam num único componente semântico.
3. As inferências do reasoner (0 nestas ontologias) não são propagadas aos grupos; as arestas `inferred_subsumption` do grafo são fecho transitivo estrutural, não inferências de reasoner.
4. Anotações personalizadas nunca chegam ao motor.
5. Os indicadores de utilização não distinguem conhecimento de estrutura (§4).
6. O arnês B3.1/B3.2 só usa a **partição** (preferência), não hierarquia, disjunção ou restrições (limite estrutural reconhecido no Doc. 44).

## 6. Conhecimento de domínio ausente e propostas de fonte independente (candidatas — nada é axioma ainda)
Todas as entradas são **candidatas** (`DOMAIN_REVIEW_PENDING`): precisam de fonte citada com localizador, definição, condições de aplicação e revisão humana (Doc. 47). Nenhuma foi inferida de correlações nos dados nem escolhida por resultados. Fontes a localizar/confirmar pelos revisores:
* **iris** — definições operacionais das quatro medidas (comprimento/largura de sépala/pétala), unidade (cm) e positividade (limite inferior > 0); fonte primária da recolha (Fisher, 1936) e descrição UCI. Sem relações de ordem entre medidas (não são universais).
* **wine** — descrição UCI das 13 constituintes e método analítico de cada uma; relação **conceptual** de composição entre fenóis totais, flavonoides e não-flavonoides **sem** ordem numérica (a ordem numérica `flavanoids ≤ total_phenols` falha em 36,0 % das linhas e está **rejeitada** como axioma universal, Doc. 39); não-negatividade das concentrações.
* **breast_cancer (WDBC)** — papel estatístico (média / erro padrão / «pior» = média dos três maiores) e família de medida (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria, dimensão fractal), a partir da descrição de Street, Wolberg & Mangasarian (1993) e UCI; relação `worst ≥ mean` (`wdbc_order_relations`, **preparação documental pendente de revisão**, Doc. 39); identidades de razão válidas por núcleo mas **não** para atributos agregados (rejeitadas); modelar `measurementFamily`/`statisticRole` como axiomas (não anotações) é uma proposta de desenho a rever.
* **digits (optdigits)** — esquema de pré-processamento (bitmaps 32×32 → blocos 4×4 → contagens 0–16, limite inteiro) e **vizinhança espacial** 4-conexa da grelha 8×8 (exige um tipo de facto novo, `SPATIAL_ADJACENCY`, ainda inexistente); a partição em três faixas atual não tem fonte.
Estas entradas **não** são promovidas com base nas métricas já observadas; ontologias revistas depois de observados estes benchmarks só entram em estudos DEV **exploratórios**.

## 7. Conclusão da auditoria
As quatro ontologias DEV são **logicamente válidas e bem mapeadas**, mas são, na prática, **partições planas de atributos** com 0 axiomas de domínio substantivos, 0 inferências do reasoner e sem fontes; as relações que contêm não têm suporte estatístico mensurável nos dados de treino (todos os valores-p ≥ 0,23); e o motor atual só consome a partição, com indicadores de utilização que não separam conhecimento de controlos. Isto explica, sem o provar causalmente, por que a B3.0–B3.3 não encontraram vantagem ontológica. **Não se conclui** que ontologias de qualidade não ajudem: conclui-se que **ainda não foi testada** uma ontologia de qualidade com um mecanismo capaz de a usar.
