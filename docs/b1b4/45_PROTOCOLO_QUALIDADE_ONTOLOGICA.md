# Doc. 45 — Protocolo B3-Q: qualidade, relevância, expressividade e utilização efetiva do conhecimento ontológico

Estado: `B3Q = PROTOCOL_AUTHORIZED`. Protocolo independente, **anterior a qualquer benchmark**. Só `validation/` e documentação; nada em `core/`. `B33 = COMPLETED / NO_ONTOLOGICAL_ADVANTAGE`; `B34 = DESIGN_ONLY / NOT_EXECUTED`; `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`.
Aplicação inicial (esta fase, só leitura): Doc. 46. Processo de construção de ontologias reais: Doc. 47. Separação H1/H2 e auditoria dos cálculos do Doc. 44: Doc. 48.

## 0. Princípios

1. **Cinco camadas separadas**, avaliadas por ordem e nunca fundidas num único rótulo: (i) qualidade **intrínseca** (Q1–Q3); (ii) **relevância para os atributos** (Q4); (iii) **relevância para a tarefa** e capacidade de enriquecimento (Q5); (iv) **capacidade de utilização pelo sistema** (Q6); (v) **desempenho resultante** (fora deste protocolo: é o benchmark).
2. **Independência do desempenho.** Nenhum estado de Q1–Q5 depende de accuracy, fidelidade ou de qualquer métrica de teste. Q5 e o diagnóstico de suporte usam, no máximo, **dados de treino** e documentação de domínio. Q6 usa só o rastreamento estrutural (quais artefactos chegam a que etapa), não o ganho.
3. **Sem rótulo global de «alta qualidade».** O relatório devolve um vetor de estados por dimensão, com **razões e limitações**. Uma ontologia logicamente correta não é rejeitada por não dar ganho; uma ontologia que dá ganho não é promovida por isso.
4. **Sem limiares arbitrários.** Este protocolo não introduz cortes numéricos novos. Os cortes que já existem no gate (`min_match_score = 0,72`, `min_ambiguity_margin = 0,08`, cobertura mínima 0,35, rácio genérico 0,45) são **herdados e não validados**; ficam assinalados como tal e não são usados para classificar a qualidade. Onde um número seria necessário, o estado é **descritivo** (contagens, razões, testes estatísticos com valor-p) ou exige decisão humana explícita.
5. **Estados auditáveis**: cada estado traz `reason`, `evidence` (ficheiro/hash) e `limitations`. «Não avaliado» é um estado legítimo.
6. **Sem revisões inventadas.** Estados que dependem de revisão humana ficam `*_REVIEW_PENDING` até existirem aprovações verificáveis de pessoas designadas.

## 1. Dimensões, testes e estados

### Q1 — Validade lógica
| Teste | Como | Estados |
|---|---|---|
| Perfil OWL 2 DL | verificador de perfil (sem *punning*, só construções DL); hoje apenas **proxy** (sem punning + classificação HermiT); o verificador formal fica por implementar | `DL_PROFILE_PROXY_OK` / `DL_PROFILE_VIOLATION` / `NOT_EVALUATED` |
| Consistência | HermiT | `CONSISTENT` / `INCONSISTENT` / `REASONER_UNAVAILABLE` |
| Satisfatibilidade das classes relevantes | HermiT; lista de classes insatisfazíveis | `ALL_RELEVANT_SATISFIABLE` / `UNSATISFIABLE_RELEVANT` |
| Comportamento do reasoner | tempo, erros, mundo aberto, inferências produzidas | descritivo |
| Contradições indevidas | classes insatisfazíveis, disjunções violadas por assertivas | `NONE_FOUND` / lista |
Estado Q1 = o pior dos testes avaliados; `NOT_EVALUATED` nunca se promove a «OK».

### Q2 — Proveniência e verdade de domínio
Por **axioma** (ficha de axioma, Doc. 47): fonte documental verificável com localizador; definição dos conceitos; relação afirmada; condições de validade; independência dos dados de teste; revisão humana independente. Estados por axioma: `SOURCED_REVIEWED` / `SOURCED_REVIEW_PENDING` / `UNSOURCED` / `CONTRADICTED_BY_SOURCE` / `REJECTED_AS_UNIVERSAL`. A «verdade» de domínio **não é verificável por máquina**; a máquina verifica apenas que há fonte, localizador, âmbito e que a fonte não é o dataset de avaliação. Verificação empírica contra dados (como a de `wine_phenol_subset_rejected`, 36 % de violações) serve para **refutar** um axioma universal, nunca para o **criar**.

### Q3 — Expressividade (descritiva; mais axiomas ≠ melhor)
Perfil funcional, não uma pontuação: contagem por função (declaração, anotação, tipagem, agrupamento atributo→conceito, taxonomia, disjunção, **substantivo-relacional**: equivalências, subpropriedades, restrições de valor, cardinalidades, relações de objetos, cadeias, inversas, facetas de tipos de dados). Um axioma só é marcado **informativo** se (a) não é uma consequência trivial das declarações e (b) tem pelo menos uma entidade mapeada a um atributo (Q4). Axiomas puramente organizacionais (raiz, declaração, anotação) são reportados mas não contam como expressividade de domínio.

### Q4 — Matching e cobertura
Cobertura das *features* originais; confiança; ambiguidades (com os diagnósticos P3); colisões; relações aplicáveis às *features* mapeadas; atributos sem correspondência e consequências da rejeição (que axiomas deixam de se aplicar). Estados: descritivos (`coverage = n_map/n_features`, listas) + `MATCHING_COMPLETE` / `MATCHING_PARTIAL(unmapped=…)` / `MATCHING_REJECTED`.

### Q5 — Capacidade de enriquecimento
Cada relação/axioma é classificado numa categoria **exclusiva**:
| Categoria | Definição | Evidência |
|---|---|---|
| `EXTERNAL_GENUINE` | informação de domínio ausente da tabela e não recuperável dos dados de treino disponíveis | fonte externa + teste de recuperabilidade (abaixo) |
| `RECOVERABLE_FROM_DATA` | recuperável do treino com um método genérico a partir de n exemplos | teste de recuperabilidade (ex.: ARI do agrupamento por correlação vs a partição declarada, Doc. 48) |
| `REDUNDANT_TRANSFORM` | determinístico face às *features* originais (ex.: agregados lineares, razões) | derivação exata |
| `ANNOTATION_ONLY` | informação presente só em anotações, invisível ao reasoner e ao núcleo | censo |
| `LOGICALLY_USABLE_CONSTRAINT` | restrição que um mecanismo pode aplicar a candidatos | tipo de axioma + mecanismo existente |
| `ORGANIZATIONAL_ONLY` | agrupamento/taxonomia sem consequência lógica adicional | censo |
| `UNDETERMINED` | sem evidência suficiente | — |
Distingue-se **informação externa genuína** de **transformações redundantes das features** (já identificadas: agregados hierárquicos de grupo são combinações lineares das originais). A recuperabilidade a partir dos dados é medida **no regime de n do estudo**, nunca no n máximo.
**Diagnóstico de relevância para a tarefa (separado de Q1–Q5):** sinergia por pares dentro vs fora dos grupos nos dados de **treino** com teste de permutação das atribuições (`b3q_audit.data_support`). Reporta suporte estatístico da partição; **não** é qualidade da ontologia.

### Q6 — Utilização efetiva pelo TREPAN
Cadeia por relação: `Axioma OWL → Reasoner → Fact Registry → Feature Mapping → Candidato m-of-n → Decisão → Árvore`.
| Estado | Definição operacional | Evidência |
|---|---|---|
| `AVAILABLE` | o axioma existe no ficheiro | censo |
| `VALIDATED` | passa Q1 (consistência/satisfatibilidade) | HermiT |
| `MAPPED` | as entidades do axioma casam com *features* (Q4) | matching sem rejeição |
| `COMPILED` | origina um facto utilizável no registo (`permitted_use ≠ NONE`, não rejeitado) | `FactRegistry.usable()` |
| `USED` | o facto é consumido por um mecanismo (unidade/relatedness/restrição) na construção de candidatos | rastreamento do mecanismo |
| `DECISION_INFLUENCED` | existe uma decisão de split cuja escolha mudou **por causa** desse facto **e** a mudança é **maior** com a ontologia verdadeira do que com os controlos permutado/aleatório (senão é efeito do mecanismo, não do conhecimento) | rastreamento + controlos |
| `BENEFIT_OBSERVED` | ganho conjunto atribuído (accuracy e fidelidade) com controlos, nível 1 | benchmark (fora deste protocolo) |
Os estados são **ordenados mas não equivalentes**; um estado só se atribui se o anterior se verificar para essa relação. **Aviso metodológico (achado do Doc. 46):** os indicadores `ontology_influenced_splits`/`semantic_decision_changed_count` do núcleo atual são praticamente iguais para a ontologia verdadeira e para os controlos, e o rótulo `ontology_effectively_used` é atribuído **pelo papel do braço** (`not control`), não por uma medição que distinga conhecimento de estrutura arbitrária; portanto **não servem** como prova de `DECISION_INFLUENCED`.

## 2. Relatório de qualidade ontológica (formato)
Por ontologia e versão (hash): `{ontology_id, version, sha256, q1:{estado,razões,limitações}, q2:{por axioma}, q3:{perfil funcional}, q4:{cobertura, ambiguidades, não mapeados}, q5:{por relação}, q6:{por relação, estados máximos atingidos}, task_relevance_diagnostic:{…}, human_review:{domain, ontology, methodology: *_REVIEW_PENDING}, limitations:[…]}`. O relatório é gerado **antes** do benchmark, é versionado e não é editado depois de se ver o desempenho.

## 3. Regras de uso
* Ontologia congelada não se altera; correções são **propostas versionadas** (`…/proposals/<id>/vN`).
* Ontologias revistas depois de se observarem os benchmarks DEV só podem participar em estudos DEV **exploratórios**; nunca como prova confirmatória independente.
* Nenhum axioma é escolhido com base em resultados de teste; nenhum axioma de domínio é derivado de correlações dos dados.
* Uma ontologia só é «independentemente validada» quando existirem aprovações verificáveis dos três revisores designados.

## 4. Testes
Implementados nesta fase (`tests/test_b3q_audit.py`): classificação do censo de axiomas num OWL sintético; deteção de anotação personalizada como `ANOTACAO_PERSONALIZADA`; verificação do hash da partição de treino; recuperabilidade de partições (agrupamento por correlação); evidência de auditoria presente e coerente. A implementar (fora desta fase, se autorizado): verificador formal OWL 2 DL; rastreamento por relação até `DECISION_INFLUENCED` com controlos; ficha de axioma validada por esquema; geração automática do relatório do §2.

## 5. Limitações do protocolo
Q2 e parte de Q1 dependem de pessoas; Q5 depende de assumir um regime de n; o diagnóstico de tarefa mede suporte estatístico e não verdade; o rastreamento do Q6 está limitado pelo que o núcleo regista (hoje só partições, ver Doc. 46 §6).
