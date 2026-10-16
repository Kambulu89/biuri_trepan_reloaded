# Doc. 33 — Proposta B3 revista: semântica expressiva, verificável e auditável (DESENHO; nada implementado)

Estado: **PROPOSTA PARA APROVAÇÃO.** Nenhum mecanismo é implementado, nenhum parâmetro do núcleo é alterado, nenhuma campanha é executada. Substitui, como candidata, a proposta B3-a do Doc. 27 (restrições sobre grupos), que **não** é aprovada na arquitetura anterior. Base factual: Docs. 30 (resultados M29), 31 (auditoria do percurso OWL) e 32 (relatedness e axiomas). Requisitos permanentes: `CLAUDE.md` §2/§2b; Docs. 7 (EQ), 28 (hierarquia conjunta), 29 (controlos).

## 0. Princípios e contratos herdados (não negociáveis)
- `Reloaded(ontologia=None) ≡ Original`; idem se a ontologia for rejeitada ou não produzir estrutura ativa (EQ S0–S2). Mecanismos novos só atuam em S3, sob *flag* **desligada por omissão**, com tentativa semântica auditada (`oracle_accounting_`).
- O TREPAN Original, o C4.5, o MLP/oráculo congelado e os resultados históricos não se alteram. Teste externo nunca entra na seleção. Dataset-agnóstico (`scripts/check_dataset_agnosticism.py`).
- Objetivo e avaliação em dois níveis (Nível 1 ganho direto vs Original equivalente + atribuição com controlos; Nível 2 cadeia global U1–U5); não-inferioridade ≠ superioridade; estado lógico ≠ evidência; unidade de inferência = dataset; DEV nunca confirmatório.
- Mundo aberto: **ausência de afirmação não é negação.** Um valor/relação desconhecido é «desconhecido» (propaga-se como tal); nunca se infere ¬P de «P não afirmado».

## 1. O que a auditoria exige da nova arquitetura
| achado (Docs. 31/32) | exigência |
|---|---|
| A informação OWL colapsa em partição + agregados + 4 vetores pobres; relatedness constante por *hub* `DatatypeProperty`; `OntoDepth` constante | corrigir o **modelo de relações** antes de qualquer restrição (senão verdadeiro ≡ permutado) |
| Grupos usam só o pai direto; subsunção inferida e equivalência entre classes perdem-se | usar o **fecho** do reasoner (inferido) com origem registada |
| `disjointWith`, restrições, cardinalidades são ignorados | distinguir o que **pode** ser traduzido para atributos tabulares sem inventar regras |
| Candidatos não são limitados pelos grupos (1 componente) mas, com relatedness discriminativa, restringir por relações **falsas** prejudica (recuperação 9/12 vs 12/12) | **restrição só quando logicamente implicada**; relações plausíveis ⇒ preferência limitada |
| Influência operacional ≈ 30 % de decisões, igual nos controlos; «splits influenciados» inflaciona por constantes | métricas de influência **diferenciais** e rastreabilidade ao axioma |
| Ontologias DEV são partições planas | conhecimento mais expressivo **verdadeiro e verificável** (protocolo de §7); não se acrescentam axiomas arbitrários |
| Controlo em falta; E29 não comparável; BC sem permutação válida | desenho de controlos revisto (§6) |

## 2. Comparação com a proposta antiga (B3-a, Doc. 27)
| aspeto | B3-a (Doc. 27) | B3 revisto (este documento) |
|---|---|---|
| Unidade de conhecimento | grupo (partição por domínio) | registo tipado `SemanticFact` com estatuto epistémico, proveniência (IRI do axioma) e escopo de aplicação |
| Efeito | restringe pares/subconjuntos **permitidos** por grupo | (H) elimina só o que é **logicamente impossível/vazio**; (S) preferência **limitada** entre candidatos estatisticamente indistinguíveis; (R) representação |
| Risco principal | descartar splits úteis (permitidos: 0–11 % dos subconjuntos de 3, Doc. 27 §1) | nenhum descarte por mera plausibilidade; elimina-se só o implicado |
| Relatedness | assume a existente | substituída por modelo taxonómico (flag) com testes de sensibilidade; legado preservado |
| Controlos | permutado/aleatório de grupos (pares autorizados iguais ⇒ efeito igual por construção) | permutado **ao nível da relação** (mantém tipo/forma/escala), estrutura aleatória com o mesmo histograma de tipos de axioma, ontologia **válida sem axiomas úteis**, convencional de dimensão igualada, Original na **mesma representação** com estrutura neutra, orçamento igualado |
| Candidatos | filtra o espaço dos candidatos | **não aumenta** candidatos: orçamento semântico fixo (≤ `semantic_candidate_budget`); (H) pode diminuí-los |
| Rastreabilidade | por split até entidades | cadeia completa axioma → facto → candidato → decisão → efeito (`semantic_trace_`) |
| Dependência dos dados | grupos das TBoxes DEV (poucas relações) | exige ontologia com relações além de partição; sem isso o mecanismo **abstém-se** (S2) |

## 3. Arquitetura proposta (B3-S)
```
OWL → carga → reasoner (consistência + classificação, fecho inferido)
    → matching (inalterado; auditado)           ──┐
    → SemanticKnowledgeCompiler (novo, validation/ então core/ sob aprovação)
         entrada: ontologia + reasoner + matches ; saída: SemanticFacts tipados
    → [R] Representação: agregados por nível (fecho), features de região de valores (classes definidas por restrição de dados)
    → [H] Plausibilidade de consultas: limites declarados, ordens declaradas, exclusividade ; consultas rejeitadas CONTADAS
    → [H] Eliminação de candidatos logicamente vazios/tautológicos/redundantes
    → [S] Prioridade/desempate semântico com relatedness taxonómica (orçamento fixo; só entre candidatos indistinguíveis)
    → Reloaded (≡ Original se nenhum facto aplicável)  →  semantic_trace_  →  explicação por regra
```
### 3.1 `SemanticFact` (contrato)
`{id, kind, attrs, params, status, source_axioms[], provenance, scope}` com
- `kind ∈ {ValueBound, ValueRegion, OrderRelation, ExclusiveGroup, ConceptMembership(Closure), Disjoint, FamilyRole}`;
- `status ∈ {ENTAILED (reasoner), ASSERTED (axioma explícito), ANNOTATED (anotação não lógica), VERIFIED_DATA_CONSISTENT (só diagnóstico, nunca fonte de verdade)}`;
- `source_axioms`: IRIs/serialização dos axiomas de origem; `scope`: atributos mapeados com confiança de matching ≥ limiar.
Só factos `ENTAILED`/`ASSERTED` podem alimentar mecanismos (H). `ANNOTATED` só (R)/(S) e com estatuto declarado. Um facto cuja aplicação exija valor desconhecido **não se aplica** (mundo aberto).

### 3.2 Mecanismos (cada um: flag própria, auditável, com controlo negativo)
| id | classe | mecanismo | quando é logicamente obrigatório | efeito |
|---|---|---|---|---|
| **M0** | infra | relatedness taxonómica (Wu–Palmer/Lin, Doc. 32 §3) sem nós genéricos; fecho do reasoner; profundidade informativa | — (substitui um defeito; o legado fica como `relatedness_model="legacy"`) | grupos/relatedness passam a depender da ontologia |
| **M1** | (H) | **plausibilidade das consultas**: rejeita pontos de consulta que violam `ValueBound`, `OrderRelation` (ex.: «máximo ≥ média» **declarado**) e `ExclusiveGroup`; reamostra até preencher o lote | sim: um ponto que viola um axioma é impossível para qualquer indivíduo consistente | altera o **fluxo de consultas** (contabilizado em S3) |
| **M2** | (H) | **eliminação de candidatos vazios/tautológicos/redundantes**: m-of-n cujos literais exigem regiões disjuntas do mesmo atributo, ou um literal implicado por outro (subsunção de regiões) | sim: o teste é vazio/redundante em qualquer modelo da ontologia | menos candidatos, regras mais curtas |
| **M3** | (S) | **prioridade semântica** dos candidatos: unidades semânticas (taxonomia) ordenam a geração dentro do **mesmo** orçamento (24) | não: é preferência | altera a ordem/seleção do orçamento, não o tamanho |
| **M4** | (S) | **desempate semântico** só em bandas de empate estatístico (`tiebreak`, τ/√n_real do B1) usando coesão taxonómica | não | só reordena candidatos indistinguíveis |
| **M5** | (R) | features de **região de valores** (a partir de classes definidas por restrição de dados) e agregados por nível hierárquico | classes definidas ⇒ equivalência lógica entre predicado e região | novas colunas, com o gate de redundância existente |
| **M6** | explic. | texto da regra com conceitos e axiomas de origem | — | só relato |
**Não** se faz: converter toda relação OWL em condição m-of-n; ler ausência como negação; descartar divisões estatisticamente úteis por não serem «plausíveis»; usar rótulos reais; aumentar candidatos além do orçamento; modificar o Original.

### 3.3 Hard vs heurística (regra de decisão)
Um mecanismo é (H) **se e só se** o facto de que depende é `ENTAILED`/`ASSERTED` e a eliminação é consequência lógica (prova: o que se elimina não tem modelo consistente com a ontologia). Qualquer outra utilização é (S) com influência limitada e nunca elimina. Cada eliminação (H) regista o axioma e a derivação; testes de *soundness* verificam que nenhuma eliminação ocorre sem entailment.

### 3.4 Onde fica o código
Fase de desenho/prova: `validation/` (compilador, medidas, controlos). Só após aprovação e prova de EQ (golden) migra para `core/` atrás de *flags* desligadas, sob manifest versionado novo. O compilador não consulta nomes de datasets: opera sobre tipos de axioma e confiança de matching.

## 4. Contratos executáveis a criar (com testes)
1. **EQ-B3:** com `flags=off` ou ontologia sem factos aplicáveis, `Reloaded ≡ Original` (E1–E9, golden existente + novos cenários S2-sem-factos).
2. **Soundness (H):** para ontologias sintéticas geradas, M1/M2 nunca eliminam pontos/testes que um modelo da ontologia satisfaz (teste de propriedade).
3. **Mundo aberto:** facto com atributo desconhecido não se aplica; ausência de `disjointWith` não cria compatibilidade nem incompatibilidade.
4. **Orçamento:** nº de candidatos avaliados ≤ Original + `semantic_candidate_budget`; M2 só reduz.
5. **Sensibilidade (identificabilidade):** relações distintas ⇒ decisões distintas em sintéticos controlados; relações permutadas ⇒ decisões **diferentes das verdadeiras** (o que o caminho atual não consegue: Doc. 32 §6).
6. **Rastreabilidade:** toda decisão influenciada tem cadeia completa (§5).
7. **Agnosticismo:** teste metamórfico (renomear/permutar colunas e nomes de conceitos preserva as decisões) e verificador estático.
8. **Dois níveis:** reutiliza `core/hierarchy_contract.py` sem alterações.

## 5. Rastreabilidade — `semantic_trace_` (só relato; fora de E1–E9)
Por split e por consulta rejeitada: `fact_ids[]` → `source_axioms[]` → `candidate_origin ∈ {statistical_beam, semantic_budget}` → `decision ∈ {accepted_hard_filter, rejected_hard_filter, preference_applied, no_effect}` → `effect {delta_score, delta_local_fidelity, changed_final_tree, nodes_affected}`. Agregado por unidade: nº de consultas rejeitadas por M1 (e custo), candidatos eliminados por M2, decisões alteradas por M3/M4, **contributo por tipo de facto**. A rastreabilidade mostra influência **operacional**; a atribuição de ganho continua a depender dos controlos (Nível 1).

## 6. Controlos e desenho experimental (a aprovar; sem execução)
| braço | descrição | separa |
|---|---|---|
| A | Original canónico (+ amostrador C quando aplicável) | referência |
| A′ | **Original sobre a representação OWL enriquecida, estrutura semântica neutra** (controlo em falta) | efeito da **representação** |
| B-legado | Reloaded atual (M29) | continuidade |
| B3-real | Reloaded com B3-S, ontologia verdadeira | efeito da ontologia nos mecanismos novos |
| B3-perm | permutação **ao nível da relação** (mantém tipo, forma, escala e nº de factos; troca os atributos a que cada facto se aplica) | verdadeiro vs falso, mesma «forma» |
| B3-rand | estrutura aleatória com o **mesmo histograma de tipos de axioma** | conteúdo vs forma |
| B3-vazio | ontologia **válida sem axiomas úteis** (só partição trivial) | mecanismo sem conteúdo |
| E′ | engenharia convencional com **dimensão igualada** (agregados + diferenças/rácios entre pares correlacionados) | conhecimento vs correlação estatística |
| G | Original com orçamento de consultas igualado (A2) | orçamento |
Breast Cancer: o controlo permutado de features relacionais exige definição própria (Doc. 31 §2.3): propõe-se calcular as derivadas sobre colunas padronizadas **em todos os braços** (nova versão versionada) ou declarar `NOT_ASSESSABLE` e mantê-la no relatório global. Regras mantidas: ρ de orçamento, dimensão ±25 %, portão de validade; um controlo inválido nunca conta como evidência desfavorável.
Avaliação: Nível 1/2 inalterados; unidade = dataset (n pequeno ⇒ `observed_only` até haver confirmatórios); relatório de ~17 métricas por unidade; sem arredondar antes das desigualdades estritas; nenhuma unidade excluída.

## 7. Conhecimento verdadeiro: protocolo de engenharia de ontologias
Requisitos para qualquer ontologia enriquecida: (1) cada axioma com **fonte citável** (padrão, documentação do atributo, norma científica) registada na própria ontologia; (2) escrita e **congelada por hash antes** de qualquer experiência; (3) independente dos rótulos e dos resultados; (4) **verificação de consistência com o treino** só como diagnóstico (taxa de violação), nunca como seleção; (5) revisão independente; (6) nenhum axioma «para aumentar a expressividade».
Exemplos a **validar** (não assumidos), medidos só com atributos (sem rótulos), em `scikit-learn` DEV:
- Cancro da mama (10 famílias de medições, `média`/`pior`): **ordem por definição** `máximo ≥ média` → satisfeita em **100 %** das linhas de treino (5 sementes, 10 famílias). É candidata a `OrderRelation` ASSERTED/ENTAILED *se a ontologia a declarar* (a TBox DEV atual só tem as anotações família×papel, que não são axioma). **Medido** (`b30_query_constraint_violation.py`, só atributos de treino): com os amostradores de consultas, **92–93 %** dos pontos do amostrador marginal do Original e **43–45 %** dos do amostrador C violam a ordem em pelo menos uma família (treino real: 0 %), i.e. são pontos impossíveis para qualquer indivíduo consistente com a definição. Isto motiva M1 (hipótese: rejeitá-los melhora a relevância das consultas; **não** se afirma ganho). Relações geométricas aproximadas (`perímetro ≈ 2π·raio`: razão mediana 1,03 [P5–P95 1,006–1,072]; `área ≈ π·raio²`: 0,98 [0,958–1,003]) são **aproximadas** ⇒ só (S)/(R), nunca (H).
- Vinho: «flavonoides ≤ fenóis totais» **só 64 %** (e a soma com não-flavonoides 35 %) ⇒ **rejeitado como axioma** (exemplo do que a verificação deve travar).
- Dígitos: topologia espacial da grelha (vizinhança de píxeis) é conhecimento estrutural verdadeiro ⇒ candidato a unidades semânticas por proximidade (M3/M4), a rever com especialista.
- Iris: sem relações adicionais conhecidas a priori além da partição ⇒ o mecanismo deve **abster-se** (serve de teste do ramo S2).
Para os testes de identificabilidade, usam-se ontologias **sintéticas** com verdade conhecida (Doc. 32 §6), não valores favoráveis.

## 8. Testes previstos (resumo)
Unitários de compilador (um por tipo de facto, incl. mundo aberto); soundness por propriedade (M1/M2); EQ golden com *flags* desligadas; sensibilidade/identificabilidade sintética (verdadeiro ≠ permutado); metamórficos (renomear colunas/conceitos); regressão de custo (candidatos ≤ orçamento; tempo ≤ k× Original); cadeia de rastreabilidade completa; verificador de agnosticismo; reprodutibilidade (sementes). Já existentes e verdes: 15 testes B3.0 + EQ (311 passed, 40 skipped nesta fase, subconjunto relevante).

## 9. Custo estimado (estimativa; nada executado)
Compilação semântica: segundos por unidade (reasoner ≈ 0,6 s). Execução: o custo por *fit* na M29 foi ≈ 25 s (CPU) em média. Piloto (iris/42 e wine/42, 9 braços × 3 réplicas): ≈ 0,15–0,2 CPU-h. Campanha DEV completa (18 unidades × 8 braços novos × 3 réplicas): ≈ 3–4 CPU-h (digits ≈ 30 % do total). Etapas e *gates* de custo como na M29 (guarda de CPU real, paragem para aprovação antes de exceder). Pedido de orçamento só após aprovação do desenho e da ontologia enriquecida.

## 10. Riscos
1. **Falta de conhecimento verdadeiro expressivo** nos DEV (Iris/Wine/Digits): o mecanismo pode abster-se e não haver ganho — resultado legítimo (reporta-se).
2. M1 muda o fluxo de consultas (custo e distribuição): contabilizar e controlar com G e com A′.
3. Correção de relatedness altera o Reloaded (mecanismo novo, não bug-fix retroativo): manifest versionado; legado intacto.
4. Matching lexical: falsos negativos (sinónimos) limitam cobertura; falsos positivos por subcadeia — manter auditoria e limiar; nunca ampliar matching por similaridade dos dados.
5. Com n=4 datasets o poder é baixo; evidência `observed_only`.
6. Armadilha de «ontologia à medida»: mitigada por §7 (fontes, hash prévio, revisão).

## 11. Critérios de aceitação para autorizar a implementação (estado hoje)
| # | critério (pedido do utilizador) | estado após B3.0 |
|---|---|---|
| 1 | Relações OWL verdadeiras disponíveis | **parcial**: DEV só partições (+ anotações em BC); relações expressivas exigem §7 |
| 2 | Reconhecidas corretamente | **parcial**: partição/domínio sim; disjunção, restrições, inferência: não (Doc. 32 §4) |
| 3 | Matching seguro | **sim nos DEV** (100 %, sem falsos positivos observados); limitado a nomes (falsos negativos) |
| 4 | Reasoner produz/valida inferências aplicáveis | **não**: valida consistência; 0 inferências DEV; a inferência sintética não propaga |
| 5 | Relações influenciam candidatos m-of-n | **sim com relatedness discriminativa** (diagnóstico GRP); **não** no caminho atual (hub) |
| 6 | Influência diferente da das permutadas | **demonstrada só com relatedness discriminativa** (GRP: 9/12 vs 12/12); a demonstrar com M0–M4 |
| 7 | Complexidade controlável | **desenho**: orçamento fixo + M2 só reduz; a provar |
| 8 | Identidade do Original sem OWL | **sim** (EQ verde, CI verde); a repetir com *flags* novas |
| 9 | Protocolo de ganhos fidelity/accuracy | **sim** (Nível 1/2, Docs. 28–30) + controlos §6 |
| 10 | Independência do dataset | **por construção** (tipos de axioma, não nomes) + verificador |
Conclusão: **ainda não** estão reunidos os critérios 1, 2, 4 e 6 para pedir implementação do núcleo. Recomenda-se uma **fase B3.1 sem alterar o núcleo** (ver §12).

## 12. Decisões pedidas
1. Aprovar a **abstenção** da proposta B3-a antiga e a adoção do desenho B3-S como candidato.
2. Aprovar a fase **B3.1 (validation/ apenas)**: compilador de factos + medidas taxonómicas + testes de soundness/mundo aberto/identificabilidade em sintéticos + `semantic_trace_` em protótipo, **sem tocar** `core/` nem executar campanha DEV.
3. Aprovar o **protocolo de ontologias enriquecidas** (§7) e indicar quem revê/valida as fontes (a escolha de axiomas é científica e do utilizador).
4. Decidir se o controlo de BC passa a calcular derivadas padronizadas em **todos** os braços (nova versão) ou se BC fica `NOT_ASSESSABLE` na atribuição.
5. Só depois: pedir orçamento (piloto de 2 unidades) para M1–M5 com *flags* desligadas por omissão.
