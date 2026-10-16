# Doc. 35 — Relatedness taxonómica (B3.1)

Estado: **EXPERIMENTAL EM `validation/`; a relatedness histórica do núcleo (`OntologySemanticGraph.relatedness`) NÃO foi alterada.** Código: `validation/benchmark/b31_relatedness.py` (+ `b31_relatedness_report.py`); testes: `tests/test_b31_facts_relatedness.py` (secção «relatedness»). Evidência: `docs/evidence/b1b4/b31/relatedness_dev_synthetic.json`. Antecedentes: Doc. 32 §2–3.

## 1. Problema (resumo medido na B3.0)
A medida histórica devolve **0,712448 = 0,88·0,92·0,88** para quaisquer duas propriedades de dados, pelo nó genérico `DatatypeProperty` (ancestral comum sem valor discriminativo). Nas 4 ontologias DEV tem **1 valor distinto** entre todos os pares originais (média «mesmo conceito» = «conceitos distintos» = 0,712).

## 2. Requisitos (pedido B3.1 §5) → como são satisfeitos
| # | requisito | solução |
|---|---|---|
| 1 | classes e propriedades efetivamente relacionadas | conceitos aplicáveis a cada atributo vêm de factos `ATTRIBUTE_APPLIES_TO_CONCEPT` (domínio + fecho); propriedades via pseudo-conceitos `prop:q` |
| 2 | distância taxonómica | Wu–Palmer sobre o DAG de conceitos informativos |
| 3 | ancestrais semanticamente informativos | só contam conceitos com **0 < \|E(c)\| < N** (extensão = atributos mapeados a que se aplica); IC(c) = −log(\|E(c)\|/N) |
| 4 | profundidade real | profundidade = caminho mais longo desde a raiz informativa (atributo-folha 1 nível abaixo do seu conceito) |
| 5 | sub-propriedades | pseudo-conceitos `prop:q` herdados por fecho (super-propriedades) |
| 6 | equivalências verificadas | classes: fusão de nós (asseridas ou inferidas); propriedades: identificador canónico partilhado ⇒ relatedness 1 |
| 7 | relações explicitamente documentadas | factos `SAME_MEASUREMENT_FAMILY` utilizáveis (≥ `PREFERENCE`) viram pseudo-conceito `doc:<id>` |
| 8 | relacionadas vs não relacionadas | valores distintos conforme a relação; sem conceito informativo comum ⇒ 0 |
| 9 | ancestrais genéricos sem valor | **excluídos** por construção (IC = 0); teste: envolver tudo num novo ancestral genérico não altera a matriz |
| 10 | relações inferidas pelo reasoner | factos `INFERRED_BY_REASONER` entram no fecho (teste com HermiT) |
Sem constantes ajustáveis: a medida depende apenas da estrutura (testado: mesma estrutura ⇒ mesmos números; invariante a renomeação de conceitos/atributos e à ordem dos atributos).

## 3. Definições
Seja S(f) o conjunto de conceitos/pseudo-conceitos aplicáveis ao atributo f; E(c) = {f : c ∈ S(f)}; N = nº de atributos mapeados; c informativo ⇔ 0 < |E(c)| < N.
- **Lin:** `lin(f,g) = 2·IC(lcs)/(IC(f)+IC(g))`, lcs = conceito informativo comum de maior IC; IC(f) = max IC dos seus conceitos informativos. Em [0,1]; lin(f,f)=1.
- **Wu–Palmer:** `wup(f,g) = 2·d(lcs)/(d(f)+d(g))`, d = profundidade (caminho mais longo desde a raiz do DAG informativo).
- **Saída tri-estado (mundo aberto):** `RELATED` (score > 0) · `DISTINCT_CONCEPTS_DISJOINT` (disjunção **declarada** entre conceitos dos atributos: evidência de conceitos distintos, **não** de incompatibilidade de combinação) · `NO_RELATIONAL_EVIDENCE` (ausência de evidência; score 0). Nunca existe um estado «incompatível» inferido por ausência.
- **Unidades semânticas:** extensões de conceitos informativos com ≥ 2 atributos (extensões idênticas colapsam; prefere-se o conceito real ao pseudo-conceito de propriedade).

## 4. Propriedades verificadas (testes)
Simetria, reflexividade, limites [0,1]; ordem «mesmo conceito > conceitos irmãos > sem relação» (Lin); sub-propriedade e conceito partilhado > só ancestral comum; propriedades equivalentes = 1; invariância a ancestral genérico, a renomeação e à permutação de atributos; relação documentada aumenta a relatedness só dos atributos referidos; relação inferida pelo reasoner aumenta a relatedness (`C ⊑ A` inferido relaciona atributos em `A` e `C`); ausência de domínio ⇒ `NO_RELATIONAL_EVIDENCE` e nenhuma unidade.

## 5. Comparação histórica vs taxonómica
### 5.1 Ontologia sintética de 3 níveis (`Root ⊃ Mid1{L1,L2} , Mid2{L3}`; atributos `fa,fb∈L1`, `fc∈L2`, `fd∈L3`)
| par | histórica | Lin | Wu–Palmer |
|---|---|---|---|
| fa–fb (mesmo conceito L1) | 0,7124 | 0,500 | @@WUP_SAME@@ |
| fa–fc (irmãos sob Mid1) | 0,7124 | 0,208 | @@WUP_SIB@@ |
| fa–fd (ramos distintos) | 0,7124 | 0,000 | 0,000 |
A histórica é **constante**; a taxonómica ordena segundo a hierarquia.

### 5.2 Ontologias DEV (partições planas de 2 níveis)
| ontologia | valores distintos (hist./Lin/WuP) | média mesmo conceito (hist./Lin/WuP) | média conceitos distintos (hist./Lin/WuP) |
|---|---|---|---|
| iris | 1 / 2 / 2 | 0,712 / 0,500 / 0,500 | 0,712 / 0,000 / 0,000 |
| wine | 1 / 3 / 2 | 0,712 / 0,399 / 0,500 | 0,712 / 0,000 / 0,000 |
| breast_cancer | 1 / 2 / 2 | 0,712 / 0,323 / 0,500 | 0,712 / 0,000 / 0,000 |
| digits | 1 / 3 / 2 | 0,712 / 0,253 / 0,500 | 0,712 / 0,000 / 0,000 |
Nas TBoxes DEV as medidas taxonómicas **reduzem-se ao indicador «mesmo conceito»**: removem o *hub* mas **não acrescentam conhecimento** (as ontologias não o têm). O valor da medida só se manifesta com taxonomias mais profundas (§5.1) ou relações adicionais — daí a necessidade de ontologias enriquecidas validadas (Doc. 37).

## 6. Limitações
1. Usa só o que está nos factos: atributos não mapeados ficam sem evidência; o matching lexical herdado limita a cobertura.
2. Um atributo é «folha» do seu conceito: a relatedness de atributos do mesmo conceito é < 1 (são atributos distintos); só atributos equivalentes têm 1.
3. Lin depende de N (atributos mapeados): comparações entre ontologias com N diferentes não são diretamente comparáveis; dentro de uma unidade é consistente.
4. Em DAGs com herança múltipla a profundidade é a do caminho mais longo (convenção documentada).
5. É um diagnóstico em `validation/`; integrar no núcleo exige aprovação, *flag* desligada por omissão, manifesto novo e demonstração de EQ.
