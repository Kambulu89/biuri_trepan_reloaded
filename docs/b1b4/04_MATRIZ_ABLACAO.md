# Documento 4 — Matriz de ablação (PROPOSTA)

Notação: **O** = TREPAN Original (congelado); **C** = C4.5 canónico (congelado); **R0** = Reloaded atual (braço E do manifest v1: features OWL + score + projetor + EFSR). Todos
partilham partições, oráculo congelado, orçamento estrutural comum e unidades. Sufixos de controlo: `~perm` (ontologia permutada), `~rand` (grupos aleatórios), `~conv` (m-of-n
convencional com orçamento de candidatos igualado), `~fe` (engenharia convencional de features por nomes, sem OWL).

## 1. Braços principais (pedidos R0–R7, com subvariantes pré-registadas)

| Id | Algoritmo | B1 | B2 | B3 | B4 | Pergunta que responde |
|---|---|---|---|---|---|---|
| O | TREPAN Original | – | – | – | – | baseline |
| C | C4.5 canónico | – | – | – | – | baseline (rótulos reais) |
| **R0** | Reloaded atual | legacy | legacy | – | convencional (beam + ≤24 semânticos atuais) | estado atual |
| R1a / R1b / R1c | R0 + B1 | A desempate / B limitado (λ por CV interna) / C normalizado | legacy | – | atual | efeito isolado da calibração |
| **R2** | R0 + B2 | legacy | evidência (cross-fit + LCB) | – | atual | efeito isolado da aceitação |
| R3a / R3b / R3c | R0 + B1x + B2 | x | evidência | – | atual | B1 + B2 |
| **R4** | R0 + B3 | legacy | legacy | seleção por regra lexicográfica sobre a grelha *legacy* (λ_legacy ∈ {0; 0,5; 1; 2,5}) | atual | efeito da seleção sozinha |
| **R5** | R0 + B4 | legacy | legacy | – | grupos semânticos reais | efeito isolado de m-of-n semântico |
| **R6** | R0 + B1 + B2 + B3 | grelha {A, B(λ∈{.1,.25,.5}), C} escolhida por B3 | evidência | sim | atual | B1+B2+B3 |
| **R7** | R6 + B4 | idem | evidência | sim | grupos reais | pacote completo |
| R8 (híbrido, **Modo B**) | R7 com γ∈{0;.25;.5} | idem | idem | sim | grupos reais | custo/benefício de usar rótulos reais (reportado à parte) |

Subvariantes de B1 (a/b/c) são **pré-registadas e todas reportadas** (correção de Holm); nenhuma é escolhida a posteriori. Em R6/R7 a escolha entre elas é feita **por unidade**
na CV interna (B3), não por mim.

## 2. Controlos negativos e de capacidade (obrigatórios para qualquer afirmação ontológica)

| Id | Descrição | Isola |
|---|---|---|
| R0~perm, R6~perm, R7~perm | mesma variante com ontologia permutada (`ShuffledSemanticProvider`) | conhecimento específico da OWL vs. estrutura/prior |
| R5~rand, R7~rand | B4 com grupos aleatórios de igual distribuição de tamanhos e nº de candidatos | grupos reais vs. grupos arbitrários |
| R5~conv, R7~conv | B4 com m-of-n convencional e **mesmo** nº de candidatos avaliados | «mais procura» vs. conhecimento |
| O~fe, R0~fe, R6~fe | Original / Reloaded com features de engenharia convencional por nomes (sem OWL; `validation/ontology_feature_audit.name_heuristic_structure`) | OWL vs. engenharia convencional |
| R0~noprior | R0 sem o prior constante das features derivadas (`onto_weight=1`, depth uniforme) | efeito «preferir features derivadas» |

## 3. Cenário de nomes opacos
Repetir {O, R0, R6, R7, R6~perm, R6~fe, O~fe} nos datasets com colunas `f1…fn`: (i) sem sidecar → ontologia rejeitada → R≡O (fallback correto, registado como `ontology_rejected`);
(ii) com sidecar de mapeamento (D-6) → mede-se o valor do conhecimento quando os nomes não o revelam. É o único cenário em que ganho de R vs `~fe` pode ser atribuído ao conhecimento OWL.

## 4. Fatorial completo
Fatorial 2×2×2×2 (B1, B2, B3, B4) com B1 fixado na subvariante mais escolhida por B3 em R6: **só após** os resultados do Estágio 2, e **em datasets CONFIRM novos**, se o orçamento
computacional o permitir. Omissões justificadas *antes* de ver resultados: (a) 3 subvariantes de B1 só aparecem isoladas (R1/R3) e em R6/R7 via B3; (b) interações B3×B4 só em R7;
(c) o fatorial completo não é corrido no CONFIRM-1 por custo (≈ 25 braços × 120 unidades × seleção aninhada).

## 5. Estágios (ordem de execução e prioridade de custo)
| Estágio | Conteúdo | Datasets | Uso |
|---|---|---|---|
| 0 | testes unitários/propriedade/metamórficos; medição de custo e de equivalência `legacy` | DEV | não confirmatório |
| 1 | prioridade máxima: O, C, R0, R6, R7, R6~perm, R7~rand, R7~conv, R6~fe | CONFIRM (≥12) | resultados primários |
| 2 | isolados: R1a–c, R2, R3a–c, R4, R5 (+ controlos de B4) | CONFIRM (os mesmos) | atribuição por intervenção |
| 3 | nomes opacos + R8 (híbrido) | subconjunto CONFIRM | secundário |
| 4 | fatorial completo | CONFIRM novos (se orçamento) | opcional |
Se o custo medido no Estágio 0 inviabilizar tudo, reduz-se **pela ordem acima**, declarando as omissões no manifest antes de correr.

## 6. Pareamento e equivalência
* Mesma partição/oráculo/orçamento por unidade ⇒ diferenças pareadas.
* `legacy` (R0) deve ser **bit-a-bit** igual ao braço E atual (teste de equivalência; `validation/benchmark/equivalence_report.py`).
* Arms que dependem de seleção interna guardam **todos** os candidatos avaliados e a razão da escolha (`SelectionRecord`).
