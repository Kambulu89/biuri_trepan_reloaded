# Documento 4 — Matriz de ablação (PROPOSTA, revista pela correção arquitetural; ver Doc. 7)

Notação: **O** = TREPAN Original (congelado); **C** = C4.5 canónico (congelado); **Ω** = informação ontológica efetiva (ontologia aceite pelo quality gate e com estrutura semântica ativa);
**R_OWL** = Reloaded com Ω; **R∅** = Reloaded sem ontologia (controlo de equivalência). Todos partilham partições, oráculo congelado, orçamento estrutural comum (ontology-blind), seed e unidades.
**O(repr. OWL)** = Original a correr na representação enriquecida pela ontologia (mesmo oráculo via projeção), isola o efeito da representação do efeito da orientação semântica.

## 1. Benchmark principal (os seis grupos pedidos)
| Grupo | Braço | Definição | Papel |
|---|---|---|---|
| **A** | C | C4.5 canónico (rótulos reais) | baseline |
| **B** | O | TREPAN Original | baseline |
| **C** | **R∅** (+ todas as flags B1–B4 ligadas) | Reloaded sem ontologia | **controlo de equivalência EQ**: tem de igualar O bit-a-bit (E1–E9, Doc. 7); qualquer diferença = `EQ_VIOLATION` |
| **D** | **R_OWL** | Reloaded com ontologia real (variantes abaixo) | tratamento |
| **E** | R_OWL~perm / ~rand | mesmos mecanismos com ontologia permutada / grupos aleatórios (igual nº e distribuição) | controlo de conhecimento |
| **F** | O(repr. conv.) / R_conv | engenharia convencional de features por nomes (sem OWL), mesma operação/tamanho; Original e (R∅ nessa representação) | controlo de engenharia convencional |
| ref. | MLP (oráculo) | MLP original congelado | oráculo e referência de accuracy |
Acrescentam-se **O(repr. OWL)** (isolar Δ_repr/Δ_guia) e, na pista M, o MLP ontológico (secção 4). O braço `mlp_ontological`/`reloaded_e2e` do manifest v1 pertence à pista M, não ao grupo D.

## 2. Variantes do tratamento R_OWL (cada mecanismo é ontológico; nenhum atua com R∅)
| Id | B1 (calibração) | B2 (aceitação de splits semânticos) | B3 (seleção de configs do R_OWL) | B4 (m-of-n semântico) | Pergunta |
|---|---|---|---|---|---|
| **R0** | legacy | legacy (EFSR atual) | – | m-of-n atual (beam + ≤24 candidatos semânticos) | estado atual |
| R1a / R1b / R1c | A desempate / B limitado (λ por CV interna) / C normalizado | legacy | – | atual | calibração isolada |
| **R2** | legacy | evidência (cross-fit + LCB) | – | atual | aceitação isolada |
| R3a / R3b / R3c | x | evidência | – | atual | B1 + B2 |
| **R4** | legacy | legacy | seleção lexicográfica sobre a grelha legacy (λ_legacy ∈ {0; 0,5; 1; 2,5}) | atual | seleção isolada |
| **R5** | legacy | legacy | – | grupos semânticos reais | B4 isolado |
| **R6** | grelha {A, B(λ∈{.1,.25,.5}), C} escolhida por B3 | evidência | sim | atual | B1+B2+B3 |
| **R7** | idem | evidência | sim | grupos reais | pacote completo |
Removido: **R8 / Modo B (rótulos reais)** — independente da ontologia (Doc. 7 §8). `λ_legacy=0` com ontologia **não** é «Original»: a representação enriquecida continua a existir; o «sem ontologia» é R∅.

## 3. Controlos (obrigatórios para qualquer afirmação ontológica)
| Id | Descrição | Isola |
|---|---|---|
| R∅ + flags (grupo C) | Reloaded sem ontologia com B1–B4/EFSR ligados | inércia das flags sem Ω (EQ) |
| X~perm (X ∈ {R0, R6, R7}) | ontologia permutada (`ShuffledSemanticProvider`) | conhecimento vs estrutura/prior |
| R5~rand, R7~rand | B4 com grupos aleatórios (mesma distribuição de tamanhos, mesmo nº de candidatos) | grupos reais vs arbitrários |
| R5~conv, R7~conv | B4 com m-of-n convencional e **mesmo nº** de candidatos avaliados | «mais procura» vs conhecimento |
| O(repr. OWL) | Original na representação OWL | Δ_repr |
| O(repr. conv.) | Original com engenharia convencional por nomes (`validation/ontology_feature_audit.name_heuristic_structure`) | Δ_convencional |
| X~noprior | R_OWL sem o prior constante das features derivadas (`onto_weight=1`, depth uniforme) | efeito «preferir derivadas» |

## 4. Pistas separadas
* **Pista T (enriquecimento da árvore):** oráculo = MLP original congelado, igual em O, R∅, R_OWL e controlos. Todas as comparações de fidelidade do grupo A–F são desta pista.
* **Pista M (enriquecimento do MLP):** oráculo = MLP ontológico aceite pelo gate interno; compara-se O(repr. OWL, oráculo MLP_onto) vs R_OWL (mesmo oráculo). **Nunca** se misturam fidelidades das duas pistas.

## 5. Nomes opacos
Repetir {O, R∅, R0, R6, R7, R6~perm, O(repr. conv.)} com colunas `f1…fn`: (i) sem sidecar a ontologia é rejeitada ⇒ **R≡O** (verifica também o fallback do contrato EQ); (ii) com sidecar (D-6) mede-se o
valor do conhecimento quando os nomes não o revelam — único cenário em que o ganho sobre O(repr. conv.) se atribui ao conhecimento OWL.

## 6. Fatorial e estágios
Fatorial 2×2×2×2 (B1,B2,B3,B4; B1 fixado na subvariante mais escolhida em R6) só após o Estágio 2 e em datasets CONFIRM novos, se o orçamento permitir. Omissões declaradas antes de correr:
subvariantes de B1 só isoladas (R1/R3) e dentro de B3 (R6/R7); interação B3×B4 só em R7.
| Estágio | Conteúdo | Datasets | Uso |
|---|---|---|---|
| 0 | testes de contrato EQ, unitários, propriedade e metamórficos; custo | DEV | não confirmatório |
| 1 | C, O, **R∅(EQ)**, R0, R6, R7, R6~perm, R7~rand, R7~conv, O(repr. OWL), O(repr. conv.) | CONFIRM (≥12) | resultados primários |
| 2 | R1a–c, R2, R3a–c, R4, R5 (+controlos B4) | CONFIRM | atribuição por intervenção |
| 3 | nomes opacos; pista M | subconjunto CONFIRM | secundário |
| 4 | fatorial completo | CONFIRM novos | opcional |
Se o custo medido no Estágio 0 inviabilizar tudo, reduz-se pela ordem acima, declarando as omissões no manifest.

## 7. Pareamento e equivalência
* Mesma partição/oráculo/orçamento por unidade ⇒ diferenças pareadas; `legacy` (R0) bit-a-bit igual ao braço E atual do manifest v1; **R∅ bit-a-bit igual a O** (hoje violado em E4, Doc. 7 §5).
* Seleções internas guardam todos os candidatos e a razão (`SelectionRecord`).
