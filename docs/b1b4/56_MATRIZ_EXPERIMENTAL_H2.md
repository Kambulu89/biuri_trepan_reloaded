# Doc. 56 — Matriz experimental H2 (German Credit, DEV exploratório): braços A–J, comparabilidade, custos, controlos e critérios

Estado: **desenho, `frozen = false`, NÃO executado, NÃO autorizado a executar** (campanha H2 de 2,7 CPU-h não autorizada). Só `validation/`; nada em `core/`; TREPAN Original, C4.5 e MLP intocados; ontologia congelada intocada. Requisito central (objetivo experimental, não consequência garantida): `Acc(Reloaded_OWL) > Acc(Original) > Acc(C4.5)` e `Fid(Reloaded_OWL) > Fid(Original)`, `Fid(Reloaded_OWL) > Fid(C4.5)`, em simultâneo, no mesmo MLP e nas mesmas instâncias.

## 1. Braços (A–J) e implementação em `validation/`
| Braço | Definição | Representação | Informação prévia | Orientação semântica | Implementação |
|---|---|---|---|---|---|
| **A** | TREPAN Original canónico | 61 colunas base | nenhuma | — | `TrepanOriginalClassifier` |
| **B** | Original WIDE | base | nenhuma | — | `max_features_per_node = p` |
| **C** | Original **na mesma representação enriquecida**, sem orientação semântica | base + 10 colunas derivadas (O2) | os axiomas (como colunas) | **nenhuma** (as colunas são tratadas como quaisquer outras; M-P desligado) | `DerivedLiteralTrepan(prune_redundant=False)` |
| **D** | TREPAN com **grupos corretos sem OWL** (a partição temática como lista simples) | base | partição | unidades de pesquisa B3.1 | arnês `CountSteered` com unidades da lista |
| **E** | TREPAN com **ontologia plana** (O1) | base | partição (OWL) | unidades via registo B3.1 | idem, grupos vindos do registo |
| **F** | TREPAN com **ontologia expressiva** (O2) | base + derivadas | partição + axiomas | M-R + M-P (poda lógica) | `DerivedLiteralTrepan(prune_redundant=True)` (+ unidades de E) |
| **G** | TREPAN com ontologia expressiva **neutralizada** (O3: mesmas classes, sem definições) | base (sem derivadas) | partição | sem derivadas | equivale a E por construção; mantém-se como controlo da estrutura |
| **H** | **Permutada** (ordem/composição trocadas, OWL válido) ou **relações aleatórias comparáveis** (uniões aleatórias, mesmos tamanhos) | base + derivadas **erradas** (mesmo n.º e tamanhos) | informação errada | M-R + M-P | OWL H e I do `v0_draft` |
| **I** | **Engenharia convencional de features** | base + colunas cumulativas com **ordem aprendida da taxa de classe do TREINO** + agrupamento por correlação | derivada dos dados | nenhuma | `control_*` + ordenação por treino |
| **J** | **C4.5 canónico** treinado nos rótulos reais | base | nenhuma | — | `C45Classifier` |
Extra suplementar (não faz parte da lista A–J, fica opcional): pesquisa genérica de interações por pares (rastreio por pares com controlo de comparações múltiplas) como baseline forte adicional, e `WIDE` com `beam_width` calibrado para igualar candidatos (K1) / CPU (K2).

## 2. Contrato de comparabilidade (quando uma comparação é interpretável)
Uma comparação só se interpreta se, nos dois braços: (1) **as mesmas instâncias** (mesmo `split_hash` de treino/teste); (2) **o mesmo MLP/oráculo** (mesmo `oracle_id`; nunca comparar fidelidades face a MLPs diferentes); (3) **a mesma representação**, ou a diferença de representação é exatamente o objeto do contraste; (4) **os mesmos recursos**: `min_sample`, `max_queries`, `max_nodes`, `max_depth`, `max_n`, `beam_width`, `max_features_per_node`, semente; consultas e candidatos efetivos **reportados**; (5) **o conhecimento fornecido** contabilizado (n.º de fichas, de literais, bits quando aplicável).
Se qualquer condição falha, a comparação fica `NOT_COMPARABLE` e não entra em conclusões. Ganhos devidos só a mais consultas ou candidatos **não se atribuem à OWL**: cada contraste declara a única dimensão igualada.
| Contraste | Mede | Dimensão que difere | Condição para ser interpretável |
|---|---|---|---|
| **F vs A / B / C** | benefício relativamente a baselines fortes | F vs A: representação + orientação; F vs B: idem com pesquisa alargada; **F vs C: só a orientação (M-P)**, mesma representação | mesmos candidatos adicionais; n.º de colunas igual em F e C; consultas reportadas |
| **C vs A** | só a **representação enriquecida** (conhecimento como colunas) | colunas derivadas | idem |
| **F vs D / E** | benefício **adicional da semântica expressiva** sobre a partição (H2a) | axiomas (O2 vs O1/lista) | mesma partição de base; derivadas são a única diferença de informação |
| **D vs E** | efeito da representação **OWL plana** vs lista | só a representação da partição | esperado igual; igualdade **não refuta H2** |
| **F vs G / H** | benefício das **relações verdadeiras** (H2b) | conteúdo das relações, com estrutura e n.º de colunas iguais | H com os mesmos tamanhos de literais; G sem derivadas ⇒ F−G inclui a representação, F−H isola o conteúdo |
| **F vs I** | conhecimento externo vs engenharia convencional dos dados | origem da informação | I só usa treino |
| **F vs J** | contribuição para a hierarquia global | modelo | fidelidade de J face ao **mesmo** MLP; J treinado em rótulos reais |

## 3. Métricas e utilização efetiva
Accuracy (rótulos reais, principal); fidelity ao MLP; precision macro e macro-F1 (complementares); nós, profundidade, condições m-of-n; candidatos gerados/avaliados e consultas ao oráculo; CPU de ajuste, de compilação OWL/reasoner e total; memória (`tracemalloc`); estabilidade entre divisões. **Por relação OWL**: `AVAILABLE`, `VALIDATED`, `MAPPED`, `COMPILED`, `CANDIDATE_GENERATED`, `DECISION_INFLUENCED`, `BENEFIT_OBSERVED` (Doc. 55 §2), com evidência observacional (candidatos realmente avaliados; escolhas de split alteradas pela ablação contrafactual); o nome do braço nunca prova utilização.

## 4. Custos (estimativa; nada executado)
Base de medição: B3.3 (≈ 40–45 s por conjunto de sementes para 13 braços em 32–48 colunas, `n_train` ≈ 1000, memória incluída). German Credit: 750 treino × 61–71 colunas, MLP próprio (`≈ 1–3 s`), compilação HermiT ≈ 1–3 s por ontologia.
| Componente | CPU-h | Notas |
|---|---|---|
| Divisões holdout (20) × 10 braços (A–J) | ≈ 0,25 | cada divisão com o seu MLP congelado partilhado pelos braços |
| Compilação OWL/reasoner (O1, O2, O3, H, I × 20 divisões ou 1× por ontologia) | ≈ 0,05 | cache por hash da ontologia (chave exata) |
| Rastreio contrafactual (ablação de 10 literais × 20 divisões, só braço F) | ≈ 0,3 | aditivo |
| Memória e estabilidade (10 divisões extra) | ≈ 0,1 | |
| **Total fase R (German)** | **≈ 0,7** | limite duro proposto: 1,0 |
| Fase P (sintético, mecanismo com verdade conhecida; opcional) | ≈ 1,5 | só com o gerador congelado e as portas aprovadas (Docs. 49, 57) |
| Revisão humana | não estimável em CPU | ordem de grandeza: algumas horas por revisor para 10 fichas substantivas + fontes (estimativa de planeamento, a confirmar pelos designados) |
O valor anterior de 2,7 CPU-h correspondia à fase P + R + calibração; a fase R sozinha é ≈ 0,7. **Nenhum está autorizado.**

## 5. Controlos negativos e validade
G, H (permutada, aleatória), engenharia convencional I, uma ontologia **logicamente correta mas irrelevante** (axiomas verdadeiros sobre variáveis sem relação com o rótulo) e a ablação por axioma. Nenhuma escolha de axiomas, sementes ou limiares por desempenho; H/I com sementes fixadas antes; coincidência de conteúdo com O2 deteta-se e re-sorteia-se segundo regra pré-fixada e registada.

## 6. Critérios de aceitação (por contraste) e estados
Estados: `RULE_RECOVERY_GAIN` (não aplicável: sem regra verdadeira conhecida), `FIDELITY_GAIN`, `ACCURACY_GAIN`, `JOINT_GAIN`, `ONTOLOGY_ATTRIBUTED_GAIN`, `NO_ONTOLOGICAL_ADVANTAGE`, `INSUFFICIENT_EVIDENCE`, `NOT_COMPARABLE`, `UNTESTABLE`, `NOT_ADDITIVE_TO_PARTITION`, `HIERARCHY_INFEASIBLE`.
* `JOINT_GAIN(F vs X)`: Δacc > 0 **e** Δfid > 0 em simultâneo, com IC95 (bootstrap sobre divisões) inferior > 0 em ambos, no mesmo MLP/instâncias.
* `ONTOLOGY_ATTRIBUTED_GAIN`: JOINT_GAIN de F sobre A **e** sobre C (a orientação conta), D/E (H2a, ou `NOT_ADDITIVE_TO_PARTITION`/`UNTESTABLE` explícito), G/H e I; os negativos sem vantagem espúria; estabilidade entre divisões independentes; custo e complexidade documentados; fichas com revisão (ou o rótulo explícito «rascunho de IA não validado»); **e** pelo menos um literal em `DECISION_INFLUENCED` por ablação.
* `HIERARCHY_INFEASIBLE`: se `Acc(A) ≤ Acc(J)` na unidade, a cadeia global fica infeasível com os baselines fixados — reporta-se, **não se corrige**.
* `UNTESTABLE` (pré-condição de H2): se F e C/D produzirem traços de candidatos idênticos, ou se nenhum literal derivado for avaliado.
* Evidência: **`observed_only`** — um único dataset exposto; unidade de inferência = dataset ⇒ nenhuma afirmação estatística de generalização; exploratório (DEV).

## 7. Pré-requisitos para executar a fase R (nenhum cumprido hoje)
1. Dicionário de dados `german.doc` obtido (ficheiro local ou acesso) e fontes verificadas; 2. fichas revistas pelos três papéis (ou execução rotulada «rascunho de IA não validado»); 3. conjunto avaliável não vazio (hoje vazio, Doc. 54); 4. limiares/portas congelados (Doc. 57), se houver fase P; 5. adaptador ligado ao arnês dos braços D/E/F com rastreio por literal; 6. pré-registo (hipóteses, braços, métricas, sementes 8000+, critérios) congelado em commit separado; 7. aprovação explícita do orçamento.

## 8. Riscos
Exposição do conjunto (EC1) e um só dataset; axiomas de rascunho de IA; ordem de `job` e agrupamentos de `purpose` frágeis; literais maioritariamente já dentro do espaço do Original (ganho só por orientação); MLP moderado (limita a fidelidade); comparações múltiplas (Holm); risco de ajuste implícito se os limiares ou os literais mudarem depois de ver resultados — proibido.
