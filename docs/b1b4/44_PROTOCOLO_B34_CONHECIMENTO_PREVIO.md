# Doc. 44 — Protocolo B3.4: valor do conhecimento prévio ontológico quando a estatística do treino é insuficiente

Estado: **DESENHO E PRÉ-REGISTO EM RASCUNHO — `frozen = false`.** Nada foi implementado nem executado. A avaliação B3.4 **não** está autorizada; este documento pede apenas a aprovação do desenho.
Restrições: só `validation/`; nada em `core/`; `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`; sem datasets DEV/confirmatórios; sem alterar Original, C4.5, MLP, baselines, protocolos ou resultados congelados.
Manifest em rascunho: `docs/b1b4/manifest_v3/B34_EXPERIMENT_MANIFEST.DRAFT.json` (`frozen=false`). O rascunho `BENCHMARK_MANIFEST_v3.DRAFT.json` (hash fixado por gerador e testes) **não foi tocado**; o B3.4 tem manifest próprio para não perturbar protocolos existentes.

## 1. Conclusões B3.1–B3.3 (o que o desenho tem de respeitar)

| Fase | Conclusão |
|---|---|
| B3.0 | A ontologia só entra no algoritmo por relatedness e agrupamentos de atributos; os axiomas (disjunção, restrições, cardinalidades, cadeias, inversas) são ignorados; as ontologias DEV são partições planas. |
| B3.1 | Com uma pesquisa por unidades semânticas (preferência pura, pontuação estatística original), a ontologia verdadeira recupera a regra mais vezes que o Original limitado, com rastreabilidade completa. |
| B3.2 | Esse ganho **não** é atribuível à ontologia: o Original alargado (WIDE), o rastreio univariado e o beam calibrado igualam-no, a menor custo (`NO_ONTOLOGICAL_ADVANTAGE`, regimes P1/P2). |
| B3.3 | Mesmo para interações sem sinal univariado, com 1000 exemplos o rastreio genérico (incluindo por pares) iguala ou supera a ontologia (`B33_NO_ONTOLOGICAL_ADVANTAGE`). Dois achados que orientam a B3.4: (i) a **ontologia permutada produz o mesmo ganho sobre o Original limitado** em I2 — parte do ganho é um artefacto da pesquisa restrita por unidades, não de conhecimento; (ii) a ontologia verdadeira supera a permutada em I3–I5 (+0,014 a +0,037 de accuracy), ou seja, **a informação da partição certa tem valor, mas os dados chegam lá sozinhos quando há 1000 exemplos**. |
| Limite estrutural | O arnês só usa a **partição** em conceitos; logo, o que se pode testar é o valor de *conhecimento estrutural prévio*, não de raciocínio OWL (hierarquia, disjunção, restrições). Esta limitação é declarada e entra nas conclusões possíveis. |

O que a B3.3 **não** testou, e a B3.4 testa: o regime em que a estatística disponível no treino não chega para identificar as relações relevantes (poucos exemplos, ruído, dimensionalidade elevada), com um gerador em que a ontologia **não** coincide com o suporte da regra.

## 2. Hipótese científica B3.4 (testável e refutável)

**H1.** Se o conhecimento estrutural contido numa ontologia (que atributos medem o mesmo conceito) é verdadeiro e foi fixado independentemente dos dados e rótulos da avaliação, então, quando o número de exemplos de treino é pequeno face ao espaço de candidatos, a pesquisa m-of-n orientada por esse conhecimento (E) obtém, nas mesmas instâncias e contra o mesmo MLP, **accuracy e fidelidade simultaneamente superiores** às do Original equivalente (A), da ontologia permutada/aleatória/neutralizada (F, G, H) e de pelo menos um método genérico forte de recursos comparáveis (B, C, D, I, K).

Mecanismo proposto (a testar, não a pressupor): o conhecimento reduz o espaço de hipóteses (de ≈ 36 bits para ≈ 4,8 bits de incerteza sobre *onde* está o suporte da regra, §4) e, com isso, o problema de comparações múltiplas que degrada o rastreio estatístico com poucos dados.

**Predições refutáveis**
* P1 — o ganho (E−F) deve ser **não crescente em n** (M1 ≥ M2 ≥ M3) e desaparecer onde os métodos genéricos já recuperam a regra; um ganho maior com mais dados contradiz o mecanismo.
* P2 — a ontologia incompleta (M6) deve dar um ganho **menor ou igual** ao da completa (M2); a enganadora (N_MISLEADING) não deve dar ganho.
* P3 — no controlo nulo (M0) nenhum braço tem ganho conjunto (teste do erro de tipo I do próprio protocolo).
* P4 — E ≈ J (a mesma partição sem OWL): pré-registada como **igualdade esperada**; se se confirmar, o que se demonstrou é o valor do conhecimento prévio, **não** de semântica OWL.

**Refutação:** H1 é refutada num regime se, nesse regime, E não superar a ontologia permutada, ou não superar nenhum método genérico forte, ou se os controlos negativos mostrarem o mesmo ganho. Poucos exemplos **não** implicam vantagem; o protocolo reporta os regimes em que os genéricos continuam a igualar a ontologia.

## 3. Famílias (seleção mínima justificada)

Um único gerador, `LATENT_CONCEPT_V1`, com sete células (6 de teste + 1 nula). Não se selecionaram células porque os genéricos falhem: o eixo *n* inclui uma célula em que se espera que os genéricos recuperem (M3) e a B3.3 serve de teto (n=1000).

| Célula | Eixo | n treino | K×g (p) | ruído η | outra | Sementes | Justificação / poder informativo |
|---|---|---|---|---|---|---|---|
| M0_NULL | controlo nulo | 120 | 8×6 (48) | 0,5 (rótulos aleatórios) | — | 20 | mede falsos positivos do protocolo; barato |
| M1_FEW_60 | S1+S2 | 60 | 8×6 (48) | 0,05 | q=0,65 | 30 | regime mais escasso; dose-resposta |
| M2_FEW_120 | S1+S2 | 120 | 8×6 | 0,05 | q=0,65 | 30 | célula de referência |
| M3_FEW_240 | S1+S2 | 240 | 8×6 | 0,05 | q=0,65 | 30 | prediz-se que os genéricos recuperam (refutação de P1 se a ontologia ganhar mais aqui) |
| M4_NOISE_HIGH | S3 | 120 | 8×6 | 0,20 | q=0,65 | 30 | ruído elevado (M2 é o ruído moderado) |
| M5_HIGHDIM | S4 | 120 | 16×6 (96) | 0,05 | q=0,65 | 30 | 90 atributos irrelevantes; candidatos × 4 |
| M6_INCOMPLETE | S5 | 120 | 8×6 | 0,05 | cada indicador omitido da ontologia com prob. 0,4, independentemente da regra | 30 | ontologia verdadeira mas incompleta |

S6 (conhecimento irrelevante/enganador) não é uma célula: são **braços** de controlo negativo em todas as células (F, G, H, N_IRRELEVANT, N_INCOMPLETE, N_MISLEADING), mais M0. Cobrir S1–S6 com 7 células em vez de um produto cartesiano é deliberado (orçamento §13).

**Gerador (definição congelável).** K conceitos de g indicadores cada. Latentes z_c ~ N(0,1) por exemplo; indicador x_cj = λ_cj·z_c + sqrt(1−λ_cj²)·ε_cj, λ_cj ~ U(0,6; 0,85), escala aleatória por indicador (nomes `tema_medida`, ordem de colunas aleatória). Regra-alvo: **y = 1[z_A > Φ⁻¹(q)] XOR 1[z_B > Φ⁻¹(q)]** para dois conceitos A,B ocultos, com q=0,65 (sinal marginal fraco, S2) e ruído de rótulo η. Interação **entre conceitos**, não dentro de um conceito, e **o suporte da regra (2g atributos) é maior que um par**: a regra não está escrita na ontologia e exige identificar quais dos K conceitos importam. Oráculo: MLP (StandardScaler + MLP(32), max_iter 800) treinado nos n rótulos de treino.
Expressividade: cada conceito exige um teste m-of-n sobre os seus indicadores (a média de indicadores ruidosos aproxima z_c) e a XOR exige duas camadas (`max_depth=3`, `max_nodes=7` chegam).
Tecto do oráculo: com n pequeno o MLP é fraco. **Porta de qualidade do oráculo**: se a accuracy média do MLP nas sementes de calibração da célula for < 0,65, a célula é declarada `INSUFFICIENT_EVIDENCE` (explicar um oráculo sem sinal não mede recuperação). A porta é decidida nas sementes de calibração, nunca por semente de avaliação.

## 4. Independência do conhecimento ontológico (requisito principal)

**Classificação do estudo.** Os braços E–H e N_* recebem informação que A–D, I, K não têm (a partição dos indicadores em conceitos). A comparação E vs {A,B,C,D,I,K} é, portanto, um **estudo de benefício de conhecimento externo**, não uma comparação de algoritmos com informação igual. A comparação **com informação igual** é E vs **J** (mesma partição, lista simples, sem OWL). Todos os resultados indicam qual das duas leituras suporta.

**Quantidade e natureza da informação prévia** (calculadas): a partição de p atributos em K blocos de g contém ≈ 111,7 bits (p=48) / ≈ 302,2 bits (p=96) — informação **estrutural**. Sobre a regra, só restringe *onde* procurar: a incerteza sobre o suporte passa de log₂C(48,12) ≈ 36,0 bits (96: 49,2) para log₂C(8,2) ≈ 4,8 bits (96: 6,9). **Não revela** a identidade dos conceitos A,B, o tipo de regra (XOR), o quantil/limiar, o sentido das desigualdades, o ruído nem qualquer rótulo.

**Separação de informação**

| Categoria | Conteúdo | Disponível aos modelos? |
|---|---|---|
| Conhecimento estrutural | especificação de domínio (conceitos e indicadores) → ontologia (E–H, N_*) / lista de grupos (J) | E–H, J, N_*; **não** A–D, I, K |
| Escondido do gerador | z_c, λ_cj, conceitos A,B, q, tipo de regra, η, escalas | nunca |
| Rótulos de treino | n rótulos (com ruído η) → treino do MLP | só o MLP; o TREPAN vê o MLP |
| Rótulos de validação | não existem para os modelos; só nas sementes de calibração, para a calibração técnica | não |
| Rótulos de teste | 4000 exemplos independentes, com o mesmo processo de ruído | nunca (só métricas finais) |
| Parâmetros não observáveis | tudo o que está em «escondido do gerador» | nunca |

**Garantias verificáveis (testes pré-registados, a implementar antes da calibração)**
* T-IND-1 — o construtor de ontologia recebe **apenas** o mapa conceito→indicadores (fluxo aleatório «domínio», separado dos fluxos «regra» e «dados»); duas instâncias com o mesmo domínio e **regras diferentes** produzem **ontologias idênticas byte a byte**.
* T-IND-2 — verificação estática (AST): a função de construção não importa nem recebe rótulos, regra, quantil nem conceitos-alvo.
* T-IND-3 — a partição de J é idêntica à de E; as de F/G têm os mesmos tamanhos e o suporte da regra não fica contido num só bloco; H remove as relações dos conceitos relevantes sem alterar o resto.
* T-IND-4 — a especificação de domínio e o hash da ontologia ficam registados **antes** de gerar dados da avaliação.
* T-IND-5 — nenhum braço lê rótulos de teste nem a identidade de A,B (spy no oráculo e nos geradores, como na B3.1).
Se o conhecimento prévio fosse a regra completa, a experiência seria declarada estudo de conhecimento externo explícito; aqui não é (ver acima).

## 5. Baselines fortes (A–K)

| Braço | Descrição | Informação prévia |
|---|---|---|
| A | TREPAN Original canónico, `max_features_per_node` limitado | nenhuma |
| B | Original WIDE (`mf=p`) | nenhuma |
| C | Original com **pesquisa genérica de interações**: unidades formadas pelos pares de maior ganho de `(x_i⋚t_i)∧(x_j⋚t_j)` face ao oráculo | nenhuma |
| D | **Rastreio estatístico por pares com controlo de comparações múltiplas**: igual a C, mas só se retêm pares cujo ganho excede o quantil 95 % do máximo sob permutação das previsões do oráculo (10 permutações, só treino); sem pares significativos → comportamento do Original | nenhuma |
| E | Variante experimental orientada pela ontologia verdadeira (harness B3.1/B3.2, preferência pura) | partição em conceitos (OWL) |
| F | Ontologia permutada (mesmos tamanhos; suporte da regra repartido) | partição errada |
| G | Ontologia aleatória | partição errada |
| H | Ontologia verdadeira com as relações dos conceitos relevantes neutralizadas | parcial |
| I | Engenharia convencional: agrupamento hierárquico por \|Spearman\| | derivada dos dados |
| J | **Conhecimento prévio genérico equivalente**: a partição de E como lista simples de grupos, sem OWL, com a mesma pesquisa | partição verdadeira |
| K1/K2 | WIDE com `beam_width` calibrado (só recursos) a igualar candidatos avaliados / CPU de ajuste de E | nenhuma |
| N_IRRELEVANT, N_INCOMPLETE, N_MISLEADING | negativos: ontologia sobre outros atributos / um conceito relevante amputado / relações semanticamente erradas | nenhuma útil |

O rastreio univariado deixa de ser baseline principal (a B3.3 mostrou que não vê interações). C, D e I são os baselines estatísticos fortes; D é a resposta genérica «de manual» ao problema de poucos dados.

## 6. Controlos negativos
F, G, H, N_IRRELEVANT, N_INCOMPLETE, N_MISLEADING em todas as células; M0 (rótulos independentes dos atributos: qualquer «ganho» é falso positivo); M3 (célula onde se prevê que os genéricos recuperam, para detetar vantagem espúria da ontologia). Os controlos medem especificamente o **artefacto da pesquisa restrita por unidades** observado na B3.3-I2.

## 7. Orçamentos comparáveis (cada modalidade separada)
| Modalidade | Como se iguala | Contraste |
|---|---|---|
| Consultas ao oráculo | mesmos `min_sample`/`max_queries` em todos os braços; contagem real reportada | todos |
| Literais | `pool_parity`: mesmo orçamento de literais por pesquisa em C–J, N_* | E vs C, D, I, J, F, G, H |
| Candidatos avaliados | K1: `beam_width` de WIDE calibrado (só recursos) | E vs K1 |
| CPU de ajuste | K2: calibrado em CPU | E vs K2 |
| Custo total | CPU de ajuste + compilação OWL (reasoner) + rastreio (C, D) | reportado sempre; D inclui as permutações |
| Complexidade | mesmos `max_nodes`, `max_depth`, `max_n`; reportam-se nós, profundidade, condições m-of-n | todos |
| Memória | `tracemalloc` nas sementes 7000–7009 (A, B, E, K2) | descritivo |
Não se afirma igualdade global: cada contraste declara a única dimensão igualada. O benefício da **informação adicional** de E–H (conhecimento externo) relata-se à parte da **eficiência algorítmica**: E vs J (informação igual) mede só o processamento; E vs A–D,I,K mede conhecimento + processamento.

## 8. Métricas e critérios
Métricas: accuracy (rótulos reais; principal), fidelity (ao mesmo MLP), precision_macro, macro-F1, `rule_recovery`, nós/profundidade/condições m-of-n, candidatos avaliados, consultas ao oráculo, CPU, memória, estabilidade entre sementes, diferença face a F e face aos genéricos, estado da cadeia (Original vs C4.5, exploratório).

`rule_recovery` (primário, estrito): num mesmo caminho raiz→folha existe um teste cujos literais (≥ 2) pertencem todos a A e outro cujos literais (≥ 2) pertencem todos a B. Secundário: pelo menos um literal de A e um de B no mesmo caminho; fração dos atributos usados que pertencem a A∪B.

Contrastes: pares emparelhados por semente (diferenças médias, bootstrap 4000, IC95). Estados por célula:
* `RULE_RECOVERY_GAIN`: Δrecovery(E−A) ≥ 0,15 e IC inferior > 0.
* `ACCURACY_GAIN` / `FIDELITY_GAIN`: Δ(E−A) > 0 com IC inferior > 0 na métrica respetiva.
* `JOINT_GAIN`: ambos, em simultâneo, sobre o **Original equivalente (A)**.
* `ONTOLOGY_ATTRIBUTED_GAIN`: JOINT_GAIN **e** os 8 critérios abaixo.
* `NO_ONTOLOGICAL_ADVANTAGE`: nenhum dos anteriores com evidência suficiente.
* `INSUFFICIENT_EVIDENCE`: porta do oráculo falhada, >10 % de ajustes em falta, ou largura do IC do contraste principal > 0,08 sem significância.
**A recuperação da regra sozinha nunca declara ganho ontológico.**

## 9. Critério de sucesso por célula (exige os 8 pontos)
1. Ganho observável sobre A (E−A).
2. Accuracy **e** fidelidade em simultâneo (JOINT_GAIN).
3. E supera F **e** G em ambas as métricas (IC inferior > 0).
4. E supera, em ambas as métricas, **pelo menos um** método genérico forte de recursos comparáveis (B, C, D, I, K1, K2). Regista-se `ATTRIBUTION_STRENGTH = ALL_GENERICS` se supera todos, `SOME_GENERICS` se só alguns; só `ALL_GENERICS` autoriza dizer «melhor que as alternativas genéricas».
5. Sem vantagem espúria nos negativos: N_MISLEADING e N_IRRELEVANT não igualam o ganho de E (E−N > 0 em ambas as métricas, IC inferior > 0), e M0 não mostra JOINT_GAIN.
6. Estabilidade: nas sementes de estabilidade independentes (6000+) os contrastes E−A, E−F, E−(genérico) mantêm o sinal positivo em ambas as métricas e a taxa de vitórias de E sobre F nas 50 sementes combinadas é ≥ 0,65.
7. Custo e complexidade documentados (§7).
8. Conhecimento anterior e independente (T-IND-1..5 verdes e hash registado antes dos dados).

**Veredicto global (regime sintético):** `B34_HYPOTHESIS_SUPPORTED_IN_REGIME` se ≥ 2 das células M1–M6 têm `ONTOLOGY_ATTRIBUTED_GAIN`, M0 está limpa e P1 não é contradita; `B34_PARTIAL` com 1; `B34_NO_ONTOLOGICAL_ADVANTAGE` com 0; células `INSUFFICIENT_EVIDENCE` reportam-se e não contam a favor nem contra. Um resultado exploratório sintético **não** constitui prova de superioridade em datasets reais. Se a ontologia não superar WIDE e a pesquisa genérica, reporta-se o resultado negativo.

## 10. Política de sementes
Calibração 4000–4019 (partilhada); avaliação `5000 + 100·célula + i` (i=0..29; M0 i=0..19); estabilidade `6000 + 100·célula + i` (i=0..19, só para células que passam os critérios principais, regra fixada a priori); diagnóstico/memória 7000–7009. Disjuntas de todas as sementes anteriores (0–39, 200–239, 300–319, 400–429, 500–509, 1000–1029, 2000–2029, 3000–3019). Faixas fixadas aqui, antes de qualquer execução; nunca repetidas, substituídas nem escolhidas depois de ver resultados; ajustes que falham contam como falha reportada.

## 11. Análise estatística prevista
Diferenças emparelhadas por semente; bootstrap percentil (4000, semente fixa 34034). Decisões por **teste de interseção-união** (todos os contrastes exigidos num critério têm de passar) com **correção de Holm** sobre as 6 células de teste a α=0,05 (valores-p bootstrap unilaterais; as duas métricas e todos os contrastes exigidos devem passar). IC95 não ajustados para descrição. Taxa de vitórias (teste do sinal) para estabilidade. Sem paragem opcional, sem exclusão de sementes. Poder: a partir da B3.3 (n=1000) o desvio das diferenças emparelhadas C−D foi 0,027–0,037 (accuracy) e 0,015–0,020 (fidelidade) → diferença mínima detetável (80 % de poder, 30 sementes) ≈ 0,014–0,019 e 0,008–0,010; com n pequeno a variância será maior, daí o critério de largura de IC (`INSUFFICIENT_EVIDENCE`) em vez de alargar sementes. Verificação de poder só com diferenças **entre braços genéricos** nas sementes de calibração (sem tocar em E).

## 12. Critérios de paragem
Hipóteses, 7 células, 15 braços, métricas, sementes e critérios ficam congelados antes de executar; **sem novas famílias, braços, sementes ou critérios depois de ver resultados**; limite duro de CPU (§13); um único veredicto global; a B3.4 é o último estudo desta linha antes da revisão de estratégia: **não se propõe B3.5/B3.6**. Se `B34_NO_ONTOLOGICAL_ADVANTAGE`, recomenda-se formalmente a revisão da estratégia (§15).

## 13. Estimativa de CPU (a aprovar antes de qualquer execução)
Base: medição da B3.3 — 2,26 CPU-h para 180 conjuntos de sementes × 13 braços (≈ 40–45 s/conjunto, memória incluída). B3.4 tem 15 braços e a pesquisa por permutação do braço D (≈ +20 s a p=48; ≈ +75 s a p=96, com 10 permutações).
| Componente | CPU-h |
|---|---|
| Testes de funcionamento e T-IND | 0,1 |
| Calibração técnica única (mf, beam K1/K2, porta do oráculo) | 1,2 |
| Avaliação: M1–M4, M6 (5×30×≈70 s) | 2,9 |
| Avaliação: M5 (30×≈175 s) | 1,5 |
| Avaliação: M0 (20×≈25 s) | 0,15 |
| Estabilidade (≤ 3 células×20×≈40 s) | 0,7 |
| Memória (7×10×4 braços) | 0,4 |
| **Total estimado** | **≈ 7** |
**Limite duro proposto: 9 CPU-h** (≈ 2–2,5 h de relógio em 4 núcleos). Alternativa mais leve, se preferido: omitir M1 e M4 (−1,3 CPU-h) à custa de perder o ponto mais escasso da dose-resposta e o eixo S3. A paralelização é por célula; otimizações de D só se provadas equivalentes (comparação exata).

## 14. Riscos de enviesamento
1. **Viés de desenho pós-B3.3**: o gerador e q=0,65 foram escolhidos depois de conhecer o resultado negativo. Mitigação: pré-registo, M3/M0 como contrafactos, sem ajustes posteriores.
2. **Privilégio de informação**: a partição é informação que os genéricos não têm. Mitigação: classificação do estudo, J, T-IND-1..5, relato separado.
3. **Aderência gerador–arnês**: o gerador define conceitos como grupos de indicadores correlacionados; o arnês só usa grupos. Pode sobrestimar o valor de ontologias reais (menos limpas).
4. **Artefacto da pesquisa por unidades**: controlado por F/G/H e M0.
5. **Oráculo fraco** com n pequeno: porta do oráculo; células insuficientes não contam.
6. **Hierarquia (Nível 2)**: com n pequeno pode ser `Original > C4.5` (a favor) ou não (`HIERARCHY_INFEASIBLE`, reporta-se sem corrigir); não é critério de seleção de células.
7. **Comparações múltiplas**: Holm; interseção-união.
8. **Graus de liberdade do investigador**: calibração única, sementes por faixas, sem substituições.
9. **Ontologia mínima**: sem axiomas além da partição; resultado não generaliza a ontologias reais ricas.
10. **Custo de compilação**: incluído; a vantagem de tempo da ontologia é improvável (B3.2/B3.3).

## 15. Alternativas se o resultado for negativo
Recomendação formal (sem propor B3.5): rever a estratégia de enriquecimento da tese, privilegiando (i) **ontologias de domínio reais**, suficientemente expressivas e de **fontes independentes**; (ii) **semântica de axiomas verificável** (inferência, disjunção, restrições) em vez de partições; (iii) **revisão humana** por domínio, ontologia e metodologia (nomes por designar; pacotes e formulários de revisão preparados no Doc. 39; `wdbc_order_relations` pendente de revisão documental; `wine_phenol_subset_rejected` rejeitada como axioma universal); (iv) **representações alternativas** (ontologia como vocabulário de explicação e restrição de regras pós-hoc, não como guia da pesquisa); (v) foco em **explicabilidade e rastreabilidade** (mapeamento regra→conceito→axioma→facto), onde a B3.1 já mostrou valor mensurável independentemente da accuracy.

## 16. Ordem de trabalho se este desenho for aprovado (não autorizada ainda)
Implementar gerador e testes T-IND em `validation/` → testes de funcionamento → calibração técnica única → congelamento (hash do código, commit separado) → estimativa de CPU final → avaliação → análise e Doc. 45. Nada disto começa sem nova autorização.

## 17. Revisão humana
Mantêm-se três responsabilidades (domínio, engenharia de ontologias, metodologia). Nomes por designar; nenhuma assinatura ou aprovação foi inventada; as ontologias sintéticas da B3.4 são construídas por código e **não** são «ontologias independentes validadas» (a independência é de dados e rótulos, não uma validação humana). Nenhuma ontologia é declarada independentemente validada antes de existirem aprovações verificáveis.

---

## Histórico de versões (acrescentado; o corpo acima, v1 = commit `66ad1a9`, não foi alterado)
* **v1** (`66ad1a9`): desenho original.
* **v2 (PROPOSTA, não vigente)**: revisões R1–R7 do Doc. 48 §5 — P1/P2 passam a observações descritivas, estudo declarado H1-only, cargas λ ~ U(0,25; 0,45) com porta de recuperabilidade, nomes opacos, texto de informação substituído pela especificação do Doc. 48 §3. Só entram em vigor por decisão explícita **antes** do congelamento. Estado: `B34 = DESIGN_ONLY / NOT_EXECUTED`; manifest `frozen = false`.
* Motivo principal: a auditoria B3-Q mostrou que, com λ ~ U(0,60; 0,85), a partição de conceitos é recuperável dos dados (ARI ≥ 0,99 com n ≥ 60), pelo que o conhecimento seria redundante e H1 não seria testável (Doc. 48 §3, F4).
