# Doc. 49 — Revisão B3.4 v2 (proposta, NÃO congelada): R1–R7, porta de recuperabilidade e protocolo revisto

Estado: `B34 = DESIGN_ONLY / NOT_EXECUTED`; v2 = **proposta**; nada congelado; campanha de ≈ 7 CPU-h **não autorizada**. A v1 (`44_PROTOCOLO_B34_CONHECIMENTO_PREVIO.md`, commit `66ad1a9`) fica **integralmente preservada**; esta v2 é um documento separado e o manifest v1 não foi tocado (`B34_EXPERIMENT_MANIFEST.v2.DRAFT.json`, `frozen = false`). `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`; só `validation/`.
Evidência nova (só dados sintéticos de X/gerador, sem ontologia nem TREPAN nem braços OWL, sementes de calibração; `docs/evidence/b1b4/b3q/`): `recoverability_protocol.json`, `learnability.json`, `additive_gate.json` (scripts `b3q_recoverability_protocol.py`, `b3q_learnability.py`, `b3q_additive_gate.py`).

## 1. Problemas da v1 (cumulativos)
| # | Problema | Evidência |
|---|---|---|
| V1 | Partição recuperável dos dados com cargas λ ~ U(0,60; 0,85): o conhecimento seria redundante | melhor-de-6 métodos: ARI 0,95 (n=30), 1,00 (n ≥ 60) |
| V2 | Porta de recuperabilidade com **um** método (hierárquico por Spearman) subestima a recuperabilidade: métodos espectral/k-means/fatorial recuperam mais | λ U(0,45;0,65), n=60: hierárquico 0,69, espectral **0,85** |
| V3 | **O MLP não aprende a tarefa** (XOR de dois conceitos latentes) nos n planeados: accuracy ≈ classe maioritária, mesmo com cargas fortes | XOR: 0,58/0,54 (n=60), 0,59/0,54 (120), 0,64/0,55 (240), 0,68/0,54 (1000); com λ fraco 0,52–0,54 vs 0,54. A porta do oráculo (0,65) falharia em **todas** as células M1–M6 ⇒ v1 seria `INSUFFICIENT_EVIDENCE` por completo |
| V4 | **Tensão estrutural**: quando a estatística é escassa para recuperar a partição, também o é para o MLP aprender o conceito (as duas regiões coincidem) | tarefa «single» λ U(0,25;0,45): MLP 0,63/0,63 (n=60), 0,64/0,64 (120) |
| V5 | Nomes `tema_medida` revelam a partição lexicalmente | — |
| V6 | P1/P2 usadas como critérios de rejeição/veredicto apesar de plausíveis mas não necessárias | Doc. 44 §2, §9 |
| V7 | Texto de informação prévia mal interpretado e sem condicionar a X | Doc. 48 §3 |
| V8 | Estudo apresentado como se pudesse falar de OWL; só testa partição | Doc. 48 §2 |
| V9 | Limiares sem justificação (ARI ≤ 0,5; MLP ≥ 0,65) | — |

## 2. R1–R7 explícitas
Convenção: **Antes** = texto da v1; **Depois** = texto proposto; impactos científico / controlos / critérios de aceitação; risco metodológico.

### R1 — Previsões P1 e P2 passam a expectativas descritivas
* **Antes (Doc. 44 §2):** «P1 — o ganho (E−F) deve ser não crescente em n (M1 ≥ M2 ≥ M3) e desaparecer onde os métodos genéricos já recuperam a regra; um ganho maior com mais dados contradiz o mecanismo. P2 — a ontologia incompleta (M6) deve dar um ganho menor ou igual ao da completa (M2)…». §9: «… e P1 não é contradita».
* **Depois:** «Expecta-se, sem garantia, maior benefício com poucos exemplos e maior benefício com conhecimento mais completo e relevante. A curva ganho(n) e a comparação completa/incompleta reportam-se como **observações descritivas** com intervalos; não rejeitam H1 nem entram no veredicto. A ontologia mais completa pode juntar conhecimento verdadeiro mas irrelevante ou custoso; uma incompleta pode expor, por acaso, o suporte relevante.» «P1 não é contradita» sai do §9.
* **Justificação:** não há razão lógica para monotonia (ruído de amostragem, oráculo fraco, porta de recuperabilidade, efeito de teto dos genéricos).
* **Impacto científico:** evita refutar H1 por uma expectativa não necessária; mantém a curva como evidência descritiva.
* **Controlos:** inalterados.
* **Critérios de aceitação:** veredicto global deixa de depender de P1; permanecem os 8 critérios (ganho conjunto sobre A, E>F/G/H, ≥1 genérico forte, negativos limpos, estabilidade, custo, independência).
* **Risco:** a descrição poder ser lida seletivamente. Mitigação: reportar sempre a curva completa e todas as células, incluindo as `INSUFFICIENT_EVIDENCE`.

### R2 — Estudo declarado H1-only; H2 separado (Doc. 52)
* **Antes:** título «valor do conhecimento prévio ontológico» e discussão de OWL sem separar o que é testado.
* **Depois:** «A B3.4 testa H1 (conhecimento estrutural prévio). Não testa H2. Qualquer frase sobre axiomas, hierarquias ou inferências OWL fica proibida nos resultados. A igualdade entre OWL plana e partição sem OWL é o resultado **esperado** e não refuta H2.»
* **Justificação:** o arnês só usa a partição (Doc. 48 §2).
* **Impacto científico:** conclusões delimitadas; H2 tem protocolo próprio.
* **Controlos:** acrescenta-se o braço J (mesma partição sem OWL) como contraste obrigatório; o contraste D vs E (OWL plana vs partição) fica descritivo.
* **Critérios:** nenhum critério de H2 entra no veredicto de H1.
* **Risco:** uma leitura de «H1 sem H2» como «ontologia inútil». Mitigação: declaração explícita de limites.

### R3 — Cargas, tarefa e porta de recuperabilidade (ver §3)
* **Antes:** «cargas λ ~ U(0,60; 0,85)», «porta de qualidade do oráculo: accuracy do MLP < 0,65 ⇒ INSUFFICIENT_EVIDENCE», sem porta de recuperabilidade.
* **Depois:** (a) as cargas e a tarefa escolhem-se por **portas só de dados** (§3), pré-registadas antes de qualquer braço OWL; (b) porta de recuperabilidade multi-método com recuperação parcial; (c) porta do oráculo por **aprendibilidade** (accuracy do MLP acima da classe maioritária com margem pré-registada), não por um 0,65 fixo; (d) limiares definitivos **a decidir pelos revisores** e pré-registados; ARI ≤ 0,5 só como referência exploratória.
* **Justificação:** V1–V4, V9.
* **Impacto científico:** garante que as células medem H1 (conhecimento não redundante, oráculo informativo) em vez de ruído.
* **Controlos:** braço I (agrupamento por dados) torna-se a medida operacional de redundância; braços de screening supervisionado genérico (C, D) ganham papel na porta.
* **Critérios:** células `REDUNDANT_KNOWLEDGE` ou `ORACLE_NOT_LEARNABLE` não contam a favor nem contra.
* **Risco:** escolher λ/tarefa a olhar para o desempenho do braço OWL — **proibido**: as portas usam só X (e y de treino/validação do gerador nas sementes de calibração), nunca E/F/G/H.

### R4 — Nomes opacos
* **Antes:** indicadores `tema_medida`.
* **Depois:** indicadores `x000…x047`, rótulos opacos iguais nas entidades da ontologia (matching por nome/rótulo mantém-se, sem pista lexical sobre conceitos); o mapa conceito→indicadores vive só na especificação de domínio.
* **Justificação:** V5. **Impacto científico:** o único canal da partição é a ontologia. **Controlos:** J recebe a mesma partição sem usar nomes. **Critérios:** T-IND-2 estendido (nenhum nome contém o conceito). **Risco:** nomes opacos tornam o matching trivialmente exato; não testa robustez de matching (reservada a outros estudos).

### R5 — Texto de informação prévia
* **Antes:** «≈ 111,7 bits … reduz de ≈ 36 para ≈ 4,8 bits a incerteza sobre o suporte».
* **Depois:** especificação do Doc. 48 §3 (espaço amostral, prior uniforme, independência, I(S;P)=31,2/42,2 bits, bits ≠ custo algorítmico, informação externa líquida condicionada a X). A informação é rotulada **conhecimento externo disponibilizado às variantes**.
* **Justificação:** V7. **Impacto:** nenhum nos dados; muda a interpretação. **Controlos/critérios:** nenhum. **Risco:** nenhum material.

### R6 — Decomposição D/E/F de H2 só como estudo futuro
* **Antes:** sem decomposição.
* **Depois:** a matriz A–J com ontologia plana, partição sem OWL, expressiva, neutralizada e permutada fica no Doc. 52 e **não** é executada na B3.4.
* **Justificação:** evitar contaminar H1 com H2. **Impacto:** nenhum na B3.4. **Controlos:** os braços F–H da B3.4 mantêm-se. **Critérios:** nenhum. **Risco:** perder a oportunidade de testar H2 de graça; aceitável (H2 exige mecanismo ausente).

### R7 — Orçamento
* **Antes:** ≈ 6,9 CPU-h (limite 9). **Depois:** o mesmo, mais ≈ 0,3 CPU-h das portas (recuperabilidade multi-método e aprendibilidade já medidas: < 0,1 CPU-h na B3-Q). Nova estimativa na proposta §5.
* **Justificação:** transparência. **Impacto/controlos/critérios:** nenhum. **Risco:** subestimar CPU de métodos espectrais em p maior (M5); monitorizado.

### R8 (acréscimo proposto, fora das sete) — Família de tarefa aditiva por conceito
* **Antes:** regra XOR de dois conceitos latentes (V3).
* **Depois:** (proposta, a decidir) regra **aditiva por conceito**: `y = 1[Σ_{j∈A} x_j + s·ε > 0]` com indicadores independentes — um limiar de contagem ponderada, o alvo natural de um m-of-n sobre os indicadores de um conceito A oculto.
* **Justificação (dados, §3.3):** é o único gerador testado em que o MLP aprende (0,63–0,66 vs 0,51 a n=60–120, s=2) enquanto a identificação supervisionada dos 6 indicadores por métodos genéricos é **incompleta** (57–85 %) e a recuperabilidade não supervisionada é **nula por construção**.
* **Impacto:** troca a interação (XOR) por agregação; perde-se o paralelo com a B3.3. **Controlos:** os mesmos. **Critérios:** idem. **Risco:** o gerador favorece «unidades» por construção; justificado como estudo de benefício de conhecimento externo (H1), não de algoritmo.

## 3. Porta de recuperabilidade (protocolo)
Princípios: só `X` de treino para a porta de recuperabilidade; `y` de treino/validação **do gerador** só nas sementes de calibração para a aprendibilidade; nenhum rótulo/teste da avaliação; nenhum braço OWL.

### 3.1 Recuperabilidade não supervisionada (8 verificações)
1. **Só X:** `b3q_recoverability_protocol.py` (labels_used = false, test_data_used = false).
2. **Variabilidade entre sementes:** p10/mediana/p90 por método (40 réplicas).
3. **Tamanhos de treino:** n ∈ {30, 60, 120, 240, 1000}.
4. **Recuperação parcial dos conceitos:** pureza média por conceito (fração dos indicadores de um conceito no cluster que contém mais deles).
5. **Facilidade de inferir a organização sem ontologia:** melhor-de-todos os métodos (o mais favorável aos dados ⇒ o mais conservador para o conhecimento externo).
6. **Sensibilidade ao método:** hierárquico (Spearman/Pearson), espectral, k-means sobre o perfil de correlação, análise fatorial (varimax), hierárquico com K estimado por silhueta; para atributos categóricos usar associação adequada (V de Cramér / informação mútua) — **por implementar**, não necessário no gerador numérico.
7. **Capacidade de aprender a tarefa mesmo sem partição:** §3.2–3.3.
8. **Sem consulta aos rótulos de teste:** garantido pelo desenho (testes `tests/test_b3q_audit.py`).

| λ \ n | 30 | 60 | 120 | 240 | 1000 |
|---|---|---|---|---|---|
| U(0,60; 0,85) | 0,95 / 0,98 | 1,00 | 1,00 | 1,00 | 1,00 |
| U(0,45; 0,65) | 0,36 / 0,67 | 0,85 / 0,94 | 1,00 | 1,00 | 1,00 |
| U(0,35; 0,55) | 0,13 / 0,50 | 0,36 / 0,67 | 0,79 / 0,91 | 1,00 | 1,00 |
| U(0,25; 0,45) | 0,05 / 0,42 | 0,13 / 0,47 | 0,23 / 0,58 | 0,58 / 0,79 | 1,00 |
(mediana do ARI do melhor método / pureza mediana por conceito; K=8, g=6; sensibilidade ao método: a diferença entre o melhor e o pior método é de 0.00–0.19 em ARI mediano.) A pureza parcial (0,42–0,58 com ARI baixo) mostra que **parte** de cada conceito é recuperável mesmo quando o ARI global é baixo — a porta deve usar os dois.

### 3.2 Aprendibilidade da tarefa pelo MLP (accuracy vs maioritária, mediana de 15 réplicas)
| Tarefa | λ \ n | 60 | 120 | 240 | 1000 |
|---|---|---|---|---|---|
| single (z_A) | U(0,60;0,85) | 0,74/0,64 | 0,77/0,63 | 0,79/0,64 | 0,79/0,64 |
|  | U(0,45;0,65) | 0,70/0,63 | 0,71/0,64 | 0,73/0,63 | 0,74/0,63 |
|  | U(0,35;0,55) | 0,65/0,63 | 0,68/0,63 | 0,70/0,64 | 0,70/0,64 |
|  | U(0,25;0,45) | 0,63/0,63 | 0,64/0,64 | 0,66/0,64 | 0,66/0,63 |
| and (A∧B) | todos | 0,78–0,84 em **todos** os n (maioritária 0,84) | | | **nunca acima da maioritária** |
| xor (A⊕B) | U(0,60;0,85) | 0,58/0,54 | 0,59/0,54 | 0,64/0,55 | 0,68/0,54 |
|  | U(0,25;0,45) | 0,52/0,54 | 0,52/0,54 | 0,53/0,54 | 0,54/0,54 |
**V3/V4:** XOR e AND não são aprendidos; «single» só o é com cargas fortes. Nas condições em que a partição não é recuperável (λ fraco, n ≤ 120) o MLP está ao nível da maioritária. As duas condições do H1 (partição não recuperável **e** oráculo informativo) quase não coincidem neste gerador.

### 3.3 Alternativa aditiva por conceito (R8): identificabilidade supervisionada e aprendibilidade
`y = 1[Σ_{j∈A} x_j + s·ε > 0]`, K=8, g=6, indicadores independentes (partição **irrecuperável de X por construção**).
| s \ n | 60 | 120 | 240 | 1000 |
|---|---|---|---|---|
| 1,0 — MLP / maioritária | 0,66 / 0,51 | 0,72 | 0,80 | 0,84 |
| 1,0 — acertos nos 6 melhores (corr / L1) | 0,69 / 0,82 | 0,89 / 0,99 | 0,99 / 1,00 | 1,00 |
| **2,0** — MLP | **0,63** | **0,66** | 0,70 | 0,72 |
| **2,0** — acertos (corr / L1) | **0,57 / 0,60** | **0,78 / 0,85** | 0,92 / 0,95 | 1,00 |
| 3,0 — MLP | 0,58 | 0,61 | 0,64 | 0,65 |
| 3,0 — acertos (corr / L1) | 0,47 / 0,45 | 0,62 / 0,66 | 0,83 / 0,87 | 1,00 |
Maioritária ≈ 0,51 em todas. Em s=2 e n=60–120 existe a **janela** pretendida: o MLP está claramente acima da maioritária (+0,12 a +0,15) e a identificação supervisionada genérica é parcial (57–85 %); em n ≥ 240 os genéricos já identificam quase tudo (conhecimento redundante) — dose-resposta natural.

### 3.4 Regra de decisão (a pré-registar; limiares definitivos pelos revisores)
Uma célula é **informativa para H1** se, nas sementes de calibração: (G1) o oráculo é **aprendível** — accuracy mediana do MLP − classe maioritária ≥ Δ_o; (G2) o conhecimento é **não redundante** — melhor-de-métodos de recuperação não supervisionada (ARI mediano e pureza) ≤ τ_u **e** acertos supervisionados genéricos (melhor de corr/L1) ≤ τ_s. Valores **exploratórios de referência** (não validados): Δ_o = 0,05; τ_u: ARI ≤ 0,5 (com sensibilidade 0,3 / 0,5 / 0,7); τ_s = 0,9. Sensibilidade reportada para todos; o corte definitivo e a decisão sobre células `REDUNDANT_KNOWLEDGE` / `ORACLE_NOT_LEARNABLE` ficam para os revisores, **antes** de qualquer braço OWL.
**Regra de escolha de cargas/ruído (pré-especificada, sem OWL):** entre as configurações candidatas (as quatro faixas de λ e s ∈ {1, 2, 3, 4} nas duas famílias), escolher a que torna informativas pelo menos duas células do eixo n e deixa pelo menos uma célula de controlo redundante (para observar a transição), pelos critérios G1–G2; empates desfazem-se pela menor complexidade (menor s / maior λ). Com as referências exploratórias, a família aditiva com s=2 satisfaz G1–G2 em n=60 e 120 e viola G2 em n=240 e 1000; a família «single» só satisfaz G1 com cargas moderadas/fortes e aí a partição é recuperável (viola G2); o XOR só satisfaz G1 com cargas fortes e n ≥ 240 (viola G2); o AND nunca satisfaz G1. Logo, **só a família aditiva tem uma janela G1∧G2**. **Isto não é uma escolha pelo desempenho do braço OWL** (nenhum foi executado).

## 4. Protocolo v2 proposto (resumo; o resto como na v1)
* **Hipótese:** H1 apenas (R2). **Expectativas descritivas** (R1). Refutação de H1 num regime: E não supera F e G, ou nenhum genérico forte de recursos comparáveis, ou os negativos mostram o mesmo ganho; células fora das portas = `INSUFFICIENT_EVIDENCE`/`REDUNDANT_KNOWLEDGE`.
* **Gerador v2:** família a decidir entre as duas (aditiva por conceito recomendada, §3.3), nomes opacos (R4), fluxos aleatórios de domínio/regra/dados separados, T-IND-1..5 inalterados.
* **Células (propostas, a decidir após as portas):** n ∈ {60, 120, 240} (dose-resposta), ruído s ∈ {2, 3}, p=96 e ontologia incompleta como na v1; controlo nulo M0.
* **Braços:** como v1 (A, B, C, D, E–H, I, J, K1/K2, N\_\*), com E = ontologia, J = mesma partição sem OWL.
* **Critérios, estatística, sementes, paragem:** como v1 (Holm, interseção-união, 30 sementes, faixas disjuntas), exceto R1.
* **Orçamento v2:** ≈ 7 CPU-h (limite 9) + ≈ 0,3 h das portas; **não autorizado**.

## 5. Decisões necessárias antes de congelar a v2
1. Aprovar R1–R7 e decidir R8 (família aditiva vs XOR). 2. Fixar os limiares definitivos Δ_o, τ_u, τ_s (revisores). 3. Designar os revisores. 4. Decidir se, dado V4, o estudo H1 com oráculo treinado nos mesmos n exemplos vale a pena, ou se se prefere enquadrar H1 apenas como validação do mecanismo em sintético (Doc. 52 fase P) e deslocar o esforço para H2 em dados reais.
