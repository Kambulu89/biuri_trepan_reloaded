# Doc. 57 — Proposta técnica dos limiares das portas (Δ_o, τ_s, τ_u) e adequação do oráculo (G1b)

Estado: **PROPOSTA para revisão do responsável pela metodologia científica; nada está congelado; nenhum limiar foi escolhido pelo desempenho de qualquer ontologia ou braço OWL** (nenhum foi executado). Os limiares serão **congelados antes da avaliação das variantes**. Calibração **independente** (sementes novas), só gerador + MLP + métodos genéricos: `docs/evidence/b1b4/b3q/threshold_calibration.json` (sementes base 9500, 30 réplicas/célula), `oracle_adequacy.json` (base 9700, 15 réplicas), `recoverability_protocol.json` (base 6161). Exploração anterior (sementes 9191, `additive_gate.json`) é **independente e não reutilizada** para calibrar. Sem testes da avaliação; sem rótulos de teste.
Contexto: família aditiva R8 (exploratória; Doc. 49); a recuperabilidade não supervisionada (τ_u) só é informativa nas famílias com indicadores correlacionados.

## 1. Estatísticas e definições matemáticas
Seja uma célula c = (família, ruído s, n) e r = 1…R as réplicas de calibração (R = 30). Para cada réplica, o gerador produz treino (n) e validação nova (2000); o MLP (o mesmo do protocolo) treina no treino.
* **D_r** = Acc_MLP,r(val) − max_k P̂_r(y=k | val). **D_o^{(q10)}(c)** = quantil 10 % de {D_r}. **Gate G1 (oráculo aprendível):** D_o^{(q10)}(c) ≥ Δ_o.
* **R_r(g)** = |{indicadores verdadeiros} ∩ topo-g do ranking| / g, para dois estimadores supervisionados genéricos (|corr(x_j, y)| e regressão logística L1, C=0,3), usando só (X, y) de treino. **H_s(c) = média_r max(R_r^corr, R_r^L1)**. **Gate G2s (conhecimento não redundante face ao rastreio supervisionado):** H_s(c) ≤ τ_s.
* **U(c) = mediana_r max_{m∈M} ARI(partição verdadeira, partição_m(X_treino))** sobre os métodos não supervisionados M (Doc. 49 §3.1). **Gate G2u:** U(c) ≤ τ_u.
* **Gate G1b (adequação estrutural do oráculo):** quota da importância por permutação que cai nos indicadores verdadeiros, Q_r = Σ_{j∈A} imp_j / Σ_j imp_j (importâncias negativas truncadas a 0). **G1b:** quantil 10 % de {Q_r} ≥ κ.
Célula **informativa para H1/R8** ⇔ G1 ∧ G1b ∧ G2s ∧ G2u.

## 2. Finalidade, interpretação e erros
| Limiar | Finalidade | Interpretação | Falso positivo (declarar informativa quando não é) | Falso negativo (descartar uma informativa) |
|---|---|---|---|---|
| Δ_o | o oráculo tem de saber mais do que a classe maioritária; senão a fidelidade é a de ruído | em ≥ 90 % das réplicas o MLP supera a maioritária em ≥ Δ_o | extrair/explicar um oráculo sem sinal; resultados de ganho/perda sem sentido | perder poder: descartar células onde o oráculo é ligeiramente informativo |
| κ (G1b) | «acima da maioritária» não implica ter aprendido a estrutura certa | o MLP atribui ≥ κ da importância aos indicadores verdadeiros (acaso = g/p = 0,125) | oráculo informativo por razões erradas (atalhos) | rejeitar oráculos adequados mas com importância difusa |
| τ_s | o conhecimento externo só vale se métodos genéricos supervisionados não o recuperam | os genéricos identificam, em média, ≤ τ_s dos indicadores relevantes | tratar como «não redundante» uma célula onde um estimador melhor (não incluído) identificaria tudo ⇒ vantagem espúria da ontologia | descartar células onde a ontologia ainda orientaria a pesquisa m-of-n (p. ex. agregação em unidades) |
| τ_u | o conhecimento externo só vale se a partição não for recuperável sem rótulos | os melhores métodos não supervisionados recuperam ARI ≤ τ_u | idem, com um agrupamento melhor que os testados | idem |
Assimetria declarada: um falso positivo **infla** a hipótese a favor da ontologia (risco de enviesamento); por isso, na dúvida, os cortes devem ser conservadores (mais exigentes) e a lista de estimadores genéricos deve ser forte.

## 3. Resultados da calibração independente
### 3.1 G1 e G1b (R = 30 e 15 réplicas)
| s \ n | 60 | 120 | 240 | 1000 |
|---|---|---|---|---|
| s=1.0: D̄_o / SD / q10 | 0.159 / 0.022 / 0.128 | 0.210 / 0.025 / 0.172 | 0.286 / 0.018 / 0.267 | 0.329 / 0.011 / 0.314 |
| s=2.0: D̄_o / SD / q10 | 0.108 / 0.021 / 0.089 | 0.150 / 0.019 / 0.123 | 0.196 / 0.015 / 0.174 | 0.210 / 0.011 / 0.197 |
| s=3.0: D̄_o / SD / q10 | 0.076 / 0.022 / 0.040 | 0.102 / 0.021 / 0.074 | 0.141 / 0.017 / 0.125 | 0.131 / 0.014 / 0.116 |
| s=4.0: D̄_o / SD / q10 | 0.055 / 0.019 / 0.033 | 0.067 / 0.024 / 0.037 | 0.094 / 0.018 / 0.071 | 0.088 / 0.016 / 0.070 |

| s \ n | 60 | 120 | 240 | 1000 |
|---|---|---|---|---|
| s=1.0: Q mediana (q10) | 0.73 (0.57) | 0.84 (0.69) | 0.89 (0.80) | 0.89 (0.84) |
| s=2.0: Q mediana (q10) | 0.77 (0.59) | 0.74 (0.61) | 0.84 (0.73) | 0.76 (0.64) |
| s=3.0: Q mediana (q10) | 0.56 (0.45) | 0.65 (0.53) | 0.73 (0.57) | 0.71 (0.60) |

Observações: (i) em todas as células a fração de réplicas com D_r > 0 é 1,00; o desvio-padrão de D_r entre réplicas é 0,011–0,025 (erro-padrão da média ≈ 0,002–0,005); (ii) a quota de importância nos indicadores verdadeiros tem mediana 0,56–0,89 (q10 ≥ 0,44) contra 0,125 de acaso, mesmo nas células com D_o pequeno — **neste gerador**, quando o MLP está acima da maioritária, está-o pelas razões certas.
### 3.2 G2s
| s \ n | 60 | 120 | 240 | 1000 |
|---|---|---|---|---|
| s=1.0: H_s (EP) | 0.81 (0.021) | 0.99 (0.008) | 1.00 (0.000) | 1.00 (0.000) |
| s=2.0: H_s (EP) | 0.63 (0.026) | 0.86 (0.020) | 1.00 (0.000) | 1.00 (0.000) |
| s=3.0: H_s (EP) | 0.50 (0.028) | 0.73 (0.026) | 0.90 (0.021) | 1.00 (0.000) |
| s=4.0: H_s (EP) | 0.40 (0.026) | 0.54 (0.036) | 0.78 (0.023) | 1.00 (0.000) |

(H_s é a média do máximo entre os dois estimadores por réplica, logo ≥ os valores por estimador da exploração anterior.)
### 3.3 G2u
Na família aditiva os indicadores são independentes: o ARI mediano do melhor método é ≈ 0 em todas as 16 células (−0,02 a 0,01), pelo que G2u passa para qualquer τ_u ≥ 0,1 (trivial por construção). Para famílias com indicadores correlacionados (gerador latente), o ARI do melhor método (Doc. 49 §3.1): λ~U(0,25;0,45): 0,05 / 0,13 / 0,23 / 0,58 / 1,00 em n=30/60/120/240/1000; λ~U(0,35;0,55): 0,13 / 0,36 / 0,79 / 1,00 / 1,00; λ~U(0,45;0,65): 0,36 / 0,85 / 1,00 / 1,00 / 1,00.

## 4. Valores candidatos e sensibilidade
Células informativas (G1 ∧ G2s; G2u passa em todas; **G1b só foi medido para s ≤ 3** — as células com s=4 não têm evidência de G1b) segundo a combinação de limiares:
| Δ_o \ τ_s | 0,7 | 0,8 | 0,9 |
|---|---|---|---|
| 0.02 | 4: s=2.0 n=60, s=3.0 n=60, s=4.0 n=60, s=4.0 n=120 | 6: s=2.0 n=60, s=3.0 n=60, s=3.0 n=120, s=4.0 n=60, s=4.0 n=120, s=4.0 n=240 | 9: s=1.0 n=60, s=2.0 n=60, s=2.0 n=120, s=3.0 n=60, s=3.0 n=120, s=3.0 n=240, s=4.0 n=60, s=4.0 n=120, s=4.0 n=240 |
| 0.05 | 1: s=2.0 n=60 | 3: s=2.0 n=60, s=3.0 n=120, s=4.0 n=240 | 6: s=1.0 n=60, s=2.0 n=60, s=2.0 n=120, s=3.0 n=120, s=3.0 n=240, s=4.0 n=240 |
| 0.1 | 0 | 0 | 3: s=1.0 n=60, s=2.0 n=120, s=3.0 n=240 |

Leitura: o resultado depende fortemente dos limiares (de 0 a 9 células informativas). Isso é uma razão para os **congelar antes** e para **reportar a tabela completa** em vez de um único corte.

### 4.1 Candidatos e critérios de decisão (a decidir pelo responsável pela metodologia)
* **Δ_o ∈ {0,02; 0,05; 0,10}.** Critério proposto: Δ_o = duas vezes o desvio-padrão máximo entre réplicas de D_r, arredondado (2 × 0,025 ≈ 0,05): uma margem que o ruído do próprio estimador não produz. 0,02 ≈ um desvio-padrão (permissivo); 0,10 elimina quase todas as células de ruído moderado/alto (0–3 células). **Candidato: 0,05.**
* **κ ∈ {0,25; 0,375; 0,50}** (2×, 3×, 4× o acaso 0,125). Todas as células medidas (s ≤ 3) passam κ = 0,375 (mínimo do q10: 0,446); para s=4 falta medir. **Candidato: 0,375.**
* **τ_s ∈ {0,7; 0,8; 0,9}.** Critério: (1 − τ_s) é a fração mínima dos indicadores relevantes que os genéricos deixam por identificar; o erro-padrão de H_s é 0,02–0,036, logo as diferenças entre 0,8 e 0,9 são ≈ 3–5 erros-padrão. Dado o risco de falso positivo (estimadores genéricos fracos), exige-se também **adicionar estimadores mais fortes** (seleção por estabilidade, informação mútua) antes de congelar; sem isso, o candidato conservador é **0,8**.
* **τ_u ∈ {0,3; 0,5; 0,7}.** Critério: ARI ≤ 0,5 como referência exploratória; trivial na família aditiva; relevante só para famílias latentes (cruzamento 0,5 ≈ n=240 em λ~U(0,25;0,45)). **Candidato: 0,5, com sensibilidade 0,3/0,7.**
**Regras de congelação:** (1) definições e valores congelados em commit próprio antes de qualquer braço OWL; (2) reportar todas as células sob todos os candidatos; (3) proibido ajustar limiares, estimadores ou cargas depois de ver o desempenho de E/F/G/H; (4) o responsável pela metodologia pode rejeitar os candidatos; (5) uma célula que falhe qualquer porta é `INSUFFICIENT_EVIDENCE`/`REDUNDANT_KNOWLEDGE`, nunca «resultado negativo da ontologia».

## 5. Limitações
Um único tipo de oráculo (MLP) e de gerador; dois estimadores supervisionados; réplicas de calibração limitadas (30/15); a adequação G1b mede atribuição de importância, não correção funcional completa; os valores candidatos não estão validados por revisão humana.
