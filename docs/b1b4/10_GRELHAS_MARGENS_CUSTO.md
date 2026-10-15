# Documento 10 — Grelhas, margens, seeds e custo computacional (Fase 0.5) — D-8 e estimativa de custo

Tudo aqui é **proposta fundamentada para congelar**; nada foi corrigido a posteriori com resultados confirmatórios. As medições de custo vêm de **2 unidades DEV** (iris e breast_cancer, seed 42) com o protocolo congelado v1 — não são resultados científicos.

## 1. Definição matemática do λ (B1-B) e revisão da grelha `[0, 0.05, 0.10, 0.20, 0.40, 0.80]`
Notação por nó: `s_i` = componente estatística do candidato *i* (IG normalizado, ∈[0,1]); `m_i` = contributo semântico líquido, definido para que o score atual seja exatamente `s_i + m_i`:
`m_i = α·OntoDepth_i + β·ErrorCoverage_i + s_i·(peso_i^k − 1)` (o fator multiplicativo do peso e do grupo é reescrito de forma aditiva). Resolve a mistura de escalas identificada na auditoria (Doc. 1 §3.1).
**B1-B:** `score_i = s_i + clip(m_i, −λ·s_i, +λ·s_i)`, com `λ` **adimensional** (fração de `s_i`), distinto do `λ_legacy` do runner (que multiplica α, β, expoente e força de grupo). Para evitar confusão: `lambda_cap` (B1-B) vs `lambda_legacy`.
Propriedades (testáveis): (i) `λ=0` ⇒ ranking puramente estatístico; (ii) monotonia: aumentar λ só alarga a influência semântica; (iii) `s_i ≤ 1e-12 ⇒ score_i=0` (mantém-se); (iv) um candidato `i` só vence o melhor estatístico `j` se `s_i/s_j > (1−λ)/(1+λ)`, logo o **desvio estatístico máximo** por nó é `2λ/(1+λ)`:
| λ | 0 | 0,05 | 0,10 | 0,20 | 0,40 | 0,80 |
|---|---|---|---|---|---|---|
| desvio estatístico máx. | 0 % | 9,5 % | 18,2 % | 33,3 % | 57,1 % | 88,9 % |
**Revisão:** (a) a escala é adequada — progressão ~geométrica (×2) em λ, uniforme em log, com `λ=0` como controlo interno; (b) `λ=0,80` aproxima a magnitude do comportamento atual (razão mediana termo semântico/estatístico = 0,83 no audit, Doc. 1), logo mantém um membro «≈ legacy» na grelha;
(c) **atenção:** com `λ = 0,40` um candidato com apenas > 43 % do ganho estatístico do melhor já pode vencer (até 57 % de perda por nó; com `λ = 0,80`, > 11 %) — o gate B2 é a salvaguarda; o desvio efetivamente usado é medido e reportado por nó (`regret_i = (s_j − s_i)/s_j`);
(d) o `λ=0` com ontologia **não** é o Original (a representação enriquecida existe): é o ponto de referência «ranking estatístico na representação OWL» — deve igualar, bit a bit, **Original na representação OWL** quando B2, B4, EFSR e projetor estão desligados (propriedade `EQ-λ0`, a testar na Fase 1).
**Proposta a congelar:** manter exatamente a tua grelha `[0, 0.05, 0.10, 0.20, 0.40, 0.80]` para `lambda_cap`. Grelha de sensibilidade do `lambda_legacy` (R4): `[0, 0.5, 1, 2.5]` (a do runner, `semantic_lambdas`).
Outras constantes **fixas, não afinadas** (declaradas antes de correr): B1-A `τ = 1·SE` (regra 1-SE) com `τ ≥ 1/n_eff`; B1-C peso `w=0,5` em ranks; B2 confiança unilateral 95 % (`z=1,645`); `K=3` dobras internas; `bootstrap B=200` quando usado.
Custo da grelha por seleção: candidatos `|G| = 1 (B1-A) + 6 (B1-B) + 1 (B1-C) = 8` (λ=0 conta como candidato «sem bónus»; duplicados bit-a-bit são memoizados).

## 2. Margens (D-8) — interpretação, análise e proposta
Adotadas como propostas por ti: `delta_accuracy = 0` (estrito), `delta_fidelity = 0` (estrito), `nu_fidelity = 0` (hipótese principal), cinco seeds DEV, dez seeds CONFIRM.
Interpretação a **confirmar**:
* `δ_A=0` e `δ_F=0` definem o veredito de **superioridade** (`>` estrito, valores não arredondados). `ν_F=0` é a restrição de **não-inferioridade** na seleção: `F_R ≥ F_O` (não estrita; um empate exato não exclui — senão o `λ=0` seria sempre excluído).
* **Granularidade e empates:** num teste de `n_test` amostras a accuracy move-se em passos de `1/n_test` (iris com 25 % de teste: `n_test=38` ⇒ 2,6 pp); empates por unidade são frequentes e contam como `NOT_SATISFIED` ao nível da unidade. Ao nível de dataset/portfólio usa-se a diferença média pareada, não contagens de vitórias; reporta-se a **fração de empates**.
* **Ruído da restrição `ν_F=0`:** a diferença de fidelidade OOF com K=3 dobras tem erro-padrão da ordem de 1–3 pp (ordem observada: DP por unidade 2,5–2,9 pp); uma configuração com `ΔF` verdadeiro 0 passa a restrição ~50 % das vezes. Consequência: a restrição é um **filtro ruidoso**, não uma garantia; pode haver `no_config_validated`. Mitigação sem relaxar ν: guardar as métricas OOF de **todos** os candidatos (`SelectionRecord`) para permitir **análise de sensibilidade offline** de ν ∈ {0; 0,005; 0,01; 0,02} (repetição da regra de seleção sem retreinar), reportada como exploratória, **nunca** usada para declarar superioridade de fidelidade (cumpre a tua restrição).
* Distinção de vereditos (Doc. 5): *observada* vs *estatisticamente sustentada*; *hierarquia não satisfeita*; *inviável pelos baselines* (`A_O ≤ A_C`); *evidência insuficiente*.

## 3. Seeds
* **DEV (5):** `[42, 7, 123, 2024, 11]` (as do manifest v1; já usadas, ficam em desenvolvimento).
* **CONFIRM (10):** definidas por regra determinística independente de resultados: `seed_i = int(SHA-256("BIURI-CONFIRM-v2/i"), 16) mod (2³¹−1)`, `i=1..10` ⇒ `[742495737, 722389263, 45964685, 307577131, 1546787794, 9826868, 550199366, 1548718903, 496223488, 514059117]` (distintas entre si e das seeds de desenvolvimento). Cada seed gera um *holdout* estratificado de 25 % (esquema do runner v1), pareado entre braços.
* As 10 seeds de um dataset **não** são independentes (partilham dados); a inferência entre datasets usa a unidade = dataset (Doc. 2 §7).

## 4. Custo computacional (medido em DEV, extrapolado com cautela)
**Medições (protocolo v1, seed 42, CPU de utilizador):**
| | iris (n_treino 112, p 4) | breast_cancer (n_treino 426, p 30) |
|---|---|---|
| unidade completa (braços A–F) | 40,2 s | 334 s |
| tuning estrutural ontology-blind (partilhado) | 31,7 s (79 %) | 263 s (79 %) |
| Original (1 árvore) | 0,14 s | 2,5 s |
| Reloaded Core (sem ontologia) | 0,38 s | 3,4 s |
| Reloaded + OWL real | 4,0 s | 46,4 s |
| Reloaded + OWL permutada | 2,5 s | 16,4 s |
| consultas ao oráculo: Original / Reloaded Core / Reloaded OWL | 9 896 / 20 272 / 20 592 | 15 054 / 32 426 / 35 137 |
A última linha **quantifica a V1 no protocolo real**: sem ontologia o Reloaded consulta o oráculo ≈ **2,05× (iris)** e **2,15× (breast_cancer)** o que o Original consulta, para a mesma árvore. A correção da Fase 1a repõe a razão em 1,0.
Expoentes observados iris→breast_cancer: tuning `n^1,59`, Reloaded+OWL `n^1,84` (confundidos com p e nº de nós; **duas medições**: indicativo, não lei).

**Modelo de custo do Estágio 1 (Doc. 4):** por unidade `T = T_tune + 3·T_R + 5·(K·|G|·T_R,interno + T_R)`, com 5 braços a selecionar (R6, R7, R6~perm, R7~rand, R7~conv), `K=3`, `|G|=8` (⇒ ~125 ajustes Reloaded por unidade), `T_R,interno` à escala de `2/3·n_treino`. CPU-horas para 10 seeds (um núcleo; dividir pelo nº de núcleos):
| Cenário | otimista (linear em n) | pessimista (expoentes observados) |
|---|---|---|
| A. proposta (G=8, 5 braços a selecionar) — 10 elegíveis | **≈ 1 990 h** | ≈ 33 400 h |
| A, sem Bank Marketing | ≈ 1 030 h | ≈ 7 400 h |
| A, só 6 datasets com n ≤ 5 000 | ≈ 300 h | ≈ 900 h |
| B. G=4, só R6 e R7 selecionam (controlos herdam a config) | ≈ 570 h | ≈ 9 200 h |
| D. sem seleção (B3 fora) | ≈ 180 h | ≈ 2 500 h |
(`Bank Marketing` ≈ 960 h/45 211 linhas no cenário A otimista; `Dry Bean`/`Shoppers` ≈ 260–290 h cada.) Os valores ignoram MLP, raciocínio OWL (1–2 s) e I/O; são ordens de grandeza.
**Conclusão:** o plano completo, tal como proposto, **não é exequível** em ambiente de nuvem com 4 núcleos para os datasets maiores. Isto é um achado de planeamento, não um resultado científico.

**Opções (decisão D-14, tua):**
1. **Teto global de dimensão** (regra EC3, dataset-agnóstica, decidida já): `n ≤ 5 000` (ou outro limiar) — retira Shoppers, Dry Bean, Mushroom, Bank; obriga a repor com reservas pequenas e a rever EC9 (categóricos: só Car restaria entre os primários).
2. **Reduzir seleção:** controlos herdam a configuração do tratamento (perde-se «igual oportunidade de seleção»; declarar), `G=4`.
3. **Mais computação** (máquina dedicada/cluster); o runner já suporta retoma (`tuning_checkpoint`) e `--max-wall-time-per-unit` (política global, não altera a ciência).
4. **Aceleração exata** (vetorização/memoização verificadas por `equivalence_report`), sem mudar resultados — recomendada em qualquer caso (79 % do tempo é tuning; 125 ajustes dominam a seleção).
5. **Faseamento:** Estágio 1 só com datasets pequenos; grandes só no Estágio 3 se houver orçamento.
Recomendação: (1)+(4) com limiar a fixar por ti, e seleção completa só para R6/R7 com controlos a re-selecionar apenas nos datasets de menor custo.

## 5. O que fica por congelar (e depende de ti)
D-8 (confirmar a interpretação de §2 e a grelha de §1) · D-14 (orçamento/teto de dimensão, §4) · D-13 (deriva de hashes do Original, Doc. 12). Nada disto foi ajustado com resultados confirmatórios.
