# Documento 17 — B1: pré-registo da análise (escrito ANTES de ver os resultados B1) e resultados

## 0. Pré-registo (data de escrita anterior à análise dos resultados das variantes B1; commit registado)
**Âmbito:** só datasets DEV (iris, wine, breast_cancer × seeds {42, 7, 123, 2024, 11}; digits × {42, 7, 123}); sem datasets confirmatórios; protocolo v1, MLP/FrozenOracle, ontologias e tuning estrutural ontology-blind inalterados; ambiente numérico fixado; referência pré-B1 congelada (`docs/evidence/b1b4/pre_b1_reference/`, código `db417ff`).
**Variantes:** `tiebreak` (τ = 1/√n_real); `bounded` e `calibrated` com `lambda_cap ∈ {0, 0.05, 0.10, 0.20, 0.40, 0.80}` (grelha D-8); cada variante com o seu **controlo negativo** (ontologia permutada, mesma calibração). Referência da comparação: (i) **Original** (braço C) — objetivo da hierarquia; (ii) **Reloaded+OWL histórico** (braço E) — para saber se B1 melhora o que já existe; (iii) o controlo permutado da própria variante — ganho atribuível ao *conhecimento* e não à calibração.
**Métricas (por unidade; relatadas por dataset e agregadas por dataset-média; unidade de inferência = dataset):** `Fidelity` ao FrozenOracle e `Accuracy` com rótulos reais no teste do split (relato; nunca para seleção), nós/folhas (complexidade), amostras totais ao oráculo e custo extra face ao Original, `b1_switch_rate`, `b1_regret_median`, `b1_ratio_m_over_s_median`, estado semântico. Margens: `δ_A = δ_F = 0` estritos; **distinguir** `F_R ≥ F_O` (não-inferioridade *observada*) de `F_R > F_O` (superioridade *observada*); empates contam como não-vitórias; reportar fração de empates. Todos os níveis de evidência são `observed_only` (sem inferência estatística formal com N = 4 datasets).
**Critérios de uma variante «a levar para B2» (todos obrigatórios; avaliados nos agregados por dataset, nunca num dataset isolado):**
1. **Não-inferioridade de fidelidade ao Original**: `mean ΔF(variante − Original) ≥ 0` em ≥ 3/4 datasets (se nenhuma o satisfaz, regista-se `NOT_ACHIEVED_ON_DEV`, sem ajustar critérios).
2. **Melhora o histórico**: `mean ΔF(variante − E) > 0` e `mean ΔAcc(variante − E) ≥ 0` em ≥ 3/4 datasets.
3. **Ganho atribuível ao conhecimento**: `mean ΔF(variante − controlo permutado) > 0` e `mean ΔAcc ≥ 0` em ≥ 3/4 datasets.
4. **Complexidade**: nós médios ≤ 1,10 × nós médios de E em cada dataset.
5. **Custo**: amostras totais ao oráculo ≤ as de E (em média por dataset).
6. **Reprodutibilidade/integridade**: com B1 desligado os braços A–F são idênticos à referência (`equivalence_report` PASS); `b1_legacy_logged ≡ E`; resultados repetíveis.
**Regras:** não se escolhe λ nem variante por um único dataset ou seed; se várias variantes satisfazem os critérios, ordena-se por (fidelidade média vs Original, depois accuracy, depois custo) e levam-se **todas as que passam** para B2 como candidatas; uma melhoria só se declara se **o efeito nas árvores e nas métricas** a sustentar — diminuir o bónus semântico (`b1_ratio_m_over_s_median`) não é, por si, evidência.
**Honestidade prévia:** a referência pré-B1 mostra que o braço E (OWL, histórico) tem fidelidade **inferior** à do Original em 4/4 datasets e o controlo permutado iguala ou supera E em fidelidade em 3/4; a hierarquia `F_R > F_O` **não** está satisfeita no ponto de partida. B1 só altera o *peso* do bónus (e não EFSR, consultas ativas, projetor nem representação), pelo que pode não ser suficiente; se assim for, reporta-se.

## 1. Referência pré-B1 (congelada)
Ver `docs/evidence/b1b4/pre_b1_reference/pre_b1_reference.json` (sha256 `75edc11048830f8e6da69cdec5bed7014a2187de7bc6f62633444028cc629ebf`; raw em `raw/`, `UNIT_SHA256.json` por unidade). Médias por dataset (5 seeds; digits 3):

| dataset | braço | Fidelity | Accuracy | nós |
|---|---|---|---|---|
| breast_cancer | Original (C) = Reloaded Core (D) | 0,9371 | 0,9329 | 32,2 |
| | E (OWL) | 0,9259 | 0,9357 | 30,6 |
| | F (OWL permutada) | 0,9273 | 0,9427 | 30,2 |
| | C4.5 | 0,9231 | 0,9301 | 17,0 |
| digits | C = D | 0,8215 | 0,8230 | 229,7 |
| | E | 0,8170 | 0,8230 | 231,7 |
| | F | 0,8348 | 0,8341 | 236,3 |
| | C4.5 | 0,8415 | 0,8489 | 165,0 |
| iris | C = D | 0,9632 | 0,9368 | 35,8 |
| | E | 0,9579 | 0,9316 | 36,2 |
| | F | 0,9526 | 0,9474 | 34,6 |
| | C4.5 | 0,9632 | 0,9579 | 8,6 |
| wine | C = D | 0,9200 | 0,9289 | 38,2 |
| | E | 0,8889 | 0,9022 | 41,0 |
| | F | 0,9244 | 0,9333 | 42,2 |
| | C4.5 | 0,9156 | 0,9289 | 9,8 |
Confirma-se no protocolo real que **D ≡ C** (Reloaded sem ontologia = Original: mesmos valores em todas as 18 unidades; contrato S0–S2).


## 2. Resultados DEV de B1 (18 unidades × 13 variantes + controlos; análise executada **exatamente** segundo o pré-registo §0)
Dados: `docs/evidence/b1b4/b1_dev_results/` (`raw/` das 18 unidades, `b1_analysis.json` sha256 `2c124685…c75c`, `B1_RESULTS.md` com todas as tabelas por dataset). Código avaliado: `bde1831`/`122185f` (B1 desligado por omissão). Ambiente numérico fixado; tuning estrutural reutilizado da referência (resultados idênticos).

### 2.1 Integridade (critério 6)
* **B1 desligado ≡ referência: 108/108 braços (6 braços × 18 unidades) idênticos** (previsões, fidelidade, accuracy, nós, folhas, profundidade); `b1_legacy_logged ≡ E`: **18/18**. Sem regressões nos braços A–F. A instrumentação de candidatos não altera decisões.
* Sem *leakage*: B1 só usa o ganho estatístico do nó, o número de linhas **reais** do nó e as estruturas OWL; o teste só entra no relato.

### 2.2 Escala real do bónus semântico (instrumentação de todos os candidatos)
Razão mediana `m/s` (termo semântico / estatístico) **no score histórico (E)**: breast_cancer **168,8**, digits **44,1**, iris **11,8**, wine **28,6** — o bónus domina a componente estatística por 1–2 ordens de grandeza (a auditoria anterior com 30 nós dava 0,83: era uma ontologia/seed; a escala depende do dataset e do nó). Com B1 a razão é exatamente `λ` (0,05 … 0,80) ou 2–28 no `tiebreak` (|bónus| ≤ τ, mas s pequeno nos nós fundos). A cota `|bónus| ≤ λ·s` verifica-se (testes).

### 2.3 Variantes vs Original (médias por dataset; macro = média dos 4 datasets; evidência *observed_only*)
| variante | média ΔF vs Original | média ΔAcc vs Original | F≥O | F>O | Acc>O | C1/C2/C3/C4/C5 | consultas vs Original | → B2 |
|---|---|---|---|---|---|---|---|---|
| E (histórico, B1 off) | -0.0130 | -0.0073 | 0/4 | 0/4 | 1/4 | — | — | — |
| b1_tiebreak | +0.0040 | +0.0053 | 1/4 | 1/4 | 3/4 | 1/4/2/4/2 | 2.22× | não |
| b1_bounded_l0 | -0.0041 | -0.0034 | 1/4 | 1/4 | 1/4 | 1/2/1/4/4 | 2.17× | não |
| b1_bounded_l0p05 | -0.0033 | -0.0039 | 1/4 | 1/4 | 1/4 | 1/2/1/4/3 | 2.17× | não |
| b1_bounded_l0p1 | -0.0041 | -0.0034 | 1/4 | 1/4 | 1/4 | 1/2/1/4/4 | 2.17× | não |
| b1_bounded_l0p2 | -0.0046 | -0.0039 | 1/4 | 1/4 | 1/4 | 1/2/1/4/4 | 2.17× | não |
| b1_bounded_l0p4 | -0.0037 | +0.0012 | 1/4 | 1/4 | 2/4 | 1/2/1/4/3 | 2.18× | não |
| b1_bounded_l0p8 | -0.0022 | +0.0027 | 1/4 | 1/4 | 3/4 | 1/3/3/4/3 | 2.20× | não |
| b1_calibrated_l0 | -0.0041 | -0.0034 | 1/4 | 1/4 | 1/4 | 1/2/1/4/4 | 2.17× | não |
| b1_calibrated_l0p05 | -0.0027 | -0.0016 | 1/4 | 1/4 | 2/4 | 1/3/1/4/3 | 2.18× | não |
| b1_calibrated_l0p1 | -0.0010 | -0.0006 | 2/4 | 2/4 | 1/4 | 2/2/1/4/3 | 2.18× | não |
| b1_calibrated_l0p2 | +0.0029 | +0.0016 | 2/4 | 2/4 | 3/4 | 2/2/2/4/3 | 2.18× | não |
| b1_calibrated_l0p4 | +0.0023 | +0.0038 | 2/4 | 2/4 | 3/4 | 2/3/3/4/3 | 2.19× | não |
| b1_calibrated_l0p8 | +0.0030 | +0.0013 | 2/4 | 2/4 | 1/4 | 2/2/2/4/3 | 2.19× | não |

Notas de leitura: «F≥O»/«F>O» = nº de datasets (dos 4) com média ΔFidelity ≥ 0 / > 0 (estritos, δ=0, valores não arredondados); C1–C5 = nº de datasets que cumprem cada critério pré-registado (mínimo 3/4). Custo: amostras totais ao oráculo / amostras do Original; **todas** as variantes (e o histórico E) fazem ≈ **2,1–2,3×** as consultas do Original (custo da tentativa semântica: contabilizado, não oculto).

### 2.4 Decisão segundo o pré-registo
**Nenhuma variante B1 cumpre os critérios (estado `NOT_ACHIEVED_ON_DEV` para «levar para B2»).** O **C1** (não-inferioridade de fidelidade ao Original em ≥ 3/4 datasets) falha em **todas**: o melhor resultado é 2/4 (`calibrated` com λ ≥ 0,10). Falham também C3 (bater o controlo permutado em ≥ 3/4) na maioria das variantes e C5 (custo ≤ E) no `tiebreak` e em várias λ. As **duas variantes mais próximas** — `bounded` λ = 0,8 e `calibrated` λ = 0,4 — cumprem C2–C5 e **falham só C1**. **Granularidade (para ler o C1 sem o relaxar):** a margem é estrita (δ = 0) e a unidade de teste é pequena: em iris (38 amostras de teste/seed × 5 seeds) 1 amostra = 0,0053 de ΔF médio; em wine (45 × 5) = 0,0044. O ΔF de `calibrated` λ = 0,4 em iris (−0,0053) e wine (−0,0044) corresponde, assim, a **uma única amostra de teste** a menos, enquanto bc (+0,0070) e digits (+0,0119) são positivos. O critério não foi alterado; regista-se que o resultado está na fronteira do que 5 seeds de DEV conseguem distinguir (os MDE do Doc. 2 §7 aplicam-se à fase confirmatória).
### 2.5 O que B1 mostra (leitura científica, sem relaxar critérios)
1. **Calibrar reduz o dano do histórico**: E fica **abaixo** do Original em fidelidade nos 4 datasets (macro −0,0130 F; −0,0073 Acc). Com B1 a macro ΔF passa para −0,004…+0,004: recupera a maior parte da perda (wine: −0,031 → −0,013 com λ = 0), mas **não** produz, de forma consistente, `F_R > F_O`.
2. **λ = 0** (ranking puramente estatístico, representação OWL) já fica abaixo do Original em 3/4 datasets (bc −0,0014; iris −0,0053; wine −0,0133; digits +0,0037): **parte da perda vem de mecanismos que B1 não toca** (representação enriquecida, consultas ativas, refinamento EFSR e respetivo gate de aceitação), não do peso do bónus.
3. **Pouca sensibilidade a λ**: `switch_rate` e `regret` quase não mudam entre λ = 0 e 0,8 (p.ex. bc 0,76→0,73; wine 0,83→0,76) — a decisão final é dominada pelo gate/candidatos, não pela escala do bónus. *Cautela:* estes diagnósticos agregam candidatos pontuados em contextos diferentes (busca semântica, candidatos guiados, avaliação do gate); servem para comparar variantes entre si, não como medida absoluta de «dano».
4. **Ganhos observados vs Original existem, mas não são robustos**: `tiebreak` e `calibrated` (λ = 0,2–0,8) têm macro ΔF/ΔAcc ligeiramente positiva, **quase toda de Digits** (3 seeds: `tiebreak` +0,0274 F / +0,0259 Acc), enquanto bc/iris/wine ficam ≤ 0 (wine −0,004…−0,009). Contra o controlo permutado, `tiebreak` ganha em fidelidade em 3/4 mas perde em accuracy em 2/4 (iris −0,021). Com 4 datasets e seeds de desenvolvimento isto é **hipótese**, não resultado.
5. **Controlo negativo:** o controlo permutado iguala ou supera o E histórico em 3/4 datasets em fidelidade (E−F: bc −0,001, digits −0,018, iris +0,005, wine −0,036) — no histórico o efeito atribuível ao *conhecimento* é nulo ou negativo; em B1 só `tiebreak`/`calibrated` λ ≥ 0,2 mostram sinal positivo sobre o controlo, de forma parcial.
6. **Custo:** nenhuma variante reduz as consultas (≈ 2,2× o Original); a tentativa semântica é custo estrutural do desenho atual (Doc. 15). Ganho por consulta adicional: ver `semantic_cost.efficiency_report` (a calcular quando houver ganho sustentado).

### 2.6 Regressões identificadas
Nenhuma nos braços A–F (108/108). B1 desligado preserva o comportamento histórico bit-a-bit; S0–S2 ≡ Original. Limitação operacional: o executor de tarefas em segundo plano termina aos 120 min — as 3 unidades de Digits (≈ 90–105 min cada com 38 braços B1) exigiram relançar as unidades em falta (retoma por checkpoint; resultados idênticos).

### 2.7 Recomendação fundamentada para B2 (não implementado)
* **Por pré-registo nenhuma variante B1 é «levada» para B2.** A hierarquia `F_R > F_O` **não é atingida** em DEV com B1 sozinho (`NOT_ACHIEVED_ON_DEV`); isto é um resultado, reporta-se como tal.
* **B2 é o passo seguinte justificado**, porque (i) a perda residual com λ = 0 e a insensibilidade a λ localizam o problema na **aceitação/seleção** (gate EFSR, que aceita perdas locais até 1,5 pp e valida na mesma amostra de decisão), não no peso do bónus; (ii) a instrumentação por candidato (já disponível) permite medir benefício real vs flutuação amostral por nó.
* **Proposta para B2 (a decidir por ti antes de implementar):** critério de aceitação de divisões semânticas com **validação cruzada nas linhas reais** do nó (as consultas sintéticas servem para propor/ordenar, não como observações independentes), limite inferior de confiança do ganho local, não-inferioridade (`ν = 0`) e fallback explícito para o vencedor *data-only*; avaliar B2 **fatorialmente** com os candidatos de calibração `{histórico, tiebreak, calibrated 0,2/0,4}` em vez de fixar um B1 agora, selecionando só com treino/validação interna (B3) e nunca com o teste. Manter controlos: ontologia permutada/aleatória e Original equivalente com o mesmo orçamento de consultas.
* **Pré-condição científica registada:** ganhos que apareçam apenas em Digits não sustentam o objetivo; a hierarquia só se declara com evidência em múltiplos datasets e, na fase confirmatória, com ambiente congelado e N pré-definido.
