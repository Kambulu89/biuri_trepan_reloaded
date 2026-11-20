# Fase 9 — resultados: EFSR com gate validado em consultas novas

Análise segundo `FASE9_PREREGISTO.md` (commit `21d4f29`, anterior à campanha; emenda 1 no commit `42a8314`, antes de qualquer análise). Nenhuma hipótese, critério ou braço foi alterado. As fases 1 a 8 não foram alteradas. O núcleo (`core/`) não foi alterado. Dados em `evidencia/fase9/`.

## 1. Execução

- 30 de 30 unidades com código de saída 0 (6 datasets × 5 sementes; 10 braços TREPAN × 3 réplicas, mais MLP otimizado e C4.5). 1050 linhas brutas.
- Recalculei as métricas a partir das previsões guardadas: 6270 valores, com diferença máxima de 0.
- **Integridade 1:** em todas as 30 unidades (420 braços), as previsões de MLP, C4.5, B, B7, R e R7 são iguais às das fases 7 (phys_*, german_credit) e 8 (digits, adult). A elegibilidade das fases 7 e 8 aplica-se, portanto, sem recálculo.
- **Integridade 2:** em todas as 44 árvores V/V7 sem nenhuma tentativa do EFSR, V é igual a E (mesmas previsões).

### Incidentes

1. **Reinício do contentor** com 8 de 30 unidades concluídas. As unidades em falta foram executadas do início. O registo parcial do EFSR de german_credit/7 (21 linhas da execução interrompida) foi separado para `evidencia/fase9/interrompidas/`. A unidade digits/7 retomou 238 ajustes do tuning a partir do checkpoint; o tuning é determinístico e as previsões de B e R coincidem com as da fase 8.
2. **Defeito de implementação (emenda 1):** em phys_power/42 o braço `sc_V_r1` foi descartado porque o lote novo não pôde ser gerado num nó estreito. Corrigi o código para tratar esse caso como rejeição, que já era o comportamento previsto para um lote vazio. Voltei a correr a unidade: os 34 braços que já existiam ficaram com previsões idênticas e apareceu o braço que faltava. A saída original está em `interrompidas/`. Nenhuma outra unidade teve braços descartados.
3. Uma unidade (phys_bmi/7) começou na janela entre a correção e o commit e ficou registada como `code_dirty`. O código só difere no caso da exceção, que não ocorreu nessa unidade (nenhum braço descartado).

## 2. O gate novo em números

Tentativas e aprovações do EFSR por árvore, na estrutura do tuning:

| Dataset | E: tentativas | E: aprovações | V: tentativas | V: aprovações | Consultas extra de V por árvore |
|---|---|---|---|---|---|
| digits | 85,5 | 19,5 | 85,1 | 14,1 | 138 354 |
| german_credit | 28,3 | 6,8 | 28,3 | 1,2 | 33 500 |
| phys_bmi | 2,1 | 1,1 | 2,0 | 0,3 | 1 800 |
| phys_power | 5,9 | 3,5 | 6,3 | 1,9 | 7 500 |
| phys_kinetic | 14,1 | 8,2 | 13,9 | 4,1 | 20 820 |
| adult | 26,9 | 5,5 | 26,9 | 1,5 | 43 267 |

O gate novo aprova entre 2 e 6 vezes menos propostas do que o original, exceto no digits (19,5 → 14,1). Custa 40 a 50 % mais consultas ao oráculo do que E nos datasets reais, e 10 a 35 % nos físicos.

## 3. Mecanismo (estrutura do tuning)

Diferença média e IC a 95 %:

| Contraste | digits | german_credit | phys_bmi | phys_power | phys_kinetic | adult |
|---|---|---|---|---|---|---|
| E − R accuracy (EFSR original) | +0,003 [−0,001; +0,006] | +0,006 [−0,001; +0,012] | +0,001 [+0,000; +0,004] | +0,000 [+0,000; +0,000] | +0,001 [−0,002; +0,006] | −0,001 [−0,010; +0,008] |
| E − R fidelidade | +0,003 [−0,000; +0,006] | +0,010 [−0,004; +0,025] | +0,000 [+0,000; +0,000] | **−0,004 [−0,007; −0,001]** | −0,003 [−0,009; +0,003] | +0,002 [−0,005; +0,010] |
| G1 accuracy: V − E | −0,002 [−0,009; +0,005] | +0,005 [−0,005; +0,015] | +0,001 [+0,000; +0,002] | −0,002 [−0,006; +0,000] | −0,001 [−0,009; +0,003] | +0,003 [−0,002; +0,008] |
| G1 fidelidade: V − E | −0,002 [−0,007; +0,003] | −0,004 [−0,015; +0,007] | −0,001 [−0,002; +0,000] | **+0,003 [+0,001; +0,006]** | **+0,003 [+0,001; +0,003]** | −0,004 [−0,009; +0,001] |
| G2 accuracy: V − R | +0,001 [−0,007; +0,009] | **+0,011 [+0,005; +0,017]** | +0,002 [+0,000; +0,006] | −0,002 [−0,006; +0,000] | +0,000 [−0,003; +0,003] | +0,002 [−0,002; +0,005] |
| G2 fidelidade: V − R | +0,002 [−0,005; +0,009] | +0,006 [−0,003; +0,016] | −0,001 [−0,002; +0,000] | −0,001 [−0,002; +0,000] | +0,000 [−0,006; +0,006] | −0,002 [−0,005; +0,002] |

Com 7 nós: E7 − R7 é negativo e sustentado no digits (accuracy −0,005, fidelidade −0,005). V7 − E7 é positivo e sustentado no digits (accuracy e fidelidade +0,003) e na fidelidade do phys_power (+0,003), e negativo e sustentado na accuracy do phys_power (−0,003); nos restantes o intervalo toca 0. V7 − R7 fica a menos de 0,004 de zero em todos os datasets, sem nenhum intervalo estritamente acima ou abaixo de 0 (tabela completa em `fase9_stats.log`).

**Regra de adoção (pré-registada, estrutura do tuning, só elegíveis):** G2 tem limite inferior > 0 no german_credit (accuracy +0,011, 5 de 5 sementes positivas) e não tem limite superior < 0 em nenhum elegível. **Pela regra, V é útil.** Como a regra é fraca, a decisão também é: é cumprida por **um** dataset e **uma** métrica entre oito contrastes G2, sem correção para comparações múltiplas, e a fidelidade do german_credit não melhora (+0,006, intervalo a incluir 0).

## 4. Hierarquia com V como Reloaded

### 4.1 Estrutura escolhida pelo tuning

| Contraste | digits | german_credit | phys_bmi | phys_power | phys_kinetic | adult |
|---|---|---|---|---|---|---|
| H1 accuracy: V − B | +0,002 [−0,007; +0,011] | **+0,013 [+0,000; +0,026]** ¹ | **+0,015 [+0,005; +0,025]** | +0,012 [−0,003; +0,027] | **+0,019 [+0,009; +0,028]** | +0,001 [−0,001; +0,004] |
| H2 accuracy: V − C4.5 | +0,020 [−0,003; +0,044] | +0,024 [−0,006; +0,050] | **+0,027 [+0,007; +0,044]** | **+0,048 [+0,033; +0,063]** | **+0,035 [+0,018; +0,054]** | −0,003 [−0,012; +0,006] |
| H3 accuracy: B − C4.5 | +0,019 [−0,000; +0,038] | +0,010 [−0,009; +0,030] | +0,011 [−0,005; +0,029] | **+0,036 [+0,013; +0,061]** | **+0,017 [+0,003; +0,033]** | −0,005 [−0,016; +0,008] |
| H4 fidelidade: V − B | +0,003 [−0,007; +0,013] | +0,003 [−0,006; +0,011] | +0,006 [−0,000; +0,013] | **+0,011 [+0,004; +0,018]** | **+0,020 [+0,014; +0,026]** | **−0,003 [−0,004; −0,002]** |
| H5 fidelidade: V − C4.5 | +0,021 [−0,002; +0,046] | −0,018 [−0,047; +0,013] | **+0,028 [+0,017; +0,037]** | **+0,073 [+0,049; +0,098]** | **+0,054 [+0,037; +0,071]** | **+0,009 [+0,002; +0,015]** |
| AT accuracy: V − Vq | +0,000 [−0,011; +0,012] | **+0,011 [+0,004; +0,019]** | **+0,148 [+0,122; +0,173]** | +0,011 [−0,005; +0,027] | **+0,033 [+0,019; +0,047]** | **+0,008 [+0,003; +0,011]** |
| AT fidelidade: V − Vq | +0,000 [−0,008; +0,012] | +0,014 [+0,000; +0,026] | **+0,128 [+0,101; +0,149]** | **+0,015 [+0,007; +0,023]** | **+0,025 [+0,009; +0,043]** | +0,006 [+0,000; +0,014] |

¹ Limite inferior +0,0003.

### 4.2 Máximo de 7 nós

| Contraste | digits | german_credit | phys_bmi | phys_power | phys_kinetic | adult |
|---|---|---|---|---|---|---|
| H1 accuracy: V7 − B7 | +0,002 [−0,004; +0,005] | **+0,012 [+0,006; +0,019]** | **+0,077 [+0,049; +0,098]** | **+0,067 [+0,043; +0,091]** | **+0,076 [+0,062; +0,090]** | −0,000 [−0,007; +0,006] |
| H3 accuracy: B7 − C4.5 | **−0,492** | +0,016 [−0,014; +0,040] | **−0,048 [−0,066; −0,035]** | −0,017 [−0,039; +0,007] | **−0,032 [−0,052; −0,019]** | −0,001 [−0,009; +0,008] |
| H4 fidelidade: V7 − B7 | +0,002 [−0,003; +0,005] | +0,003 [−0,003; +0,011] | **+0,047 [+0,025; +0,071]** | **+0,085 [+0,057; +0,125]** | **+0,065 [+0,056; +0,074]** | −0,004 [−0,015; +0,005] |
| AT accuracy: V7 − Vq7 | −0,000 | +0,006 [−0,004; +0,013] | **+0,137** | **+0,054** | **+0,041** | −0,002 |

Com 7 nós o Original (B7) fica abaixo do C4.5 sem limite de nós nos físicos e no digits; a cadeia `Original > C4.5` falha por construção do orçamento. O C4.5 não tem o mesmo limite.

### 4.3 Decisão (critérios da fase 6)

| Orçamento | Dataset | Elegível | Sustentada | Observada | Atribuível |
|---|---|---|---|---|---|
| estrutura do tuning | digits | sim | não | sim | não |
| estrutura do tuning | german_credit | sim | não | não | não |
| estrutura do tuning | phys_bmi | sim | não | sim | não |
| estrutura do tuning | phys_power | sim | não | sim | não |
| estrutura do tuning | phys_kinetic | não | **sim** | sim | **sim** |
| estrutura do tuning | adult | não | não | não | não |
| máx. 7 nós | digits | sim | não | não | não |
| máx. 7 nós | german_credit | sim | não | não | não |
| máx. 7 nós | phys_bmi | sim | não | não | **sim** |
| máx. 7 nós | phys_power | sim | não | não | **sim** |
| máx. 7 nós | phys_kinetic | não | não | não | **sim** |
| máx. 7 nós | adult | não | não | não | não |

Unidades com as cinco condições em conjunto, na estrutura do tuning: phys_kinetic 4/5, digits, phys_bmi e phys_power 2/5, german_credit 1/5, adult 0/5.

## 5. Leitura

- **Veredicto principal (elegíveis: digits, german_credit, phys_bmi, phys_power):** com V como Reloaded, a hierarquia `Reloaded > Original > C4.5` **não se sustenta em nenhum** elegível. É observada (todas as médias positivas) no digits, no phys_bmi e no phys_power, com a estrutura do tuning.
- **Comparação com R (fases 7 e 8):** com R, o phys_power tinha a hierarquia sustentada (fase 7). Com V deixa de ter (H1 V − B com limite inferior −0,003), porque V fica ligeiramente abaixo de R em accuracy no phys_power (−0,002). Nos datasets físicos, V e R são praticamente iguais: o ganho atribuível nos físicos vem dos atributos da ontologia (R), não do EFSR.
- **O EFSR original quase não prejudica aqui.** Na estrutura do tuning, E − R só é negativo e sustentado na fidelidade do phys_power (−0,004). A perda grande da fase 5 (wine, breast_cancer) não se repete nestes datasets. Com 7 nós volta a prejudicar no digits.
- **O gate novo é mais conservador, não claramente melhor:** aprova muito menos propostas, e V − E é pequeno e de sinal misto (±0,005).
- **german_credit é o único caso de ganho do EFSR validado.** A accuracy sobe +1,1 pontos face a R e +1,3 face ao Original (5 de 5 sementes face a R), e supera a ontologia permutada em accuracy (+1,1, sustentado). A **fidelidade não melhora** (V − R +0,006 e V − B +0,003, ambos a incluir 0), e o C4.5 tem mais fidelidade do que V (−0,018). É uma melhoria de **accuracy**, não de fidelidade, e não satisfaz a hierarquia conjunta. Na fase 7, com R, o german_credit não tinha nenhum ganho.
- **Atribuição com 7 nós (phys_bmi, phys_power, phys_kinetic):** V7 bate B7 e Vq7 em accuracy e fidelidade com margens grandes. O mesmo já acontecia com R7 na fase 7: é o efeito dos atributos que codificam a relação física verdadeira, e V7 − R7 ≈ 0.
- **Adult:** não elegível; V tem menos fidelidade do que o Original (−0,003, 5 de 5 sementes).

## 6. Limitações

- V faz 40 a 50 % mais consultas ao oráculo do que E nos datasets reais. As consultas extra só decidem aceitar ou rejeitar, mas são informação adicional: o ganho do german_credit pode vir em parte delas. Não testei um controlo com o mesmo número de consultas sem gate.
- A regra de adoção é cumprida por um único dataset e uma única métrica; sem correção para comparações múltiplas, este resultado é frágil.
- O canal de score (λ = 1) vem sempre junto com o EFSR em E e V e não é isolado aqui.
- Todos os datasets já tinham sido observados com B e R; n = 5 sementes por dataset.
- Com 7 nós o C4.5 não fica limitado ao mesmo tamanho.

## 7. Conclusões

1. O gate validado em consultas novas funciona como desenhado: aprova menos propostas e nunca piora de forma sustentada um dataset elegível face a R. Pela regra pré-registada é **útil**, mas só por causa do german_credit, e só em accuracy.
2. Não produz a hierarquia conjunta em nenhum dataset elegível. Perde a que R tinha no phys_power.
3. O ganho atribuível à ontologia continua a vir dos atributos derivados de relações verdadeiras (R), não do EFSR. Nos dados reais, o único sinal novo é o +1,1 pontos de accuracy no german_credit, sem ganho de fidelidade.
4. Recomendação: não integrar V no núcleo com base nesta fase. Uma confirmação exigiria datasets novos, um controlo com o mesmo número de consultas e um pré-registo próprio.

Não avancei para nenhuma fase seguinte.
