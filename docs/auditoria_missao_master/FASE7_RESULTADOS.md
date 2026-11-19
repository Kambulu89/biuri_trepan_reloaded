# Fase 7 — resultados: auditoria e otimização do MLP

Análise segundo `FASE7_PREREGISTO.md` (commit `50cce82`, anterior à campanha). Nenhuma configuração, hipótese ou critério foi alterado durante a campanha. As fases 1 a 6 e os seus resultados não foram alterados. Dados em `evidencia/fase7/`.

## 1. Execução

- 20 de 20 unidades com código de saída 0 (4 datasets × 5 sementes; 6 braços TREPAN × 3 réplicas); nenhum braço saltado.
- Verificação independente das métricas: 2740 valores, diferença máxima 0.
- **Mesmas partições de teste que a fase 6** em todas as unidades: True.
- Elegibilidade por validação cruzada aninhada só no treino: concluída para as 20 unidades.

## 2. Tratamento estatístico (sem pseudorreplicação)

- A **unidade experimental é a semente**: cada uma define uma partição treino/teste e um MLP. As 3 réplicas de cada braço TREPAN variam só a aleatoriedade do extrator; são **promediadas dentro da unidade** antes de qualquer comparação. Nunca entram como observações independentes.
- Cada contraste usa 5 diferenças emparelhadas por dataset (uma por unidade) e um intervalo a 95 % por bootstrap percentil dessas 5 diferenças (20 000 reamostragens). Com n = 5 os intervalos são aproximados.
- O MLP e o C4.5 têm uma única execução por unidade (são determinísticos dada a semente).
- Não há teste entre datasets: com 4 datasets nenhum teste de sinais pode ser significativo.
- **Comparações múltiplas:** reporto muitos intervalos (4 datasets × 2 orçamentos × 7 contrastes, mais os do MLP) sem correção; leio-os como evidência exploratória por dataset, não como confirmação.

## 3. Ponto 1 — o MLP original memorizava o ruído?

Com a semente fixa do gerador reconstruí, para cada linha, se o rótulo foi trocado e qual era o rótulo limpo. Os MLP reajustados reproduzem **exatamente** as previsões de teste guardadas nas fases 6 e 7 em 15 de 15 unidades. Análise **post hoc**, descritiva, não pré-registada (`evidencia/fase7/memorization.py`, `memorizacao.csv`).

| Dataset | Modelo | Linhas trocadas em que prevê o rótulo trocado | Accuracy de treino nas linhas limpas | Accuracy de teste (rótulos observados) | Accuracy de teste (rótulos limpos, descritiva) |
|---|---|---|---|---|---|
| phys_bmi | C4.5 | 0,337 | 0,993 | 0,922 | 0,958 |
| phys_bmi | MLP original | 1,000 | 0,997 | 0,874 | 0,914 |
| phys_bmi | MLP otimizado | 0,036 | 0,982 | 0,926 | 0,966 |
| phys_kinetic | C4.5 | 0,386 | 0,991 | 0,888 | 0,942 |
| phys_kinetic | MLP original | 0,930 | 1,000 | 0,838 | 0,884 |
| phys_kinetic | MLP otimizado | 0,086 | 0,985 | 0,896 | 0,950 |
| phys_power | C4.5 | 0,308 | 0,985 | 0,860 | 0,918 |
| phys_power | MLP original | 0,954 | 1,000 | 0,846 | 0,904 |
| phys_power | MLP otimizado | 0,012 | 0,975 | 0,898 | 0,960 |

- **O MLP original memoriza o ruído:** nas linhas de treino com rótulo trocado, prevê o rótulo trocado em 93 a 100 % dos casos. Um modelo que seguisse a regra verdadeira preveria-o em perto de 0 %.
- **Isto é memorização e não sobreajuste genérico.** Nas linhas limpas, a accuracy de treino é praticamente a mesma nos três modelos (97,5–100 %): a diferença está quase toda nas linhas trocadas. O C4.5 podado memoriza 31–39 %; o MLP otimizado, 0–9 %.
- **A memorização custa accuracy de teste:** face aos rótulos limpos, o MLP original fica 5 a 6 pontos abaixo do otimizado. A coluna com rótulos limpos é só descritiva; nada foi escolhido com ela.

## 4. Ponto 2 — MLP original contra otimizado e contra o C4.5

Teste, rótulos reais. «MLP original» = previsões da fase 6 nas mesmas partições.

| Contraste | german_credit | phys_bmi | phys_kinetic | phys_power |
|---|---|---|---|---|
| accuracy: MLP otimizado − MLP original | +0,010 [-0,020; +0,036] | +0,052 [+0,024; +0,080] | +0,058 [+0,042; +0,074] | +0,052 [+0,038; +0,068] |
| accuracy: MLP otimizado − C4.5 | +0,042 [+0,023; +0,060] | +0,004 [-0,014; +0,022] | +0,008 [-0,010; +0,026] | +0,038 [+0,016; +0,060] |
| accuracy: MLP original − C4.5 | +0,031 [-0,009; +0,071] | -0,048 [-0,086; -0,018] | -0,050 [-0,074; -0,026] | -0,014 [-0,024; -0,002] |
| macro-F1: MLP otimizado − MLP original | +0,086 [+0,032; +0,139] | +0,054 [+0,023; +0,084] | +0,058 [+0,043; +0,074] | +0,052 [+0,038; +0,068] |
| macro-F1: MLP otimizado − C4.5 | +0,050 [+0,023; +0,078] | +0,004 [-0,015; +0,022] | +0,008 [-0,010; +0,026] | +0,039 [+0,017; +0,061] |
| macro-F1: MLP original − C4.5 | -0,035 [-0,114; +0,043] | -0,050 [-0,091; -0,019] | -0,050 [-0,074; -0,027] | -0,013 [-0,024; -0,000] |
| balanced accuracy: MLP otimizado − MLP original | +0,084 [+0,045; +0,124] | +0,052 [+0,019; +0,086] | +0,058 [+0,043; +0,076] | +0,052 [+0,039; +0,067] |
| balanced accuracy: MLP otimizado − C4.5 | +0,049 [+0,018; +0,080] | +0,001 [-0,017; +0,019] | +0,008 [-0,009; +0,026] | +0,040 [+0,018; +0,062] |
| balanced accuracy: MLP original − C4.5 | -0,036 [-0,102; +0,031] | -0,052 [-0,090; -0,021] | -0,050 [-0,073; -0,029] | -0,012 [-0,024; +0,001] |

- **Datasets físicos:** a otimização melhora o MLP em 5,2 a 5,8 pontos de accuracy, com intervalo acima de zero nos três. **M1 sustentada.**
- **German Credit:** +1,0 ponto de accuracy, com intervalo a incluir zero; mas +8,6 pontos de macro-F1 e +8,4 de balanced accuracy, com intervalo acima de zero. O MLP original quase não previa a classe minoritária; o otimizado sim.
- **Face ao C4.5:** o MLP otimizado supera-o com intervalo acima de zero no German Credit e no phys_power; no phys_bmi e no phys_kinetic fica acima, mas com intervalo a incluir zero. O MLP original ficava abaixo do C4.5 nos três físicos.

Configurações mais escolhidas (validação cruzada aninhada): α = 10 com 16 neurónios na maioria das dobras do phys_bmi e do phys_power; α = 1 no phys_kinetic. **O α escolhido está muitas vezes no limite superior do espaço congelado (10)**; o ótimo pode estar acima. Não alarguei o espaço.

### Elegibilidade (M2)

| Dataset | MLP otimizado − C4.5 (CV aninhada no treino) | Elegível |
|---|---|---|
| german_credit | +0,022 | sim |
| phys_bmi | +0,045 | sim |
| phys_kinetic | +0,017 | não |
| phys_power | +0,026 | sim |

Elegíveis: german_credit, phys_bmi, phys_power. O phys_kinetic fica abaixo da margem (1,7 pontos).

## 5. Ponto 3 — a melhoria do MLP chega às árvores?

Accuracy face aos rótulos reais, fase 7 − fase 6, nas mesmas partições. Não comparo fidelidades entre fases: medem a concordância com oráculos diferentes.

| Dataset | Braço | Δ accuracy (fase 7 − fase 6) | IC 95 % | Fração do ganho do MLP |
|---|---|---|---|---|
| german_credit | B | -0,013 | [-0,025; +0,006] | -1,28 |
| german_credit | B7 | -0,004 | [-0,013; +0,002] | -0,38 |
| german_credit | MLP | +0,010 | [-0,020; +0,036] | +1,00 |
| german_credit | R | -0,003 | [-0,009; +0,004] | -0,28 |
| german_credit | R7 | +0,013 | [+0,006; +0,020] | +1,23 |
| german_credit | Rq | -0,005 | [-0,019; +0,010] | -0,51 |
| german_credit | Rq7 | +0,018 | [+0,014; +0,023] | +1,72 |
| phys_bmi | B | +0,025 | [+0,009; +0,042] | +0,49 |
| phys_bmi | B7 | +0,007 | [-0,014; +0,025] | +0,14 |
| phys_bmi | MLP | +0,052 | [+0,024; +0,080] | +1,00 |
| phys_bmi | R | +0,016 | [-0,007; +0,040] | +0,31 |
| phys_bmi | R7 | -0,003 | [-0,007; +0,001] | -0,06 |
| phys_bmi | Rq | +0,020 | [-0,001; +0,041] | +0,38 |
| phys_bmi | Rq7 | -0,003 | [-0,009; +0,004] | -0,06 |
| phys_kinetic | B | +0,022 | [+0,005; +0,039] | +0,38 |
| phys_kinetic | B7 | +0,007 | [-0,002; +0,016] | +0,12 |
| phys_kinetic | MLP | +0,058 | [+0,042; +0,078] | +1,00 |
| phys_kinetic | R | +0,013 | [-0,006; +0,032] | +0,23 |
| phys_kinetic | R7 | -0,005 | [-0,008; -0,002] | -0,09 |
| phys_kinetic | Rq | +0,020 | [-0,001; +0,043] | +0,34 |
| phys_kinetic | Rq7 | +0,007 | [-0,009; +0,024] | +0,13 |
| phys_power | B | +0,032 | [+0,019; +0,042] | +0,62 |
| phys_power | B7 | -0,004 | [-0,024; +0,016] | -0,08 |
| phys_power | MLP | +0,052 | [+0,038; +0,068] | +1,00 |
| phys_power | R | +0,009 | [-0,003; +0,023] | +0,18 |
| phys_power | R7 | -0,012 | [-0,020; -0,006] | -0,23 |
| phys_power | Rq | +0,023 | [+0,011; +0,035] | +0,44 |
| phys_power | Rq7 | -0,006 | [-0,031; +0,016] | -0,12 |

- **TREPAN Original com a estrutura do tuning:** ganha 2,2 a 3,2 pontos nos três físicos (intervalo acima de zero), entre 38 % e 62 % do ganho do MLP. **A melhoria do oráculo é transmitida em parte.**
- **Com 7 nós:** a transmissão é nula ou negativa. Uma árvore de 3 divisões não tem capacidade para representar o que o MLP aprendeu a mais.
- **Reloaded:** ganha menos (+0,9 a +1,6 pontos, intervalos a incluir zero). Já na fase 6 o Reloaded estava acima do MLP original, porque o atributo derivado lhe dá a regra verdadeira; tinha menos a ganhar. Com 7 nós perde ligeiramente no phys_kinetic e no phys_power (−0,5 e −1,2 pontos, intervalo abaixo de zero).
- **German Credit:** o Original perde 1,3 pontos (intervalo a incluir zero); o Reloaded com 7 nós ganha 1,3 pontos (intervalo acima de zero).

## 6. Pontos 4 e 5 — Original contra Reloaded e contributo da ontologia (mesmo MLP otimizado)

### 6.1 Estrutura escolhida pelo tuning

| Contraste | german_credit | phys_bmi | phys_kinetic | phys_power |
|---|---|---|---|---|
| H1 accuracy: R − B | +0,002 [-0,012; +0,014] | +0,013 [+0,003; +0,024] | +0,019 [+0,011; +0,027] | +0,014 [+0,001; +0,029] |
| H2 accuracy: R − C4.5 | +0,013 [-0,019; +0,039] | +0,025 [+0,005; +0,044] | +0,035 [+0,021; +0,053] | +0,050 [+0,033; +0,069] |
| H3 accuracy: B − C4.5 | +0,010 [-0,009; +0,030] | +0,011 [-0,005; +0,028] | +0,017 [+0,003; +0,033] | +0,036 [+0,013; +0,061] |
| H4 fidelidade: R − B | -0,004 [-0,012; +0,003] | +0,007 [+0,001; +0,013] | +0,020 [+0,013; +0,029] | +0,011 [+0,005; +0,019] |
| H5 fidelidade: R − C4.5 | -0,025 [-0,055; +0,002] | +0,029 [+0,017; +0,037] | +0,054 [+0,037; +0,071] | +0,074 [+0,049; +0,099] |
| AT accuracy: R − Rq | +0,006 [-0,007; +0,018] | +0,143 [+0,130; +0,155] | +0,034 [+0,031; +0,037] | +0,015 [-0,002; +0,028] |
| AT fidelidade: R − Rq | -0,005 [-0,016; +0,005] | +0,131 [+0,113; +0,151] | +0,029 [+0,015; +0,043] | +0,020 [+0,011; +0,029] |

### 6.2 Máximo de 7 nós

| Contraste | german_credit | phys_bmi | phys_kinetic | phys_power |
|---|---|---|---|---|
| H1 accuracy: R7 − B7 | +0,012 [+0,006; +0,019] | +0,077 [+0,049; +0,096] | +0,077 [+0,059; +0,095] | +0,070 [+0,050; +0,094] |
| H2 accuracy: R7 − C4.5 | +0,028 [+0,006; +0,047] | +0,029 [+0,009; +0,044] | +0,045 [+0,029; +0,067] | +0,053 [+0,037; +0,070] |
| H3 accuracy: B7 − C4.5 | +0,016 [-0,014; +0,040] | -0,048 [-0,067; -0,035] | -0,032 [-0,052; -0,019] | -0,017 [-0,039; +0,007] |
| H4 fidelidade: R7 − B7 | +0,003 [-0,003; +0,011] | +0,046 [+0,023; +0,070] | +0,064 [+0,056; +0,074] | +0,085 [+0,057; +0,125] |
| H5 fidelidade: R7 − C4.5 | -0,019 [-0,051; +0,010] | +0,025 [+0,012; +0,037] | +0,044 [+0,023; +0,065] | +0,071 [+0,047; +0,095] |
| AT accuracy: R7 − Rq7 | +0,007 [-0,004; +0,013] | +0,143 [+0,128; +0,163] | +0,069 [+0,052; +0,088] | +0,075 [+0,054; +0,100] |
| AT fidelidade: R7 − Rq7 | +0,011 [-0,004; +0,026] | +0,118 [+0,094; +0,142] | +0,053 [+0,046; +0,061] | +0,101 [+0,074; +0,139] |

### 6.3 Decisão (critérios da fase 6)

| Orçamento | Dataset | Elegível | Sustentada | Observada | Atribuível |
|---|---|---|---|---|---|
| estrutura do tuning | german_credit | sim | não | não | não |
| estrutura do tuning | phys_bmi | sim | não | sim | sim |
| estrutura do tuning | phys_kinetic | não | sim | sim | sim |
| estrutura do tuning | phys_power | sim | sim | sim | não |
| máx. 7 nós | german_credit | sim | não | não | não |
| máx. 7 nós | phys_bmi | sim | não | não | sim |
| máx. 7 nós | phys_kinetic | não | não | não | sim |
| máx. 7 nós | phys_power | sim | não | não | sim |

Unidades em que as cinco condições se verificam (médias das réplicas por unidade):

| Orçamento | Dataset | H1 | H2 | H3 | H4 | H5 | Conjunta |
|---|---|---|---|---|---|---|---|
| estrutura do tuning | german_credit | 3/5 | 3/5 | 3/5 | 2/5 | 1/5 | 0/5 |
| estrutura do tuning | phys_bmi | 4/5 | 3/5 | 3/5 | 3/5 | 5/5 | 2/5 |
| estrutura do tuning | phys_kinetic | 5/5 | 5/5 | 4/5 | 5/5 | 5/5 | 4/5 |
| estrutura do tuning | phys_power | 4/5 | 5/5 | 4/5 | 5/5 | 5/5 | 3/5 |
| máx. 7 nós | german_credit | 5/5 | 4/5 | 3/5 | 3/5 | 1/5 | 1/5 |
| máx. 7 nós | phys_bmi | 5/5 | 4/5 | 0/5 | 5/5 | 5/5 | 0/5 |
| máx. 7 nós | phys_kinetic | 5/5 | 5/5 | 0/5 | 5/5 | 5/5 | 0/5 |
| máx. 7 nós | phys_power | 5/5 | 5/5 | 1/5 | 5/5 | 5/5 | 1/5 |

## 7. Leitura

**Hierarquia completa, estrutura do tuning:**

- **phys_power (elegível): hierarquia sustentada.** As cinco condições têm intervalo acima de zero. **Não é atribuível**, porque o Reloaded supera a ontologia permutada em fidelidade (+2,0 pontos, intervalo acima de zero) mas não em accuracy (+1,5 pontos, intervalo a incluir zero).
- **phys_kinetic (não elegível): hierarquia sustentada e atribuível.** Mas o dataset não cumpriu a elegibilidade pré-registada, por isso não conta para o veredicto da hipótese.
- **phys_bmi (elegível): hierarquia observada, não sustentada** (`Original > C4.5`: +1,1 pontos, intervalo a incluir zero). O ganho do Reloaded é atribuível.
- **German Credit (elegível): sem hierarquia.** O Reloaded não supera o Original em accuracy nem em fidelidade.

**Hierarquia completa, 7 nós:** não se sustenta em nenhum dataset. O Reloaded supera o Original e o C4.5, com ganho atribuível nos três físicos, mas o Original com 7 nós fica abaixo do C4.5 canónico (que usa 20 a 23 nós).

**Contributo da ontologia (ponto 5):** nos três físicos e nos dois orçamentos, o Reloaded supera a ontologia permutada, quase sempre com intervalo acima de zero (a exceção é a accuracy no phys_power com a estrutura do tuning). No German Credit, a ontologia real não se distingue da permutada.

**Fidelidade e accuracy (ponto 8):** onde o Reloaded ganha fidelidade ao MLP otimizado, ganha também accuracy face aos rótulos reais, nos físicos. No German Credit não ganha nenhuma das duas. Não trato uma como a outra: estão em linhas separadas de todas as tabelas.

## 8. EFSR (ponto 7)

Não foi alterado nem usado nesta fase: o Reloaded avaliado é O, sem mecanismos. A variante com validação independente está descrita no `FASE7_PREREGISTO.md` §6. Antes de qualquer implementação precisa de pré-registo próprio, de avaliação em datasets diferentes dos usados para a desenhar e de correção para comparações múltiplas se forem testadas várias variantes.

## 9. Limitações

- **Controlos positivos sintéticos**, desenhados por mim, com a relação verdadeira declarada na ontologia. A hierarquia sustentada no phys_power mostra que o sistema consegue cumpri-la quando todas as condições se verificam por construção; não mostra que o faça em dados reais.
- **O único dataset real** (German Credit) não mostra ganho do Reloaded.
- **5 unidades por dataset** e muitos contrastes sem correção para comparações múltiplas.
- **O α escolhido está muitas vezes no limite do espaço de pesquisa.**
- **A análise de memorização é post hoc.**
- **Problema operacional encontrado:** reajustar o MLP fora do executor sem fixar primeiro o ambiente numérico dá modelos diferentes (P13). Com o ambiente fixado, a reprodução foi exata.

## 10. Conclusões

1. **O problema do MLP está resolvido nos controlos positivos.** A causa era a memorização dos rótulos com ruído, por regularização insuficiente no perfil para poucos dados. Uma seleção de α e de arquitetura feita só no treino melhora o MLP em 5 a 6 pontos e põe-no acima do C4.5.
2. **A melhoria do oráculo chega ao TREPAN Original com a estrutura do tuning** (38–62 % do ganho), mas não às árvores de 7 nós.
3. **Com o oráculo melhorado, a hierarquia `Reloaded > Original > C4.5` sustenta-se num dataset elegível (phys_power)**, sem atribuição completa à ontologia, e sustenta-se com atribuição num dataset não elegível (phys_kinetic). Fica observada, sem sustentação, no phys_bmi.
4. **O enriquecimento ontológico continua a produzir ganhos mensuráveis e atribuíveis** nos controlos positivos, nos dois orçamentos. No dataset real não produz.
5. **A hipótese não fica demonstrada em geral:** só se sustenta em dados sintéticos construídos para a favorecer, e falha no único dataset real disponível.
