# Fase 6 — resultados

Análise feita segundo `FASE6_PREREGISTO.md` (commit `521ce2d`, anterior aos resultados da fase 5 e a esta campanha). Reloaded corrigido congelado pela regra da fase 5: **O** (atributos da ontologia v2, sem mecanismos). Dados em `evidencia/fase6/`.

## 1. Execução

- 20 de 20 unidades (4 datasets × 5 sementes) com código de saída 0; 6 braços × 3 réplicas; nenhum braço saltado. A unidade mais longa (German Credit) demorou 537 s.
- Verificação independente das métricas: 2740 valores, diferença máxima 0.
- O limite de 7 nós foi respeitado em todos os braços com esse orçamento.

## 2. Elegibilidade (só treino)

| Dataset | MLP − C4.5 (CV no treino) | Elegível |
|---|---|---|
| german_credit | +0,011 | não |
| phys_bmi | +0,005 | não |
| phys_kinetic | -0,021 | não |
| phys_power | -0,028 | não |

**Nenhum dataset é elegível.** Nos três controlos positivos o MLP fica **abaixo** do C4.5, e no German Credit a vantagem (1,1 pontos) fica abaixo da margem de 2 pontos. O regime que a hipótese pressupõe (MLP claramente melhor do que o C4.5) não foi criado: com 300 linhas de treino e uma fronteira monótona em duas variáveis, o C4.5 aproxima bem a fronteira, e o MLP do projeto não a aprende melhor. Pelo pré-registo, não há nenhum dataset em que a hipótese possa ser avaliada como tal; reporto todos os resultados.

## 3. Contrastes — estrutura escolhida pelo tuning

| Contraste | german_credit | phys_bmi | phys_kinetic | phys_power |
|---|---|---|---|---|
| H1 accuracy: R − B | -0,008 [-0,019; +0,002] | +0,023 [+0,003; +0,041] | +0,027 [+0,011; +0,045] | +0,037 [+0,015; +0,055] |
| H2 accuracy: R − C4.5 | +0,015 [-0,012; +0,042] | +0,009 [-0,007; +0,024] | +0,022 [-0,001; +0,045] | +0,041 [+0,016; +0,069] |
| H3 accuracy: B − C4.5 | +0,024 [-0,014; +0,055] | -0,014 [-0,035; +0,002] | -0,005 [-0,020; +0,009] | +0,004 [-0,017; +0,023] |
| H4 fidelidade: R − B | -0,011 [-0,025; -0,001] | +0,004 [-0,012; +0,022] | -0,002 [-0,013; +0,011] | +0,023 [+0,007; +0,038] |
| H5 fidelidade: R − C4.5 | +0,124 [+0,088; +0,165] | +0,007 [-0,009; +0,026] | -0,001 [-0,024; +0,020] | +0,038 [+0,027; +0,053] |
| AT accuracy: R − Rq | +0,004 [-0,006; +0,012] | +0,147 [+0,131; +0,163] | +0,041 [+0,023; +0,055] | +0,028 [+0,014; +0,041] |
| AT fidelidade: R − Rq | +0,006 [-0,002; +0,014] | +0,148 [+0,135; +0,161] | -0,005 [-0,016; +0,007] | +0,028 [+0,004; +0,046] |
| MLP accuracy: MLP − C4.5 | +0,031 [-0,009; +0,071] | -0,048 [-0,088; -0,016] | -0,050 [-0,074; -0,028] | -0,014 [-0,024; -0,002] |

## 4. Contrastes — orçamento restrito (máx. 7 nós)

| Contraste | german_credit | phys_bmi | phys_kinetic | phys_power |
|---|---|---|---|---|
| H1 accuracy: R7 − B7 | -0,005 [-0,013; +0,001] | +0,087 [+0,075; +0,102] | +0,089 [+0,076; +0,103] | +0,078 [+0,061; +0,093] |
| H2 accuracy: R7 − C4.5 | +0,015 [-0,012; +0,036] | +0,032 [+0,017; +0,044] | +0,051 [+0,033; +0,071] | +0,065 [+0,051; +0,082] |
| H3 accuracy: B7 − C4.5 | +0,020 [-0,012; +0,048] | -0,055 [-0,063; -0,048] | -0,039 [-0,055; -0,023] | -0,013 [-0,028; -0,000] |
| H4 fidelidade: R7 − B7 | -0,002 [-0,011; +0,009] | +0,066 [+0,049; +0,083] | +0,025 [+0,003; +0,052] | +0,065 [+0,042; +0,089] |
| H5 fidelidade: R7 − C4.5 | +0,126 [+0,087; +0,174] | +0,012 [+0,003; +0,021] | +0,011 [-0,007; +0,028] | +0,048 [+0,033; +0,065] |
| AT accuracy: R7 − Rq7 | +0,012 [+0,005; +0,020] | +0,143 [+0,129; +0,158] | +0,082 [+0,068; +0,096] | +0,081 [+0,059; +0,097] |
| AT fidelidade: R7 − Rq7 | -0,000 [-0,009; +0,013] | +0,131 [+0,098; +0,165] | +0,026 [+0,003; +0,049] | +0,075 [+0,043; +0,104] |
| MLP accuracy: MLP − C4.5 | +0,031 [-0,009; +0,071] | -0,048 [-0,088; -0,016] | -0,050 [-0,074; -0,026] | -0,014 [-0,024; -0,002] |

## 5. Decisão por dataset e orçamento

| Orçamento | Dataset | Elegível | Sustentada | Observada | Atribuível |
|---|---|---|---|---|---|
| estrutura do tuning | german_credit | não | não | não | não |
| estrutura do tuning | phys_bmi | não | não | não | não |
| estrutura do tuning | phys_kinetic | não | não | não | não |
| estrutura do tuning | phys_power | não | não | sim | sim |
| máx. 7 nós | german_credit | não | não | não | não |
| máx. 7 nós | phys_bmi | não | não | não | sim |
| máx. 7 nós | phys_kinetic | não | não | não | sim |
| máx. 7 nós | phys_power | não | não | não | sim |

Unidades em que as cinco condições se verificam:

| Orçamento | Dataset | H1 | H2 | H3 | H4 | H5 | Conjunta |
|---|---|---|---|---|---|---|---|
| estrutura do tuning | german_credit | 1/5 | 4/5 | 4/5 | 1/5 | 5/5 | 0/5 |
| estrutura do tuning | phys_bmi | 4/5 | 3/5 | 1/5 | 2/5 | 2/5 | 0/5 |
| estrutura do tuning | phys_kinetic | 4/5 | 3/5 | 2/5 | 2/5 | 3/5 | 1/5 |
| estrutura do tuning | phys_power | 4/5 | 5/5 | 3/5 | 4/5 | 5/5 | 1/5 |
| máx. 7 nós | german_credit | 2/5 | 3/5 | 3/5 | 2/5 | 5/5 | 0/5 |
| máx. 7 nós | phys_bmi | 5/5 | 5/5 | 0/5 | 5/5 | 4/5 | 0/5 |
| máx. 7 nós | phys_kinetic | 5/5 | 5/5 | 0/5 | 4/5 | 3/5 | 0/5 |
| máx. 7 nós | phys_power | 5/5 | 5/5 | 1/5 | 5/5 | 5/5 | 1/5 |

## 6. Leitura

**Elo «Reloaded > Original»: confirmado nos controlos positivos, e atribuível à ontologia.**

- Com 7 nós, o Reloaded supera o Original nos três controlos positivos, com intervalos acima de zero: accuracy +7,8 a +8,9 pontos e fidelidade +2,5 a +6,6 pontos.
- Supera também a versão com a ontologia permutada em accuracy e em fidelidade nos três (por exemplo, +14,3 pontos de accuracy no phys_bmi). O ganho vem do significado da relação declarada, não de ter mais colunas.
- O ganho é maior com o orçamento restrito do que com a estrutura do tuning (no phys_bmi, +8,7 contra +2,3 pontos de accuracy). Isto confirma a previsão: uma única divisão sobre a relação certa substitui uma escada de cortes ortogonais.
- O Reloaded com 7 nós usa em média 3 nós (uma divisão sobre o atributo derivado) e supera o C4.5 canónico, que usa 21 a 23 nós.

**Elo «Original > C4.5»: falha, porque o oráculo é pior do que o C4.5.**

- O MLP fica 1,4 a 5 pontos abaixo do C4.5 nos controlos positivos (intervalos abaixo de zero). O Original, que imita o MLP, fica abaixo ou empatado com o C4.5.
- Com 7 nós, o Original fica abaixo do C4.5 com intervalo abaixo de zero nos três controlos.
- **A hierarquia completa não se sustenta em nenhum dataset nem orçamento.** É apenas observada (cinco médias positivas) no phys_power com a estrutura do tuning, sem sustentação, porque `Original > C4.5` tem intervalo a incluir zero.

**German Credit (real):** o Reloaded não supera o Original (fidelidade −1,2 pontos com intervalo abaixo de zero na estrutura do tuning). A relação montante / prazo não melhora a árvore. Com 7 nós, a versão real supera a permutada em accuracy (+1,2 pontos), mas não em fidelidade.

**Observação não pré-registada:** nos controlos positivos, o Reloaded com 7 nós tem accuracy **acima do próprio MLP** (por exemplo 0,954 contra 0,874 no phys_bmi). A árvore com a relação certa aproxima a regra verdadeira melhor do que o oráculo que imita. Isto mostra que a ontologia pode corrigir um oráculo fraco; não é uma conclusão testada.

## 7. Médias por braço

| Dataset | Braço | Accuracy | Fidelidade | Nós | Profundidade | Consultas | Tempo (s) |
|---|---|---|---|---|---|---|---|
| german_credit | MLP | 0,731 | N/A | N/A | N/A | N/A | 0,0 |
| german_credit | C4.5 | 0,700 | 0,729 | 172,2 | 28,2 | N/A | 1,5 |
| german_credit | B | 0,724 | 0,865 | 59,3 | 9,5 | 34592 | 3,2 |
| german_credit | R | 0,715 | 0,853 | 57,9 | 9,4 | 34615 | 6,4 |
| german_credit | Rq | 0,712 | 0,847 | 59,1 | 10,1 | 34268 | 6,4 |
| german_credit | B7 | 0,720 | 0,857 | 4,6 | 1,8 | 2368 | 0,5 |
| german_credit | R7 | 0,715 | 0,855 | 5,5 | 2,3 | 2312 | 0,7 |
| german_credit | Rq7 | 0,703 | 0,855 | 3,5 | 1,3 | 2168 | 0,7 |
| phys_bmi | MLP | 0,874 | N/A | N/A | N/A | N/A | 0,0 |
| phys_bmi | C4.5 | 0,922 | 0,896 | 23,4 | 6,6 | N/A | 0,1 |
| phys_bmi | B | 0,908 | 0,899 | 38,7 | 7,5 | 20752 | 0,4 |
| phys_bmi | R | 0,931 | 0,903 | 38,6 | 9,2 | 21647 | 1,5 |
| phys_bmi | Rq | 0,784 | 0,755 | 39,0 | 7,5 | 20656 | 1,2 |
| phys_bmi | B7 | 0,867 | 0,842 | 5,9 | 2,3 | 2658 | 0,1 |
| phys_bmi | R7 | 0,954 | 0,908 | 3,7 | 1,3 | 2678 | 0,2 |
| phys_bmi | Rq7 | 0,811 | 0,777 | 6,1 | 2,5 | 2696 | 0,2 |
| phys_kinetic | MLP | 0,838 | N/A | N/A | N/A | N/A | 0,0 |
| phys_kinetic | C4.5 | 0,888 | 0,878 | 21,0 | 6,4 | N/A | 0,1 |
| phys_kinetic | B | 0,883 | 0,879 | 46,6 | 8,7 | 29205 | 0,5 |
| phys_kinetic | R | 0,910 | 0,877 | 42,1 | 8,7 | 29467 | 1,6 |
| phys_kinetic | Rq | 0,869 | 0,882 | 45,1 | 8,1 | 29466 | 1,4 |
| phys_kinetic | B7 | 0,849 | 0,863 | 5,7 | 2,3 | 3182 | 0,1 |
| phys_kinetic | R7 | 0,939 | 0,889 | 3,0 | 1,0 | 3203 | 0,2 |
| phys_kinetic | Rq7 | 0,857 | 0,863 | 6,1 | 2,4 | 3176 | 0,2 |
| phys_power | MLP | 0,846 | N/A | N/A | N/A | N/A | 0,0 |
| phys_power | C4.5 | 0,860 | 0,866 | 23,0 | 5,8 | N/A | 0,1 |
| phys_power | B | 0,864 | 0,881 | 47,8 | 7,3 | 21329 | 0,5 |
| phys_power | R | 0,901 | 0,904 | 44,7 | 8,5 | 21587 | 1,7 |
| phys_power | Rq | 0,873 | 0,876 | 46,3 | 8,1 | 21003 | 1,2 |
| phys_power | B7 | 0,847 | 0,849 | 6,6 | 2,6 | 2141 | 0,1 |
| phys_power | R7 | 0,925 | 0,914 | 3,0 | 1,0 | 2127 | 0,2 |
| phys_power | Rq7 | 0,844 | 0,839 | 6,3 | 2,7 | 2146 | 0,2 |

## 8. Limites

- Os controlos positivos foram desenhados por mim e a relação verdadeira está declarada na ontologia por construção. Mostram que o mecanismo **consegue** ganhar quando o conhecimento é certo e relevante; não estimam o ganho em dados reais.
- O German Credit foi observado pelo projeto antes, e o único dataset real desta fase não mostrou ganho.
- Não consegui obter datasets reais novos (OpenML e UCI bloqueados neste ambiente).
- O MLP do projeto ficou abaixo do C4.5 nos controlos positivos; não investiguei se isso é limitação do MLP (por exemplo, configuração para poucos dados) ou da tarefa.
