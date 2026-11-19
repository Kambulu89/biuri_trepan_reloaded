# Fase 4 — resultados

Análise feita segundo `FASE4_PREREGISTO.md` (commit `e45e8cd`, anterior à campanha). Dados em `evidencia/fase4/`. Natureza: **DEV, exploratória**.

## 1. Execução

- 15 unidades × 4 braços × 3 réplicas. 14 unidades terminaram na primeira passagem.
- **Incidente:** a unidade iris/42 falhou em 2 s, antes de qualquer cálculo, porque as quatro unidades iniciais escreveram ao mesmo tempo o ficheiro que regista o hash das ontologias v2. Corrigi o executor com um bloqueio de ficheiro (commit separado) e voltei a correr só essa unidade, que terminou normalmente. O registo da falha está guardado (`logs/iris_seed42_falha_corrida.log` na pasta da campanha).
- Nenhum braço foi saltado em nenhuma unidade.
- Verificação independente das métricas: 1515 valores, diferença máxima 0.
- **Integridade: cumprida.** B reproduz a fase 3 em todas as unidades (diferença máxima 1.1e-16).

## 2. J1 — os atributos da ontologia, sem os mecanismos, ajudam o Original?

O = Original-equivalente no espaço enriquecido pela OWL v2 (mecanismos do Reloaded desligados); B = Original no espaço original. Ambos com amostrador conjunto.

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| J1 accuracy: O − B | -0,000 [-0,018; +0,012] | -0,006 [-0,033; +0,012] | +0,005 [-0,005; +0,014] |
| J1 fidelidade: O − B | +0,000 [-0,014; +0,011] | -0,013 [-0,038; +0,004] | +0,002 [-0,005; +0,009] |

**Não.** As diferenças estão todas a menos de 1,4 pontos de zero e todos os intervalos incluem zero. Com o mesmo algoritmo e o mesmo número de consultas, acrescentar os atributos da ontologia não muda a árvore de forma mensurável.

## 3. J2 — atribuição ao significado da ontologia

Q = O com a ontologia permutada; K = O com atributos convencionais.

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| J2 accuracy: O − Q | +0,011 [+0,002; +0,019] | -0,019 [-0,038; +0,000] | +0,001 [-0,015; +0,016] |
| J2 fidelidade: O − Q | +0,007 [+0,000; +0,018] | -0,025 [-0,036; -0,012] | +0,001 [-0,009; +0,010] |
| J2 accuracy: O − K | +0,007 [-0,002; +0,018] | +0,018 [+0,007; +0,028] | -0,002 [-0,017; +0,008] |
| J2 fidelidade: O − K | +0,000 [-0,014; +0,009] | +0,012 [-0,010; +0,028] | -0,001 [-0,010; +0,007] |

- No iris, O fica acima da permutada em accuracy (+1,1 pontos, intervalo acima de zero); em fidelidade o limite inferior toca zero.
- No wine é o contrário: a permutada fica **acima** de O em fidelidade (−2,5 pontos, intervalo abaixo de zero) e em accuracy (limite superior em zero).
- No wine, O fica acima dos atributos convencionais em accuracy (+1,8 pontos, intervalo acima de zero).
- Como J1 não se cumpre, o pré-registo não permite interpretar estes contrastes como atribuição. Os sinais opostos entre iris e wine são compatíveis com ruído.

**Estado: nenhum efeito atribuível ao significado da ontologia.**

## 4. J3 — os mecanismos do Reloaded ajudam ou prejudicam, dados os mesmos atributos?

D2 = Reloaded completo com a OWL v2 (fase 3); O = o mesmo espaço sem os mecanismos.

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| J3 accuracy: D2 − O | -0,012 [-0,019; -0,009] | -0,019 [-0,038; +0,000] | -0,009 [-0,019; +0,001] |
| J3 fidelidade: D2 − O | +0,009 [-0,005; +0,023] | -0,007 [-0,024; +0,007] | -0,008 [-0,013; -0,001] |

- **iris:** accuracy −1,2 pontos com intervalo abaixo de zero (5 de 5 unidades negativas); fidelidade sem diferença distinguível.
- **breast_cancer:** fidelidade −0,8 pontos com intervalo abaixo de zero; accuracy sem diferença distinguível.
- **wine:** accuracy −1,9 pontos (4 de 5 unidades negativas), com o limite superior do intervalo em zero; fidelidade sem diferença distinguível.

**Pelo critério pré-registado, os mecanismos do Reloaded prejudicam no iris e no breast_cancer** (intervalo abaixo de zero numa métrica e não acima na outra). No wine a direção é a mesma, mas o intervalo não exclui zero.

## 5. Hierarquia com O no lugar do Reloaded

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| H accuracy: O − C4.5 | -0,002 [-0,011; +0,007] | +0,013 [-0,016; +0,038] | +0,012 [-0,003; +0,023] |
| H fidelidade: O − C4.5 | +0,002 [-0,016; +0,021] | +0,010 [-0,025; +0,044] | +0,013 [+0,005; +0,022] |
| H accuracy: B − C4.5 | -0,002 [-0,011; +0,009] | +0,019 [+0,000; +0,038] | +0,007 [-0,010; +0,026] |
| H fidelidade: B − C4.5 | +0,002 [-0,005; +0,011] | +0,024 [-0,003; +0,047] | +0,012 [+0,001; +0,025] |

| Dataset | O>B acc | O>C4.5 acc | B>C4.5 acc | O>B fid | O>C4.5 fid | conjunta |
|---|---|---|---|---|---|---|
| iris | 3/5 | 2/5 | 2/5 | 2/5 | 2/5 | 1/5 |
| wine | 2/5 | 4/5 | 3/5 | 1/5 | 3/5 | 0/5 |
| breast_cancer | 2/5 | 4/5 | 2/5 | 3/5 | 5/5 | 0/5 |
| **Total** | 7/15 | 10/15 | 7/15 | 6/15 | 10/15 | 1/15 |

Nenhum dataset tem a hierarquia sustentada. O supera o C4.5 em fidelidade no breast_cancer com intervalo acima de zero; tudo o resto inclui zero.

## 6. Médias por braço

| Dataset | Braço | Accuracy | Fidelidade | Nós | Consultas da árvore final | Amostras do oráculo | Tempo (s) |
|---|---|---|---|---|---|---|---|
| iris | MLP | 0,953 | N/A | N/A | N/A | N/A | 0,0 |
| iris | C4.5 | 0,958 | 0,963 | 8,6 | N/A | N/A | 0,0 |
| iris | B Original + conjunto | 0,956 | 0,965 | 35,0 | 10337 | 10337 | 0,3 |
| iris | O Original-equivalente, espaço OWL v2 | 0,956 | 0,965 | 32,6 | 10856 | 23140 | 1,8 |
| iris | Q O com OWL v2 permutada | 0,946 | 0,958 | 34,2 | 10636 | 23050 | 1,5 |
| iris | K O com engenharia convencional | 0,949 | 0,965 | 34,2 | 10593 | 22871 | 0,5 |
| iris | D2 Reloaded + OWL v2 (fase 3) | 0,944 | 0,974 | 34,7 | N/A | 22929 | 2,2 |
| wine | MLP | 0,969 | N/A | N/A | N/A | N/A | 0,0 |
| wine | C4.5 | 0,929 | 0,916 | 9,8 | N/A | N/A | 0,1 |
| wine | B Original + conjunto | 0,948 | 0,939 | 29,0 | 10615 | 10615 | 1,9 |
| wine | O Original-equivalente, espaço OWL v2 | 0,942 | 0,926 | 37,1 | 11332 | 24507 | 10,0 |
| wine | Q O com OWL v2 permutada | 0,961 | 0,951 | 33,7 | 11826 | 25785 | 11,9 |
| wine | K O com engenharia convencional | 0,924 | 0,914 | 28,7 | 9687 | 20891 | 2,1 |
| wine | D2 Reloaded + OWL v2 (fase 3) | 0,923 | 0,919 | 34,2 | N/A | 25185 | 12,5 |
| breast_cancer | MLP | 0,962 | N/A | N/A | N/A | N/A | 0,0 |
| breast_cancer | C4.5 | 0,930 | 0,923 | 17,0 | N/A | N/A | 0,4 |
| breast_cancer | B Original + conjunto | 0,937 | 0,935 | 35,9 | 16725 | 16725 | 2,6 |
| breast_cancer | O Original-equivalente, espaço OWL v2 | 0,942 | 0,937 | 34,5 | 17357 | 38693 | 10,4 |
| breast_cancer | Q O com OWL v2 permutada | 0,940 | 0,936 | 34,7 | 17609 | 39094 | 11,4 |
| breast_cancer | K O com engenharia convencional | 0,944 | 0,938 | 34,7 | 17155 | 38026 | 4,2 |
| breast_cancer | D2 Reloaded + OWL v2 (fase 3) | 0,933 | 0,929 | 35,3 | N/A | 40135 | 17,3 |

«Amostras do oráculo» de O, Q e K inclui a construção da porção Original equivalente que o Reloaded faz no estado S3; as «consultas da árvore final» são comparáveis às de B.

## 7. Leitura

1. **A perda da fase 3 vinha dos mecanismos do Reloaded, não dos atributos.** Com os mecanismos desligados, os mesmos atributos deixam a árvore igual à do Original; com eles ligados, a árvore piora em dois dos três datasets.
2. **Os atributos da ontologia não acrescentam informação útil ao TREPAN nestes datasets.** É coerente com a fase 1: o MLP já capta essas grandezas a partir das colunas originais, e as consultas ao oráculo dão ao TREPAN acesso a essa função.
3. **Para o projeto, a consequência prática é simples:** nas condições testadas, a melhor árvore extraída é o Original com o amostrador conjunto. Nenhuma das formas de usar a ontologia testadas nas quatro fases a melhorou.

## 8. Limites

- O não é o Original canónico: é o Reloaded com os mecanismos desligados, num espaço com mais colunas e com a pré-seleção de 12 atributos por nó aplicada a esse espaço.
- 5 unidades por dataset, 3 datasets já observados, relações escolhidas por mim (contaminação declarada na fase 3).
- J3 junta todos os mecanismos; não separei qual deles (peso fixo, Gain_Reloaded, refinamento focado no erro, consultas ativas) é responsável pela perda.
