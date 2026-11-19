# Fase 3 — resultados

Análise feita segundo `FASE3_PREREGISTO.md` (commit `15558b5`, anterior à execução). Dados em `evidencia/fase3/`. Natureza: **DEV, exploratória**.

## 1. Execução

- **Braços principais** (`run_fase3 sampler`): 15 de 15 unidades com código de saída 0; a mais longa demorou 163 s.
- **Controlos** (`run_fase3 controls`): 15 de 15 unidades com código de saída 0, sem falhas de comparabilidade; a mais longa demorou 335 s.
- 3 réplicas por braço em todas as unidades. Nenhuma unidade excluída.
- Verificação independente das métricas: 1605 + 1335 valores recalculados, diferença máxima 0.
- **Condição de integridade do pré-registo: cumprida.** O Original com amostrador conjunto reproduz os valores da fase 2 em todas as unidades (diferença máxima 1,1 × 10⁻¹⁶).
- Incidente: a unidade iris/42 dos braços principais foi executada por engano em primeiro plano antes do lançamento da campanha, com o mesmo comando e configuração; foi mantida.
- As ontologias v2 passaram o quality gate com cobertura 100 % e geraram 8 atributos derivados no iris (2 na v1), 8 no wine (3 na v1) e 59 no breast_cancer (53 na v1). Em média, as árvores usaram 4,1, 4,8 e 7,3 atributos derivados, respetivamente.

## 2. Hierarquia

B = Original + amostrador conjunto; D2 = Reloaded + OWL v2 + amostrador conjunto.

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| H1 accuracy: D2 − B | -0,012 [-0,030; +0,002] | -0,025 [-0,052; +0,003] | -0,004 [-0,013; +0,002] |
| H2 accuracy: D2 − C4.5 | -0,014 [-0,026; -0,002] | -0,006 [-0,042; +0,030] | +0,003 [-0,018; +0,024] |
| H3 accuracy: B − C4.5 | -0,002 [-0,011; +0,009] | +0,019 [-0,000; +0,038] | +0,007 [-0,010; +0,026] |
| H4 fidelidade: D2 − B | +0,009 [-0,004; +0,026] | -0,021 [-0,049; +0,004] | -0,006 [-0,009; -0,003] |
| H5 fidelidade: D2 − C4.5 | +0,011 [-0,005; +0,037] | +0,003 [-0,034; +0,040] | +0,006 [-0,006; +0,021] |

| Dataset | H1 | H2 | H3 | H4 | H5 | Conjunta |
|---|---|---|---|---|---|---|
| iris | 1/5 | 0/5 | 1/5 | 2/5 | 1/5 | 0/5 |
| wine | 2/5 | 2/5 | 3/5 | 1/5 | 3/5 | 1/5 |
| breast_cancer | 2/5 | 2/5 | 2/5 | 0/5 | 3/5 | 0/5 |
| **Total** | 5/15 | 4/15 | 6/15 | 3/15 | 7/15 | 1/15 |

**Decisão segundo o pré-registo:**

- **Hierarquia sustentada:** em nenhum dataset.
- **Hierarquia observada:** em nenhum dataset. H1 tem média negativa nos três.
- **H2 no iris** e **H4 no breast_cancer** têm intervalo abaixo de zero: com a ontologia v2, o Reloaded fica abaixo do C4.5 em accuracy no iris e é menos fiel do que o Original no breast_cancer.
- **Por unidade:** as cinco condições verificam-se em 1 de 15 unidades (wine, semente 123).

## 3. As relações declaradas acrescentam algo às ontologias v1?

D1 = Reloaded + OWL v1 + amostrador conjunto (fase 2).

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| K1 accuracy: D2 − D1 | -0,011 [-0,026; +0,004] | -0,012 [-0,028; +0,006] | -0,003 [-0,012; +0,007] |
| K1 fidelidade: D2 − D1 | +0,021 [+0,005; +0,040] | -0,003 [-0,019; +0,013] | -0,003 [-0,012; +0,006] |

- **No iris a fidelidade sobe com intervalo acima de zero** (+2,1 pontos), mas a accuracy desce 1,1 pontos. O critério pré-registado exigia ganho nas duas métricas; não se cumpre.
- No wine e no breast_cancer não há diferença distinguível de zero.

**Estado: as relações declaradas não ajudam segundo o critério pré-registado.**

## 4. Atribuição à ontologia

Pelo pré-registo, só interpretável se H1 e H4 se observassem. Não se observam; os contrastes ficam como descrição.

| Contraste | iris | wine | breast_cancer |
|---|---|---|---|
| accuracy: D2 − F | -0,012 [-0,030; +0,002] | -0,025 [-0,052; +0,003] | -0,004 [-0,013; +0,002] |
| fidelidade: D2 − F | +0,009 [-0,004; +0,026] | -0,021 [-0,049; +0,004] | -0,006 [-0,009; -0,003] |
| accuracy: D2 − P2 | +0,011 [-0,004; +0,032] | -0,006 [-0,036; +0,033] | +0,002 [-0,006; +0,008] |
| fidelidade: D2 − P2 | +0,018 [-0,005; +0,040] | -0,003 [-0,031; +0,034] | -0,003 [-0,011; +0,005] |
| accuracy: D2 − Dg2 | +0,014 [-0,009; +0,037] | -0,016 [-0,044; +0,018] | +0,001 [-0,013; +0,011] |
| fidelidade: D2 − Dg2 | +0,028 [+0,009; +0,047] | -0,010 [-0,024; +0,004] | +0,001 [-0,010; +0,012] |
| accuracy: D2 − Ec2 | -0,005 [-0,012; +0,000] | -0,002 [-0,022; +0,021] | -0,011 [-0,033; +0,009] |
| fidelidade: D2 − Ec2 | +0,009 [-0,004; +0,025] | +0,004 [-0,018; +0,027] | -0,009 [-0,021; +0,005] |
| accuracy: D2 − G2 | -0,009 [-0,018; -0,002] | -0,037 [-0,050; -0,024] | -0,008 [-0,023; +0,002] |
| fidelidade: D2 − G2 | +0,005 [-0,007; +0,018] | -0,025 [-0,040; -0,010] | -0,011 [-0,017; -0,004] |

- A ontologia v2 só fica acima de um controlo com intervalo a excluir zero num caso: fidelidade no iris face aos grupos aleatórios.
- Fica **abaixo** do Original com as mesmas consultas (G2) em accuracy no iris e no wine, e em fidelidade no wine e no breast_cancer, sempre com intervalo abaixo de zero.
- Não se distingue da ontologia permutada (P2) em nenhum dataset nem métrica.

**Estado: nenhum ganho atribuível à ontologia.**

## 5. Médias por condição

| Dataset | Condição | Accuracy | Fidelidade | Nós | Amostras do oráculo | Tempo (s) |
|---|---|---|---|---|---|---|
| iris | MLP | 0,953 | N/A | N/A | N/A | 0,0 |
| iris | C4.5 | 0,958 | 0,963 | 8,6 | N/A | 0,0 |
| iris | B Original + conjunto | 0,956 | 0,965 | 35,0 | 10337 | 0,2 |
| iris | G2 Original, consultas igualadas | 0,953 | 0,968 | 33,8 | 24280 | 0,3 |
| iris | D1 Reloaded+OWL v1 + conjunto (fase 2) | 0,954 | 0,953 | 35,7 | 23132 | 1,8 |
| iris | D2 Reloaded+OWL v2 + conjunto | 0,944 | 0,974 | 34,7 | 22929 | 2,2 |
| iris | F Reloaded sem OWL + conjunto | 0,956 | 0,965 | 35,0 | 10449 | 0,2 |
| iris | P2 permutada | 0,933 | 0,956 | 34,6 | 22892 | 4,1 |
| iris | Dg2 grupos aleatórios | 0,930 | 0,946 | 35,1 | 22955 | 3,0 |
| iris | Ec2 eng. convencional | 0,949 | 0,965 | 34,2 | 22871 | 0,5 |
| wine | MLP | 0,969 | N/A | N/A | N/A | 0,0 |
| wine | C4.5 | 0,929 | 0,916 | 9,8 | N/A | 0,1 |
| wine | B Original + conjunto | 0,948 | 0,939 | 29,0 | 10615 | 1,7 |
| wine | G2 Original, consultas igualadas | 0,960 | 0,944 | 33,1 | 32340 | 1,5 |
| wine | D1 Reloaded+OWL v1 + conjunto (fase 2) | 0,935 | 0,921 | 26,3 | 20483 | 9,4 |
| wine | D2 Reloaded+OWL v2 + conjunto | 0,923 | 0,919 | 34,2 | 25185 | 12,5 |
| wine | F Reloaded sem OWL + conjunto | 0,948 | 0,939 | 29,0 | 10748 | 1,7 |
| wine | P2 permutada | 0,929 | 0,921 | 34,5 | 26303 | 16,3 |
| wine | Dg2 grupos aleatórios | 0,939 | 0,929 | 36,5 | 25800 | 14,6 |
| wine | Ec2 eng. convencional | 0,924 | 0,914 | 28,7 | 20891 | 2,0 |
| breast_cancer | MLP | 0,962 | N/A | N/A | N/A | 0,0 |
| breast_cancer | C4.5 | 0,930 | 0,923 | 17,0 | N/A | 0,4 |
| breast_cancer | B Original + conjunto | 0,937 | 0,935 | 35,9 | 16725 | 2,1 |
| breast_cancer | G2 Original, consultas igualadas | 0,941 | 0,940 | 35,8 | 42700 | 3,0 |
| breast_cancer | D1 Reloaded+OWL v1 + conjunto (fase 2) | 0,936 | 0,931 | 33,8 | 40075 | 13,9 |
| breast_cancer | D2 Reloaded+OWL v2 + conjunto | 0,933 | 0,929 | 35,3 | 40135 | 17,3 |
| breast_cancer | F Reloaded sem OWL + conjunto | 0,937 | 0,935 | 35,9 | 17151 | 2,2 |
| breast_cancer | P2 permutada | 0,931 | 0,931 | 35,5 | 39639 | 43,2 |
| breast_cancer | Dg2 grupos aleatórios | 0,932 | 0,928 | 34,2 | 40244 | 17,7 |
| breast_cancer | Ec2 eng. convencional | 0,944 | 0,938 | 34,7 | 38026 | 3,6 |

Desvio-padrão médio entre réplicas: accuracy 1,2 a 2,8 pontos; fidelidade 1,3 a 2,7 pontos.

## 6. Leitura

1. **O mecanismo funciona tecnicamente**: as relações são lidas, geram atributos e as árvores usam-nos.
2. **Não melhora os resultados.** Dar à árvore grandezas derivadas com significado de domínio não a tornou mais exata nem, no geral, mais fiel ao MLP.
3. **O que continua a ganhar é gastar o mesmo orçamento sem ontologia.** O Original com as consultas igualadas é a melhor árvore no wine (0,960) e fica acima do Reloaded com ontologia nos três datasets em accuracy.
4. **Hipótese para o efeito negativo** (não testada): o Reloaded dá um peso de ganho fixo às colunas derivadas e gasta consultas em intervenções semânticas; isso desvia a procura das divisões estatisticamente melhores. É o problema P06 da auditoria, que esta fase não alterou.

## 7. Limites

- 5 unidades por dataset e 3 datasets; todos já observados; contaminação declarada no pré-registo.
- As relações são poucas e simples; não é uma avaliação de «ontologias ricas» em geral.
- O digits não entrou (sem ontologia v2).
- Não separei o efeito dos atributos derivados do efeito do peso fixo que o Reloaded lhes dá.
