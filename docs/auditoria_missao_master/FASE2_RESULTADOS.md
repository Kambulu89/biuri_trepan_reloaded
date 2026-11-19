# Fase 2 — resultados

Análise feita segundo `FASE2_PREREGISTO.md` (commit `794f25c`, anterior à execução). Dados em `evidencia/fase2/`. Natureza: **DEV, exploratória**.

## 1. Execução

- **Campanha 1** (`run_sampler_c`, `--reps 3 --c-replicas all`): 20 de 20 unidades com código de saída 0; a mais longa demorou 1099 s.
- **Campanha 2** (`run_controls_m29`, `--reps 3`): 20 de 20 unidades com código de saída 0; a verificação de comparabilidade estrita do executor não falhou em nenhuma; a mais longa demorou 1267 s.
- Nenhuma unidade atingiu o limite operacional de 5400 s. Nenhuma foi excluída.
- Partições, MLP (`oracle_id`) e estrutura das árvores: os da fase 1; o Original histórico reproduz os valores da fase 1.
- Verificação independente das métricas: 2380 valores na campanha 1 e 1780 na campanha 2, diferença máxima 0.
- 3 réplicas em todos os braços com réplicas, em todas as unidades.

## 2. Hierarquia sob a referência com amostrador conjunto

B = Original + amostrador conjunto; D = Reloaded + OWL + amostrador conjunto. Diferença média emparelhada e intervalo a 95 %.

| Contraste | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| H1 accuracy: D − B | -0,002 [-0,007; +0,004] | -0,013 [-0,049; +0,015] | -0,001 [-0,008; +0,007] | -0,012 [-0,022; +0,000] |
| H2 accuracy: D − C4.5 | -0,004 [-0,009; +0,004] | +0,006 [-0,034; +0,040] | +0,006 [-0,009; +0,020] | +0,007 [-0,014; +0,027] |
| H3 accuracy: B − C4.5 | -0,002 [-0,012; +0,009] | +0,019 [-0,000; +0,038] | +0,007 [-0,010; +0,026] | +0,019 [+0,004; +0,029] |
| H4 fidelidade: D − B | -0,012 [-0,025; -0,004] | -0,018 [-0,058; +0,004] | -0,003 [-0,009; +0,002] | -0,009 [-0,021; +0,005] |
| H5 fidelidade: D − C4.5 | -0,011 [-0,025; +0,002] | +0,006 [-0,044; +0,047] | +0,008 [-0,005; +0,024] | +0,013 [-0,011; +0,035] |

**Decisão segundo o pré-registo:**

- **Hierarquia sustentada:** em nenhum dataset.
- **Hierarquia observada** (cinco médias positivas): em nenhum dataset. H1 e H4 têm média negativa nos quatro.
- **H3 (Original > C4.5 em accuracy):** sustentada no digits; observada, não sustentada, no wine e no breast_cancer; negativa no iris.
- **H4 no iris:** o intervalo fica abaixo de zero. O Reloaded com ontologia é menos fiel do que o Original.
- **Por unidade:** as cinco condições verificam-se em 1 de 20 unidades (digits, semente 42). Na fase 1 eram 0 de 20.

| Condição | Unidades (de 20) |
|---|---|
| H1 | 4 |
| H2 | 10 |
| H3 | 10 |
| H4 | 3 |
| H5 | 9 |
| Conjunta | 1 |

## 3. Atribuição à ontologia

Pelo pré-registo, estes contrastes só seriam interpretados como atribuição se H1 e H4 se observassem. Não se observam. Reporto-os como descrição.

| Contraste | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| accuracy: D − Reloaded sem OWL | -0,002 [-0,007; +0,004] | -0,013 [-0,049; +0,015] | -0,001 [-0,008; +0,007] | -0,012 [-0,022; +0,000] |
| fidelidade: D − Reloaded sem OWL | -0,012 [-0,025; -0,004] | -0,018 [-0,058; +0,004] | -0,003 [-0,009; +0,002] | -0,009 [-0,021; +0,005] |
| accuracy: D − ontologia permutada | +0,012 [-0,000; +0,025] | +0,013 [-0,025; +0,056] | -0,003 [-0,013; +0,010] | -0,007 [-0,014; -0,001] |
| fidelidade: D − ontologia permutada | -0,007 [-0,021; +0,011] | +0,010 [-0,030; +0,053] | -0,001 [-0,015; +0,010] | -0,004 [-0,013; +0,005] |
| accuracy: D − grupos aleatórios | +0,011 [-0,007; +0,025] | -0,010 [-0,036; +0,013] | -0,007 [-0,017; +0,002] | -0,003 [-0,011; +0,008] |
| fidelidade: D − grupos aleatórios | +0,000 [-0,011; +0,011] | -0,002 [-0,034; +0,027] | -0,004 [-0,017; +0,009] | -0,004 [-0,011; +0,009] |
| accuracy: D − engenharia convencional | +0,005 [-0,007; +0,018] | -0,025 [-0,053; -0,009] | -0,008 [-0,024; +0,005] | -0,020 [-0,029; -0,006] |
| fidelidade: D − engenharia convencional | -0,012 [-0,025; +0,000] | -0,033 [-0,067; -0,006] | -0,006 [-0,023; +0,010] | -0,018 [-0,026; -0,004] |
| accuracy: D − Original com consultas igualadas | -0,000 [-0,016; +0,023] | -0,015 [-0,042; +0,004] | -0,001 [-0,007; +0,006] | -0,021 [-0,031; -0,011] |
| fidelidade: D − Original com consultas igualadas | -0,014 [-0,025; -0,005] | -0,012 [-0,040; +0,006] | -0,007 [-0,023; +0,008] | -0,016 [-0,031; +0,000] |

- A ontologia real **não supera nenhum controlo com intervalo acima de zero**, em nenhum dataset e em nenhuma métrica.
- Fica **abaixo** com intervalo a excluir zero: da engenharia convencional no wine (accuracy e fidelidade) e no digits (ambas); do Original com consultas igualadas no digits (accuracy) e no iris (fidelidade); da permutada no digits (accuracy); do Reloaded sem ontologia no iris (fidelidade).
- D − F é igual a D − B porque o Reloaded sem ontologia continua idêntico ao Original também com o amostrador conjunto.

**Estado: nenhum ganho atribuível à ontologia.** Coerente com os documentos 30 e 42 do projeto (`NO_ONTOLOGICAL_ADVANTAGE`).

## 4. Efeito do amostrador conjunto (descritivo)

A e C usam o amostrador marginal histórico.

| Contraste | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| Original, accuracy: B − A | +0,019 [+0,007; +0,032] | +0,030 [+0,004; +0,053] | +0,005 [-0,003; +0,013] | +0,047 [+0,032; +0,063] |
| Original, fidelidade: B − A | +0,009 [-0,016; +0,032] | +0,022 [+0,000; +0,046] | +0,004 [+0,001; +0,006] | +0,043 [+0,027; +0,058] |
| Reloaded + OWL, accuracy: D − C | +0,014 [+0,005; +0,025] | +0,030 [-0,000; +0,062] | +0,001 [-0,015; +0,013] | +0,034 [+0,029; +0,042] |
| Reloaded + OWL, fidelidade: D − C | -0,004 [-0,018; +0,011] | +0,019 [-0,010; +0,050] | -0,002 [-0,015; +0,006] | +0,034 [+0,028; +0,041] |

O amostrador conjunto melhora a accuracy do Original nos quatro datasets, com intervalo acima de zero em três, e a fidelidade também nos quatro, com intervalo acima de zero em dois.

## 5. Médias por condição

| Dataset | Condição | Accuracy | Fidelidade | Nós | Amostras do oráculo | Tempo (s) |
|---|---|---|---|---|---|---|
| iris | MLP | 0,953 | N/A | N/A | N/A | 0,0 |
| iris | C4.5 | 0,958 | 0,963 | 8,6 | N/A | 0,0 |
| iris | A Original (marginal) | 0,937 | 0,956 | 34,1 | 9813 | 0,2 |
| iris | B Original + conjunto | 0,956 | 0,965 | 35,0 | 10337 | 0,2 |
| iris | G Original + conjunto, consultas igualadas | 0,954 | 0,967 | 35,0 | 23453 | 0,3 |
| iris | C Reloaded+OWL (marginal) | 0,940 | 0,956 | 34,9 | 22033 | 1,9 |
| iris | D Reloaded+OWL + conjunto | 0,954 | 0,953 | 35,7 | 23132 | 1,8 |
| iris | F Reloaded sem OWL + conjunto | 0,956 | 0,965 | 35,0 | 10449 | 0,2 |
| iris | P permutada + conjunto | 0,942 | 0,960 | 35,3 | 23627 | 4,5 |
| iris | Dg grupos aleatórios + conjunto | 0,944 | 0,953 | 34,6 | 23803 | 2,4 |
| iris | Ec eng. convencional + conjunto | 0,949 | 0,965 | 34,2 | 22871 | 0,5 |
| wine | MLP | 0,969 | N/A | N/A | N/A | 0,0 |
| wine | C4.5 | 0,929 | 0,916 | 9,8 | N/A | 0,0 |
| wine | A Original (marginal) | 0,919 | 0,917 | 39,7 | 10165 | 0,8 |
| wine | B Original + conjunto | 0,948 | 0,939 | 29,0 | 10615 | 1,5 |
| wine | G Original + conjunto, consultas igualadas | 0,950 | 0,933 | 32,1 | 26894 | 1,7 |
| wine | C Reloaded+OWL (marginal) | 0,905 | 0,902 | 40,7 | 23086 | 11,6 |
| wine | D Reloaded+OWL + conjunto | 0,935 | 0,921 | 26,3 | 20483 | 9,4 |
| wine | F Reloaded sem OWL + conjunto | 0,948 | 0,939 | 29,0 | 10748 | 1,5 |
| wine | P permutada + conjunto | 0,921 | 0,911 | 32,6 | 25697 | 15,6 |
| wine | Dg grupos aleatórios + conjunto | 0,945 | 0,923 | 35,4 | 26734 | 15,3 |
| wine | Ec eng. convencional + conjunto | 0,960 | 0,954 | 27,8 | 20676 | 1,5 |
| breast_cancer | MLP | 0,962 | N/A | N/A | N/A | 0,0 |
| breast_cancer | C4.5 | 0,930 | 0,923 | 17,0 | N/A | 0,3 |
| breast_cancer | A Original (marginal) | 0,932 | 0,931 | 31,5 | 17386 | 1,6 |
| breast_cancer | B Original + conjunto | 0,937 | 0,935 | 35,9 | 16725 | 1,8 |
| breast_cancer | G Original + conjunto, consultas igualadas | 0,936 | 0,938 | 34,9 | 43371 | 3,0 |
| breast_cancer | C Reloaded+OWL (marginal) | 0,935 | 0,934 | 31,0 | 39312 | 14,8 |
| breast_cancer | D Reloaded+OWL + conjunto | 0,936 | 0,931 | 33,8 | 40075 | 13,9 |
| breast_cancer | F Reloaded sem OWL + conjunto | 0,937 | 0,935 | 35,9 | 17151 | 1,8 |
| breast_cancer | P permutada + conjunto | 0,938 | 0,932 | 34,9 | 40722 | 39,2 |
| breast_cancer | Dg grupos aleatórios + conjunto | 0,943 | 0,935 | 34,9 | 40172 | 17,0 |
| breast_cancer | Ec eng. convencional + conjunto | 0,944 | 0,938 | 34,7 | 38026 | 3,3 |
| digits | MLP | 0,960 | N/A | N/A | N/A | 0,0 |
| digits | C4.5 | 0,853 | 0,843 | 169,4 | N/A | 1,6 |
| digits | A Original (marginal) | 0,824 | 0,823 | 229,3 | 180717 | 10,5 |
| digits | B Original + conjunto | 0,872 | 0,866 | 252,3 | 178284 | 11,0 |
| digits | G Original + conjunto, consultas igualadas | 0,882 | 0,872 | 253,0 | 390663 | 22,1 |
| digits | C Reloaded+OWL (marginal) | 0,826 | 0,822 | 233,1 | 385710 | 103,6 |
| digits | D Reloaded+OWL + conjunto | 0,860 | 0,856 | 252,2 | 383506 | 96,9 |
| digits | F Reloaded sem OWL + conjunto | 0,872 | 0,866 | 252,3 | 179631 | 10,9 |
| digits | P permutada + conjunto | 0,868 | 0,860 | 253,1 | 383476 | 116,2 |
| digits | Dg grupos aleatórios + conjunto | 0,863 | 0,860 | 252,3 | 384019 | 113,8 |
| digits | Ec eng. convencional + conjunto | 0,880 | 0,874 | 252,7 | 375580 | 18,1 |

Desvio-padrão médio entre réplicas (tudo o resto fixo): accuracy 1,0 a 3,1 pontos; fidelidade 1,2 a 2,5 pontos.

## 6. Observação posterior, fora do pré-registo

Estes contrastes não estavam pré-registados; foram calculados depois de ver os dados e valem como hipótese para uma fase seguinte, não como resultado.

| Contraste | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| accuracy: G − C4.5 | -0,004 [-0,030; +0,016] | +0,021 [+0,003; +0,039] | +0,006 [-0,007; +0,022] | +0,029 [+0,013; +0,045] |
| fidelidade: G − C4.5 | +0,004 [-0,005; +0,016] | +0,018 [-0,012; +0,047] | +0,015 [+0,003; +0,028] | +0,029 [+0,015; +0,046] |
| accuracy: Ec − C4.5 | -0,009 [-0,023; +0,000] | +0,031 [+0,009; +0,053] | +0,014 [+0,010; +0,018] | +0,028 [+0,013; +0,039] |
| fidelidade: Ec − C4.5 | +0,002 [-0,011; +0,016] | +0,039 [+0,022; +0,055] | +0,015 [+0,008; +0,021] | +0,031 [+0,012; +0,049] |

G = Original + amostrador conjunto com as consultas igualadas às do Reloaded; Ec = Reloaded com atributos de engenharia convencional (sem ontologia) + amostrador conjunto. Ambos superam o C4.5 com intervalo acima de zero em dois ou três datasets. Os dois são melhorias não ontológicas.

## 7. Conclusão da fase

1. **A hipótese continua por confirmar.** Com a referência melhorada, o primeiro elo (Original > C4.5) passa a verificar-se num dataset e a observar-se em mais dois; os elos que dependem da ontologia (Reloaded > Original) são negativos nos quatro.
2. **O que aproxima o sistema do C4.5 e do MLP não é a ontologia:** é o amostrador conjunto, mais consultas e atributos derivados convencionais.
3. **Com as ontologias estruturais atuais e os mecanismos atuais, a ontologia não acrescenta nada mensurável e por vezes prejudica.**

## 8. Limites

- 5 unidades por dataset, 4 datasets, todos já observados. Nada é confirmatório.
- Só ontologias estruturais; a enriquecida do Breast Cancer não entrou nesta fase.
- A estrutura das árvores foi escolhida pelo tuning com o amostrador marginal; não foi reafinada para o conjunto.
- O iris continua inviável para H3: o MLP não supera o C4.5.
