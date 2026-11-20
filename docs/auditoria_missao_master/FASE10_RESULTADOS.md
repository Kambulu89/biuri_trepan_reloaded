# Fase 10 — resultados: orçamento de consultas efetivo ao oráculo

Análise segundo `FASE10_PREREGISTO.md` (commit `cf54508`, anterior à campanha). Nenhuma hipótese, critério ou braço foi alterado. O núcleo não foi alterado. As fases 1 a 9 não foram alteradas. Dados em `evidencia/fase10/`.

## 1. Execução

- 45 de 45 unidades com código de saída 0 (9 datasets × 5 sementes; 6 braços TREPAN × 3 réplicas). Nenhum braço descartado.
- Recalculei as métricas a partir das previsões guardadas: 6165 valores, com diferença máxima de 0.
- **Integridade:** 630 de 630 previsões (MLP, C4.5, B, B7, R e R7) iguais às das fases 8 e 9 na mesma unidade, já com o código atual da `main` (com P20/P21). O registador do oráculo não muda nenhum resultado.

## 2. Resultados

| Dataset | Orçamento | Pontos novos B (média) | Pontos novos R (média) | Razão novos R/B (mín.–máx.) | Razão linhas contadas R/B (mín.–máx.) | Reconsultas de probabilidade em R (média) | H10.1 (unidades) | H10.2: Rnp ≡ R (árvores) | Integridade (unidades) |
|---|---|---|---|---|---|---|---|---|---|
| iris | tuning | 7104 | 7304 | 0,97–1,18 | 2,15–2,54 | 8780 | 4/5 | 15/15 | 5/5 |
| iris | 7 nós | 1004 | 985 | 0,89–1,00 | 2,14–2,38 | 1435 | 5/5 | 15/15 | 5/5 |
| wine | tuning | 10570 | 8761 | 0,41–1,67 | 0,87–3,62 | 10178 | 3/5 | 15/15 | 5/5 |
| wine | 7 nós | 1674 | 1648 | 0,89–1,00 | 2,14–2,38 | 2182 | 5/5 | 15/15 | 5/5 |
| breast_cancer | tuning | 19971 | 19722 | 0,96–1,05 | 2,08–2,29 | 23212 | 5/5 | 15/15 | 5/5 |
| breast_cancer | 7 nós | 2785 | 2811 | 0,97–1,08 | 2,22–2,49 | 4389 | 5/5 | 15/15 | 5/5 |
| digits | tuning | 139364 | 140223 | 1,00–1,02 | 2,10–2,17 | 156969 | 5/5 | 15/15 | 5/5 |
| digits | 7 nós | 2093 | 2075 | 0,99–1,01 | 2,88–3,08 | 6766 | 5/5 | 15/15 | 5/5 |
| adult | tuning | 38129 | 37943 | 0,98–1,00 | 2,22–2,45 | 50457 | 5/5 | 15/15 | 5/5 |
| adult | 7 nós | 2140 | 2024 | 0,73–1,01 | 2,47–3,04 | 6376 | 5/5 | 15/15 | 5/5 |
| german_credit | tuning | 30200 | 30050 | 0,98–1,02 | 2,12–2,46 | 37950 | 5/5 | 15/15 | 5/5 |
| german_credit | 7 nós | 2175 | 2128 | 0,86–1,07 | 2,39–2,98 | 5072 | 5/5 | 15/15 | 5/5 |
| phys_bmi | tuning | 25723 | 6641 | 0,15–1,01 | 0,35–2,26 | 8359 | 5/5 | 15/15 | 5/5 |
| phys_bmi | 7 nós | 4298 | 4289 | 0,98–1,04 | 2,14–2,33 | 5431 | 5/5 | 15/15 | 5/5 |
| phys_kinetic | tuning | 33801 | 28657 | 0,16–1,08 | 0,37–2,27 | 32543 | 5/5 | 15/15 | 5/5 |
| phys_kinetic | 7 nós | 4256 | 4341 | 1,00–1,04 | 2,18–2,37 | 5379 | 5/5 | 15/15 | 5/5 |
| phys_power | tuning | 20786 | 12259 | 0,13–1,05 | 0,30–2,25 | 14141 | 5/5 | 15/15 | 5/5 |
| phys_power | 7 nós | 3229 | 3226 | 0,98–1,04 | 2,14–2,39 | 4334 | 5/5 | 15/15 | 5/5 |

«Pontos novos»: linhas pedidas ao oráculo que a árvore ainda não conhecia (nem linhas de treino, nem já rotuladas), em média por árvore. «Linhas contadas»: todas as linhas enviadas ao oráculo, que é o que as fases 6 a 9 reportavam como consultas.

- **Em todas as 270 árvores R e R7, `pontos_novos_so_probabilidade` = 0.** Todas as chamadas extra a `predict_proba` são sobre pontos que a árvore já conhecia.
- **H10.2 verifica-se em 270 de 270 árvores:** retirar `predict_proba` ao oráculo (Rnp, R7np) dá previsões iguais em todas as instâncias de teste. As reconsultas de probabilidade não influenciam nenhuma decisão do R nem do R7.
- **H10.1 verifica-se em 16 dos 18 pares dataset×orçamento.** Falha com a estrutura do tuning em:
  - iris: 1 de 5 unidades, semente 123, razão 1,18;
  - wine: 2 de 5 unidades, sementes 42 e 7, razões 1,67 e 1,33.
  Nessas unidades o R cresceu uma árvore maior do que o Original (wine 42: 15,7 contra 9,7 nós em média) e pediu mais pontos novos. Com 7 nós a razão fica entre 0,73 e 1,08 em todos os datasets.
- Nos controlos positivos físicos com a estrutura do tuning, o R pede **menos** pontos novos do que o Original (razões a partir de 0,13), porque as árvores são muito menores.

## 3. Decisão (regra pré-registada)

| Comparações R − B e R7 − B7 (oráculo: MLP selecionado só no treino) | Ressalva das consultas |
|---|---|
| breast_cancer, digits, adult, german_credit, phys_bmi, phys_kinetic, phys_power (os dois orçamentos); iris e wine com 7 nós | **Substituída:** o braço com OWL recebeu um número de pontos novos igual ou inferior a 1,10 vezes o do Original; as consultas adicionais contadas são probabilidades de pontos já conhecidos e não influenciam as decisões |
| iris e wine com a estrutura do tuning | **Mantém-se:** confundida pela informação em 1 e 2 das 5 unidades. Nestas comparações não há ganho a explicar: R − B foi −0,18/−0,53 pp (iris) e −0,44/+0,15 pp (wine) na fase 8 |

## 4. Leitura

1. A razão de 2,2 vezes nas consultas, repetida nas fases 6 a 9, **não era informação adicional** para o Reloaded O: era a reavaliação, com probabilidades, de pontos que a árvore já tinha, e essas probabilidades não mudam nenhuma decisão do R ou do R7.
2. **Os ganhos sustentados nos controlos positivos com 7 nós** (R7 − B7, fases 7 e 9) ocorreram com o mesmo número de pontos novos (razão 0,98 a 1,04). Não se explicam por mais informação do oráculo.
3. Com a estrutura do tuning, a quantidade de informação acompanha o tamanho da árvore. Nos físicos, o R usou muito menos pontos do que o Original e mesmo assim ganhou (fases 7 e 9).

## 5. Limitações

- A medição usa o pipeline das fases 7 a 9 (MLP selecionado só no treino). As comparações da fase 6 (MLP por omissão) não foram medidas. O caminho de código que gera as chamadas extra é o mesmo, mas os números da fase 6 não estão verificados.
- Os braços com mecanismos (E, V, D, A) usam as probabilidades nas decisões e não foram avaliados. A ressalva das consultas mantém-se para eles.
- O `reloaded_owl_full` histórico ficou registado, mas não foi analisado (outro amostrador).

Não avancei para nenhuma fase seguinte.
