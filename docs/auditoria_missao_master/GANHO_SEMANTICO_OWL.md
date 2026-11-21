# Ganho semântico com OWL: comparador, consultas e nível de evidência de cada conclusão

Consolida todas as comparações «TREPAN com OWL − TREPAN sem OWL» das campanhas da auditoria (fases 1 a 9). Não houve execuções novas: a tabela foi reconstruída dos resultados brutos guardados.

- Tabela completa: `evidencia/owl_pairs/owl_pairs.csv` (102 comparações distintas; pares repetidos entre campanhas aparecem uma vez, com as campanhas em `campanhas`).
- Script: `evidencia/owl_pairs/owl_pairs.py` (`python owl_pairs.py RAIZ_DAS_CAMPANHAS SAIDA.csv`).

## 1. Como ler cada comparação

- **Diferença:** com OWL − sem OWL, em pontos percentuais, média de 5 sementes; as réplicas são promediadas dentro de cada semente. A accuracy é medida face aos rótulos reais e a fidelidade face ao MLP da mesma unidade.
- **Comparador** (coluna `comparador`), um de três:
  - **Reloaded sem OWL com os mesmos mecanismos.** Pares `reloaded_owl_full − reloaded_core` (todas as fases) e `C − E`, `D − F` (fases 2 e 3).
  - **TREPAN Original (B).** Equivale ao Reloaded sem OWL pelo contrato EQ (`docs/b1b4/07_CONTRATO_EQUIVALENCIA.md`). A equivalência foi **observada** no benchmark da fase 1: o braço sem ontologia deu accuracy, fidelidade e número de nós iguais aos do Original nas 20 unidades (`AUDITORIA_ALGORITMOS.md`, achado 1).
  - **TREPAN Original com no máximo 7 nós (B7).**
- **Oráculo** (coluna `oraculo`): MLP por omissão nas fases 1 a 6; MLP selecionado só no treino nas fases 7 a 9. Nos registos das fases 7 a 9, este último continua a chamar-se «MLP Original». **Nenhuma comparação usou o MLP_Onto.**
- **Verificado em todas as 102 comparações, semente a semente:**
  - as mesmas linhas de teste;
  - o mesmo oráculo congelado (`oracle_id`);
  - o mesmo orçamento **configurado** (`tree_max_nodes`, `query_budget`, `tree_max_n`).
- **Consultas efetivamente realizadas** (colunas `consultas_com_owl`, `consultas_sem_owl`, `razao_consultas`): o braço com OWL **contou** entre 0,57 e 5,16 vezes as linhas enviadas ao oráculo pelo braço sem OWL (mediana 2,22). «Mesmo orçamento» significa apenas o mesmo limite. A fase 10 decompôs estas linhas para R e R7 (§6): a diferença são probabilidades de pontos já conhecidos, sem efeito nas decisões; os pontos novos são iguais ou inferiores.
- **Estado:**

| Estado | Critério |
|---|---|
| `sustentado_ambas` | Intervalos bootstrap a 95 % da accuracy **e** da fidelidade acima de 0 |
| `observado_ambas` | As duas médias acima de 0, com pelo menos um intervalo a incluir 0 (observado nas médias, não sustentado) |
| `misto` | Uma média positiva e outra negativa ou nula, sem perda sustentada |
| `perda_sustentada` | Pelo menos um intervalo abaixo de 0 |
| `empate` | As duas médias iguais a 0 |

Os intervalos usam 5 sementes, sem correção para comparações múltiplas: são evidência exploratória.

## 2. Resumo por família de comparação

| Família | Dados | n | Sustentado nas duas | Observado nas médias | Misto | Perda sustentada | Empate | Consultas com/sem OWL (mín.–mediana–máx.) |
|---|---|---|---|---|---|---|---|---|
| Reloaded completo vs Reloaded sem OWL (histórico) | reais | 14 | 0 | 1 | 11 | 2 | 0 | 2,06–2,20–2,37 |
| | sintéticos | 6 | 0 | 2 | 3 | 1 | 0 | 2,05–2,17–2,24 |
| Reloaded completo vs Reloaded sem OWL (fases 2–3, réplicas) | reais | 11 | 0 | 2 | 7 | 2 | 0 | 1,91–2,21–2,34 |
| Atributos OWL sem mecanismos (O, R, W, T, L) vs Original | reais | 19 | 0 | 5 | 9 | 1 | 4 | 1,78–2,25–2,28 |
| | sintéticos | 6 | **4** | 1 | 1 | 0 | 0 | 0,59–1,95–2,19 |
| Atributos OWL, 7 nós (R7) vs Original 7 nós | reais | 7 | 0 | 3 | 3 | 1 | 0 | 2,19–2,55–2,96 |
| | sintéticos | 6 | **6** | 0 | 0 | 0 | 0 | 2,18–2,24–2,34 |
| Mecanismos (E, A, D) vs Original | reais | 12 | 0 | 4 | 5 | 2 | 1 | 2,12–2,22–2,34 |
| | sintéticos | 3 | **3** | 0 | 0 | 0 | 0 | 0,57–1,23–1,79 |
| EFSR com lote novo (V) vs Original | reais | 3 | 0 | 2 | 0 | 1 | 0 | 3,15–3,45–3,52 |
| | sintéticos | 3 | **1** | 2 | 0 | 0 | 0 | 0,66–1,59–2,43 |
| EFSR (E7, V7), 7 nós vs Original 7 nós | reais | 6 | 0 | 2 | 4 | 0 | 0 | 2,72–3,71–5,16 |
| | sintéticos | 6 | **6** | 0 | 0 | 0 | 0 | 2,18–2,23–2,51 |
| **Total** | reais | **72** | **0** | 19 | 39 | 9 | 5 | 1,78–2,25–5,16 |
| | sintéticos | **30** | **20** | 5 | 4 | 1 | 0 | 0,57–2,18–2,51 |

«Sintéticos» são os controlos positivos `phys_bmi`, `phys_kinetic` e `phys_power`: a classe é gerada por uma fórmula, e a OWL v2 declara exatamente essa relação (`ratio:hasHeightM:1:2`, `product:hasSpeed:1:2`, `ratio:hasResistance:2:1`).

## 3. As 20 comparações com ganho sustentado nas duas métricas (todas em controlos positivos sintéticos)

| Campanhas | Dataset | Com OWL − sem OWL | Comparador | Oráculo | Accuracy (pp) | Fidelidade (pp) | Consultas com/sem OWL |
|---|---|---|---|---|---|---|---|
| fase6 | phys_power | R − B | Original | por omissão | +3,67 | +2,33 | 2,14 |
| fase6 | phys_bmi | R7 − B7 | Original 7 nós | por omissão | +8,73 | +6,60 | 2,29 |
| fase6 | phys_kinetic | R7 − B7 | Original 7 nós | por omissão | +8,93 | +2,53 | 2,26 |
| fase6 | phys_power | R7 − B7 | Original 7 nós | por omissão | +7,80 | +6,47 | 2,34 |
| fase7, fase9 | phys_bmi | R − B | Original | selecionado no treino | +1,33 | +0,67 | 0,59 |
| fase7, fase9 | phys_kinetic | R − B | Original | selecionado no treino | +1,87 | +2,00 | 1,80 |
| fase7, fase9 | phys_power | R − B | Original | selecionado no treino | +1,40 | +1,13 | 1,27 |
| fase7, fase9 | phys_bmi | R7 − B7 | Original 7 nós | selecionado no treino | +7,67 | +4,60 | 2,18 |
| fase7, fase9 | phys_kinetic | R7 − B7 | Original 7 nós | selecionado no treino | +7,73 | +6,40 | 2,20 |
| fase7, fase9 | phys_power | R7 − B7 | Original 7 nós | selecionado no treino | +7,00 | +8,47 | 2,23 |
| fase9 | phys_bmi | E − B | Original | selecionado no treino | +1,47 | +0,67 | 0,57 |
| fase9 | phys_kinetic | E − B | Original | selecionado no treino | +2,00 | +1,73 | 1,79 |
| fase9 | phys_power | E − B | Original | selecionado no treino | +1,40 | +0,73 | 1,23 |
| fase9 | phys_bmi | E7 − B7 | Original 7 nós | selecionado no treino | +7,33 | +4,93 | 2,18 |
| fase9 | phys_kinetic | E7 − B7 | Original 7 nós | selecionado no treino | +7,40 | +6,33 | 2,20 |
| fase9 | phys_power | E7 − B7 | Original 7 nós | selecionado no treino | +6,93 | +8,27 | 2,23 |
| fase9 | phys_kinetic | V − B | Original | selecionado no treino | +1,87 | +2,00 | 2,43 |
| fase9 | phys_bmi | V7 − B7 | Original 7 nós | selecionado no treino | +7,73 | +4,67 | 2,23 |
| fase9 | phys_kinetic | V7 − B7 | Original 7 nós | selecionado no treino | +7,60 | +6,53 | 2,45 |
| fase9 | phys_power | V7 − B7 | Original 7 nós | selecionado no treino | +6,67 | +8,53 | 2,51 |

As 20 linhas não são independentes: dentro de cada campanha partilham o mesmo Original e o mesmo MLP, e as linhas das fases 7 e 9 repetem os mesmos B e R.

## 4. Conclusões da auditoria com o comparador, as consultas e o nível de evidência

| # | Conclusão | Comparador | Oráculo | Consultas com/sem OWL | Nível de evidência |
|---|---|---|---|---|---|
| C1 | Em dados reais, nenhuma comparação «com OWL − sem OWL» mostra ganho nas duas métricas | Os três comparadores (72 comparações, 6 datasets reais) | Ambos os MLP | 1,78–5,16 (mediana 2,25) | **Nenhuma sustentada nas duas**; 19 observadas nas médias; 9 com perda sustentada em pelo menos uma métrica |
| C2 | Nos controlos positivos, os atributos derivados da relação declarada na OWL melhoram accuracy e fidelidade face ao Original | Original (B) e Original 7 nós (B7) | Ambos os MLP | 0,59–2,34 linhas contadas; **pontos novos 0,13–1,08 vezes os do Original** (fase 10, MLP selecionado no treino), e as reconsultas não mudam decisões | **Sustentado nas duas** em 10 de 12 comparações R/R7. As 2 restantes (fase 6, R − B no phys_bmi e no phys_kinetic) têm a fidelidade com intervalo a incluir 0 |
| C3 | O Reloaded completo com OWL não supera o Reloaded completo sem OWL (mesmos mecanismos) | Reloaded sem OWL (`reloaded_core`, E, F) | Ambos os MLP | 1,91–2,37 | **Nenhuma sustentada nas duas** em 31 comparações (25 reais, 6 sintéticas); 5 observadas nas médias; 5 com perda sustentada |
| C4 | Os mecanismos do Reloaded (score, EFSR) não acrescentam ganho aos atributos | **Reloaded O / R** (não o Original): contrastes G1 e G2 de `FASE5_RESULTADOS.md` e `FASE9_RESULTADOS.md`, fora desta tabela | Fase 5: por omissão; fase 9: selecionado no treino | V usa 1,15–1,55 vezes as consultas de E (`FASE9_RESULTADOS.md`, tabela de braços) | Na fase 9, G2 (V − R) só é **sustentado** na accuracy do german_credit (+1,1 pp), **sem** ganho de fidelidade |
| C5 | O ganho nos controlos positivos é atribuível ao significado da OWL | **OWL permutada** (Rq, Vq): contrastes AT de `FASE6_RESULTADOS.md`, `FASE7_RESULTADOS.md` e `FASE9_RESULTADOS.md` | Por fase | Vq faz mais consultas do que V nos físicos (fase 9, §2) | Sustentado nas duas métricas só onde o relatório de cada fase o indica; não é inferível desta tabela |

Os relatórios de cada fase mantêm a redação original (são históricos). Para qualquer frase sobre «ganho da ontologia», esta tabela e o CSV são a referência quanto ao comparador, às consultas e ao nível de evidência.

## 5. Frases para a tese

1. **Agnosticismo.** «No núcleo do BIURI, a identidade do dataset não altera o resultado: com os mesmos dados, a mesma OWL, a mesma partição, a mesma configuração e a mesma semente, mudar apenas o nome do ficheiro ou apenas o identificador do dataset deixa iguais, instância a instância, as linhas de teste, as previsões do oráculo e as previsões de cada árvore, no benchmark e no percurso de produção; e um atributo nominal declarado no ARFF mantém as suas categorias mesmo quando parecem números.»
   - **Evidência:** `tests/test_identity_metamorphic.py` (6 testes) e `tests/test_arff_declared_schema.py` (13 testes), executados com sucesso; os testes de identidade detetam uma semente diferente e uma única previsão alterada.
   - **Limitação:** dados sintéticos pequenos e uma ontologia de agrupamento. A produção não recebe `dataset_id`; aí varia-se o identificador do artefacto. Um CSV sem esquema continua a inferir `01` e `1` como o mesmo número.
2. **Ganho semântico.** «Com o mesmo MLP como oráculo, as mesmas instâncias de teste e o mesmo orçamento configurado, os atributos derivados da relação declarada na OWL melhoraram simultaneamente a accuracy (+1,3 a +8,9 pp) e a fidelidade (+0,7 a +8,5 pp) do TREPAN face ao TREPAN Original em 10 de 12 comparações nos três controlos positivos sintéticos, em que a OWL declara a relação que gera a classe; nas 72 comparações com seis conjuntos de dados reais, nenhuma melhoria das duas métricas foi sustentada pelos intervalos.»
   - **Evidência:** §3 e §4 (C1, C2) e `evidencia/owl_pairs/owl_pairs.csv`.
   - **Limitações:** os controlos foram construídos para que a OWL contenha a regra exata, por isso dão um limite superior do benefício. O comparador é o Original, equivalente ao Reloaded sem OWL pelo contrato EQ. As linhas contadas foram 0,59 a 2,34 vezes as do Original, mas nas comparações das fases 7 e 9 os pontos novos foram iguais ou inferiores e as reconsultas não mudam decisões (fase 10); nas da fase 6 isto não foi medido. Há 5 sementes por dataset, sem correção para comparações múltiplas.

## 6. Consultas efetivas: fase 10

Medição de `FASE10_PREREGISTO.md` e `FASE10_RESULTADOS.md`: 45 unidades, 270 árvores R e R7, com o MLP selecionado só no treino.

- Todas as linhas que R e R7 enviam ao oráculo a mais do que B são `predict_proba` sobre pontos que a árvore já conhecia (0 pontos novos só por probabilidade).
- Retirar `predict_proba` ao oráculo deixa as previsões iguais em 270 de 270 árvores: essas reconsultas não influenciam nenhuma decisão.
- Pontos novos R/B: 0,73 a 1,08 com 7 nós (todos os datasets); com a estrutura do tuning, 0,13 a 1,08, exceto iris (1 unidade com 1,18) e wine (2 unidades com 1,33 e 1,67), onde o R cresceu árvores maiores.

**Ressalva revista para R − B e R7 − B7** (fases 7 a 9): o braço com OWL não recebeu mais informação nova do oráculo do que o Original, exceto no iris e no wine com a estrutura do tuning, onde a comparação continua confundida pela informação e onde, de qualquer modo, não há ganho a explicar. As linhas da fase 6 (MLP por omissão) e os braços com mecanismos (E, V, D, A) não foram medidos e mantêm a ressalva original.

## 7. Fase 11: MLP por omissão, braços com mecanismos e complexidade igual

Fonte: `FASE11_RESULTADOS.md`.

- **Comparações da fase 6 (MLP por omissão), R e R7:** pontos novos R/B ≤ 1,10 em 39 de 40 unidades (exceção: phys_bmi, R, 1 unidade, razão 1,16). Rnp ≡ R em 120 de 120 árvores. A ressalva revista do §6 aplica-se também a estas linhas.
- **Braços com mecanismos (E, D, A):** recebem os mesmos pontos novos do que o Original em 42, 40 e 41 de 45 unidades. As exceções estão no wine, no iris, no breast_cancer, no phys_kinetic e no phys_power. Mas usam probabilidades do MLP sobre pontos conhecidos nas decisões. Para estas linhas, a ressalva passa a ser «informação de confiança do MLP que o Original não usa», e não «mais consultas».
- **V:** o lote novo é informação adicional por desenho (1,35 a 2,47 vezes os pontos novos nos dados reais).
- **Complexidade igual (P07):** comparadas com o C4.5 com o mesmo limite de nós, as conclusões sobre `Original > C4.5` e `Reloaded > C4.5` reforçam-se. As conclusões sobre o ganho semântico (R − B) não mudam: o limite é o mesmo nos dois braços TREPAN.
