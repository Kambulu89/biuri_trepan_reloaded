# Fase 9 — pré-registo: EFSR com gate validado em consultas novas

Escrito antes da campanha. As fases 1 a 8 e os seus resultados não são alterados. O núcleo (`core/`) não é alterado.

## 1. Pergunta

A fase 5 mostrou que, sem o EFSR, a ontologia não muda nenhuma decisão de divisão, e que as decisões aprovadas pelo EFSR pioram a árvore. A causa provável (fase 7, `FASE7_PREREGISTO.md` §6): os dois gates do EFSR medem a fidelidade **dentro** da amostra usada para escolher a divisão, e há uma tolerância de 1,5 pontos que compra coerência semântica. As fases 6 a 8 avaliaram o Reloaded O (só atributos). A pergunta desta fase é:

> Um gate do EFSR que só aprova uma divisão proposta pela semântica quando ela é melhor num **lote novo** de consultas ao oráculo (1) evita a perda do EFSR original e (2) dá ao Reloaded um ganho sobre o Original atribuível à ontologia?

Esta é a única variante testada nesta fase.

## 2. A variante (V)

Implementação isolada em `validation/benchmark/efsr_holdout_fase9.py` (subclasse do Reloaded; só atua nos braços V e Vq).

1. Chamado exatamente onde o gate original é chamado: proposta semântica diferente da divisão estatística, numa região de erro elegível (critério de elegibilidade inalterado).
2. Sorteia um lote novo de `effective_min_sample_` consultas na região do nó, com o mesmo modelo de distribuição local e o mesmo gerador do lote de decisão (restrições do caminho, projeção), mas com uma RNG própria `SeedSequence([random_state, 0xEF59, node_id])`. O lote não entra na escolha de nenhuma das duas divisões e não avança a RNG do Original.
3. Cada divisão prevê em cada ramo a classe maioritária do lote de decisão.
4. Aceita a proposta só se o teste de McNemar exato unilateral no lote novo (proposta melhor do que a divisão estatística) der p < 0,05. Sem tolerância. Os dois gates originais deixam de decidir.
5. As consultas do lote novo são feitas ao mesmo oráculo congelado e entram em `oracle_queries` do braço; não consomem o orçamento de consultas do Original. **V usa mais consultas do que B e E**; reporto quantas.

## 3. Pipeline (igual às fases 7 e 8)

MLP otimizado por seleção só no treino (`mlp_selection_fase7.py`), amostrador conjunto, 3 réplicas, holdout estratificado de 25 %, sementes 42, 7, 123, 2024, 11, ontologias v2, limite de 7200 s por unidade. Executor `validation/benchmark/run_fase9.py`.

Braços, em dois orçamentos (estrutura do tuning; máximo de 7 nós, sufixo «7»):

| Braço | O que é |
|---|---|
| B | Original |
| R | Reloaded O (atributos, sem mecanismos), como nas fases 5 a 8 |
| E | Reloaded com canal de score (λ = 1) e EFSR original (braço E da fase 5) |
| V | E com o gate do §2 |
| Vq | V com a ontologia v2 permutada (controlo) |

Mais MLP otimizado e C4.5 canónico, do runner, sem alterações.

## 4. Datasets

A variante foi desenhada a partir da fase 5 (iris, wine, breast_cancer). Avalio-a **noutros** datasets:

| Dataset | Elegível (MLP otimizado − C4.5 > 0,02, só no treino) | Fonte da elegibilidade |
|---|---|---|
| digits | sim (+0,130) | fase 8 |
| german_credit | sim (+0,022) | fase 7 |
| phys_bmi | sim (+0,045) | fase 7 |
| phys_power | sim (+0,026) | fase 7 |
| phys_kinetic | não (+0,017) | fase 7 |
| adult | não (−0,022) | fase 8 |

A elegibilidade não é recalculada: depende só do treino, do MLP e do C4.5, que são os mesmos (mesmas sementes, partições e seleção). Verifico isso na campanha (§6).

**Contaminação:** todos estes datasets foram observados nas fases 6 a 8 com B e R. Os braços E, V e Vq nunca foram executados neles. O desenho do gate não usou nenhum deles.

## 5. Contrastes e critérios

Estatística igual às fases 7 e 8: unidade = semente; 3 réplicas promediadas dentro da unidade; IC 95 % por bootstrap percentil de 5 diferenças emparelhadas (20 000 reamostragens); «sustentado» = limite inferior estritamente > 0.

**Hierarquia com V como Reloaded** (definições da fase 6): H1 accuracy V − B; H2 accuracy V − C4.5; H3 accuracy B − C4.5; H4 fidelidade V − B; H5 fidelidade V − C4.5; AT accuracy e fidelidade V − Vq. Sustentada = H1–H5 com limite inferior > 0; observada = H1–H5 com média > 0; atribuível = H1, H4 e AT com limite inferior > 0.

**Mecanismo:**
- G1: V − E (accuracy, fidelidade): o gate novo é melhor do que o original?
- G2: V − R (accuracy, fidelidade): o EFSR validado acrescenta alguma coisa aos atributos?
- Diagnóstico: tentativas e aprovações do EFSR por árvore em E e V; consultas ao oráculo por braço.

**Regra de adoção (decidida agora):** V é considerado **útil** se, na estrutura do tuning, V − R tiver limite inferior > 0 em accuracy ou em fidelidade em pelo menos um dataset elegível **e** em nenhum dataset elegível tiver limite superior < 0 em accuracy ou fidelidade. Caso contrário, não é adotado. Uma decisão positiva serve só para propor V para um manifest futuro; não altera o núcleo nesta fase.

**Veredicto principal:** só sobre os quatro elegíveis; todos os seis são reportados. Uma variante, 4 datasets elegíveis × 2 orçamentos, sem correção para comparações múltiplas: leitura exploratória por dataset.

## 6. Verificações de integridade (antes da análise)

1. Para cada unidade, as previsões de MLP, C4.5, B e R são iguais às da mesma unidade nas fases 7 (phys_*, german_credit) e 8 (digits, adult). Se não forem, reporto e não analiso essa unidade como comparável.
2. Onde V não fez nenhuma tentativa do EFSR, V é igual a E (mesmas previsões).
3. Recalculo das métricas a partir das previsões guardadas.

## 7. Smoke test (feito antes deste pré-registo, só para verificar a execução)

iris, semente 42, 1 réplica (iris foi usado no desenho; este resultado não entra em nenhuma análise). Tempo 48 s. MLP, C4.5, B e R com previsões iguais às da fase 8 na mesma unidade. E7 e V7 sem tentativas e com previsões iguais. Na estrutura do tuning: E fez 6 tentativas e aprovou 3; V fez 6 e aprovou 1; Vq fez 9 e aprovou 2; V fez 2016 consultas extra (12 901 contra 10 864 de E).

## 8. Limitações conhecidas antes de executar

- V usa mais consultas ao oráculo do que B e E: um ganho de V pode vir em parte da informação adicional. Reporto as consultas.
- O canal de score (λ = 1) vem com o EFSR em E e V; a fase 5 mostrou que sem EFSR ele não muda decisões, mas não é isolado aqui.
- Todos os datasets foram observados antes com B e R; nada é confirmatório.

## 9. Emenda 1 (durante a campanha, antes de qualquer análise)

**Defeito de implementação:** na unidade phys_power/42, o braço `sc_V_r1` foi descartado pelo runner com `RuntimeError: Não foi possível gerar membership queries coerentes que satisfaçam as restrições do nó.` O lote novo pede `effective_min_sample_` consultas numa região onde o gerador não as consegue produzir; o lote de decisão desse nó pedia menos e não falhou.

**Correção:** essa exceção passa a ser tratada como «lote indisponível», que o §2 e o código já tratavam como **rejeição** da proposta semântica (decisão conservadora: fica a divisão estatística). Nenhuma outra regra muda.

**Efeito nas unidades:** o código só difere quando a exceção ocorreria, e nesse caso o braço era descartado. As unidades sem braços descartados são idênticas com e sem a correção e mantêm-se. As unidades com braços V/Vq descartados são executadas de novo com a correção, no fim da campanha; as saídas originais ficam guardadas em `interrompidas/`. Nenhum resultado foi consultado para decidir esta emenda (só a lista de braços descartados).

**Outro incidente:** reinício do contentor com 8 de 30 unidades concluídas; as em falta foram retomadas do início. O registo parcial do EFSR de german_credit/7 (21 linhas da execução interrompida) foi separado para `interrompidas/`.
