# Fase 4 — atributos da ontologia contra mecanismos do Reloaded: pré-registo

Escrito e registado em commit **antes** da campanha desta fase. Foi feito um teste de funcionamento do executor em duas unidades com 1 réplica (iris/42 e wine/7), num diretório temporário fora da evidência; esses valores não entram na análise. Os resultados irão para `FASE4_RESULTADOS.md`.

## 1. Pergunta

Na fase 3, o Reloaded com a ontologia v2 (D2) ficou abaixo do Original. D2 junta duas coisas: (a) os atributos derivados da ontologia e (b) os mecanismos do Reloaded (peso de ganho fixo nas colunas derivadas, Gain_Reloaded com profundidade ontológica e cobertura de erro, refinamento focado no erro, consultas ativas). Esta fase separa (a) de (b).

## 2. Braços novos

Executor `validation/benchmark/run_fase4.py` (reutiliza `run_sampler_c` e as ontologias v2, sem os alterar). Todos com o amostrador conjunto e 3 réplicas.

| Braço | O que é |
|---|---|
| B | Original + amostrador conjunto, espaço original (verificação de integridade face às fases 2 e 3) |
| O | «Original-equivalente» no espaço enriquecido pela OWL v2: λ = 0 (sem pesos, sem Gain_Reloaded, sem grupos nem relatedness), refinamento focado no erro desligado, consultas ativas desligadas; só a projeção das consultas (que recalcula as derivadas) fica ativa |
| Q | O com a ontologia v2 permutada (controlo negativo do runner) |
| K | O com atributos de engenharia convencional (médias de grupos escolhidos pelos dados) |

D2 (Reloaded + OWL v2 + conjunto) e C4.5 vêm da fase 3, com as mesmas partições, MLP e sementes.

**Nota de contabilidade, verificada no teste de funcionamento:** no estado S3 o Reloaded constrói também a porção Original equivalente, pelo que as «amostras do oráculo» de O incluem essa construção. O número de consultas da árvore final (`membership_queries`) de O é comparável ao de B. Reporto os dois.

## 3. Unidades

iris, wine, breast_cancer × sementes 42, 7, 123, 2024, 11 (15 unidades), 3 réplicas. Limite operacional de 5400 s por unidade.

## 4. Contrastes e critério

Medida por unidade: média das 3 réplicas. `X > Y` num dataset: diferença média emparelhada positiva e intervalo a 95 % (bootstrap percentil, 20 000 reamostragens das 5 unidades) acima de zero.

| # | Contraste | Pergunta |
|---|---|---|
| J1 | O − B, accuracy e fidelidade | Os atributos da ontologia, usados sem mecanismos, ajudam o Original? |
| J2 | O − Q e O − K, accuracy e fidelidade | Esse efeito vem do significado da ontologia? |
| J3 | D2 − O, accuracy e fidelidade | Os mecanismos do Reloaded ajudam ou prejudicam, dados os mesmos atributos? |
| H | O > B, O > C4.5, B > C4.5 (accuracy); O > B, O > C4.5 (fidelidade) | Hierarquia com O no lugar do Reloaded |

## 5. Regras de decisão

- **Os atributos ajudam:** J1 com intervalo acima de zero nas duas métricas.
- **Atribuível à ontologia:** J1 cumprido **e** O acima de Q e de K, com intervalo acima de zero, nas duas métricas.
- **Os mecanismos do Reloaded prejudicam:** J3 com intervalo abaixo de zero em pelo menos uma métrica e não acima de zero na outra. **Ajudam:** J3 com intervalo acima de zero nas duas.
- **Integridade:** B tem de reproduzir os valores da fase 3; se não reproduzir, a fase é inválida.
- Resultado negativo ou nulo é reportado como tal. Não altero ontologias, sementes, datasets nem braços depois de ver resultados.

## 6. Limites conhecidos à partida

- O não é o TREPAN Original canónico: é o Reloaded com os mecanismos desligados, num espaço com mais colunas; a pré-seleção de 12 atributos por nó (P04) aplica-se ao espaço enriquecido.
- K gera médias de grupos, não razões nem produtos; Q é o controlo que aplica as mesmas operações sem o significado.
- Datasets DEV já observados; a contaminação declarada na fase 3 mantém-se.
