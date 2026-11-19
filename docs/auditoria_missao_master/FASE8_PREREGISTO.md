# Fase 8 — o pipeline melhorado em dados reais: pré-registo

Escrito e registado em commit **antes** de qualquer execução da fase 8. Resultados em `FASE8_RESULTADOS.md`. Fases anteriores inalteradas.

## 1. Pergunta

A fase 7 mostrou que, com o MLP otimizado, a hierarquia `Reloaded > Original > C4.5` se sustenta em controlos positivos sintéticos, mas não no German Credit. A pergunta desta fase é se o mesmo pipeline, sem alterações, produz a hierarquia e um ganho atribuível à ontologia em **dados reais**.

## 2. Pipeline (congelado nas fases 5 e 7, sem alterações)

- MLP: seleção congelada de `validation/benchmark/mlp_selection_fase7.py` (16 candidatos, validação cruzada de 5 dobras só no treino).
- Reloaded: **O** (atributos da ontologia, sem mecanismos), congelado na fase 5.
- Amostrador conjunto em todos os braços TREPAN, 3 réplicas.
- Orçamentos: estrutura escolhida pelo tuning do projeto, e máximo de 7 nós.
- Executor `validation/benchmark/run_fase8.py` (igual ao da fase 7, só regista o Adult).

## 3. Datasets

| Dataset | Origem | Ontologia | Relação declarada | Já observado? |
|---|---|---|---|---|
| iris | sklearn / UCI | v2 | produtos e razões de comprimento e largura | sim, fases 1 a 5 |
| wine | sklearn / UCI | v2 | frações fenólicas, razões sobre as cinzas | sim, fases 1 a 5 |
| breast_cancer | sklearn / UCI | v2 | perímetro² / área e outras | sim, fases 1 a 5 |
| digits | sklearn / UCI | cópia literal da v1 | nenhuma (só agrupamentos) | sim, fases 1 e 2 |
| adult | `data/dataset.arff` (UCI Adult), amostra estratificada fixa de 2000 linhas | nova (agrupamento temático do dicionário UCI) | ganho de capital − perda de capital | **não**, nesta auditoria |

- **Amostra do Adult:** 2000 de 48 842 linhas, estratificada pela classe, semente 20261124, fixada antes de qualquer execução (`validation/benchmark/datasets_adult.py`). A razão é o custo da seleção do MLP e do TREPAN por unidade. Pré-processamento genérico igual ao do German Credit (one-hot, «?» como nível próprio). Fica com 103 colunas.
- **German Credit** não é repetido: já foi avaliado com este pipeline na fase 7.
- **Contaminação:** iris, wine, breast_cancer e digits foram observados nas fases anteriores. O Adult é o único dataset real novo nesta auditoria; não verifiquei se o projeto o usou em experiências próprias além dos contrafactuais.

## 4. Desenho

- Sementes 42, 7, 123, 2024, 11; holdout estratificado de 25 %. Para iris, wine, breast_cancer e digits as partições são as do baseline (mesmas sementes e mesmo protocolo).
- Braços: MLP otimizado (oráculo), C4.5, B e B7 (Original), R e R7 (Reloaded O), Rq e Rq7 (ontologia permutada).
- MLP original para comparação: `validation/benchmark/fase8_mlp_original.py` (perfil por omissão, ajustado só no treino, nas mesmas partições). Verifica a reprodução das previsões do baseline onde ele existe.
- **Limite operacional:** 7200 s por unidade, igual para todos. Uma unidade que o exceda é reportada como `resource_limit_exceeded` e não entra nos contrastes. Não altero a configuração para a fazer terminar.

## 5. Elegibilidade

`validation/benchmark/fase8_eligibility.py`, igual à da fase 7: validação cruzada aninhada só no treino, MLP otimizado contra C4.5, margem 0,02.

## 6. Contrastes e critérios

Iguais aos da fase 7:

- M1: MLP otimizado − MLP original, em accuracy, macro-F1 e balanced accuracy.
- H1 a H5 e AT, para cada dataset e cada orçamento. Hierarquia sustentada, observada e atribuível com as definições da fase 6.
- Unidade = semente; réplicas promediadas dentro da unidade; intervalo a 95 % por bootstrap percentil das 5 diferenças emparelhadas; sem teste entre datasets.
- **Veredicto principal:** só sobre os datasets **elegíveis**. Todos são reportados.
- Accuracy e fidelidade sempre separadas.
- Resultado negativo ou nulo é reportado como tal. Não altero datasets, amostra, ontologias, braços, sementes nem limites depois de ver resultados.

## 7. Limites conhecidos à partida

- Quatro dos cinco datasets já foram observados; só o Adult é novo, e mesmo assim numa amostra de 2000 linhas.
- As ontologias de iris, wine e breast_cancer foram escritas por mim (fase 3); a do digits não tem relações; a do Adult tem uma única relação.
- Muitos contrastes sem correção para comparações múltiplas.
