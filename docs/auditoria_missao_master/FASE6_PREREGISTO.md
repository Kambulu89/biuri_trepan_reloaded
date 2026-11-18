# Fase 6 — Reloaded corrigido em datasets novos e sob orçamento restrito: pré-registo

Escrito e registado em commit **antes** de conhecer os resultados da fase 5 e antes de qualquer execução avaliada da fase 6. Foi feito um teste de funcionamento do executor numa unidade (phys_bmi/42, 1 réplica) fora da evidência; dele só verifiquei que os seis braços existem, que o limite de 7 nós é respeitado e que o Reloaded usa o espaço enriquecido, sem ler métricas. Resultados em `FASE6_RESULTADOS.md`.

## 1. Pergunta

A hipótese `Reloaded > Original > C4.5` verifica-se onde tem base teórica para funcionar: (a) o MLP supera claramente o C4.5, (b) a ontologia declara relações que uma árvore de cortes ortogonais só aproxima com uma «escada» de cortes, (c) o orçamento de complexidade é restrito?

## 2. Reloaded corrigido

É o escolhido pela regra congelada no `FASE5_PREREGISTO.md` §5 (uma de O, T, L, W), sem qualquer ajuste. Se a regra escolher O, o «Reloaded corrigido» é o Original no espaço enriquecido pela ontologia, sem mecanismos.

## 3. Datasets

| Dataset | Tipo | Relação declarada na ontologia v2 | Estatuto |
|---|---|---|---|
| german_credit | real (UCI Statlog, 1000 linhas, 61 colunas após one-hot), local no repositório | montante do crédito / prazo em meses | PROVADO (definição aritmética) |
| phys_bmi | sintético, controlo positivo: obesidade = massa / altura² ≥ 30 | massa / altura² | PROVADO (definição do IMC, OMS) |
| phys_kinetic | sintético, controlo positivo: dano = ½ m v² ≥ 1 kJ | massa × velocidade² | PROVADO (mecânica clássica) |
| phys_power | sintético, controlo positivo: sobreaquecimento = V²/R ≥ 60 W | tensão² / resistência | PROVADO (leis de Ohm e de Joule) |

- **Sintéticos:** 400 linhas (300 de treino), 2 colunas relevantes e 3 irrelevantes, 2 % de erro de medição e 5 % de rótulos trocados; semente de geração fixa (`validation/benchmark/datasets_physics.py`). São controlos positivos: mostram se o mecanismo consegue ganhar quando o conhecimento é verdadeiro e relevante por construção. Um resultado positivo neles **não** é evidência de ganho em dados reais.
- **German Credit:** não foi usado nas fases 1 a 5 desta auditoria, mas o projeto já o observou (documentos 53 a 59) e marcou-o como inelegível para confirmação. Os resultados nele são exploratórios.
- Não consegui obter datasets reais novos: o acesso a OpenML e UCI está bloqueado neste ambiente.

## 4. Desenho

- Executor `validation/benchmark/run_fase6.py`; sementes 42, 7, 123, 2024, 11; holdout estratificado de 25 %; MLP e estrutura escolhidos pelo runner do projeto (tuning estrutural só no treino).
- Braços, todos com amostrador conjunto e 3 réplicas: B (Original), R (Reloaded corrigido, ontologia v2), Rq (R com a ontologia v2 permutada). Cada um em dois orçamentos: o da estrutura escolhida pelo tuning, e **no máximo 7 nós** (B7, R7, Rq7).
- C4.5 canónico (sem limite de nós) e MLP vêm do runner. Com orçamento de 7 nós, o C4.5 continua canónico; reporto o seu tamanho ao lado.

## 5. Elegibilidade (antes de olhar para o teste)

`validation/benchmark/fase6_eligibility.py`: em cada unidade, validação cruzada de 5 dobras **só no treino**, MLP contra C4.5. O dataset é **elegível** se a diferença média de accuracy MLP − C4.5 for > 0,02. Todos os datasets são reportados; o veredicto da hipótese é dado separadamente para os elegíveis.

## 6. Contrastes e critério

Medida por unidade: média das 3 réplicas. Intervalo a 95 % por bootstrap percentil (20 000 reamostragens das 5 unidades). Para cada dataset e cada orçamento:

| # | Condição |
|---|---|
| H1 | accuracy: R > B |
| H2 | accuracy: R > C4.5 |
| H3 | accuracy: B > C4.5 |
| H4 | fidelidade: R > B |
| H5 | fidelidade: R > C4.5 |
| AT | R > Rq em accuracy e em fidelidade |

- **Hierarquia sustentada:** H1 a H5 com intervalo acima de zero.
- **Hierarquia observada:** as cinco diferenças médias positivas.
- **Ganho atribuível à ontologia:** H1 e H4 sustentadas e AT com intervalo acima de zero nas duas métricas.
- Resultado negativo ou nulo é reportado como tal. Não altero datasets, ontologias, braços, orçamento nem sementes depois de ver resultados.

## 7. Limites conhecidos à partida

- Os controlos positivos foram desenhados por mim para o regime em que a hipótese deve funcionar; a contribuição deles é mostrar se o mecanismo consegue ganhar, não estimar quanto ganha no mundo real.
- 5 unidades por dataset.
- O limite de 7 nós foi escolhido por mim como «orçamento restrito» (3 divisões); não foi ajustado.
