# Plano de correção

Ordenado por dependência: nada da fase B faz sentido antes de a fase A estar fechada, porque a fase A define contra que referência se mede.

## Fase A — Referência e protocolo (sem alterar algoritmos)

| # | Ação | Resolve | Critério de conclusão |
|---|---|---|---|
| A1 | Regenerar o manifest (v2) com os hashes atuais e registar a deriva desde o v1 | P05 | `verify_manifest` sem deriva; resultados v1 preservados como históricos |
| A2 | Acrescentar ao benchmark um braço «Original WIDE» (`max_features_per_node = nº de atributos`) | P04 | Braço executado em todas as unidades; a ontologia passa a ser comparada com ele |
| A3 | Igualar o orçamento: dar ao Original o mesmo número de amostras do oráculo que o Reloaded consome, ou limitar o Reloaded ao orçamento do Original | P02 | Rácio de amostras do oráculo E/C = 1 por unidade |
| A4 | Reportar fidelidade e accuracy a complexidade igual (curva métrica × nº de nós; comparação no mesmo nº de nós) | P07 | Tabela por unidade com métricas a 7, 15, 31, 63 nós |
| A5 | Réplicas por unidade (≥ 5) com a aleatoriedade do extrator isolada da do MLP | Ruído | Desvio entre réplicas reportado; diferenças menores do que ele não são interpretadas |
| A6 | Fixar o ambiente numérico (`OMP_NUM_THREADS=1`) em todos os pontos de entrada e testar a reprodução bit a bit do MLP | P13 | Teste automático: duas execuções dão o mesmo `oracle_id` |

## Fase B — Atribuição (alterações ao Reloaded, sob manifest v2)

| # | Ação | Resolve | Critério de conclusão |
|---|---|---|---|
| B1 | Peso de divisão derivado do conteúdo da ontologia e não do prefixo `onto_*`; retirar a variável de ambiente | P06 | Ontologia permutada produz pesos diferentes dos da real; teste automático |
| B2 | Braço sem ontologia e controlos (permutada, grupos aleatórios, engenharia convencional) obrigatórios no pipeline de produção | P03 | O relatório de produção só usa «atribuível» com ganho sobre os três controlos |
| B3 | Generalizar as relações: a ontologia declara a operação (razão, diferença, parte-todo) em vez de três papéis fixos | P11 | Ontologia de outro domínio gera relações sem alterar o núcleo |
| B4 | Limites físicos passam a restringir a amostragem de consultas em vez de gerar indicadores | P10 | Nenhuma consulta fora do domínio declarado; efeito na fidelidade medido |

## Fase C — Arquitetura e qualidade

| # | Ação | Resolve |
|---|---|---|
| C1 | Um único serviço de extração usado pelo benchmark, pela produção e pela GUI, com uma única fonte de valores por omissão | P08 |
| C2 | Retirar do extrator o código de dominância e migrar os dois ficheiros de teste que o usam | P12 |
| C3 | Dividir `trepan_reloaded_extractor.py` (5109 linhas) e `metrics_comparator.py` (4029 linhas); subir a cobertura dos módulos abaixo de 60 % | P14 |
| C4 | Mover os relatórios históricos da raiz para `docs/historico/` com uma nota de que não foram reverificados | P17 |
| C5 | Documentar `libEGL` para testes de GUI em Linux | P18 |

## O que o plano não promete

Nenhuma destas ações garante `Reloaded > Original > C4.5`. A fase A torna a comparação justa; pode até reduzir a vantagem aparente do Reloaded. A fase B torna um ganho atribuível, se existir.
