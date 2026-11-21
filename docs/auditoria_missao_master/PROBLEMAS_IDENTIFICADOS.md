# Problemas identificados

Cada problema indica como foi estabelecido: **MEDIDO** (experiência com resultado em `evidencia/`), **EXECUTADO** (reproduzido em execução real), **LIDO** (análise do código, sem execução dedicada). Estado da correção em `CORRECCOES_IMPLEMENTADAS.md`.

## Críticos (comprometem conclusões científicas)

| ID | Problema | Evidência | Estado |
|---|---|---|---|
| P01 | O gate de contribuição semântica declarava `SEMANTIC_CONTRIBUTION_SUPPORTED` sem nenhuma decisão alterada pela ontologia, com limiares 0, numa única partição e sem controlos. | EXECUTADO: Breast Cancer, semente 7, impacto 0, 0 intervenções aceites, estado «SUPPORTED». Verificado de novo em 2026-11-20: o mesmo código (`af02d5d`) reproduz exatamente esta execução. Com o código atual (ordem declarada das classes, P21) a partição muda e esta semente passa a ter impacto 0,129 e 4 intervenções aceites, deixando de ser um exemplo do defeito; o defeito em si está coberto por testes unitários (`PRODUCAO_BC_RECALCULO.md`). | Corrigido |
| P02 | O Reloaded com ontologia não tem orçamento comparável ao do Original: consulta o oráculo 2,2 vezes mais e demora 11,6 vezes mais (medianas das 20 unidades). Qualquer ganho face ao Original confunde-se com mais consultas. | MEDIDO: `evidencia/baseline_por_unidade.csv`. | Medido (fases 10 e 11): as linhas a mais são probabilidades de pontos conhecidos; R e R7 recebem os mesmos pontos novos (≤ 1,10 × B em 126 de 130 combinações unidade×orçamento) e essas probabilidades não mudam decisões (390/390 árvores); E, D e A usam-nas nas decisões |
| P03 | No pipeline de produção não há braço sem ontologia nem controlos por omissão (`semantic_attribution=None`). A diferença Reloaded − Original inclui mecanismos não semânticos. | LIDO + EXECUTADO (3 execuções). | Por corrigir |

## Altos

| ID | Problema | Evidência | Estado |
|---|---|---|---|
| P04 | O TREPAN Original (e o Reloaded) só avalia os 12 atributos de maior variância bruta por nó. O resultado depende das unidades de medida e ignora atributos de escala pequena. | MEDIDO: árvore diferente em 14 de 15 unidades após reescala equivalente. | Documentado; correção exige novo manifest |
| P05 | Cinco dos onze ficheiros declarados congelados por hash no `BENCHMARK_MANIFEST_v1.json` foram alterados depois do congelamento (`trepan_original.py`, `trepan_reloaded_historical.py`, `trepan_scientific_tuning.py`, `benchmark/runner.py`, `benchmark/semantic.py`). O manifest nunca foi regenerado; as execuções registam a deriva mas prosseguem. | EXECUTADO: comparação de SHA-256. | Por corrigir |
| P06 | Qualquer coluna derivada `onto_*` recebe um peso de ganho fixo (2,5), independentemente do significado; configurável por variável de ambiente `ONTO_FEATURE_BIAS_WEIGHT`, fora do manifest. Uma ontologia permutada recebe o mesmo bónus. | LIDO. | Por corrigir (novo manifest) |
| P07 | As árvores do TREPAN são muito maiores do que as do C4.5 (mediana 2,17 vezes mais nós, até 9 vezes) e a comparação não controla a complexidade. | MEDIDO. Comparação com complexidade igual na fase 11 (C4.5 de profundidade 2 contra B7/R7; TREPAN limitado ao tamanho do C4.5). | Medido (fase 11); o C4.5 de profundidade 2 é em média mais pequeno do que B7/R7 |
| P08 | Três pipelines paralelos (benchmark, produção, GUI) com valores por omissão diferentes. Um resultado obtido num não se transfere para os outros. | LIDO. | Por corrigir (arquitetura) |
| P20 | Regressão introduzida pela correção de P09 (`af02d5d`): a conversão de texto «todo numérico» em número aplicava-se também a atributos ARFF nominais. `@attribute codigo {01,1}` com 01, 1 e ? ficava com uma só categoria, era classificado como constante e descartado: o atributo não chegava ao modelo em `pipeline_v92` nem em `production_training`. O carregador descartava o esquema declarado. | EXECUTADO: `tests/test_arff_declared_schema.py`, 11 de 13 testes falhavam antes da correção. | Corrigido |
| P21 | A ordem declarada das classes do ARFF (ex.: `{1,01}`) não chegava a `pipeline_v92` nem a `production_training` (o sklearn ordenava os rótulos); a GUI mantinha linhas com alvo `?` e o treino falhava; o benchmark numérico aceitava em silêncio atributos nominais (fundindo categorias) e tratava `?` no alvo como classe. | EXECUTADO: os mesmos testes. | Corrigido |

## Médios

| ID | Problema | Evidência | Estado |
|---|---|---|---|
| P09 | Num CSV, uma coluna numérica com marcadores de ausente (`?`, `NA`) era classificada como categórica. | EXECUTADO: 5 testes falhavam. | Corrigido |
| P10 | Limites verdadeiros (por exemplo ≥ 0) não geram atributos: o indicador é constante e é eliminado. | LIDO + EXECUTADO (30 facetas do Breast Cancer sem efeito). | Documentado |
| P11 | As relações de família só existem para os papéis `mean`, `worst`, `error`. | LIDO. | Documentado |
| P12 | Código de seleção por «dominância» face ao Original e ao C4.5 permanece no extrator (inalcançável em produção, usado por 2 ficheiros de teste). | LIDO. | Documentado |
| P13 | O MLP não é reproduzido bit a bit quando o ambiente numérico muda: em 2 de 20 unidades, o MLP reajustado fora do harness (sem `OMP_NUM_THREADS=1`) deu previsões de teste diferentes das guardadas. | EXECUTADO (iris, sementes 42 e 7). Causa não isolada. | Por investigar |
| P14 | Cobertura de testes de 72,1 % em `core/` e `counterfactuals/`; `metrics_comparator.py` 34 %, `trepan_reloaded_extractor.py` 59 %, `natural_language_explainer.py` 9 %. | EXECUTADO. | Por corrigir |
| P15 | No caminho de produção a ordem das classes é a alfabética (`np.unique`), não a declarada no ARFF. | EXECUTADO: ver P21 e `PRODUCAO_BC_RECALCULO.md` (a ordem muda também a partição estratificada). | Corrigido em `010619f` (P21) |

## Baixos

| ID | Problema | Evidência | Estado |
|---|---|---|---|
| P16 | `core/decision_tree.py` e `core/confirmatory_benchmark.py` não são importados por nenhum módulo (0 % de cobertura). | EXECUTADO (grep + cobertura). | Documentado; não removidos |
| P17 | Mais de 60 relatórios na raiz do repositório, alguns desatualizados (`TEST_REPORT.md` refere 154 testes; a suite tem 1772). | LIDO. | Documentado |
| P18 | Os testes de GUI em Linux precisam de `libEGL`, não referido na documentação de instalação. | EXECUTADO: 7 erros de coleção antes de instalar. | Documentado |
| P19 | As dependências fixadas não instalam em Python 3.13 (numpy 1.26.4). O projeto declara 3.11/3.12. | EXECUTADO. | Documentado |

## O que não é um defeito mas limita a hipótese

- **O MLP não supera o C4.5 em todos os datasets.** Quando não supera, `Original > C4.5` em accuracy é inviável por construção. Ver `ANALISE_ESTATISTICA.md`.
- **O critério de pureza (ε = 0,05) fixa um teto de fidelidade por folha.** É fiel à referência.
- **As ontologias de benchmark são estruturais.** Só permitem médias de grupo.
