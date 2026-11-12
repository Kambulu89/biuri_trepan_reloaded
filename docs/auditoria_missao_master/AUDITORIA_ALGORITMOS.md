# Auditoria dos algoritmos

Âmbito: leitura do código e execução real (ver `VALIDACAO_FUNCIONAL.md`). Estado de cada afirmação: **LIDO** (análise estática), **EXECUTADO** (corrido nesta auditoria), **MEDIDO** (com resultado numérico em `evidencia/`).

## 1. TREPAN Original (`core/trepan_original.py`, 1356 linhas)

Referência: Craven & Shavlik (1996), *Extracting Tree-Structured Representations of Trained Networks*; Craven (1996), tese.

| Componente | Referência | Implementação | Estado |
|---|---|---|---|
| Oráculo | MLP como caixa-preta, só `predict` | `oracle.predict`; rótulos reais nunca usados no ajuste (`uses_real_labels: False`) | LIDO + EXECUTADO |
| Consultas por nó | `min_sample` instâncias por nó, completadas com consultas | `_decision_sample`: completa até `effective_min_sample_`, respeitando `max_queries` | LIDO |
| Distribuição das consultas | marginais: empírica (discretas), KDE (contínuas); modelo local por nó se diferir do pai | `FeatureDistributionModel`, teste de diferença com `local_model_alpha` | LIDO |
| Restrições do caminho | consultas respeitam os testes raiz→nó | amostragem por rejeição em `draw(n, constraints)` | LIDO |
| Expansão | melhor-primeiro por `alcance × (1 − fidelidade)` | `_priority`; registo `is_best_first_choice` por expansão | LIDO |
| Divisões | binárias, m-of-n por procura em feixe com operadores (m, n+1) e (m+1, n+1) | `_best_mofn`, largura 2, `max_n = 3`, teste χ² de diferença entre partições | LIDO |
| Paragem | pureza estatística (proporção ≥ 1 − ε com confiança 1 − δ), limite de nós | limite inferior de Wilson ≥ 1 − `purity_epsilon`; `max_nodes` | LIDO + MEDIDO |
| Poda | colapsar subárvores com a mesma previsão | `_prune_identical_subtrees` | LIDO |

### Desvios em relação à referência

1. **Pré-seleção de atributos por variância bruta (ALTO).** `_candidate_literals` só gera candidatos para os `max_features_per_node` (12 por omissão) atributos de maior `np.nanvar(X)`. A referência considera todos os atributos. A variância bruta depende das unidades de medida, e a árvore opera no espaço não normalizado (a normalização está dentro do pipeline do MLP). **MEDIDO:** reescalar as colunas com a função do oráculo compensada (a mesma função) alterou a árvore (métricas de teste ou número de nós) em 14 das 15 unidades com mais de 12 atributos; o iris, com 4 atributos, não é afetado (`evidencia/exp_variance_prefilter.json`).
2. **Semente única da procura m-of-n.** A procura parte só do melhor literal; a referência parte do melhor teste e do seu complemento. Efeito não medido.
3. **Limiares limitados a 24 por atributo** (amostragem uniforme dos pontos de fronteira). Simplificação de custo; efeito não medido.
4. **`max_depth` e `min_samples_leaf`** não existem na referência. No benchmark o tuning escolhe `max_depth` suficientemente alto para não ser ativo na maioria das unidades (não verificado em todas).
5. **Fidelidade dos filhos estimada só com exemplos reais** no momento de entrar na fila (antes de sortear consultas). A prioridade melhor-primeiro usa portanto uma estimativa com poucas instâncias.
6. **Desempate m-of-n não totalmente determinístico** (limitação já registada pelo projeto em `docs/b1b4/19`).

### Comportamento medido

- **Critério de pureza limita a fidelidade.** Com `purity_epsilon = 0,05` um nó com 97 % de concordância pára. No Breast Cancer (semente 7, todos os atributos) a árvore ficou com 3 nós: os dois filhos pararam com 97,2 % e 97,1 % (`STOP_PURE_NODE`, `wilson_lower_bound`). É fiel à referência, mas fixa um teto de fidelidade próximo de 1 − ε por folha.
- **Árvores muito maiores do que as do C4.5 sem vantagem.** Mediana do rácio de nós Original/C4.5 nas 10 primeiras unidades: 2,45 (até 9 no iris). O iris é resolvido pelo C4.5 com 7 nós; o Original usa 27–63.
- **Fidelidade no teste semelhante à de uma árvore treinada com rótulos reais.** Ver a comparação por dataset em `ANALISE_ESTATISTICA.md` e os valores por unidade em `RESULTADOS_COMPARATIVOS.csv`.

**Conclusão:** a implementação segue a estrutura do TREPAN, com os desvios acima. Não a declaro «cientificamente equivalente» ao TREPAN de Craven & Shavlik; o desvio 1 altera o conjunto de divisões candidatas e depende da escala dos dados.

## 2. C4.5 (`core/c45_j48_tree.py`, 921 linhas)

Referência: Quinlan (1993), *C4.5: Programs for Machine Learning*; J48 do WEKA.

| Componente | Implementação | Estado |
|---|---|---|
| Entropia, ganho de informação, razão de ganho | `_entropy`; `gain / split_info` | LIDO |
| Seleção | razão de ganho máxima entre candidatos com ganho ≥ ganho médio | `_best_split` | LIDO |
| Atributos contínuos | limiar binário; candidatos nas fronteiras entre blocos de valor; correção MDL `log2(N−1)/|D|` | `_numeric_candidate` | LIDO |
| Atributos categóricos | divisão multi-ramo, atributo retirado nos descendentes | `_categorical_candidate` | LIDO |
| Valores ausentes | ganho ponderado pela fração conhecida; distribuição fracionária na previsão | LIDO |
| Poda | erro pessimista com fator de confiança 0,25 (limite superior binomial) | `_extra_errors`, `_prune` | LIDO |
| Não é CART | classe própria; os testes do projeto verificam que não herda de árvores do scikit-learn | EXECUTADO (suite) |

Desvios: sem *subtree raising* (o J48 tem-no ligado por omissão); o limiar é o ponto médio (o C4.5 usa o maior valor observado ≤ ponto médio). Ambos menores e documentados aqui.

**Conclusão:** é um C4.5 genuíno, não um substituto CART. EXECUTADO em todas as unidades do baseline; as métricas foram recalculadas de forma independente (`VALIDACAO_FUNCIONAL.md`).

## 3. TREPAN Reloaded (`core/trepan_reloaded_historical.py`, 2064 linhas; `core/trepan_reloaded_extractor.py`, 5109 linhas)

O Reloaded é uma subclasse do Original. Diferenças reais:

| Mecanismo | O que faz | Depende da ontologia? | Estado |
|---|---|---|---|
| Pesos semânticos no ganho | `score = ganho × peso^força`; peso 2,5 por omissão para qualquer coluna `onto_*` | Só do facto de a coluna ser derivada, não do significado | LIDO |
| Prioridade de atributos | variância × peso semântico na pré-seleção | Idem | LIDO |
| Candidatos m-of-n guiados | combina literais de atributos do mesmo grupo/relacionados | Sim (grupos e *relatedness*) | LIDO |
| Atributos derivados | média padronizada por grupo, relações de família, indicadores de limite, grupos categóricos | Sim | LIDO + EXECUTADO |
| Projeção de consultas | recalcula as derivadas após sortear, para coerência algébrica | Sim | LIDO |
| EFSR | refinamento focado nas regiões de erro, com consultas adicionais | Parcial (suporte semântico); ativo também sem ontologia | LIDO + EXECUTADO |
| Critério `normalized_information_gain`, fração de consultas ativas | mecanismos não semânticos | Não | EXECUTADO |
| Espelho | se a ontologia não tiver efeito, adota a árvore do Original | — | EXECUTADO |

Achados:

1. **O contrato de equivalência verifica-se no benchmark.** O braço D (Reloaded sem ontologia) deu métricas, nós e consultas idênticos aos do Original em todas as unidades executadas. EXECUTADO.
2. **O Reloaded com ontologia consulta o oráculo ~2,3 vezes mais** (mediana, 10 unidades) e demora ~12 vezes mais do que o Original. O orçamento não é comparável.
3. **No pipeline de produção (`core/production_training.py`) não existe braço sem ontologia.** A diferença Reloaded − Original mistura mecanismos semânticos e não semânticos. Na semente 7 do Breast Cancer o Reloaded diferiu do Original (23 contra 27 nós, fidelidade +0,007) com impacto semântico 0 e 0 intervenções EFSR aceites.
4. **O gate de contribuição declarava `SEMANTIC_CONTRIBUTION_SUPPORTED` nesse caso.** Corrigido (`CORRECCOES_IMPLEMENTADAS.md`).
5. **Limites verdadeiros não geram atributos.** O indicador `valor ≥ mínimo` é constante quando o limite é um facto físico (por exemplo ≥ 0) e é eliminado pelo filtro de novidade.
6. **Relações só para os papéis `mean`/`worst`/`error`.** Estrutura específica do Breast Cancer.
7. **Código de «dominância» sem uso em produção.** `_tree_dominates`, `_select_fidelity_constrained_best` e `_pick_best_dominance_candidate` escolheriam a árvore comparando-a com o Original e o C4.5. O único caminho que os chamava termina em `RuntimeError`; `report_ontology_dominance_status` é apenas informativo. Não removido porque dois ficheiros de teste os referenciam.
8. **Três pipelines paralelos** (benchmark em `core/benchmark/`, produção em `core/production_training.py`, GUI em `gui/biuri_app_complete.py` + extrator) com configurações por omissão diferentes (por exemplo `max_features_per_node` 12, 20 e 12; `max_queries` automático, 2000 e variável).

**Cenário de utilização da ontologia (secção 12 da missão):** cenário **C** — a ontologia participa no algoritmo (atributos derivados, pesos, candidatos, projeção de consultas). Comprovado por execução: no iris/42, 15 divisões influenciadas e 10 decisões alteradas. Participar não é o mesmo que ajudar: ver `ANALISE_ESTATISTICA.md`.
