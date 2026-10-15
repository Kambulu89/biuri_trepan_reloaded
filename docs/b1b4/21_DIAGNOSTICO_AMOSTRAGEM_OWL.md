# Doc. 21 — Diagnóstico da coerência das amostras sintéticas com enriquecimento OWL (P0)

Estado: diagnóstico **concluído** (DEV; só treino e rótulos do oráculo congelado; nada do teste; sem lógica por dataset). Evidência: `docs/evidence/b1b4/coherent_sampling/` (`transform_contract/`, `repr_queries/`, `benchmark/audits/`). Scripts: `validation/benchmark/diag_transform_contract.py`, `diag_repr_queries.py`, `coherence_metrics.py`, `run_coherent.py`. Legenda: **[OBS]** observado (DEV, exploratório) · **[IMPL]** lido no código · **[HIP]** hipótese não confirmada.

## 0. Conclusão principal (e correção da hipótese de partida)
A hipótese de partida («a amostragem marginal produz combinações incoerentes que degradam o Reloaded OWL») **só se confirma para a referência «Original na representação OWL» e para a amostra-base de B2-C**, **não para o Reloaded OWL histórico**: este já recalcula as derivadas depois de sortear (projetor ontológico), com violação algébrica **0** em todas as unidades. Logo a incoerência algébrica **não explica** a perda do Reloaded histórico face ao Original; explica parte do handicap da referência equivalente. Separadamente, **coerência algébrica ≠ plausibilidade estatística**: as consultas continuam muito distinguíveis das linhas reais (AUC real-vs-sintético ≈ 0,93 em todos os braços, incluindo o Original canónico) e, em breast_cancer, as derivadas de amostras coerentes ficam fora do intervalo do treino em 100 % das linhas.

## 1. Percurso completo da amostra **[IMPL + OBS]**
```
Dataset (sklearn/loader; matriz numérica «as loaded», sem imputação/encoding/seleção)                validation/benchmark/datasets.py, core/benchmark/preprocessing.py
 → holdout estratificado 25 % (1 por semente mestra)                                               core/benchmark/runner.py::_run_split
 → MLP (pipeline sklearn; padronização INTERNA ajustada no treino) → FrozenOracle (oracle_id)       runner.fit_mlp, freeze_oracle
 → OwlSemanticProvider.build(Xtr): reasoner + quality gate + OntologyProcessor.fit (centros/escalas só do treino)
                                                                                                   core/benchmark/semantic.py, core/ontology_processor.py
 → ctx.transform: [originais | derivadas = média padronizada das fontes]  (determinística, por linha)  OntologyProcessor.transform/_apply_spec
 → oráculo das árvores = OriginalOracleProjection(FrozenOracle, orig_idx): só as p colunas originais chegam ao MLP
                                                                                                   core/controlled_trepan_experiment.py
 Árvores (todas contra o MESMO oráculo):
  A  Original canónico: espaço ORIGINAL (p cols); FeatureDistributionModel(p) → model.draw → oráculo      core/trepan_original.py::_decision_sample
  R  Original no espaço ENRIQUECIDO (p+k): FeatureDistributionModel(p+k) sorteia TODAS as colunas MARGINALMENTE; sem projetor
  C  Reloaded OWL histórico: modelo (p+k) → _draw_unconstrained → projetor(raw)=ctx.enrich(raw[:, orig_idx]) (RECALCULA derivadas)
        → restrições do caminho → [amostragem ativa: pool×4, topo por fronteira/âncoras de erro] → oráculo   core/trepan_reloaded_historical.py::_draw_membership_queries(_unfiltered)
  D/E Reloaded OWL coerente (nova variante): modelo ORIGINAL (p) = o do Original → sorteio → ctx.enrich → restrições → oráculo
                                                                                                   _OriginalSpaceModel, _decision_sample_coherent
```
Espaços: **dados/oráculo** p colunas; **árvore** p+k colunas (k = nº de derivadas: iris 2, wine 3, breast_cancer 53, digits 3); o MLP nunca vê as derivadas (a «MLP_OWL» com entrada enriquecida **não** é usada nestas experiências; ficaria numa experiência separada com outro oráculo).

## 2. As 10 verificações pedidas
| Métrica | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| dim. original p → enriquecida p+k → entrada do oráculo | 4 → 6 → 4 | 13 → 16 → 13 | 30 → 83 → 30 | 64 → 67 → 64 |
| transformação determinística / originais inalteradas | sim / sim | sim / sim | sim / sim | sim / sim |
| independência de linha/lote (máx. abs. Δ) | 0.0e+00 | 0.0e+00 | 2.2e-16 | 3.3e-16 |
| invariância à permutação (máx. abs. Δ) | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| projetor recalcula derivadas ignorando a entrada (máx. abs. Δ) | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| transformação das consultas = a do treino (máx. abs. Δ) | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| controlo permutado: coerente com a sua própria definição (abs. Δ) | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| entradas extremas (3/10/50 DP) finitas | sim | sim | sim | sim |
| entrada com NaN | rejects: OntologySchemaError | rejects: OntologySchemaError | rejects: OntologySchemaError | rejects: OntologySchemaError |
| derivadas: abs(z) máx. com extremos de 3 DP (DP do treino) | 4.6 | 5.0 | 106.6 | 6.9 |
| sorteio marginal no espaço enriquecido: incoerência média (DP) | 1.25 | 1.35 | 0.79 | 1.41 |
| após projeção / espaço original + transformador | 0.00 | 0.00 | 0.00 | 0.00 |
| derivadas fora do intervalo do treino — linhas reais (hold-out) | 0.053 | 0.022 | 0.056 | 0.009 |
| derivadas fora do intervalo do treino — sintéticas coerentes (raiz) | 0.058 | 0.017 | 1.000 | 0.000 |
| p99 abs(z) das derivadas — real / sintético coerente | 2.9 / 2.7 | 2.2 / 2.1 | 3.1 / 15.7 | 3.2 / 1.8 |

**Tabela de contrato do transformador e coerência/plausibilidade das derivadas (4 datasets DEV, semente 42, treino 75 %):** acima.

1. **Derivadas preservam relações determinísticas com as originais?** Sim: por construção (média padronizada das fontes, parâmetros ajustados só no treino), determinística, independente da linha/lote/ordem (≤ 3,3e-16, ruído de ponto flutuante), originais inalteradas **[OBS]**.
2. **A geração de consultas quebra essas relações?** Depende do braço **[OBS]**: no Original/B2-C-base no espaço enriquecido, **100 % das linhas** violam (magnitude média ≈ 0,8–1,4 DP da derivada; ver §3); no Reloaded histórico e na variante coerente a violação é **0,000** em todas as unidades (audit: taxa e magnitude).
3. **Amostras fora do domínio válido da transformação?** A transformação é definida (finita) para qualquer entrada finita, incluindo extremos de ±50 DP; rejeita NaN (`OntologySchemaError`) **[OBS]**. O «domínio válido» relevante é o **estatístico**: 38–46 % das consultas têm alguma feature original fora do intervalo do treino em **todos** os braços (incluindo o Original canónico: caudas do KDE); a coerência não altera isto.
4. **Derivadas recalculadas após a amostragem?** Reloaded histórico e coerente: sim (projetor); Original-OWL e amostra-base de B2-C: não **[IMPL+OBS]**.
5. **Mesma transformação nas consultas e no treino?** Sim: `ctx.enrich` é o objeto usado em `_enriched(ctx, X)` (treino/teste) e em `query_projector()`; diferença máxima 0 **[OBS]**; o controlo permutado é coerente com a sua própria definição (0) mas semanticamente destruído.
6. **Atributos categóricos preservam restrições válidas?** Os 4 datasets DEV são numéricos (sem categóricos nominais): **não verificável em dados reais**. Teste sintético: colunas one-hot amostradas marginalmente produzem combinações inválidas (> 30 % das linhas) em **todas** as variantes, incluindo o Original — limitação herdada, **não corrigida** (`test_marginal_sampling_does_not_preserve_joint_categorical_constraints`). Colunas inteiras: ≈ 44 % das linhas têm alguma coluna inteira com valor não inteiro (KDE), também no Original; categorias discretas inválidas: 0,1–0,4 %.
7. **Incoerências entre espaços original/enriquecido/oráculo?** Nenhuma de dimensão (tabela §2: p → p+k → p; `OriginalOracleProjection` devolve previsões idênticas às do MLP nas linhas de treino) **[OBS]**.
8. **A projeção semântica altera indevidamente a distribuição?** A projeção ontológica só recalcula as derivadas; não altera as originais (a projeção de **domínio** OWL não está ativa no benchmark: não se passa grafo). KS médio das originais contra as linhas reais é praticamente igual (0,19–0,20) com e sem projeção/coerência (Original canónico 0,186) **[OBS]**. O que altera a distribuição é a **amostragem ativa**.
9. **As consultas ativas concentram-se em regiões pouco representativas?** Concentram-se junto à fronteira por desenho: fração a < 0,25 DP dos melhores limiares sobe em iris (0,65→0,82), wine (0,42→0,57), digits (0,30→0,38) e não em breast_cancer **[OBS]**; a taxa de linhas fora do intervalo do treino **desce** com ativas (0,41 vs 0,45 sem), pelo que não há evidência de deriva para regiões mais distantes dos dados; o AUC real-vs-sintético não muda (0,936 vs 0,938).
10. **Associação com divisões prejudiciais?** Nível de nó (18 unidades, `diag_repr_queries`): o melhor teste escolhido numa amostra **coerente** generaliza melhor para as linhas reais do nó do que o escolhido na amostra-base incoerente em iris (+0,105) e wine (+0,053), ≈ 0 em digits (+0,005) e **pior** em breast_cancer (−0,014); média +0,041 (EP 0,017) contra referência nula +0,009 (EP 0,013) **[OBS]**. A associação é **sugestiva e heterogénea**, não prova causalidade (ver §5).

## 3. Demonstração das incoerências encontradas
| Braço | amostragem | violação algébrica | quem a produz |
|---|---|---|---|
| A Original canónico | espaço original, sem derivadas | n/a | — |
| R Original na repr. OWL | marginal (p+k) | **100 % das linhas, ≈ 0,8–1,4 DP** | `FeatureDistributionModel(p+k).draw` (`core/trepan_original.py`) — comportamento do Original, **não** é erro do Reloaded |
| B2-C amostra-base | idem (Original) | 100 % | `TrepanOriginalClassifier._decision_sample` no espaço enriquecido |
| C Reloaded histórico | marginal (p+k) + projetor | 0 | projetor recalcula derivadas (`query_projector`) |
| D/E coerente | original (p) + transformador | 0 | `_OriginalSpaceModel` |
Funções/ficheiros responsáveis: `core/trepan_original.py` (`FeatureDistributionModel.fit/_sample_column/_draw_unconstrained/draw`, `_decision_sample`, `fit`: modelo global em X enriquecido), `core/trepan_reloaded_historical.py` (`_draw_membership_queries`, `_draw_membership_queries_unfiltered`, `_OriginalSpaceModel`, `_decision_sample_coherent`), `core/benchmark/runner.py` (`_run_arm`: `X_fit`, `OriginalOracleProjection`, `query_projector`), `core/benchmark/semantic.py` (`OwlSemanticProvider.build`, `SemanticContext.enrich/query_projector`), `core/ontology_processor.py` (`transform`, `_apply_spec`).

## 4. Coerência algébrica vs plausibilidade estatística **[OBS]**
- Consultas coerentes continuam **muito distinguíveis** das linhas reais: AUC (floresta aleatória, 3 dobras) ≈ 0,93 em todos os braços (A canónico 0,93; R 0,95; C 0,94; D 0,94). A causa é o modelo marginal do Original (colunas independentes), partilhado por todos.
- Em **breast_cancer** (53 agregados de colunas fortemente correlacionadas) as derivadas de amostras **algebricamente coerentes** ficam fora do intervalo observado no treino em **100 %** das linhas (|z| p99 = 15,7 vs 3,1 nas linhas reais): decorrelação das originais ⇒ agregados fora do suporte. Em iris/wine/digits as derivadas sintéticas coerentes têm a mesma taxa de «fora do intervalo» que as linhas reais.
- Conclusão: a transformação garante coerência algébrica; **não** garante plausibilidade da amostra original nem das derivadas resultantes. Nenhuma inferência de qualidade semântica se faz a partir da redução de incoerência.

## 5. Nível de nó e hipóteses causais
| Métrica por nó (média por dataset; 18 unidades) | iris | wine | breast_cancer | digits |
|---|---|---|---|---|
| KS médio base vs semântica (ativa+projeção) | 0.281 | 0.138 | 0.137 | 0.044 |
| fração junto à fronteira — base | 0.650 | 0.422 | 0.670 | 0.296 |
| fração junto à fronteira — semântica | 0.823 | 0.569 | 0.654 | 0.382 |
| fidelidade nas linhas REAIS da escolha feita na amostra-base incoerente | 0.696 | 0.736 | 0.720 | 0.587 |
| … feita noutra amostra-base incoerente (referência nula) | 0.735 | 0.730 | 0.723 | 0.581 |
| … feita na amostra semântica (ativa) | 0.718 | 0.698 | 0.659 | 0.573 |
| … feita na amostra COERENTE (espaço original + transformador) | 0.801 | 0.797 | 0.707 | 0.592 |
| diferença coerente − incoerente nas linhas reais | +0.105 | +0.053 | -0.014 | +0.005 |

Hipóteses (**não confirmadas**): H1 a incoerência algébrica degrada a seleção de divisões no Original-OWL (suporte: nó iris/wine; sem suporte em breast_cancer); H2 a perda do Reloaded histórico **não** vem da incoerência algébrica (violação 0); H3 a diferença entre o histórico e o coerente vem de outros fatores (modelo local testado em todas as colunas incl. derivadas, consumo de RNG, desperdício de sorteios das derivadas), ver Doc. 22 §6; H4 a amostragem ativa pode concentrar a pesquisa em regiões não representativas das linhas reais em alguns datasets (iris/wine) — evidência indireta.

## 6. Riscos de fuga de informação
| Risco | Estado |
|---|---|
| Teste usado na seleção/desenho | **Não**: os diagnósticos usam só treino e rótulos do oráculo; as variantes não foram escolhidas pelo teste (matriz fixada antes de correr) |
| Transformador ajustado fora do treino | **Não**: `OntologyProcessor.fit` só com `Xtr`; conventional: centros/escalas/clusters só do treino |
| Rótulos reais nas consultas | **Não**: oráculo = MLP congelado; as linhas reais do nó são as do treino já usadas pelo Original |
| Modelo de consultas ajustado a dados de teste | **Não**: ajustado a `node.real_X` (treino) |
| Dobras internas do diagnóstico | n/a (sem seleção); AUC real-vs-sintético usa validação cruzada interna sobre treino+sintético |
| Seleção de variante por resultados DEV | Evitada: sem ajuste posterior; decisões em Doc. 22 declaradas exploratórias |
| Heterogeneidade/múltiplas comparações | ~10 contrastes sem correção: tratar p-valores/EP como indicativos |
