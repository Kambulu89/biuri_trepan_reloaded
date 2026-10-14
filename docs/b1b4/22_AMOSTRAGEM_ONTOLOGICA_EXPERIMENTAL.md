# Doc. 22 — Amostragem ontologicamente coerente (variante experimental) — arquitetura, testes, custo, resultados DEV e decisão

Estado: **implementada, testada e avaliada em DEV** (18 unidades). **Desligada por omissão**; não promovida; B2-A/B2-B/B3/B4 **não** implementados; nenhum dado confirmatório; sem merge.
`COHERENT_SAMPLING_TECHNICAL_STATUS = VALIDATED` · `COHERENT_SAMPLING_SCIENTIFIC_STATUS = PARTIAL_NOT_PROMOTED` (melhora o Reloaded OWL histórico, não supera o Original e não é atribuível à ontologia — §7). Registo de B2-C: `B2C_TECHNICAL_STATUS = VALIDATED`, `B2C_SCIENTIFIC_STATUS = NOT_DEMONSTRATED` (Doc. 20; implementação e evidências preservadas).
CI: commit `f3f331d` — Windows 3.11, Windows 3.12 e `agnosticism-guard` **verdes** (o commit anterior falhou só num teste novo do fingerprint, por checkout CRLF; corrigido no teste). Suite completa local: 1391 passed / 40 skipped.
Legenda: **[IMPL]** implementado · **[TEST]** testado · **[OBS]** observado (DEV, exploratório, `observed_only`) · **[HIP]** hipótese.

## 1. Arquitetura da variante **[IMPL]**
Opção `coherent_sampling=False` do `TrepanReloadedClassifier` (e `coherent_original_indices` no `fit`). Só atua com ontologia efetiva (S3); sem OWL ⇒ Original (S0–S2 inalterados). Incompatível (por agora) com `statistical_fallback` (B2-C).
- `_OriginalSpaceModel` (`core/trepan_reloaded_historical.py`): ajusta e amostra **só as colunas originais** com o **mesmo** `FeatureDistributionModel` do Original (mesmas sementes: global `random_state`, local `random_state+node_id`; mesmos lotes de rejeição) e **recalcula as derivadas com o transformador já ajustado no treino** (o `query_projector` existente). Restrições do caminho (definidas no espaço enriquecido) são avaliadas sobre as linhas levantadas (`_LiftedConstraints`), de modo que também as que usam derivadas são cumpridas.
- `_decision_sample_coherent`: cópia fiel da lógica de `TrepanOriginalClassifier._decision_sample` com modelos no espaço original (teste diferencial). O resto do Reloaded (perfil de erro, gate, score, prioridade) é inalterado; o TREPAN Original não foi tocado.
- Sem consultas ativas, o sorteio **é** o do Original; com ativas usa o *pool* histórico sobre o modelo original-space.
- Baseline de controlo de engenharia: `ConventionalFeatureProvider` (`core/benchmark/semantic.py`): mesma forma de enriquecimento (médias padronizadas de grupos de colunas) com grupos obtidos **dos dados de treino** (agrupamento hierárquico sobre 1−|ρ Spearman|), nº ≤ o de derivadas OWL; determinístico; sem ontologia.
- Braços do benchmark (grupo `COH`; `BenchmarkConfig.coherent_pilot`, `False` por omissão ⇒ listas de braços e configuração idênticas às anteriores): A `trepan_original` · B `reloaded_core` · C `reloaded_owl_full` · D `coh_full` · E `coh_active_off` · F `hist_active_off` · G1 `reloaded_owl_shuffled`, G2 `coh_full_shuf`, G3 `coh_active_off_shuf` · H `conv_coh` · referências R `trepan_original_enr` e Rc `coh_features`.

## 2. Contrato de transformação **[IMPL, TEST]**
1. `X_sint_orig ~ FeatureDistributionModel(treino original)` (+ restrições do caminho levantadas); 2. `X_sint = transform(X_sint_orig)` com o transformador `fit` **só no treino** (`ctx.enrich`); 3. oráculo: `OriginalOracleProjection` (só as colunas originais; dimensão p esperada pelo MLP); 4. ordem das classes (`classes_`) e quality gate OWL inalterados (o contexto só existe se o gate aceitar); 5. a variante nunca altera as originais amostradas (a projeção de domínio OWL, quando ativa, é o único passo adicional e já existia); 6. invariantes: violação algébrica = 0; derivadas = função das originais; sem dependência da identidade do dataset (verificador estático OK).
Limites explícitos: **não** garante plausibilidade estatística das originais (modelo marginal do Original); **não** corrige restrições categóricas conjuntas (limitação herdada do Original; testada).

## 3. Testes **[TEST]**
`tests/test_coherent_sampling.py` (14): defaults desligados; incompatibilidade com B2-C e entradas obrigatórias; flag sem efeito sem ontologia (S0 ≡ Original); **transformador identidade ⇒ árvore, previsões e consultas `predict` ≡ Original byte a byte** (3 sementes); consultas coerentes algebricamente exatas e respeitam restrições do caminho; o sorteio marginal do Original no espaço enriquecido viola as relações (> 90 %); `_OriginalSpaceModel.draw` ≡ sorteios do Original nas colunas originais; determinismo/semente/pickle; desligado ≡ comportamento anterior; limite categórico declarado; **integração OWL real + MLP** (iris, `OriginalOracleProjection`): violação 0 e previsões do oráculo iguais. `tests/test_tuning_fingerprint.py` (8). Regressão: EQ + B1 + B2-C + perf-equivalence + agnosticismo: 362 passed / 40 skipped; **126/126** comparações braço×unidade dos braços sem a flag coincidem bit a bit com a campanha anterior (A, B, C, F, G1, R, C4.5).

## 4. Custo e estratégia computacional
| Item | Valor |
|---|---|
| Benchmark DEV (18 unidades × 15 braços) | 5,1 h de CPU somando unidades; ≈ 2,4 h de relógio com 3 processos; 1 execução (sem tarefas duplicadas) |
| Tuning estrutural (recalculado uma vez: o *fingerprint* mudou) | iris 42 s · wine 231 s · breast_cancer 350 s · digits 3 005 s |
| Unidade Digits | 4 266–4 518 s (< 120 min por unidade, retoma por *checkpoint*) |
| Tempo de árvore (s, média) Original / R / C / D / E / F | iris 0,8/1,3/4,5/5,1/4,6/3,8 · wine 1,7/2,6/22,9/27,1/22,0/21,8 · bc 3,8/7,1/32,4/25,5/18,1/19,2 · digits 24,6/27,9/182/206/155/150 |
| Consultas ao oráculo vs Original | C ×2,22 · D ×2,25 · E ×2,16 · F ×2,18 · H ×2,15 · R ×1,00 |
| Reutilização de *checkpoint* (demonstração) | iris/42 repetida com o *checkpoint* novo: 120 ajustes retomados, tuning 0,3 s (vs 42 s), 14 braços **idênticos** |
O custo adicional de consultas (×2,2) vem da camada semântica (amostra ativa/projetada, `predict_proba` do perfil de erro), não da coerência: D custa o mesmo que C (×2,25 vs ×2,22).

## 5. Resultados DEV — fidelidade e accuracy **[OBS]**
ΔF/ΔAcc face ao Original canónico, emparelhados por unidade (macro por dataset; EP entre as 18 unidades; teste externo só para relato, nunca para seleção):

| Braço | ΔF vs Original | EP | piores/melhores (18) | ΔAcc | ΔNós | ΔProf. | ΔF bc / digits / iris / wine |
|---|---|---|---|---|---|---|---|
| B — Reloaded sem ontologia (≡ Original) | +0.0000 | 0.0000 | 0/0 | +0.0000 | +0.0 | +0.00 | +0.000 / +0.000 / +0.000 / +0.000 |
| C — OWL histórico, ativas ligadas | -0.0130 | 0.0074 | 9/6 | -0.0073 | +0.9 | +0.07 | -0.011 / -0.004 / -0.005 / -0.031 |
| **D — OWL coerente, ativas ligadas** | -0.0018 | 0.0064 | 6/6 | -0.0038 | -2.9 | -0.13 | -0.001 / -0.001 / -0.000 / -0.004 |
| **E — OWL coerente, ativas desligadas** | -0.0016 | 0.0067 | 8/5 | +0.0034 | +1.5 | -0.10 | -0.008 / +0.027 / -0.016 / -0.009 |
| F — OWL histórico, ativas desligadas | -0.0014 | 0.0053 | 10/4 | +0.0013 | +1.5 | +0.70 | -0.007 / +0.007 / -0.011 / +0.004 |
| G1 — permutada, histórico, ativas | -0.0006 | 0.0082 | 5/9 | +0.0090 | +1.9 | +0.52 | -0.010 / +0.013 / -0.011 / +0.004 |
| G2 — permutada, coerente, ativas | +0.0045 | 0.0063 | 5/9 | +0.0102 | -0.8 | +0.23 | +0.006 / +0.021 / -0.000 / -0.009 |
| G3 — permutada, coerente, ativas desligadas | -0.0076 | 0.0051 | 9/4 | -0.0005 | -3.5 | -0.23 | -0.013 / -0.008 / -0.005 / -0.004 |
| H — features convencionais (sem ontologia), coerente, só representação | -0.0035 | 0.0048 | 7/5 | -0.0032 | +0.3 | +0.10 | +0.003 / +0.000 / -0.021 / +0.004 |
| R — Original na representação OWL (marginal) | -0.0089 | 0.0063 | 9/6 | -0.0048 | -0.9 | +0.40 | -0.004 / -0.007 / -0.016 / -0.009 |
| Rc — OWL coerente, só representação (λ=0) | -0.0030 | 0.0067 | 5/7 | +0.0040 | -0.7 | +0.05 | +0.003 / +0.000 / -0.011 / -0.004 |

Contrastes pedidos (emparelhados por unidade; **10 contrastes sem correção para multiplicidade** — indicativos):

| Contraste (emparelhado por unidade, 18) | ΔF (macro) | EP | piores/melhores | ΔAcc | EP | ΔNós |
|---|---|---|---|---|---|---|
| Efeito da coerência na representação (Rc − R) | +0.0058 | 0.0054 | 5/9 | +0.0088 | 0.0102 | +0.2 |
| Efeito da coerência, com ativas (D − C) | +0.0112 | 0.0051 | 2/10 | +0.0035 | 0.0065 | -3.8 |
| Efeito da coerência, sem ativas (E − F) | -0.0002 | 0.0063 | 6/8 | +0.0021 | 0.0084 | -0.1 |
| Efeito das consultas ativas, histórico (C − F) | -0.0116 | 0.0070 | 10/4 | -0.0086 | 0.0096 | -0.6 |
| Efeito das consultas ativas, coerente (D − E) | -0.0002 | 0.0062 | 6/6 | -0.0072 | 0.0085 | -4.3 |
| Ontologia verdadeira vs permutada, coerente+ativas (D − G2) | -0.0064 | 0.0046 | 7/3 | -0.0140 | 0.0077 | -2.0 |
| Ontologia verdadeira vs permutada, coerente sem ativas (E − G3) | +0.0060 | 0.0060 | 3/6 | +0.0039 | 0.0069 | +5.0 |
| Ontologia verdadeira vs permutada, histórico (C − G1) | -0.0124 | 0.0074 | 10/3 | -0.0163 | 0.0070 | -1.0 |
| Features OWL vs convencionais, só representação (Rc − H) | +0.0004 | 0.0071 | 5/8 | +0.0072 | 0.0092 | -1.0 |
| OWL completo coerente vs convencionais (E − H) | +0.0019 | 0.0073 | 8/7 | +0.0066 | 0.0087 | +1.2 |
| **Interação coerência × ativas** ((D−C)−(E−F)) | +0.0144 | 0.0077 | 5/12 | +0.0045 | 0.0097 | – |

Coerência das consultas, custo e aceitação semântica por braço:

| Braço | viol. algébrica (taxa) | magnitude média (DP) | fora do intervalo do treino | AUC real-vs-sintético (enriquecido) | consultas ×Orig | tempo árvore ×Orig | taxa aceitação semântica | aceites por tolerância |
|---|---|---|---|---|---|---|---|---|
| A — Original canónico | nan | nan | 0.463 | 0.928 | 1.00 | 1.0 | – | – |
| R — Original na repr. OWL | 1.000 | 1.307 | 0.446 | 0.948 | 1.00 | 1.6 | – | – |
| C — histórico | 0.000 | 0.000 | 0.406 | 0.936 | 2.22 | 8.6 | 0.36 | 181 |
| D — coerente+ativas | 0.000 | 0.000 | 0.405 | 0.936 | 2.25 | 9.0 | 0.36 | 165 |
| E — coerente | 0.000 | 0.000 | 0.451 | 0.938 | 2.16 | 7.3 | 0.37 | 167 |
| F — histórico sem ativas | 0.000 | 0.000 | 0.454 | 0.941 | 2.18 | 7.1 | 0.36 | 155 |
| G2 — permutada coerente | 0.000 | 0.000 | 0.383 | 0.934 | 2.23 | 8.4 | 0.33 | 150 |
| H — convencional | 0.000 | 0.000 | 0.454 | 0.920 | 2.15 | 1.9 | – | 0 |

Hierarquia (objetivo pré-registado, não imposto), D e E: `Acc_R > Acc_O` em 8/18 e 9/18 unidades; `Acc_O > Acc_C4.5` em 5/18 (independente do Reloaded); ordem completa de accuracy em 0/18 e 1/18; `F_R > F_O` em 6/18 e 5/18; `F_R ≥ F_O` em 12/18 e 10/18. Estado: `HIERARCHY_NOT_SATISFIED` (DEV) com `INSUFFICIENT_EVIDENCE` estatística.
Estabilidade da seleção de divisões (11 unidades: iris×5, wine×3, breast_cancer×3; 3 réplicas por braço com sementes de amostragem diferentes; orçamento fixo): Jaccard médio das features das 5 primeiras divisões — histórico/ativas 0,686 · histórico/sem 0,710 · coerente/ativas 0,644 · coerente/sem 0,704; Jaccard da árvore toda 0,818 / 0,816 / 0,789 / 0,799; DP da fidelidade de treino 0,006–0,008. **Sem melhoria de estabilidade** com a coerência.

## 6. Interpretação — o que a coerência explica e o que não explica **[OBS + HIP]**
1. **O Reloaded histórico já era algebricamente coerente** (violação 0; Doc. 21): o efeito «D − C» **não** é a correção de violações algébricas.
2. **Coerência algébrica isolada** (só representação, sem semântica): Rc − R = +0,0058 (EP 0,0054; 5 piores/9 melhores) e projetor sozinho `reloaded_owl_features` − R = +0,0055 (EP 0,0066): pequeno positivo, **não significativo**; mudar também o modelo de amostragem para o espaço original (Rc − `reloaded_owl_features`) = +0,0004 (EP 0,0092): **sem efeito** no regime só-representação.
3. **D − C = +0,0112** (EP 0,0051; 2 piores/10 melhores; positivo em todos os 4 datasets) aparece com a camada semântica ligada e **interage com as consultas ativas** (interação +0,0144, EP 0,0077): com ativas desligadas o efeito é ≈ 0 (E − F = −0,0002). **[HIP]** a mudança de modelo/RNG de sorteio altera a pesquisa semântica (testes de modelo local só nas colunas originais, ausência de sorteios desperdiçados das derivadas, outro fluxo de RNG) e essa alteração interage com a seleção ativa; **não isolado**.
4. **Não é efeito da ontologia:** o controlo permutado coerente com ativas (G2) fica **acima** do OWL verdadeiro em fidelidade (D − G2 = −0,0064, EP 0,0046; accuracy −0,0140, EP 0,0077); sem ativas, E − G3 = +0,0060 (EP 0,0060; dominado por digits, +0,035). O ganho de D face a C é, portanto, atribuível ao regime de amostragem/interação e não à semântica verdadeira.
5. **Face ao Original:** D = −0,0018 (EP 0,0064; 6 piores/6 melhores) e E = −0,0016 (EP 0,0067): **paridade**, não superioridade; accuracy −0,0038 (D) / +0,0034 (E), dentro do ruído. A engenharia convencional de features (H) fica em −0,0035 (EP 0,0048) e E − H = +0,0019 (EP 0,0073): **a ontologia não supera a engenharia convencional** nestes dados.
6. **Consultas ativas:** no regime histórico custam ≈ −0,0116 (C − F, EP 0,0070); no regime coerente o efeito é ≈ 0 (D − E = −0,0002) — i.e. a perda atribuída às ativas no Doc. 18 desaparece com a amostragem coerente (interação). Mantêm-se ligadas por omissão; não se desligam permanentemente.
7. **Aceitação semântica:** 36 % (D) vs 33 % (G2) de aceitação; 74 % das aceitações de D passam só pela tolerância de coerência (165/222). O gate não discrimina OWL verdadeiro de permutado ⇒ o problema de B2-A/B2-B mantém-se, mas **a coerência não o resolve**.

## 7. Decisão científica
| Critério pedido | Veredito |
|---|---|
| fidelidade ao MLP | melhora o histórico (+0,011), **paridade** com o Original (−0,002 ± 0,006); não superior |
| accuracy | −0,004 (D) / +0,003 (E) vs Original; inconclusiva |
| complexidade | D: −2,9 nós vs Original (−3,8 vs C); profundidade −0,13; sem crescimento artificial |
| estabilidade | sem melhoria (Jaccard 0,64 vs 0,69) |
| custo | mesmo que C (×2,25 consultas; ×9 tempo de árvore) |
| utilização real da ontologia | **não demonstrada**: permutada ≥ real; convencional ≈ real |
**Decisão:** manter a variante **experimental e desligada**; **não** promover; **não** declarar superioridade. A estratégia corrige uma incoerência real do «Original equivalente» (R) e melhora o Reloaded histórico, mas o resultado final é neutro face ao Original e sem atribuição à ontologia. Resultado neutro/negativo documentado; sem heurísticas por dataset.

## 8. B2-A: ainda se justifica? (decisão do utilizador; nada implementado)
Sim, **conceptualmente**: com amostragem coerente o gate continua a aceitar 36 % dos nós (74 % por tolerância) e aceita tanto com a ontologia verdadeira (36 %) como com a permutada (33 %) — não discrimina informação semântica de ruído. Mas o **teto de ganho é pequeno**: D já está em paridade com o Original (−0,002) e os efeitos observados estão ao nível de 1–2 EP. Proposta de piloto pequeno (não executado): `B2-0`, `B2-A`, `B2-C`, `B2-A+C` (B2-C exigiria permitir a combinação com a amostragem coerente, hoje bloqueada) sobre as 18 unidades DEV, com validação interna separada da seleção, ganhos por linha (`d_i`) e clusters; custo estimado ≈ 15 min de CPU por configuração (árvores) ⇒ ≈ 1 h para 4 configurações + instrumentação, sem tuning (checkpoints reaproveitáveis, §9). A matriz de 30 configurações **não** se executa antes disto.

## 9. Política de *checkpoints* de tuning (fingerprint) **[IMPL, TEST]**
`core/tuning_checkpoint.py` — `FINGERPRINT_VERSION = 2`; chave = candidato + repetição/seed/dobra + hash dos índices de treino + `oracle_id` + **fingerprint**. O fingerprint (relatório auditável gravado ao lado do *checkpoint*: `*.ckpt.fingerprint.json`, com histórico) = SHA-256 dos ficheiros em `EFFECTIVE_DEPENDENCIES` (`trepan_original`, `trepan_scientific_tuning`, `controlled_trepan_experiment`, `scientific_experiment_contract`, `training_config`, `tuning_checkpoint`; fins de linha normalizados) + versões de Python/NumPy/SciPy/scikit-learn + plataforma + ambiente numérico fixado. Análise pedida: o tuning estrutural só executa ajustes do TREPAN **Original** contra o oráculo congelado (seleção estrutural, CV, objetivo e métricas em `trepan_scientific_tuning`; oráculo e pré-processamento/treino entram na chave por `oracle_id` e hash dos índices; RNG por seed da repetição na chave; critérios de aceitação/enriquecimento/geração de candidatos semânticos **não** intervêm). `trepan_reloaded_historical.py` não executa nesse caminho (etapa semântica desligada e não checkpointada): a **prova é dinâmica** — `test_executed_repo_files_during_tuning_are_declared` traça a execução real do tuning e falha se algum ficheiro do repositório executado não estiver declarado. Em dúvida invalida: qualquer alteração a um ficheiro declarado, à versão de uma biblioteca, à plataforma ou ao ambiente numérico muda o fingerprint e força o recálculo; alterar só o Reloaded **não** invalida (demonstrado por teste e pela reutilização de §4).

## 10. Desempate determinístico (D4) e manifest v3
Estudo `Original-det vs Reloaded-det` mantido separado e **sem alterações** nesta etapa (Doc. 19): resultados com regimes de desempate diferentes nunca se misturam; o benchmark acima usa exclusivamente o desempate canónico. Qualquer extensão do Reloaded-det exige protocolo prévio mostrando que os dois algoritmos usam a mesma política. Manifest v3 (rascunho, `frozen = false`): inclui as variantes de amostragem (`coherent_sampling`, `ConventionalFeatureProvider`), o *fingerprint* e os estados de B2-C/coerência; não executado em confirmação.

## 11. Limitações
DEV com 4 datasets heterogéneos; EP ≈ 0,005–0,007 por contraste; 10 contrastes sem correção; mecanismo do efeito D−C não isolado; plausibilidade estatística não resolvida (AUC ≈ 0,93); categóricos conjuntos não verificáveis em dados reais DEV; «MLP_OWL» com entrada enriquecida não avaliada (experiência separada, outro oráculo).
