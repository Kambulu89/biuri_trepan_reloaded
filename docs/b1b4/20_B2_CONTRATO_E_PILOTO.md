# Doc. 20 — B2.0 (diagnóstico e contratos) e B2-C (implementação isolada + piloto DEV)

Estado: B2.0 e B2.1 **implementados e testados**; B2.2 (piloto DEV) **executado** (resultados exploratórios, `observed_only`, DEV ≠ confirmatório). B2-A, B2-B, B3 e B4 **não** foram implementados; `cv-lite` não implementado; nenhum dado confirmatório; sem merge.
**Registo formal (decisão do utilizador após o piloto):** `B2C_TECHNICAL_STATUS = VALIDATED` · `B2C_SCIENTIFIC_STATUS = NOT_DEMONSTRATED`. B2-C permanece implementado como mecanismo experimental **desligado por omissão**; não é promovido; a implementação e as evidências ficam preservadas. Continuação: Doc. 21 (diagnóstico da coerência das amostras) e Doc. 22 (amostragem coerente).
Legenda de evidência: **[IMPL]** implementado · **[TEST]** coberto por teste · **[OBS]** observado (DEV, exploratório) · **[HIP]** hipótese não confirmada.

## 1. Diagnóstico das interações (B2.0)

### 1.1 Mapa de dependências no código **[IMPL = leitura do código; verificado por testes onde indicado]**

| De → Para | Mecanismo concreto | Consequência |
|---|---|---|
| Representação (R) → Oráculo (O) | O oráculo vê só as colunas originais (`OriginalOracleProjection`); as derivadas são funções determinísticas das originais | A rotulagem é correta em qualquer representação (verificado: previsões iguais, §1.3) |
| R → Consultas (Q) | O `FeatureDistributionModel` ajustado no espaço enriquecido sorteia **cada coluna marginalmente**; as derivadas deixam de ser função das originais nas linhas sintéticas | O «Original equivalente» na representação OWL (`trepan_original_enr`) aprende com linhas sintéticas **incoerentes** (§1.3) |
| Projeção OWL (Q) ← R | `query_projector` recalcula as derivadas a partir das originais; atua mesmo com consultas ativas desligadas (`projector is not None`) | Coerência restaurada; é um mecanismo ontológico distinto da seleção ativa |
| Amostragem ativa (Q) → RNG (N) | Sorteia um *pool* ×4 e fica com o topo por proximidade da fronteira/âncoras de erro (sem consultar o oráculo antes de selecionar) | Consome mais estado do RNG do modelo partilhado e concentra a amostra junto da fronteira |
| Perfil de erro / EFSR (F) → Q, S, G, ordem | `_build_error_region_profile` (+ `predict_proba` nas linhas reais) alimenta: elegibilidade; consulta ativa só em regiões de erro; `ErrorCoverage` no score; prioridade de expansão; contador *top-k fallback* **com estado** entre nós | F liga quatro mecanismos ao mesmo tempo |
| Prioridade (F) → N | A ordem best-first decide **que nó consome o RNG partilhado primeiro** (o `local_model` é herdado pelos filhos) | Mudar a prioridade altera as amostras de todos os nós seguintes, mesmo sem alterar nenhuma divisão |
| Score semântico (S) → candidatos | `_feature_priority_scores` (variância·peso^k) escolhe o *pool* de features candidatas; `_ontology_guided_candidates` (≤24); calibrações B1 | S altera o espaço de candidatos antes de qualquer pontuação |
| Gate (G) ← S, amostra, perfil, linhas reais | Compara fidelidade local do teste semântico com a do teste «data-only» **na mesma amostra D**; 2.º gate nas linhas reais (subconjunto de D) | O gate herda o enviesamento da amostra (ativa) e dos candidatos (S) |
| Fallback histórico (G) → ≠ Original | O «data-only» é recalculado em D semântica (ativa) e com o RNG já avançado | Não coincide com a escolha do Original (hipótese H2, Doc. 18) |

### 1.2 O que pode ser isolado e o que depende de quê
| Intervenção | Isolável? | Como (nesta implementação) | Dependências residuais |
|---|---|---|---|
| Representação (R) | Sim | `trepan_original_enr` (Original na mesma representação) | R inclui a incoerência das consultas (R×Q) |
| Projeção OWL / domínio (Q) | Parcial | `reloaded_owl_features` (λ=0: projeção + representação, sem score/gate) | Atua em conjunto com R |
| Amostragem ativa (Q) | Sim | `diag_active_off` (DIAG, Doc. 18) | Só atua onde F declara região de erro |
| Gate (G) | Sim | `diag_gate_off`; ou gate que rejeita sempre (cenário **inerte** dos testes) | Efeito depende de S e da amostra |
| Score/candidatos (S) | Sim | `reloaded_semantic_score`; B1 λ | Depende de F (ErrorCoverage) |
| Prioridade de nós (F) | Sim (B2-C) | `b2c_semantic_node_priority` (False por omissão em B2-C) | Altera N |
| Estado RNG (N) | Sim (B2-C) | `rng_original` (modelo) vs `rng_semantic` (cópia com semente derivada) | Só enquanto a ordem de expansão coincidir com a do Original |
| Oráculo/consultas (O) | Sim | Contabilidade etiquetada `original`/`semantic` | — |
Não isoláveis entre si: **Q×F** (consulta ativa só em regiões de erro), **S×F** (`ErrorCoverage`), **G×Q×S** (o gate mede ganho numa amostra escolhida por Q para candidatos de S), **N×prioridade** (ordem de expansão).

### 1.3 Representação OWL — análise específica (§10 do pedido) **[OBS, DEV, agnóstica; só treino + rótulos do oráculo; `docs/evidence/b1b4/b2_diag/repr_queries_*.json`]**
Origem: `validation/benchmark/diag_repr_queries.py` (18 unidades DEV; tuning estrutural desligado e orçamento fixo — o diagnóstico não mede desempenho final; sem teste e sem rótulos reais; um ajuste B2-C **inerte** fornece as amostras por nó). Médias por dataset (5/5/5/3 sementes).

| métrica | breast_cancer | digits | iris | wine |
|---|---|---|---|---|
| nº features originais | 30 | 64 | 4 | 13 |
| nº features derivadas | 53 | 3 | 2 | 3 |
| derivadas com R² > 0,95 a partir das originais | 0.770 | 1.000 | 1.000 | 1.000 |
| derivadas com MI(oráculo) acima do p95 nulo | 0.777 | 1.000 | 0.700 | 1.000 |
| MI mediana derivadas | 0.098 | 0.531 | 0.555 | 0.466 |
| MI mediana originais | 0.191 | 0.243 | 0.750 | 0.420 |
| desvio-padrão mediano derivadas | 0.064 | 0.292 | 0.823 | 0.623 |
| desvio-padrão mediano originais | 0.072 | 4.255 | 0.792 | 0.802 |
| candidatos (espaço original) | 571.600 | 334.667 | 71.200 | 566.800 |
| candidatos (espaço enriquecido) | 568.000 | 334.667 | 125.200 | 564.800 |
| derivadas no top-12 de candidatos da raiz | 0.733 | 0.000 | 0.500 | 0.067 |
| melhor candidato da raiz é derivado | 0.600 | 0.000 | 0.000 | 0.000 |
| incoerência das derivadas num sorteio marginal (unid. de DP) | 0.788 | 1.407 | 1.246 | 1.346 |
| incoerência após projeção OWL | 0.000 | 0.000 | 0.000 | 0.000 |
| literais derivados na árvore inerte (média) | 3.800 | 0.000 | 0.200 | 0.400 |

Resultados **[OBS]**:
1. **Correspondência e transformação (checks de erro de implementação): sem erros detetados.** Em todas as 18 unidades: transformação finita, determinística, colunas originais inalteradas, projetor idempotente (erro 0), e o oráculo projetado dá previsões idênticas nas representações original e enriquecida. Não há derivadas constantes. **Não há erro de transformação** a corrigir.
2. **Propriedade legítima do enriquecimento:** as derivadas são agregados lineares das originais (R² > 0,95 em 100 % das derivadas de iris/wine/digits e 77 % em breast_cancer); portanto **não acrescentam informação** nova ao oráculo (que só vê as originais), apenas atalhos para combinações — redundância esperada, não erro. Em breast_cancer 53 derivadas para 30 originais ampliam o espaço de candidatos (a derivada é o melhor candidato da raiz em 60 % das sementes e 73 % do top-12 são derivadas).
3. **Poder discriminativo:** a maioria das derivadas tem MI com o oráculo acima do nulo (70–100 %); em breast_cancer a MI mediana das derivadas (0,098) é metade da das originais (0,191) — há derivadas fracas, **não removidas** (regra do pedido).
4. **Escalas/normalização:** as derivadas têm DP da ordem das originais em iris/wine/breast_cancer; em digits as originais têm DP mediano 4,26 contra 0,29 das 3 derivadas (agregados normalizados) — a pesquisa de limiares é invariante a escala, mas os critérios dependentes de escala (largura de banda KDE, âncoras) usam escalas diferentes.
5. **Distribuição das consultas (achado principal):** num sorteio marginal no espaço enriquecido (o do Original e, portanto, de `trepan_original_enr` e da amostra-base de B2-C), as derivadas ficam **incoerentes** com as originais: erro médio de 0,8–1,4 desvios-padrão (0 após projeção OWL). O oráculo rotula pelas colunas originais, logo as derivadas das linhas sintéticas são ruído relativamente ao rótulo. **Isto contamina o «Original equivalente na representação OWL» e a amostra-base de B2-C** (ΔF ≈ −0,009 de `trepan_original_enr` não pode ser atribuído à ontologia: inclui este efeito de coerência).
6. **Alterações no espaço de candidatos:** +76 % de candidatos em iris, ≈ 0 nos restantes (limite de 24 limiares por feature e de `max_features_per_node`).
7. **Contribuição efetiva da semântica por divisão:** na árvore B2-C inerte (= Original na representação OWL) há literais derivados em iris (0,2), wine (0,4) e breast_cancer (3,8) — a representação influencia a árvore sobretudo em breast_cancer.
**[HIP, não confirmada]:** (a) a incoerência das consultas explica parte da perda da representação; (b) projetar também a amostra-base seria uma variante legítima de «mecanismo ontológico» (projeção) mas **deixa de ser o Original** — fica como fator experimental a autorizar (§11, R2).

### 1.4 Consultas ativas — análise específica (§11 do pedido) **[OBS]**
Mesmas 18 unidades; `D_o` = amostra-base (sorteio do Original), `D_s` = amostra semântica (projeção + ativa), `D_alt` = **segunda** amostra-base com outra semente (referência nula da variabilidade amostral). «Escolha em X» = teste m-of-n da pesquisa do Original (pesos 1) em X; avaliado como fidelidade do substituto de profundidade 1 na amostra indicada.

| métrica | breast_cancer | digits | iris | wine |
|---|---|---|---|---|
| KS médio base vs semântica (por feature) | 0.137 | 0.044 | 0.281 | 0.138 |
| fração junto à fronteira — base | 0.670 | 0.296 | 0.650 | 0.422 |
| fração junto à fronteira — semântica | 0.654 | 0.382 | 0.823 | 0.569 |
| entropia dos rótulos — base | 0.734 | 1.880 | 0.737 | 0.792 |
| entropia dos rótulos — semântica | 0.717 | 1.894 | 0.767 | 0.826 |
| mesma escolha D_o vs D_alt (nula amostral) | 0.000 | 0.024 | 0.000 | 0.000 |
| fidelidade: escolha em D_o, avaliada em D_o | 0.752 | 0.565 | 0.835 | 0.796 |
| escolha em D_o, avaliada em D_s | 0.750 | 0.540 | 0.771 | 0.767 |
| escolha em D_s, avaliada em D_o | 0.730 | 0.558 | 0.767 | 0.762 |
| escolha em D_s, avaliada em D_s | 0.760 | 0.545 | 0.845 | 0.790 |
| escolha em D_o, avaliada nas linhas REAIS | 0.720 | 0.587 | 0.696 | 0.736 |
| escolha em D_s, avaliada nas linhas REAIS | 0.659 | 0.573 | 0.718 | 0.698 |
| escolha em D_alt (outra amostra-base), linhas REAIS | 0.723 | 0.581 | 0.735 | 0.730 |

Resultados **[OBS]**:
1. **A distribuição consultada muda de forma mensurável:** KS médio base-vs-semântica 0,04–0,28; a fração de consultas junto à fronteira sobe com a amostragem ativa em iris (0,65→0,82), wine (0,42→0,57) e digits (0,30→0,38) e não em breast_cancer (0,67→0,65). A entropia dos rótulos quase não muda.
2. **Variabilidade amostral nula:** a pesquisa m-of-n escolhe um teste **diferente** em ≈100 % dos nós entre duas amostras-base independentes (mesma escolha: 0–2 %). Logo «escolha diferente» não é evidência de efeito semântico; só a **fidelidade nas linhas reais** compara de forma justa.
3. **Otimismo na própria amostra:** cada escolha rende mais na amostra onde foi feita do que na outra (p.ex. iris: 0,835 vs 0,771; wine: 0,796 vs 0,767) — o mesmo viés (H1) que o gate herda ao validar na mesma amostra.
4. **Generalização para linhas reais:** a escolha feita em `D_s` fica **abaixo** da feita em `D_o` em 3 de 4 datasets (breast_cancer 0,659 vs 0,720; wine 0,698 vs 0,736; digits 0,573 vs 0,587) e **acima** em iris (0,718 vs 0,696); a referência nula (`D_alt`) fica em 0,723/0,730/0,581/0,735. Em média dos 4 datasets: `D_o` 0,685, `D_s` 0,662, `D_alt` 0,692 ⇒ diferença `D_s`−`D_o` ≈ −0,023, dentro da ordem de grandeza da variabilidade entre sementes e **sem teste de significância** (n=3–5 unidades por dataset; heterogéneo).
5. **Quantidade vs distribuição:** o número de consultas é igual (`n` por nó) nas duas amostras, logo o efeito acima é de **distribuição**, não de quantidade. O custo adicional (≈ ×2,2 consultas ao oráculo no Reloaded) vem da amostra semântica extra e do `predict_proba` do perfil de erro (§8).
6. **Interação com o gate:** o gate mede o ganho em `D_s`, onde a escolha semântica é otimista (ponto 3); o 2.º gate (linhas reais) mitiga parcialmente, mas as linhas reais fazem parte de `D_s`.
**Conclusão limitada:** não se conclui que a amostragem ativa seja prejudicial em geral; observa-se, nestes 18 ajustes DEV, que escolhas feitas na amostra ativa generalizam em média ligeiramente pior para as linhas reais do que as feitas na amostra-base, de forma heterogénea. A amostragem ativa **não** é desligada; o piloto B2-C mede o efeito da camada semântica sem a deixar influenciar o contexto-base.

## 2. Contrato técnico de B2-C **[IMPL, TEST]**

**Flags** (construtor de `TrepanReloadedClassifier`; defaults desligados): `statistical_fallback=False`; `b2c_semantic_node_priority=False`. Só atuam com ontologia efetiva (estado S3); em S0–S2 o fit é o Original (E1–E9) e `b2c_audit_` fica vazio.

**Estrutura por nó (ontologia efetiva e `statistical_fallback=True`)**
1. **Amostra-base `D_o`** = linhas reais do nó + `n` consultas sorteadas **exatamente** como o Original (`FeatureDistributionModel.draw`, sem hooks semânticos), com o estado `rng_original`. É a amostra do laço do Original: define `node.distribution/prediction/fidelity`, critérios de pureza/`min_gain`/`min_samples_leaf`, alcance (`reach`) e filhos.
2. **Amostra semântica `D_s`** = mesmas linhas reais + `n` consultas pelo caminho semântico (projeção OWL, domínio, amostragem ativa) numa **cópia superficial do modelo com RNG própria** (`rng_semantic`). O perfil de erro (EFSR) e o gate usam `D_s`.
3. **Candidato estatístico** `c_stat` = pesquisa m-of-n do Original (pesos 1, IG cru, `_semantic_search_disabled`) em `D_o`.
4. **Candidato semântico** `c_sem` = pesquisa semântica (+ candidatos guiados) em `D_s`.
5. **Gate** (inalterado, B2-A/B não implementados): `_evaluate_error_focused_intervention(D_s, c_stat, c_sem)`; mais uma verificação nova e obrigatória: `c_sem` tem de ser também **válido na amostra-base** (`min_samples_leaf` e `min_gain` do laço do Original em `D_o`); senão `rejection_reason = semantic_invalid_in_base_context_*`.
6. **Decisão:** gate aceite e válido ⇒ `c_sem`; caso contrário ⇒ `c_stat` (**fallback estatístico local**). O laço do Original continua a partir daí sem alterações.
7. **Prioridade de expansão** = a do Original (`b2c_semantic_node_priority=False`), para a ordem de consumo do RNG partilhado coincidir com a do Original enquanto as árvores coincidirem.
8. O espelhamento (`mirror`) não se aplica a B2-C (redundante: com todo o semântico rejeitado a árvore já é a do Original na mesma representação).

**Auditoria por nó** (`b2c_audit_`, também em `semantic_split_audit_`): `node_id, candidate_ids{statistical,semantic}, candidate_type, ontology_state, statistical_gain, semantic_score, local_fidelity_gain, validated_fidelity_gain(None), uncertainty_estimate(None), effective_sample_count, base/semantic_sample_size, oracle_query_count{base,semantic}, accepted, rejection_reason, fallback_used, chosen_candidate_id, tree_complexity_delta, context_class, prefix_identical_so_far, base_stream_preserved, node_priority, reference_representation`.

**Três coisas distintas (nunca confundir)**
| Classe (`context_class`) | Significado | Prova |
|---|---|---|
| fallback estatístico no contexto atual (`fallback_used`) | `c_stat` = pesquisa do Original em `D_o` **neste** nó | por construção |
| `original_reproduction_conditions_met` | **condições necessárias** para que `c_stat` seja a decisão do Original isolado na mesma representação (nenhuma divisão semântica aceite antes; ordem do Original; corrente-base intacta) | o teste diferencial confirma `c_stat == teste do Original` nesses nós (`test_fallback_classes_and_effective_reproduction`) |
| `context_diverged_not_reproducible` | já houve divisão semântica aceite (ou prioridade semântica ligada) ⇒ reprodução **não** presumida | — |
Nunca se afirma equivalência estrutural global por haver fallback num nó: após a primeira divisão semântica aceite a árvore pode divergir do Original, e `D_o` dos nós seguintes deixa de ser a que o Original veria.

**Contabilidade de consultas (`oracle_accounting_`)**: `ledger.tag` marca cada chamada ao oráculo como `original` (rotulagem inicial + amostra-base) ou `semantic` (amostra semântica, `predict_proba` do perfil de erro). `original_oracle_*` = porção Original; `semantic_oracle_*` = tentativa semântica; `total` = o que o oráculo recebeu; `b2c_split_attribution` com o detalhe. As consultas semânticas **não** consomem o orçamento `max_queries` da porção Original (`semantic_membership_queries_` à parte).

## 3. Separação RNG (D2) **[IMPL, TEST]**
- `rng_original` = estado interno (`np.random.Generator`) dos `FeatureDistributionModel` (global: `random_state`; local: `random_state + node_id`), herdado de pai para filho. B2-C **não o toca** nas operações semânticas; o Original não foi alterado (`git diff` de `core/trepan_original.py` vazio; hash idêntico ao pré-B2).
- `rng_semantic` = `np.random.default_rng(SeedSequence([random_state, 0xB2C, node_id]))` aplicada a uma **cópia superficial** do modelo local no nó. Determinística; derivação documentada, estável e independente do dataset (`test_seed_derivation_is_stable_and_documented`).
- Garantias testadas: Reloaded sem ontologia ≡ Original (bit a bit, fluxo ao oráculo incluído); com camada semântica **inerte** (gate que rejeita sempre) a árvore, as previsões, `membership_queries_` e as chamadas/amostras da porção Original ≡ Original isolado, **com amostragens semânticas muito diferentes** (projetor/pesos/combinado/pool) ⇒ a corrente semântica não perturba a original; fits repetidos são idênticos; com `statistical_fallback=False` o comportamento é o anterior (suites EQ + B1 intactas: 232 passed, 40 skipped).

## 4. Testes (B2.1)
| Teste | Garante |
|---|---|
| `test_default_is_off` | defaults desligados |
| `test_flag_has_no_effect_without_effective_ontology` | S0: flag ligada ≡ Original (regras, previsões, fluxo, sequência de chamadas) |
| `test_inert_semantic_layer_reproduces_original_portion` (3 sementes × 3 cenários) | S3 inerte: árvore/previsões/consultas da porção Original ≡ Original; sem predict_proba na porção Original; soma das contas = total; nada oculto |
| `test_semantic_rng_is_separate_from_original_stream` | independência da corrente semântica |
| `test_semantic_stream_is_deterministic_and_depends_on_seed`, `test_seed_derivation_is_stable_and_documented` | determinismo e derivação da semente |
| `test_fallback_classes_and_effective_reproduction` (6 sementes) | as 3 classes; reprodução efetiva do teste do Original nos nós com condições cumpridas; sem afirmação global |
| `test_accepted_semantic_splits_are_valid_in_base_context` | sem crescimento artificial (nós ≤ orçamento; complexidade) |
| `test_semantic_node_priority_is_a_separate_experimental_factor` | prioridade semântica = fator separado |
| `test_b2c_off_is_bitwise_previous_behaviour` | desligado ≡ anterior |

## 5. Referência pré-B2 (congelada) **[IMPL]**
- Commit de referência: `4143db6b082b09f02066c5118a121d972c6c49fd` (último commit documental antes de B2-C; CI **verde** nesse commit: Windows 3.11/3.12 e `agnosticism-guard`). Hashes SHA-256 (prefixo) do código nesse commit: `core/trepan_original.py` `c3a4965724b78bcd` (**idêntico ao atual**; `git diff` vazio), `core/trepan_reloaded_historical.py` `3a8b5bc93556f8b6`, `core/benchmark/runner.py` `6d9ccf64fa15e62f`.
- Resultados brutos de referência (imutáveis, não versionados por tamanho; hash do conjunto dos `UNIT_SHA256.json`): campanha pré-B1 (18 unidades) `5ab121d86d61970a`; campanha B1 (18 unidades) `850d6b6cf454641a`; decomposição DEV (Doc. 18) `e88b0785254ea20a`; evidência resumida em `docs/evidence/b1b4/`.
- **Prova de que B2-C desligado não alterou nada:** no piloto, os braços sem B2-C (A–F, Original-OWL, B1) reproduzem os resultados de referência **bit a bit**: 105/105 comparações braço×unidade contra a decomposição DEV e 60/60 contra a campanha B1 (fidelidade, accuracy, nós, profundidade e nº de consultas). Testes: 232 passed/40 skipped nas suites EQ + B1 após a alteração; B2-C: 19 testes novos.

**Regressão apanhada pelo CI e corrigida (registo honesto).** O primeiro commit de B2-C (`d1043fe`) guardava o invólucro de contabilidade do oráculo (`self._ledger`) no modelo ajustado; o `__getattr__` do invólucro recursava ao fazer *pickle/deepcopy* (6 testes falharam no Windows 3.11/3.12 com `RecursionError`/`Can't pickle local object`; reproduzido localmente). Correção: o invólucro só existe durante o `fit` em B2-C (`None` depois, tal como a amostra semântica temporária) e o `__getattr__` deixou de recursar antes de o estado existir; teste de regressão novo (`test_fitted_model_is_picklable_and_copyable`). A alteração **não altera nenhuma computação**: o piloto foi executado com o código de `d1043fe` e o código corrigido reproduz as 18 linhas de braços da unidade iris/42 do piloto bit a bit (fidelidade, accuracy, nós, profundidade, consultas), pelo que os resultados do piloto se mantêm válidos.

## 6. Desenho do piloto DEV (B2.2) **[IMPL]**
`validation/benchmark/run_pilot_b2c.py` (uma unidade por processo; tuning estrutural ontology-blind recalculado — o sal do *checkpoint* inclui o código do Reloaded, logo qualquer alteração a esse ficheiro invalida o reuso; resultado idêntico). 18 unidades DEV (5 iris, 5 wine, 5 breast_cancer, 3 digits), sementes mestras pré-fixadas, mesmo split/MLP/FrozenOracle/ontologia/tuning em todos os braços. Matriz pré-definida (sem ajuste posterior): Original canónico, C4.5, MLP, Reloaded Core (≡ Original), **Original na representação OWL** (referência equivalente), E histórico, B1 tiebreak, B1 calibrado λ=0,2, λ=0,4 (todos **sem** B2-C) e as quatro configurações **com B2-C** (histórico, tiebreak, λ=0,2, λ=0,4), cada uma também com **ontologia permutada** (controlo negativo). Nenhum dado confirmatório; nenhuma seleção pelo teste; métricas de teste só reportadas.

## 7. Resultados do piloto **[OBS, DEV, exploratório, `observed_only`; evidência em `docs/evidence/b1b4/b2c_pilot/`]**
ΔF/ΔAcc emparelhados por unidade (macro por dataset; EP entre as 18 unidades; teste externo, só relato):

| Braço | ΔF vs Original | EP | piores/melhores (18) | ΔAcc vs Original | ΔF vs Orig.-OWL | ΔNós vs Original | ΔF por dataset (bc / digits / iris / wine) |
|---|---|---|---|---|---|---|---|
| Original na repr. OWL (`trepan_original_enr`) | -0.0089 | 0.0063 | 9/6 | -0.0048 | +0.0000 | -0.9 | -0.004 / -0.007 / -0.016 / -0.009 |
| E — Reloaded OWL histórico | -0.0130 | 0.0074 | 9/6 | -0.0073 | -0.0041 | +0.9 | -0.011 / -0.004 / -0.005 / -0.031 |
| B1 tiebreak | +0.0040 | 0.0066 | 7/8 | +0.0053 | +0.0129 | -0.8 | -0.007 / +0.027 / -0.000 / -0.004 |
| B1 calibrado λ=0,2 | +0.0029 | 0.0054 | 7/7 | +0.0016 | +0.0118 | -1.0 | -0.010 / +0.013 / -0.000 / +0.009 |
| B1 calibrado λ=0,4 | +0.0023 | 0.0066 | 7/8 | +0.0038 | +0.0112 | -1.0 | +0.007 / +0.012 / -0.005 / -0.004 |
| **B2-C** + histórico | -0.0051 | 0.0083 | 7/8 | +0.0048 | +0.0038 | +0.4 | +0.001 / +0.004 / -0.021 / -0.004 |
| **B2-C** + tiebreak | -0.0020 | 0.0069 | 7/7 | +0.0004 | +0.0069 | +2.4 | -0.004 / +0.019 / -0.005 / -0.018 |
| **B2-C** + calibrado λ=0,2 | -0.0056 | 0.0074 | 9/6 | -0.0046 | +0.0033 | +0.6 | -0.008 / -0.001 / -0.000 / -0.013 |
| **B2-C** + calibrado λ=0,4 | -0.0037 | 0.0070 | 8/6 | -0.0020 | +0.0052 | +0.4 | -0.001 / -0.000 / -0.000 / -0.013 |
| F — OWL permutada (histórico) | -0.0006 | 0.0082 | 5/9 | +0.0090 | +0.0083 | +1.9 | -0.010 / +0.013 / -0.011 / +0.004 |
| B2-C + histórico, ontologia PERMUTADA | -0.0048 | 0.0101 | 8/9 | +0.0060 | +0.0041 | -0.5 | -0.010 / +0.018 / -0.032 / +0.004 |
| B2-C + tiebreak, PERMUTADA | -0.0122 | 0.0108 | 11/5 | -0.0063 | -0.0033 | +1.3 | -0.021 / +0.013 / -0.032 / -0.009 |
| B2-C + calibrado 0,2, PERMUTADA | -0.0056 | 0.0101 | 8/10 | -0.0028 | +0.0033 | -2.9 | -0.006 / +0.013 / -0.021 / -0.009 |
| B2-C + calibrado 0,4, PERMUTADA | -0.0116 | 0.0088 | 10/8 | -0.0058 | -0.0027 | -1.6 | -0.008 / +0.013 / -0.016 / -0.036 |

Contrastes B2-C face ao seu par sem B2-C e face ao controlo permutado:

| Configuração B1 | B2-C − sem B2-C (ΔF; EP) | ΔAcc | ΔNós | B2-C − B2-C permutada (ΔF; EP) | ΔAcc |
|---|---|---|---|---|---|
| histórico (E) | +0.0079 (0.0080) | +0.0121 | -0.5 | -0.0003 (0.0067) | -0.0013 |
| tiebreak | -0.0060 (0.0077) | -0.0049 | +3.2 | +0.0102 (0.0086) | +0.0066 |
| calibrado λ=0,2 | -0.0085 (0.0071) | -0.0062 | +1.6 | -0.0001 (0.0084) | -0.0018 |
| calibrado λ=0,4 | -0.0060 (0.0080) | -0.0058 | +1.5 | +0.0079 (0.0080) | +0.0038 |

Auditoria B2-C, hierarquia e custo (18 unidades):

| Configuração | nós | aceites | fallback | taxa aceitação | condições de reprodução cumpridas | contexto divergido | Acc_R>Acc_O | Acc_O>Acc_C4.5 | ordem completa de Acc | F_R>F_O | F_R≥F_O | consultas ×Orig (sem B2-C → B2-C) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 670 | 207 | 463 | 0.31 | 34 | 636 | 9/18 | 5/18 | 1/18 | 8/18 | 11/18 | 2.22 → 3.19 |
| tiebreak | 670 | 231 | 439 | 0.34 | 44 | 626 | 8/18 | 5/18 | 0/18 | 7/18 | 11/18 | 2.25 → 3.18 |
| calibrated_l0p2 | 670 | 188 | 482 | 0.28 | 63 | 607 | 7/18 | 5/18 | 0/18 | 6/18 | 9/18 | 2.20 → 3.24 |
| calibrated_l0p4 | 670 | 195 | 475 | 0.29 | 33 | 637 | 9/18 | 5/18 | 1/18 | 6/18 | 10/18 | 2.21 → 3.23 |

Leitura **[OBS]**:
1. **O fallback funciona tecnicamente:** em 174/174 nós classificados `original_reproduction_conditions_met` (nas 4 configurações), o candidato estatístico coincide com o teste do Original na mesma representação (comparação por ordem de expansão com `trepan_original_enr`). Nenhuma proposta foi rejeitada por invalidade no contexto-base (0 casos `semantic_invalid_in_base_context`): a verificação existe mas não foi acionada.
2. **Mas o contexto raramente se mantém:** só 33–63 dos 670 nós (5–9 %) cumprem as condições de reprodução; a primeira divisão semântica aceite (taxa de aceitação 28–35 %, praticamente igual à do gate histórico, 34 %) faz divergir a árvore. A maior parte dos fallbacks (≈ 65–72 % dos nós) ocorre portanto em contexto divergido — **não há equivalência estrutural global**.
3. **Efeito na fidelidade:** B2-C fica entre −0,0020 e −0,0056 abaixo do Original (EP 0,007–0,008; nenhum distinguível de zero) e entre +0,003 e +0,007 acima do Original na representação OWL. Face ao par **sem** B2-C: +0,008 (histórico; ≈1 EP), −0,006 (tiebreak), −0,0085 (λ=0,2), −0,006 (λ=0,4) — **sem benefício demonstrado**; os pares B1 sem B2-C (≈ +0,002 a +0,004 vs Original) são, nestas 18 unidades, melhores do que as versões com B2-C.
4. **Contribuição da ontologia real:** B2-C real − B2-C permutada: −0,0003 (histórico), +0,010 (tiebreak), −0,0001 (λ=0,2), +0,008 (λ=0,4), EP 0,007–0,009 (accuracy +0,007/+0,004 nos dois positivos). Sinal inconsistente e não significativo: **não se demonstra contribuição da ontologia real**.
5. **Hierarquia (objetivo pré-registado, não imposto):** em nenhuma configuração se cumpre a ordem completa de accuracy (0–1 de 18 unidades). Nota estrutural: `Acc_O > Acc_C4.5` só ocorre em 5/18 unidades **independentemente do Reloaded**, pelo que nestas 18 unidades DEV a hierarquia `Acc_R > Acc_O > Acc_C4.5` não é satisfazível na maioria das unidades. Fidelidade: `F_R ≥ F_O` em 9–11/18 e `F_R > F_O` em 6–8/18. Estado por configuração: `HIERARCHY_NOT_SATISFIED` (observado em DEV), com `INSUFFICIENT_EVIDENCE` para qualquer conclusão estatística (EP ≈ 0,007).
6. **Complexidade:** das 821 aceitações semânticas, ≈ 68 % têm mais literais do que a alternativa estatística (Δliterais +1: 306, +2: 234, +3: 15 nas linhas auditadas); a contagem final de nós difere do Original em +0,4 a +2,4 (tiebreak +2,4, o maior). Sem crescimento acima do orçamento (`max_nodes`).
7. **Divisão ΔF por dataset:** heterogénea (Digits positivo em quase todas as configurações com B1, iris e wine negativos); sem explicação confirmada.

## 8. Custo e auditoria de consultas ao oráculo **[OBS]**
| Item | Valor |
|---|---|
| Tempo de treino da árvore (s, média) — Original / Original-OWL | iris 0,3/0,3 · wine 1,2/1,6 · breast_cancer 2,5/4,0 · digits 16,7/17,1 |
| Reloaded E (sem B2-C) | iris 3,2 · wine 22,7 · breast_cancer 28,2 · digits 157 |
| Reloaded + B2-C (histórico) | iris 3,1 · wine 22,7 · breast_cancer 28,3 · digits 162 (**mesmo custo** que sem B2-C: razão de tempo vs Original ≈ 13 → 13–16×) |
| Consultas ao oráculo vs Original | sem B2-C ×2,2 → **com B2-C ×3,2** (+46 % de amostras totais: ≈ 91,9 k → 133,8 k por unidade, média) |
| Consultas semânticas por unidade (B2-C) | ≈ 42 k (755–769 k no total das 18 unidades por configuração) |
| Tuning estrutural por unidade (recalculado) | iris 41 s · wine 252 s · breast_cancer 333 s · digits 2 728 s |
| Tempo total do piloto | 5,9 h de CPU em 18 unidades × 19 braços (≈ 2,5 h de relógio com 3 processos); unidades Digits ≈ 82 min |
Auditoria: a contabilidade `oracle_accounting_` separa `original` (rotulagem inicial + amostra-base; coincide com as chamadas do Original isolado no cenário inerte — testado) de `semantic` (amostra semântica + `predict_proba` do perfil de erro); `total = original + semantic − reutilizadas` coincide com o que o oráculo recebeu (testado). As consultas semânticas **não** consomem o orçamento `max_queries` da porção Original.

## 9. Entregáveis D3 / D4 / D5 / D6 / D7 (desenho; sem execução de seleção)

### 9.1 D3 — parâmetros de B2-A/B2-B (definições, unidades, grelha compacta, custo, regra pré-registada)
Notação: nó v; dobras de seleção/validação `D^sel`, `D^val` (partição determinística sem RNG principal); `d_i ∈ {−1,0,+1}` = acerto do substituto de profundidade 1 do candidato semântico menos o do estatístico na linha i de `D^val`; clusters = linha real (1) ou lote de consulta sintético.
| Quantidade | Definição | Unidade |
|---|---|---|
| `ganho_observado` | `(1/n_val) Σ d_i` na amostra em que o candidato foi escolhido (o que o gate atual mede) | diferença de fidelidade ∈ [−1,1] |
| `ganho_validado` | o mesmo estimador em `D^val`, independente da escolha | idem |
| `incerteza` | erro-padrão por clusters, `SE = sqrt(Σ_c (Σ_{i∈c} d_i − n_c·δ̂)² )/n_val` (bootstrap de clusters B=500, percentil, sementes derivadas de `node_id`) | idem |
| `amostra_efetiva` | `n_eff = (Σ_c n_c)² / Σ_c n_c²` (Kish por clusters), corrigido por `1/(1+(m̄−1)ρ̂)` | nº de linhas |
| `limite_inferior_de_confiança` | `LB_α = δ̂ − q_α·SE` (ou quantil bootstrap α); unilateral | idem |
| `g_min` | ganho validado mínimo para aceitar | idem |
| `m_ni` | margem de não-inferioridade: aceita-se ainda que `δ̂ ≤ g_min` se `LB_α > −m_ni` **e** coerência semântica > 0 (substitui a tolerância fixa 0,015 do gate histórico) | idem |
| `α` | nível unilateral do limite inferior | probabilidade |
| `n_min` | `n_eff` mínimo para decidir; abaixo ⇒ rejeição conservadora (fallback) | nº de linhas |
Granularidade: `δ̂` é múltiplo de `1/n_val`; com `n_val` ≈ 200–700 (nó de 400–1 300 linhas, dobra 50 %) a resolução é 0,0014–0,005; por isso `g_min ≥ 0,005` equivale a ≥ 1–4 unidades de resolução e valores mais finos seriam ruído. A tolerância histórica 0,015 equivale a 3–10 unidades.
**Grelha compacta** (um-fator-de-cada-vez em torno do centro `g_min=0,005`, `m_ni=0,005`, `α=0,10`, `n_min=60`): `g_min ∈ {0; 0,005; 0,01}`, `m_ni ∈ {0; 0,005; 0,01}`, `α ∈ {0,05; 0,10; 0,20}`, `n_min ∈ {30; 60; 120}` ⇒ **9 configurações** (1 centro + 2×4 variações). Intervalos de confiança: **não** se usam aproximações assintóticas (n_clusters pequeno); método por defeito = bootstrap de clusters; antes de qualquer uso estudar a cobertura por simulação com os `d_i` registados (o piloto atual **não** guardou `d_i` por linha — instrumentação a acrescentar em B2-A).
**Custo:** cada configuração = 1 passagem Reloaded sobre as 18 unidades ≈ 13 min de CPU (E/B2-C sem tuning) ⇒ 9 configs ≈ 2 h CPU (+ overhead de B2-A/B 1–70 ms/nó, Doc. 18 §12.2). A seleção usa **validação interna**: 3 dobras do treino de cada unidade (ajustes completos por dobra, rótulos do FrozenOracle) ⇒ ×3 ⇒ ≈ 6 h CPU (≈ 1,5 h com 4 núcleos) + tuning já disponível se o *checkpoint* for reaproveitado.
**Regra de seleção pré-registada** (antes de executar): escolher a configuração que maximiza a fidelidade OOF interna macro (por dataset) face ao «Original equivalente» (`trepan_original_enr` nas mesmas dobras), com desempate (i) menor nº de nós, (ii) menor taxa de aceitação semântica por tolerância, (iii) menor `α`; **nunca** teste externo, nunca a hierarquia; DEV não é validação confirmatória; a escolha fica registada antes de ver o teste.

### 9.2 D4 — estudo `Original-det vs Reloaded-det`
Ver Doc. 19 §8–§9 (4 ambientes idênticos; Original-det invariante ao ruído de 1–2 ulp; Reloaded-det **não** invariante com a política aplicada só à pesquisa m-of-n; sinal de E−Original muda com o regime; não é política adotada).

### 9.3 D5 — plano de comparação do `cv-lite` (fator experimental; **não implementado**)
Comparar `cv-lite` (K=3, re-pesquisa por dobra apenas em nós ambíguos `LB<0<δ̂+SE`) com a validação por dobra única de B2-A e com um procedimento «rigoroso» de referência (K=5 repetido, aplicado só a um subconjunto de nós para estimar a verdade-terreno do ganho): custo (s/nó e consultas), estimativa de ganho (viés vs referência), variância (entre sementes de partição), estabilidade (concordância do candidato escolhido), taxa de aceitação, taxa de fallback e fidelidade final. Não é inferência confirmatória.

### 9.4 D6 — potência/precisão com estrutura hierárquica (dataset × semente)
Modelo `ΔF_{d,s} = μ + a_d + e_{d,s}`; `Var(média) = σ_d²/m + σ_e²/(m·s)`: **sementes do mesmo dataset reduzem só o termo `σ_e²`**; 72 execuções = 12 datasets × 6 sementes **não** equivalem a 72 datasets. Estimativas DEV: `σ_e ≈ 0,032–0,037` (por unidade, dentro do dataset); `σ_d` estimado em ≈ 0 (ANOVA por momentos, truncado) mas com **4 datasets é praticamente indeterminado** (o desvio-padrão das médias por dataset é ≈ 0,0125, que é um majorante); por isso apresentam-se cenários para `σ_d` de 0 a 0,03.
Diferença mínima detetável (α=0,05 unilateral, potência 80 %, `σ_e = 0,0325`):
| σ_d \ m×s | 12×5 | 12×10 | 12×20 | 12×∞ | 24×10 |
|---|---|---|---|---|---|
| 0 | 0,0104 | 0,0074 | 0,0052 | 0,0000 | 0,0052 |
| 0,01 | 0,0127 | 0,0103 | 0,0089 | 0,0072 | 0,0073 |
| 0,0125 | 0,0138 | 0,0116 | 0,0104 | 0,0090 | 0,0082 |
| 0,02 | 0,0177 | 0,0161 | 0,0153 | 0,0144 | 0,0114 |
| 0,03 | 0,0239 | 0,0228 | 0,0222 | 0,0215 | 0,0161 |
Datasets necessários para detetar 0,01 (80 %): `σ_d=0`: 14/7/4 (s=5/10/20); `σ_d=0,01`: 20/13/10; `σ_d=0,0125`: 23/17/13; `σ_d=0,02`: 38/32/28; `σ_d=0,03`: 69/63/59.
**Cenários de orçamento (confirmatório, ainda BLOQUEADO):** *mínimo* 12 datasets × 5 sementes = 60 execuções (deteta ≈ 0,010–0,014 se `σ_d` ≤ 0,0125; não deteta 0,01 se `σ_d` ≥ 0,02); *recomendado* 12 × 10 = 120 (≈ 0,007–0,012 para `σ_d` ≤ 0,0125; ≈ 0,016 para 0,02); *alargado* 24 × 10 = 240 (≈ 0,005–0,008 para `σ_d` ≤ 0,0125; ≈ 0,011 para 0,02). Custo estimado por execução (mistura DEV: tuning médio ≈ 555 s + ≈ 6 árvores ≈ 250 s ≈ 0,22 h): ≈ 13 h / 27 h / 53 h de CPU (÷4 núcleos), **muito dependente da mistura de datasets** (Digits-like domina) e da correção do reuso do *checkpoint*. O protocolo confirmatório continua **bloqueado** até congelar datasets, ontologias, oráculos, configurações e métricas.

### 9.5 D7 — manifest v3 (rascunho)
`docs/b1b4/manifest_v3/BENCHMARK_MANIFEST_v3.DRAFT.json` (gerado por `validation/benchmark/manifest_v3_draft.py`): `frozen = false`; hashes do código, baselines, ambiente numérico, sementes, oráculos, ontologias (SHA-256 dos `.owl` DEV), particionamento, variantes B1/B2, políticas RNG, métricas, critérios, orçamento e controlos. Não substitui v1 nem v2-DRAFT; não é usado por nenhuma execução; a confirmação com v3 continua proibida até aprovação.

## 10. Critérios de conclusão técnica de B2-C
| Critério | Estado |
|---|---|
| implementação isolada e desligada por omissão | cumprido (flags `False`; só atua em S3) |
| contrato de equivalência conforme | cumprido nos testes (S0 ≡ Original; S3 inerte: porção Original ≡ Original; flag off ≡ anterior bit a bit; 232 testes EQ/B1 + 19 B2-C) e na prova empírica 165/165 |
| fallback com decisões corretamente auditadas | cumprido (`b2c_audit_`; 174/174 nós de reprodução confirmados) |
| corrente RNG controlada | cumprido (testes de independência e determinismo) |
| testes passam | cumprido localmente; ver CI abaixo |
| CI verde | `4143db6` (pré-B2): verde. `d1043fe`/`786b027` (B2-C): `agnosticism-guard` verde, **Windows vermelho** (regressão de pickle descrita em §5, corrigida); **commit corrigido `ef8b010`: Windows 3.11, Windows 3.12 e `agnosticism-guard` VERDES**; suite completa local 1367 passed / 40 skipped |
| resultados DEV documentados | cumprido (§7) |
| custo medido | cumprido (§8) |
| sem alterações indevidas ao Original | cumprido (`core/trepan_original.py` inalterado, hash idêntico) |
**Aprovação científica de B2-C: NÃO demonstrada** (resultado neutro-a-negativo, `INSUFFICIENT_EVIDENCE`): não superou o Original, não melhorou de forma consistente os pares B1, e a ontologia real não se separou da permutada.

## 11. Decisão B2.3 e recomendação para B2-A
**Conclusões:** (1) B2-C é correto e auditável, mas na prática o contexto-base só se mantém até à primeira divisão semântica aceite (5–9 % dos nós); (2) o benefício esperado do fallback é limitado porque o gate aceita 28–35 % dos nós e deixa a árvore divergir; (3) a amostra-base de B2-C herda a incoerência das derivadas (§1.3), pelo que B2-C ≈ «Original na representação OWL» (−0,009) até haver aceitação; (4) a escolha feita na amostra semântica generaliza pior para as linhas reais em 3 de 4 datasets (§1.4), o que sustenta o problema que B2-A/B2-B pretendem resolver (validar fora da amostra de seleção).
**Recomendação (para decisão do utilizador; nada foi implementado):**
- **R1 (recomendada): autorizar B2-A** (ganho validado em dobra independente, por cima do fluxo atual **e** com B2-C), mas só depois de acrescentar a instrumentação por linha (`d_i`, clusters) — sem ela não se estudam coberturas — e com fatorial pequeno `{B2-0, B2-C, B2-A, B2-A+C}` × `{histórico, tiebreak, λ=0,2}` (custo ≈ 12 configs ≈ 2,5 h CPU DEV). Justificação: o defeito do gate (seleção e validação na mesma amostra, tolerância sem custo, 80 % das aceitações sem ganho local) está medido e é o alvo direto de B2-A; B2-A não depende de B2-C ter benefício.
- **R2 (alternativa): variante B2-C′** com amostra-base **projetada** (derivadas coerentes). Resolve a incoerência da referência, mas a amostra-base deixa de ser a do Original canónico (`Original equivalente` passaria a ser um Original com projeção): exige aprovação e um baseline próprio; evidência limitada a favor: o braço `reloaded_owl_features` (projeção, λ=0) tem ΔF −0,003 vs −0,009 do Original-OWL, diferença dentro do ruído.
- **R3 (alternativa): adiar B2-A e tratar primeiro a representação (B3)**: a representação isolada e a sua interação com as consultas explicam a maior parte da perda observada (Doc. 18 §1.2, Doc. 20 §1.3); sem resolvê-la, B2 só pode igualar o «Original equivalente».
- **Não recomendado:** promover B2-C, desligar a amostragem ativa, remover derivadas, ou escolher margens olhando ao teste.

## 12. Estado e limitações
Implementado: B2-C (flags off), contabilidade etiquetada, runner/piloto, scripts de diagnóstico/análise, manifest v3 rascunho, estudo de desempate (inclui D4 estendido). **Não** implementado: B2-A, B2-B, `cv-lite`, B3, B4. Testes executados: EQ (232 passed, 40 skipped), B1 (29), B2-C (19), desempate (8), agnosticismo (verificador estático), benchmark (57). Resultados observados: §1.3–1.4, §7–§8 (DEV, 18 unidades, exploratórios, EP ≈ 0,007). Hipóteses por confirmar: causalidade das perdas (representação × consultas × gate), efeito da incoerência das consultas na perda da representação, generalização da conclusão para outros datasets. Limitações: DEV com 4 datasets (heterogéneo); tuning recalculado (resultados idênticos); `σ_d` indeterminado; sem teste de significância ajustado a múltiplas comparações; o piloto não guardou `d_i` por linha.
