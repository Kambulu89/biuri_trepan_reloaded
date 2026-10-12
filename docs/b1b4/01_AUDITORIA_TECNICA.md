# Documento 1 — Auditoria técnica (Fase 0, B1–B4)

Estado: **auditoria sem alteração de código de produção** (base: `main` @ `1789c17`). Referências `ficheiro:função` verificadas no código real; nada
foi presumido a partir de nomes de classes ou opções. Evidência empírica = leitura de código + reanálise (apenas leitura) de
`docs/evidence/ontology_isolation/split_audit_allnodes_v2.json` (30 nós, 2 ontologias × 5 seeds, **exploratório**, um só dataset).

## 1. Fluxo real ponta a ponta (benchmark científico = mesmo serviço que a GUI)

| Etapa | Onde (real) | Observação |
|---|---|---|
| Splits | `core/benchmark/splits.py:make_splits` | `holdout` (1 split por seed) ou `repeated_cv`; estratificado; `assert_disjoint`. |
| Semântica (fit só no treino) | `core/benchmark/runner.py:_run_split` → `provider.build(Xtr, …)` (`core/benchmark/semantic.py:OwlSemanticProvider.build`) | `fit` em treino, `transform` em teste; mundo OWL isolado (PR #9). |
| Oráculo | `fit_mlp` + `core/scientific_experiment_contract.py:freeze_oracle` | MLP congelado (hash dos pesos = `oracle_id`); todas as árvores do split consultam **o mesmo** objeto; `_verify_oracle_contract`. |
| Tuning estrutural | `runner._structural_protocol` → `core/trepan_scientific_tuning.py:tune_scientific_trepan` | **ontology-blind**, só TREPAN Original + FrozenOracle, só treino; mesma estrutura em C, D, E, F. |
| Braços | `runner.default_arms` / `_run_arm` | A MLP · B C4.5 · C TREPAN Original · D Reloaded Core · E Reloaded+OWL · F Reloaded+OWL permutada (controlo). |
| Métricas | `core/benchmark/metrics.py:evaluate_model` | accuracy vs **rótulos reais** e `fidelity_to_oracle` separadas por construção; floats sem arredondamento. |
| Protocolo anti-fuga | `core/evaluation_protocol.py:EvaluationProtocolGuard` | teste só em `record_final_evaluation`, uma vez por unidade. |

**Conclusão:** a infraestrutura de pareamento, oráculo partilhado, tuning cego à ontologia e guarda de teste **existe e está ativa**. B1–B4 têm de
entrar *dentro* deste fluxo, sem o contornar.

## 2. Implementações (qual código é qual)

* **TREPAN Original:** `core/trepan_original.py:TrepanOriginalClassifier` — score de split = Information Gain (`_split_selection_score`, l. 471); consulta o
  oráculo; m-of-n por beam (`_best_mofn`); **nunca usa rótulos reais** (`oracle_info_["uses_real_labels"]=False`).
* **TREPAN Reloaded:** `core/trepan_reloaded_historical.py:TrepanReloadedClassifier` — **subclasse** do Original (hooks: score, prioridade, candidatos
  m-of-n guiados, projeção de queries, gate EFSR). `core/trepan_reloaded_extractor.py` (5109 l.) é o orquestrador/GUI, não o motor.
* **C4.5:** `core/c45_j48_tree.py:C45Classifier` — implementação nativa (gain ratio, multirramo nominal, poda pessimista CF=0,25, `min_samples_leaf=2`), treinada
  só em rótulos reais, sem teto de nós; `validation/benchmark/c45_audit.py` e o manifest v1 registam as verificações de canonicidade.
* **Selecção multiobjetivo existente:** `core/multiobjective_tree_selector.py` (fronteira de Pareto pós-treino) — **não é importado por código de produção**
  (apenas testes): não está no fluxo real.

## 3. Diagnóstico B1–B4 — validação e **correções ao diagnóstico anterior**

### 3.1 O score do Reloaded (B1)
`TrepanReloadedClassifier._split_selection_score` (l. 840): quando o ganho ontológico está ativo,
`score = IG_normalizado · peso^k + α·OntoDepth + β·ErrorCoverage` (α=0,35, β=0,20 no benchmark), com `IG_normalizado = IG/H(nó) ∈ [0,1]`
(`_normalized_information_gain`). `OntoDepth` e `ErrorCoverage` são **médias por literal de vetores por feature**, não dependem do nó em si.

**ERRATA ao diagnóstico anterior (D1):** a afirmação «bónus semântico 0,90 vs ganho de informação 0,21 (razão 3,9; 5–9× fora da raiz)» usava o campo
`semantic_bonus` do audit, definido em `trepan_reloaded_historical.py` (`proposed_semantic_score − proposed_raw`) como **score Reloaded (escala normalizada, já
com o IG normalizado) menos IG cru em bits**. Mistura escalas e **não** é a contribuição semântica. Reanálise com `alpha`, `beta`, `onto_depth`, `error_coverage`
guardados (v2):

| Quantidade (30 nós) | Mediana | Intervalo |
|---|---|---|
| Termo aditivo semântico `α·OntoDepth+β·ErrorCoverage` | **0,51** | 0,42 – 0,55 |
| Componente estatística `IG_norm·peso^k` | **0,63** | 0,05 – 1,07 |
| Razão aditivo/estatístico | **0,83** | 0,51 – 9,09 |
| Nós em que o termo aditivo > componente estatística | **12/30 (40 %)** | — |
| `OntoDepth` | 1,0 em todos | (saturado) |
| `ErrorCoverage` | 0,81 | 0,35 – 0,99 |

Leitura correta: o termo semântico é **da mesma ordem de grandeza** que o estatístico (não 5–9× como se afirmou), mas **excede-o em 40 % dos nós** — sobretudo
onde o ganho estatístico é pequeno (nós fundos). Como `OntoDepth=1` é constante nestes nós, o efeito sobre a **escolha** vem das diferenças *entre candidatos*
(`ErrorCoverage`, `peso`, coerência de grupo), que o audit atual **não regista** (só guarda o vencedor). **Lacuna de observabilidade:** falta log por candidato.

Outros defeitos de desenho confirmados:
* **Escalas diferentes entre buscas:** a busca *data-only* usa IG cru (bits; mediana 0,20), a semântica usa score normalizado+aditivos; os dois scores não são
  comparáveis entre si (o gate usa fidelidade local, não scores — correto, mas o campo `semantic_bonus` induz erro).
* **Prior constante por feature derivada:** `core/benchmark/semantic.py` (l. 203–207) atribui `depth=1,0` e `peso=onto_weight (1,35)` a **todas** as features
  derivadas. No braço E (`tree_space="enr"`) existe portanto um *prior* «preferir features de engenharia» independente do conhecimento OWL.
* **Controlo F incompleto:** `ShuffledSemanticProvider` permuta grupos/relatedness/depths mas **mantém as mesmas features derivadas**; não controla (i) o conteúdo
  das features nem (ii) a hipótese «engenharia convencional por nomes». Não existe braço de engenharia convencional (`validation/ontology_feature_audit.py`
  mostrou 60/60 features replicáveis por nomes em `breast_cancer`).
* **Reloaded Core (D) ≡ Original por construção:** com estrutura semântica inativa, `_split_selection_score` devolve o IG cru e `_priority` devolve a base
  (`_semantic_structure_active()` falso). Em dados exploratórios (bc, 10 unidades) D e C têm accuracy idêntica (0,9427). D não é um algoritmo distinto.

### 3.2 O gate EFSR e os ganhos negativos (B2)
`_evaluate_error_focused_intervention` (l. 574):
* Ganho local = diferença de fidelidade de **um split** na **amostra de decisão** do nó (mistura de linhas reais com rótulos do oráculo e queries sintéticas do
  oráculo; mediana 400, 200–426 — `decision_sample_size`). **Mesma amostra** serve para escolher e avaliar o split (otimismo in-sample); sem validação cruzada local.
* Limiar `error_focus_min_local_fidelity_gain=0,002` < granularidade `1/n` da amostra de decisão (1/200–1/426 = 0,0023–0,005): na prática **qualquer
  melhoria de ≥1 amostra passa**. **Correção:** o diagnóstico anterior comparava com `1/143` (tamanho do teste), unidade errada; a unidade relevante é a amostra de decisão.
* **Aceitação com ganho negativo é desenho, não erro:** `error_focus_fidelity_tolerance=0,015` aceita perdas até 1,5 pp se `semantic_coherence_gain>0`
  (razão `semantic_coherence_gain_within_fidelity_tolerance`). Reanálise v2: 29/30 nós propõem uma divisão diferente do data-only; o gate aceita **10/30**
  (5 por «melhora local e real», 5 por «coerência dentro da tolerância»); **3/10** aceites têm ganho local negativo (−0,7 pp, −0,9 pp, −0,7 pp).
* Segundo gate (`real_training_gate_evaluated`): linhas reais do nó (`node.real_X/real_y` em `trepan_original.py:606`; `real_y` = **rótulo do oráculo**, não o
  real) — não mede accuracy real.
* Efeito agregado (v2, 10 execuções): fidelidade ao oráculo Reloaded+OWL vs Core: +0,14 pp / −0,42 pp por ontologia (DP ≈ 2,5–2,9 pp) — **indistinguível de 0**.

### 3.3 m-of-n semântico (B4)
`_ontology_guided_candidates` (l. 890): agrupa por componentes ligadas de `semantic_relatedness_matrix_ > 0,35`, orçamento `semantic_candidate_budget=24`,
valida com o mesmo score. É genérico (usa só grupos/relações do grafo). **Controlos existentes:** permutação (F). **Ausentes:** grupos aleatórios de igual
tamanho; m-of-n convencional como baseline *igualado em orçamento de candidatos* (o Original usa beam de largura 2, o Reloaded acrescenta até 24 candidatos →
mais oportunidades de seleção; o ganho pode ser «capacidade de procura», não ontologia).

### 3.4 Selecção multiobjetivo e hierarquia (B3)
* O pipeline **não tem** noção de hierarquia de desempenho: `grep -ri hierarchy` não encontra nada de relevante em `core/`, `validation/`, `gui/`.
* Qualquer seleção por **accuracy real** enfrenta um enviesamento estrutural: o MLP é treinado em todas as linhas de treino; a accuracy (real) do
  surrogate **nas mesmas linhas de treino** herda o sobreajuste do MLP. A validação honesta exige **MLP reajustado dentro de cada dobra interna** (oráculo
  por dobra). O tuning atual (`tune_scientific_trepan`) escapa ao problema porque só mede **fidelidade** ao oráculo congelado (rótulos do oráculo), nunca accuracy.
* O Reloaded/Original **nunca vêem rótulos reais**; o C4.5 só vê rótulos reais. `A_R > A_O` e `A_O > A_C` são portanto comparações entre famílias de supervisão
  diferentes; `A_O > A_C` só é plausível se o MLP for mais exato que uma árvore **e** a imitação for boa (fora do controlo do Reloaded).

## 4. Fuga de dados — resultado da auditoria

| Ponto | Resultado |
|---|---|
| Ontologia ajustada só no treino | ✔ `provider.build(Xtr)`; `enrich(Xte)` é `transform` |
| Seleção de oráculo (gate MLP ontológico) | ✔ CV interna no treino (`select_oracle`) |
| Tuning estrutural | ✔ `test_used=False`, `semantic_inputs_given_to_tuning=False` |
| Teste usado uma vez | ✔ `EvaluationProtocolGuard.record_final_evaluation` |
| Rótulos reais no Reloaded | ✔ não usados (`uses_real_labels_for_training=False`) |
| **Exposição ao nível do dataset** | ✘ iris/wine/breast_cancer/digits (e ontologias de `breast_cancer`, sonar, hepatitis, balance-scale) já foram observados em resultados durante o desenvolvimento → **não podem ser confirmatórios** para B1–B4 |
| **Ontologias não independentes** | ⚠ catálogo `data/ontology_catalog_v8.json`: `independent: false` (TBox escritas pelo projeto); `adult_census_full_ontology.owl` = `CONTAMINATED_ONTOLOGY` |
| **Potência** | ✘ manifest v1: 4 datasets × 5 seeds (holdout) = 20 unidades; 6 datasets contando sintéticos (não elegíveis a ranking) |
| **Selecção da hierarquia sobre accuracy de treino** | ✘ ver 3.4 (precisa de MLP aninhado) |

## 5. Baselines — comparabilidade

* C4.5 e TREPAN Original estão congelados por hash em `validation/benchmark/BENCHMARK_MANIFEST_v1.json` (`frozen_code_sha256`). B1–B4 **não podem** tocar
  `core/trepan_original.py`, `core/c45_j48_tree.py`, `core/trepan_scientific_tuning.py`, `core/scientific_experiment_contract.py`.
* Orçamento igual para todos os TREPAN (`TreeBudget`, `common_capacity`); o C4.5 não tem teto (canónico). Em nós: o C4.5 pode ser maior que os TREPAN com
  `max_nodes` fixo — a comparação de complexidade deve ser reportada, não escondida.
* O **espelhamento** (`mirror_when_no_semantic_effect`) está desligado nos braços científicos (`runner._run_arm`): correto; qualquer *fallback ao Original* em B3
  teria de ser declarado como tal (ver Documento 5).

## 6. Matching ontológico e cenário de nomes opacos
`core/ontology_quality.py:match_features` casa **por rótulos** (score ≥ 0,72, margem ≥ 0,08, cobertura ≥ 0,35…). Não existe mecanismo de mapeamento por índice/alias
(`grep alias|columnIndex` sem resultados em `ontology_*`). Com colunas `f1…fn` o gate **rejeita** a ontologia e o pipeline cai em fallback (comportamento correto).
Testar «conhecimento OWL com nomes opacos» exige um **mapeamento explícito** novo (ver Documento 3, decisão D-6).

## 7. Cobertura de testes atual relevante
`tests/test_reloaded_*`, `test_error_focused_semantic_refinement_v92.py`, `test_semantic_*`, `test_benchmark_*`, `test_no_ontology_mirror_original.py`,
`test_agnosticism_contract.py` (+ CI `agnosticism-guard`). **Não existem** testes de: estados de hierarquia, limiar adaptativo, calibração do bónus, seleção
multiobjetivo, controlo de grupos aleatórios, comparação sem arredondamento da hierarquia.

## 8. Resumo de problemas (prioridade)
1. **P0** — Diagnóstico D1 anterior incorreto (escalas misturadas) → corrigido aqui; propor correção do campo `semantic_bonus` (observabilidade, Fase 1).
2. **P0** — Dev datasets/ontologias já expostos; nenhuma confirmação válida sem datasets frescos e ontologias congeladas às cegas.
3. **P0** — Validação de accuracy exige MLP por dobra interna; sem isso a seleção pela hierarquia é enviesada.
4. **P1** — Sem log por candidato (impossível medir influência semântica real sobre o *ranking*).
5. **P1** — Controlos em falta: engenharia convencional por nomes, grupos aleatórios, igualação de orçamento de candidatos.
6. **P1** — Gate EFSR in-sample e com tolerância negativa por desenho.
7. **P2** — Prior constante `onto_weight`/`depth=1` em features derivadas.
8. **P2** — Reloaded Core ≡ Original (arm D redundante como «algoritmo»).
9. **P2** — Potência estatística insuficiente (20 unidades, 4 datasets).
