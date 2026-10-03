# IMPLEMENTATION REPORT — Métricas, benchmark pareado, controlos negativos e ablação

Escopo: protocolo experimental. Nenhum algoritmo de construção das árvores foi alterado; o benchmark **consome** `TrepanOriginalClassifier`,
`TrepanReloadedClassifier`, `C45Classifier` e a fábrica de MLP existentes. Todo o código novo está em `core/benchmark/` (sem nomes de datasets,
verificado por teste) + `scripts/run_benchmark.py`, `scripts/audit_metrics.py`.

## 1. Auditoria das métricas (Parte 1) — `METRICS_AUDIT.md` (reprodutível: `python scripts/audit_metrics.py`)

* 216 chamadas de métricas sklearn em 28 ficheiros (`metrics_comparator.py` 39, `mlp_optimizer.py` 20, `trepan_reloaded_extractor.py` 15, GUI 15, …).
* 31 linhas de **fidelity**: todas comparam o surrogate com predições de oráculo/professor (29 automáticas + 2 verificadas à mão em
  `active_query_engine.py`, onde `val_labels = _oracle_probabilities(oracle, ref_val)`). **Nenhuma calcula fidelity contra `y_real`.**
* Inconsistências encontradas (não alteradas para não mudar resultados históricos; ver §9): variantes `recall_fidelity`/`f1_fidelity` *weighted* em
  `metrics_comparator.py`; "fidelity" do C4.5 face ao MLP (`c45_j48_tree.compare_with_trepan`) é concordância informativa, não métrica do C4.5; vários
  módulos reimplementam `accuracy_score` sem identificar o oráculo.
* **Módulo central novo `core/benchmark/metrics.py`**: `accuracy_real_labels`, `fidelity_to_oracle(oracle_pred, surrogate_pred, OracleInfo)`
  (exige `OracleInfo`; `TypeError` caso contrário), `format_fidelity` (nunca mostra "Fidelity = 0.94" sem oráculo), `classification_bundle`
  (accuracy, balanced accuracy, precision/recall/F1 macro, F1 weighted, `minority_recall`), `minority_class` (determinada **no treino**; empate → ordem das classes).
  Colunas nomeadas `*_real_labels` e `fidelity_to_oracle` (+ `fidelity_oracle_name/type`), logo accuracy e fidelity nunca se confundem.

## 2. Arquitectura (`core/benchmark/`)

| módulo | papel |
|---|---|
| `metrics.py` | métricas canónicas, `OracleInfo` (name, type, feature_space, version, accuracy) |
| `stats.py` | describe, IC bootstrap (média e pares), Wilcoxon, t pareado (com Shapiro), Cohen's dz, Cliff's δ, rank-biserial, Holm, BH, Friedman+post-hoc, `EvidencePolicy` |
| `splits.py` | `make_splits` (holdout multi-seed ou `RepeatedStratifiedKFold`), `SplitSpec` com `split_hash`, `inner_folds` (CV interna só sobre o treino) |
| `semantic.py` | `NoSemanticProvider`, `GroupSemanticProvider` (ontologia programática), `OwlSemanticProvider` (OWL real + reasoner + quality gate), `ShuffledSemanticProvider` (controlo negativo) |
| `runner.py` | `BenchmarkConfig` (frozen), `Dataset` (CSV/ARFF com **ordem de classes preservada**), `BenchmarkRunner`, `default_arms`, `fit_mlp`, `select_oracle` |
| `analysis.py` | agregação, contrastes pareados a priori, evidência, controlo negativo, atribuição, entre datasets |
| `io.py` / `manifest.py` | persistência sem sobrescrever, manifest, `verify_metrics_from_predictions` |
| `report.py` | `SCIENTIFIC_VALIDATION_REPORT`, `NEGATIVE_CONTROL_REPORT`, `ABLATION_REPORT`, summary |

## 3. Definições formais (Partes 2–4)

* `accuracy_real_labels = accuracy_score(y_real, pred)` — MLP Original, MLP Ontológico, C4.5, TREPAN Original, TREPAN Reloaded.
* `fidelity_to_oracle = accuracy_score(oracle.predict(X_eval), tree.predict(X_eval))`, avaliado no teste, cada modelo no **seu** espaço de features
  (o MLP Ontológico vê as features enriquecidas; o MLP Original é consultado através de `OriginalOracleProjection` quando a árvore usa o espaço enriquecido).
* Cada linha de resultado guarda `oracle_name, oracle_type, oracle_feature_space, oracle_version, oracle_accuracy_real_labels`.
  C4.5 e MLPs têm `fidelity=None` e `oracle=None`; C4.5 só usa rótulos reais (`uses_real_labels_for_training=True`, `uses_oracle=False`);
  as árvores TREPAN têm `uses_real_labels_for_training=False`.

## 4. Os dois estudos (Partes 5–6) — nunca misturados

* **A — efeito do algoritmo, MESMO oráculo**: `trepan_original` e todos os braços `reloaded_*` (excepto `reloaded_e2e`) → **MLP Original**; mesmo split, seed,
  `min_sample`, `max_queries`, `max_depth`, `max_nodes`, amostragem e conjunto de avaliação (testado). `reloaded_lambda0` é uma verificação de sanidade: tem de
  coincidir com o Original (testado: predições idênticas).
* **B — pipeline completo**: `reloaded_e2e` → oráculo seleccionado pelo gate (MLP Original ou Ontológico). Como o oráculo pode diferir, a **fidelity não é comparável**
  com a do Original: o runner marca esses contrastes `not_comparable` e compara apenas a accuracy vs rótulos reais.
* **Gate do professor**: o MLP Ontológico só é oráculo se o ganho de balanced accuracy na **CV interna do treino** exceder `oracle_gate_margin` (0,01, fixado a priori).
  `select_oracle` não recebe dados de teste (assinatura testada).

## 5. Controlo negativo e ablação (Partes 7–9)

Braços (todos por split, mesma infraestrutura): `mlp_original`, `mlp_ontological`, `c45`, `trepan_original`, `trepan_original_no_mofn`, `reloaded_lambda0`
(**λ=0 ⇒ nenhum canal semântico**: alpha, beta, expoente do peso, força de grupo e estrutura passada = 0/nenhuma), `reloaded_semantic_score` (só score, features originais),
`reloaded_owl_features` (features OWL sem score), `reloaded_owl_full`, `reloaded_owl_shuffled` (**controlo negativo**), `reloaded_e2e`, e ablações da OWL completa:
`no_mofn`, `no_active_queries`, `no_error_focus`; sensibilidade a λ via `semantic_lambdas`.

* **OWL permutada** (`ShuffledSemanticProvider`): permuta, com seed, peso/grupo/depth/linhas-colunas da *relatedness* por feature e calcula as features derivadas sobre
  **colunas originais trocadas**. Preserva dimensionalidade, nº de features derivadas e distribuições (testado: multiconjuntos de pesos/grupos iguais, espectro da relatedness igual,
  assinatura estrutural diferente, determinístico por seed).
* **Veredictos** (por métrica, a partir do IC bootstrap pareado, mínimo de 5 pares): `REAL_GT_SHUFFLED`, `REAL_APPROX_SHUFFLED`, `SHUFFLED_GT_REAL`, `INSUFFICIENT_DATA`.
  Contrastes com alegação semântica só podem chegar a `STATISTICALLY_SUPPORTED` se o controlo negativo for ultrapassado.
* Decomposição da fidelity: arquitectura | features OWL | score semântico | OWL completa | total | real − shuffled (`semantic_attribution`).
* **Ablações NÃO aplicáveis** (e porquê, registado em `ABLATION_REPORT.md`/`semantic_report.json`): sem reasoner, sem features relacionais, sem agregados, sem constraints, sem poda semântica — não existem como switches independentes no pipeline actual.

## 6. Multi-seed, CV repetida, splits pareados, leakage (Partes 10–13, 36–38)

* Seeds fixadas **antes** dos resultados (`11,22,…,111`), nunca seleccionadas depois; `RepeatedStratifiedKFold` configurável; estratificação preservada (reduz `n_splits` se uma classe for rara).
* Os splits são gerados uma vez; todos os braços usam o mesmo `SplitSpec` (`split_hash` por split e global no manifest).
* `EvaluationProtocolGuard` por split: ajuste só em TRAIN/VALIDATION interna; teste avaliado **uma vez** no fim; auditoria guardada (`test_used_for_selection=False`).
* Paridade de MLPs: mesma rotina e mesmo `mlp_trials` para Original e Ontológico (testado); selecção por CV interna.
* Orçamento TREPAN igual para todos os braços (regra automática declarada no relatório; `budget_exhausted` registado por árvore).

## 7. Estatística (Partes 14–19, 41)

* Agregação: n, média, desvio (ddof=1), mediana, mín, máx, IC95% bootstrap percentil (`n_boot` configurável). Resultados brutos nunca são apagados.
* Comparação pareada A−B por split: IC bootstrap **dos pares**, Wilcoxon signed-rank (exacto em n pequeno); t pareado só informativo (`ttest_valid` exige Shapiro ok e n≥8);
  effect sizes rank-biserial, Cliff's δ, Cohen's dz; Holm (família primária: fidelity e accuracy por dataset) e BH (suplementar); em `repeated_cv` acrescenta-se o t corrigido de Nadeau–Bengio e
  o nível máximo fica `INDICATIVE` (folds não independentes).
* Entre datasets: Wilcoxon entre datasets (≥5) e Friedman + post-hoc Holm (≥3 datasets completos); caso contrário apenas descritivo, dito explicitamente.
* **Níveis de evidência** (limiares fixados a priori em `EvidencePolicy`): `MECHANISM_ONLY` (<3 pares), `INDICATIVE`, `STATISTICALLY_SUPPORTED` (≥10 pares, teste ≥150, p Holm<0,05, IC sem 0, |rank-biserial|≥0,3, controlo negativo ultrapassado em alegações semânticas); `IDENTICAL_RESULTS` quando todas as diferenças são 0.
* Nenhuma regra impõe Reloaded > Original > C4.5 e não há ajuste de parâmetros em função de resultados.

## 8. Manifest, artefactos, auditoria (Partes 28–33)

`results/<experiment_id>/`: `manifest.json` (experiment_id, timestamp, build_version, git_commit, dataset_hash, ontology_hash, config_hash, seeds, split_hash, library_versions, python_version, platform, splits),
`config.json`, `metrics.json`, `tree_metrics.json`, `semantic_report.json` (por split: provider, `ontology_valid`, `mlp_enrichment_accepted`, `trepan_semantics_available`, gate do oráculo, tempos, auditoria do guard, assinaturas real/shuffled), `predictions.csv`
(`y_real`, `oraclepred__<oráculo>`, `pred__<braço>`, índice da linha de teste — sem valores de features), `raw_results.csv`, `aggregate.csv`, `contrasts.csv`, `summary.md`. Nunca sobrescreve (sufixo `_2`, `_3`…).
`verify_metrics_from_predictions` recomputa accuracy, balanced accuracy, macro-F1 e fidelity a partir das predições e compara com os valores guardados (deteta adulteração — testado).
Custos: `mlp_training_time`, `tree_training_time`, `query_time`, `semantic_processing_time`, `reasoner_time` (`counterfactual_time` registado como `None`: não aplicável neste runner).
Casos: sem OWL ⇒ `semantic_available=false`, braços semânticos listados em `skipped` sem crash; OWL inválida ⇒ `ontology_valid=false` sem descartar a experiência; OWL válida com enriquecimento rejeitado ⇒ `ontology_valid=true, mlp_enrichment_accepted=false`.

## 9. Smoke test multi-seed (execução real, OWL real via HermiT)

`python scripts/run_benchmark.py --smoke --seeds 11 22 33 44 55 66 --min-sample 400 --max-nodes 15 --n-boot 2000 --out results/benchmark_smoke`
— 2 datasets sintéticos (binário e 3 classes), TBox gerada a partir de grupos, 6 seeds, 14 braços por split; métricas recomputadas a partir das predições: **1308 valores, máx. |Δ| = 1,1·10⁻¹⁶**.

| (fidelity Δ, 6 pares) | binário [IC95%] | 3 classes [IC95%] |
|---|---|---|
| Reloaded λ=0 − Original (sanidade) | 0,000 (idêntico) | 0,000 (idêntico) |
| score semântico − λ=0 | +0,015 [−0,028, +0,054] | +0,015 [−0,018, +0,062] |
| features OWL − λ=0 | −0,021 [−0,051, +0,010] | +0,010 [−0,010, +0,028] |
| OWL completa − Original (mesmo oráculo) | +0,036 [−0,000, +0,072] | +0,041 [−0,010, +0,103] |
| **OWL real − OWL permutada** | **−0,023 [−0,056, +0,005]** | **+0,026 [−0,028, +0,080]** |
| sem m-of-n − OWL completa | −0,082 [−0,131, −0,039] (p Wilcoxon 0,031; p Holm 0,59) | −0,103 [−0,133, −0,064] (0,031; 0,59) |

Leitura (honesta): (i) **controlo negativo = `REAL_APPROX_SHUFFLED` em ambos** ⇒ o pequeno ganho da OWL completa **não é atribuível à ontologia** (nestes dados os grupos são arbitrários, não há semântica real a explorar —
resultado nulo esperado e correctamente detectado); (ii) todos os contrastes são `INDICATIVE` (6 pares, teste de 65 amostras); nenhum é `STATISTICALLY_SUPPORTED`; (iii) a remoção de m-of-n reduz a fidelity de forma consistente (−8 a −10 pontos, IC sem 0) mas não sobrevive à correcção de Holm com 6 pares;
(iv) o gate recusou o MLP Ontológico em 3/6 splits em cada dataset (`mlp_enrichment_accepted=false`); (v) o custo do Reloaded é ≈10× o do Original (≈5 s vs ≈0,4–0,7 s por árvore, dominado pela projeção OWL das queries); (vi) o mirror do Reloaded foi aplicado em ≈11 % das árvores.
**Isto valida o mecanismo do benchmark, não a hipótese científica.**

## 10. Testes

`tests/test_benchmark_protocol.py` — **36 testes**: fidelity vs oráculo e accuracy vs y_real; `OracleInfo` obrigatório; naming; multiclasse e ordem de classes; classe minoritária do treino (não assumida); descrição/IC determinísticos;
caso conhecido de comparação pareada e effect sizes; Holm/BH; Friedman; níveis de evidência; splits estratificados/disjuntos/determinísticos e CV repetida; CV interna só com índices de treino; mesmos splits em todos os braços;
identificação do oráculo e uso de rótulos (C4.5 vs TREPAN); mesmo oráculo/orçamento/seed entre Original e Reloaded; `reloaded_lambda0 ≡ Original`; predições guardadas e métricas recomputáveis (e adulteração detectada);
reprodutibilidade por seed; execução sem ontologia; OWL inválida; multiclasse; anti-leakage (assinaturas e *spy* no runner); config congelada com seeds a priori; paridade de trials dos MLPs; controlo negativo (forma preservada, correspondência destruída, determinismo, veredictos, `INSUFFICIENT_DATA` com poucos pares);
fidelity não comparada entre oráculos diferentes; evidência limitada (teste pequeno, CV repetida); agregação; manifest e não-sobrescrita; hashes; relatórios com as secções exigidas; resultados negativos; contagem de splits semânticos só na árvore final; ARFF com ordem de classes; agnosticismo (sem nomes de datasets).
Suite completa do repositório: **453 passed, 3 failed** — os 3 são pré-existentes e alheios a esta tarefa (`test_counterfactual_gui` falha também no código original; soft tree retirada da produção; `test_fidelity_hierarchy_c45_original_reloaded` impõe Reloaded ≥ Original, i.e. codifica a hipótese da tese e deveria ser revisto, não forçado).

## 11. Critérios de aceitação

| critério | estado |
|---|---|
| Accuracy e Fidelity formalmente separadas / fidelity identifica o oráculo | ✔ (módulo central, tipos, colunas) |
| Original vs Reloaded com mesmo oráculo; end-to-end separado | ✔ (estudos A e B; contrastes não comparáveis marcados) |
| Controlo negativo; Reloaded λ=0; ablação | ✔ (ablações sem switch no pipeline: documentadas) |
| Multi-seed, splits pareados, sem tuning no teste | ✔ |
| Média/desvio, IC, testes, effect size, correcção de múltiplas comparações | ✔ |
| Multiclasse, complexidade, custo computacional | ✔ |
| Benchmark agnóstico; manifest; resultados brutos; testes | ✔ |

## 12. Limitações

* O smoke usa dados sintéticos com grupos arbitrários: **não** responde às perguntas 1–6 do enunciado; serve para provar o protocolo. A resposta exige datasets reais com ontologias independentes e ≥10 seeds/CV repetida.
* TBoxes de benchmark são de domínio e podem derivar das descrições das features; não são ontologias externas independentes. Em OWL real a `relatedness` é pouco discriminativa (≈0,71 entre todas as features do mesmo domínio).
* `semantic_split_count` conta splits da árvore final com bónus semântico ≠ 0 (`semantic_decision_changed_count` conta os que mudaram a decisão face ao split puramente estatístico).
* `mirror_when_no_semantic_effect` (produção) pode devolver o Original sobre o espaço enriquecido; fica registado por árvore.
* Fidelity medida no teste; folds repetidos não independentes (Nadeau–Bengio reportado, evidência limitada).
* Não foram refactorizados os 216 cálculos de métricas legados (risco de alterar resultados históricos); o benchmark usa exclusivamente o módulo central e `scripts/audit_metrics.py` evita regressões futuras de fidelity vs `y_real` no código legado (sem veredictos "SUSPEITA" actualmente).
* Datasets não numéricos exigem pré-processamento prévio; `counterfactual_time` não é medido.
