# Relatório de implementação — enriquecimento semântico (V9.3)

Branch `claude/repository-architecture-analysis-s1vl5a` (PR #1). Cada alteração é um commit
pequeno e testado. A auditoria feita **antes** de alterar código está em
`docs/SEMANTIC_PIPELINE_AUDIT.md`; os resultados experimentais em `SEMANTIC_VALIDATION_REPORT.md`
(gerado por `scripts/render_semantic_report.py` a partir de `results/semantic_validation/*.json`).

## 1. Arquitetura anterior

Dois pipelines paralelos. O **legado/GUI** (`core/trepan.py`, `gui/biuri_app_complete.py`) usava
`OntologyProcessor`, `SemanticUtilityGate` e um "MLP ontológico". O **headless V9.2**
(`core/production_training.py`) só usava a ontologia para pesos/grupos/relatedness do TREPAN:
nenhuma feature `onto_*` chegava ao MLP e não havia etapa de enriquecimento. Já existiam (e foram
**preservados**): quality gate, `fit/transform` train-only, refit por fold, estabilidade por MI,
comparação OOF, `ontology_stage_status`, relatedness e auditoria por nó no TREPAN.

## 2. Problemas encontrados

Lista completa L1–L19 em `docs/SEMANTIC_PIPELINE_AUDIT.md`. Os mais relevantes:

* ontologias de benchmark são esqueletos (taxonomia + domain/range) e o gate dava `VALID` na mesma;
* matching sem segundo candidato nomeado, sem tratamento de colisões entidade↔várias features;
* reasoner só verificava consistência; ABox sem estado nomeado;
* limiares aprendidos nos dados não distinguidos de conhecimento OWL; limites OWL só lidos como anotação;
* sem comparação MLP base vs MLP+OWL com o mesmo orçamento, sem IC, sem estados de decisão granulares;
* cache por `path+size+mtime`, sem versão/config semântica;
* **defeito pré-existente com impacto no TREPAN:** o grafo usava o tipo de dados do `rdfs:range`
  (`<class 'float'>`) como "grupo", pelo que todas as features `float` partilhavam grupo e recebiam
  bónus de coesão sem relação semântica;
* **conhecimento não ontológico:** `_group_by_semantics` adivinhava domínios (medical/financial…) por
  palavras no nome da feature;
* dois `NameError` no export da GUI e em `pipeline_train.py` (corrigidos antes, commit `9fc32f8`).

## 3. Alterações (commits, por ordem)

| Commit | Alteração | Partes |
|---|---|---|
| `9fc32f8` | `NameError` em `export_results` e `BASE_SEED` + testes | — |
| `4e9db3a` | `semantic_richness` e aviso no quality gate (não altera `accepted`) | 3, 24 |
| `3d2f0bc` | `linear_redundancy_r2` por feature; descarte opt-in | 13 |
| `b49e8ff` | `.gitignore` mínimo | — |
| `6caf70c` | auditoria prévia (`docs/SEMANTIC_PIPELINE_AUDIT.md`) | 1 |
| `92a0328` | reasoner: duração, axiomas inferidos, fallback explícito | 8 |
| `c32bef4` | matching auditável (2.º candidato, `status`, colisões, desempate), ABox `SAFE/REJECT_*`, TBox/ABox, métricas de ontologia genérica | 3, 4, 7, 24 |
| `cb23189` | processor: `knowledge_source`, limites OWL por facetas `ConstrainedDatatype`, `family_contrast`/`normalized_error`, filtros low-variance/near-duplicate, `generation_summary`, `reasoner_inferred` | 9–13 |
| `d76f87d` | `core/semantic_enrichment.py` (+ `semantic_controls.py`) | 6, 14–19, 26, 29, 30 |
| `56e8169` | grafo: `range` e entidades W3C de topo deixam de ser "grupos" (bug) | 20–22 |
| `8fa9899` | `core/semantic_metadata.py`, relatedness configurável, sem dupla contagem, auditoria por split | 20–22 |
| `8c9f6e2` | cache por conteúdo + versão/config/matching | 27 |
| `f6f80f2` | estados independentes + painel de diagnóstico (presenter) | 2, 19, 25 |
| `5f4fbd9` | remove adivinhação de grupos por nome | 23 |
| `54b4c22` | integração no pipeline de produção + relatedness por família | 20, 21 |
| `63f3fcd` | testes de matching categórico + script de validação | 5, 28–30 |

## 4. Decisões de design

* **Validade ≠ utilidade.** `build_ontology_stage_status` devolve estados independentes
  (estrutural, ABox, mapeamento, novidade, MLP, TREPAN). `semantic_mlp_accepted` e
  `semantic_trepan_available` nunca se implicam: rejeitar o MLP não desliga a semântica do TREPAN;
  ontologia inválida ou leakage de ABox desligam ambos.
* **Não alterar limiares.** O quality gate manteve os seus limiares; só se acrescentaram métricas,
  avisos e termos genéricos pedidos (`measurement`). A utilidade e o IC são definidos antes de correr.
* **Opt-in onde há risco de alterar o comportamento científico:** descarte de features linearmente
  redundantes (`drop_linear_redundant`), avaliação de enriquecimento no pipeline de produção
  (`semantic_enrichment=None`), relatedness por família só aumenta e é neutra sem famílias.
* **`evidence_strength`** (forte/fraca) rotula, sem mudar decisões, a aceitação por não-inferioridade
  com IC que inclui zero.
* **Honestidade nos dados sintéticos:** o teste de aceitação usa um problema em que o sinal relacional
  está escondido entre atributos irrelevantes; num problema fácil o pipeline *rejeita* (também testado).

## 5. Proteção contra leakage

* O `OntologyProcessor` é **refeito** em cada fold só com o treino (`fit`); validação só faz `transform`.
* A seleção de features de cada fold usa só o treino desse fold (MI, ranking, ganho add-one).
* MLP base e MLP+OWL são otimizados em separado com a mesma lista de candidatos, folds internos e
  semente (orçamento igual). O teste externo nunca entra (`test_used=False`, verificado em teste).
* O matching usa só **nomes** de features/entidades, nunca valores.
* ABox: registos do dataset e identificadores do teste → `REJECT_*`, sem apagar vocabulário controlado.
* Testes: alterar `X_test` não muda nada aprendido em `fit(X_train)`; reprodutibilidade por semente.

## 6. Matching, TBox/ABox, reasoner

* **Matching:** lexical (CamelCase/snake_case/espaços/hífens), `rdfs:label`/aliases, tokens, similaridade,
  desempate por tipo (propriedade de dados > classe, registado), margem mínima, **colisões** (só o melhor
  score mantém a entidade; empate rejeita todos). Cada registo: feature, entidade, score, 2.º candidato e
  score, margem, `status`. Categorias: valores nominais → classes/indivíduos (`private` →
  `PrivateSectorWorker`), grupos pela classe-pai; ambíguo/sem match não é adivinhado.
* **TBox/ABox:** `metrics.knowledge_split` e `abox.status` (`SAFE` / `REJECT_DATASET_RECORDS_IN_ABOX` /
  `REJECT_TEST_INSTANCE_LEAKAGE`).
* **Reasoner:** `reasoner_used`, `reasoning_mode` (inferred / explicit_axioms_only), `duration_seconds`,
  `inferred_axioms_count`, relações/equivalências/tipos inferidos, `fallback` quando falha. Funciona
  neste ambiente (HermiT); as ontologias de benchmark inferem 0 axiomas (são planas), e isso é reportado.

## 7. Feature engineering, gate, MLP e TREPAN

* **Features:** agregados padronizados (z-score do treino), relacionais por **família+papel** declarados na
  OWL (diferença, delta relativo, rácio de erro, contraste, erro normalizado), restrições só com limites
  OWL (anotações ou facetas); limiares do treino ficam como `statistical`, nunca como conhecimento OWL.
  Filtros: constante, baixa variância, duplicada, quase-duplicada; auditoria de redundância linear.
* **Gate (`semantic_enrichment.py`):** A qualidade → B novidade → C estabilidade (MI média/desvio, ganho
  add-one médio/desvio, frequência de seleção, `min_selection_frequency` configurável) → D comparação real
  MLP vs MLP+OWL, utilidade ponderada, IC bootstrap emparelhado, seleção **parcial** de features.
  Estados: `ACCEPT_SIGNIFICANT_GAIN`, `ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN`, `ACCEPT_PARTIAL_FEATURE_SET`,
  `REJECT_NO_NOVEL_FEATURES`, `REJECT_UNSTABLE_FEATURES`, `REJECT_DEGRADATION`, `REJECT_LEAKAGE_RISK`,
  `REJECT_INVALID_ONTOLOGY`, `REJECT_NO_INFORMATIONAL_GAIN`.
* **TREPAN:** metadata por feature, matriz de relatedness (regras e valores documentados e configuráveis),
  pesos de split sem dupla contagem (`audit_double_counting`), entidades OWL propagadas, e cada nó
  auditado com `base_score`, `final_score`, `selected_feature`, `ontology_entity`, `semantic_reason`.
* **Controlo negativo e ablação:** semântica baralhada; ablação por agregados / relacionais /
  restrições / inferidas pelo reasoner.
* **Relatório por feature:** `export_feature_audit` (CSV/JSON).

## 8. Testes

Ver a secção final com os números da execução completa. Ficheiros novos: `test_ontology_semantic_richness`,
`test_ontology_linear_redundancy`, `test_ontology_reasoner_report`, `test_ontology_matching_audit`,
`test_ontology_feature_provenance`, `test_ontology_categorical_matching`, `test_semantic_enrichment_pipeline`,
`test_semantic_graph_groups`, `test_semantic_metadata_and_split_audit`, `test_semantic_cache_invalidation`,
`test_semantic_status_and_presenter`, `test_no_name_based_semantic_groups`,
`test_production_semantic_enrichment_integration`, `test_undefined_names_regressions`.

## 9. Limitações restantes (sem esconder)

1. **Resultado científico:** nas ontologias do repositório o enriquecimento **não** traz ganho fiável
   (`SEMANTIC_VALIDATION_REPORT.md`): 0 aceitações com evidência forte e 17/30 controlos baralhados ≥ real.
   As ontologias são esqueletos; o que falta é conhecimento de domínio real e independente.
2. **Teste de hipótese sem ontologia independente:** o benchmark confirmatório V8 continua bloqueado
   (`NO_INDEPENDENT_VERSIONED_DOMAIN_ONTOLOGY`). Nada aqui prova superioridade do TREPAN Reloaded.
3. **Restrições OWL parciais:** suportados `minInclusive/maxInclusive` (anotação e faceta). **Não
   implementados:** `someValuesFrom`, `allValuesFrom`, `hasValue`, cardinalidades e propriedades de objeto
   como features. Exigem ontologias com indivíduos/relações que o dataset tabular não referencia.
4. **Famílias/papéis** só são descobertos por anotações (`measurementFamily`/`statisticRole`); não se
   infere família a partir de `subPropertyOf` ou equivalências.
5. **Professor ontológico no pipeline de produção:** a avaliação do enriquecimento é calculada e
   reportada, mas o MLP+OWL **aceite não passa a ser o professor** do TREPAN em produção (continua o MLP
   original). O caminho legado/GUI mantém o seu próprio mecanismo.
6. **Otimização:** busca aleatória com semente (orçamento igual), não Optuna; é determinística e testável,
   mas menos poderosa.
7. **`ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN`** usa o máximo de duas métricas secundárias, o que aumenta
   falsos positivos. A regra não foi alterada após ver os resultados; recomenda-se uma regra mais estrita
   numa versão futura, pré-registada. `evidence_strength` torna a diferença visível.
8. **`BLOCKED_BY_ENVIRONMENT` — GUI:** `PyQt6` não está instalado; o painel foi implementado no
   *presenter* (Python puro, testado) mas a **ligação à janela da GUI não foi executada nem verificada**.
   Causa: dependência ausente. Comando: `pip install -r requirements-gui.txt`. Validar: `python run_biuri.py`
   e treinar com OWL; ou `pytest tests/test_counterfactual_gui.py tests/test_tree_widget_feature_names.py`.
9. **`BLOCKED_BY_ENVIRONMENT` — TensorFlow/CLEAR:** não instalado; testes de CLEAR não correm aqui.
10. `_infer_concept_type` continua a escolher distribuições de amostragem sintética por palavras-chave
    (legado); mantido para não alterar resultados.
11. Falhas de testes **pré-existentes e não relacionadas** (ver secção de testes).

## 10. Testes executados (suite completa, Python 3.11, numpy 1.26.4, sklearn 1.6.1, owlready2 0.47, HermiT/Java)

| | Antes (início da sessão) | Depois |
|---|---|---|
| Passam | 329 | **449** |
| Falham | 3 | **2** |
| Ignorados | 2 | 2 |

* O guard anti-hardcode (`test_no_dataset_names_in_core_gui_outside_benchmark_exceptions`) **passou a
  passar**: era um falso positivo (parâmetro `digits` no presenter), corrigido com um renome e sem enfraquecer
  o guard.
* As 2 falhas restantes são **pré-existentes e não relacionadas**, e foram confirmadas **idênticas** num
  worktree do commit anterior às alterações desta fase (`b49e8ff`):
  * `test_soft_global_tree_selects_without_external_test`: `SoftDecisionTreeClassifier` foi retirado da
    produção, mas o teste ainda o usa (teste obsoleto);
  * `test_fidelity_hierarchy_c45_original_reloaded`: espera `fid_C4.5 < fid_Original < fid_Reloaded` e obtém
    `0,8583 < 0,8571` — uma alegação científica que **não se verifica**; é uma decisão do autor (regressão a
    corrigir ou tolerância a aceitar), não foi alterada.
* Um teste existente foi ajustado de propósito: `relational_features == 6` passou a `10` por causa das duas
  operações relacionais novas pedidas (2 famílias × 5), mantendo todas as outras asserções.
* `compileall` e build (wheel) passam; as ontologias de teste são construídas em memória com owlready2.
* Não validado aqui (ver limitações 8 e 9): GUI (PyQt6) e CLEAR (TensorFlow).
