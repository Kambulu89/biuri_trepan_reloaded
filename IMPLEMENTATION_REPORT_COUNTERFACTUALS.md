# Implementation report — pipeline contrafactual (cfkit)

## 1. Auditoria do estado encontrado
* `counterfactuals/engine.py` (+ `service.py`, `gui/counterfactual_panel.py`, `cf_tree.py`): gerava candidatos por LORE/CLEAR/CoGS/DiCE, mas
  devolvia candidatos não re-validados no modelo explicado, usava `status` inconsistente e rotulava as aproximações sem ressalvas.
* O espaço de trabalho era o codificado (one-hot/LabelEncoder), sem reconstrução reversível nem constraints hard/soft explícitas.
* **Porque falhava «Construir árvore CF»**: (a) o botão exige um CF *validado* e o legado aceitava/mostrava candidatos não validados,
  pelo que o estado do botão e o conteúdo não coincidiam; (b) `cf_tree.py` passava `sample_weight` ao TREPAN, que **ignora pesos**, e
  anunciava um `cf_weighting` inexistente; (c) o resultado não se distinguia da árvore global TREPAN; (d) não havia teste ponta-a-ponta
  (só a função interna). Corrigido: pesos removidos, `tree_kind="CF-LocalTree"`, finalidade e `causality: NOT_CLAIMED` documentados,
  teste e2e worker→worker→painel (`tests/test_counterfactual_gui.py::test_construir_arvore_cf_end_to_end_worker_to_panel`).

## 2. Arquitectura nova — `counterfactuals/cfkit/`
| módulo | papel |
|---|---|
| `schema.py` | `FeatureSchema`: espaço humano↔modelo reversível (one-hot reconstruído, sem combinações inválidas), int/binário/categórico, colunas derivadas (`derive_fn`) |
| `rules.py` | `ConstraintSet` (hard/soft, imutáveis, direcção, custos, dependências), `OntologyConstraintExtractor` (só extrai o que existe na OWL; resto em `not_extracted`) |
| `metrics.py` | Gower (espaço misto), esparsidade (one-hot conta 1), densidade kNN só no treino, diversidade, duplicados, dominância |
| `methods.py` | LORE-inspired, CLEAR-inspired, CoGS-inspired, tree-path (m-of-n respeitado) |
| `api.py` | `generate_counterfactual(instance, target_class, model, method, constraints, random_state, …)`, `compare_methods`, `run_cf_benchmark`, cache, export |
| `result.py` | `CounterfactualResult` + `CounterfactualStatus` (SUCCESS, NO_CF_FOUND, CONSTRAINT_INFEASIBLE, INVALID_TARGET, TIMEOUT, …) |
| `session.py` | adaptador da sessão da GUI (modelo activo, instância activa, ontologia independente do enriquecimento MLP) |

Garantias: todo CF é re-passado ao modelo explicado (`model_valid`) e a constraints hard (`semantic_valid` distinto);
alvo explícito (oposto só em binário; multiclasse exige alvo → `INVALID_TARGET`); ranges OWL > metadata > treino > fallback, **nunca teste**;
sem ontologia → `semantic_validation = NOT_AVAILABLE`; budgets (`max_iterations`, `max_time`) registados; mesma seed ⇒ mesmo resultado;
chave de cache = hash do modelo, instância, alvo, método, constraints, seed, hash da ontologia, versão do pipeline;
exportação `counterfactuals.csv`, `counterfactual_report.json`, `summary.md` com proveniência; texto determinístico e **nunca causal**.

## 3. Honestidade dos métodos
LORE/CLEAR/CoGS são **-inspired**, não canónicos (`METHOD_REGISTRY` lista divergências; `canonical=False`). Não foi feita verificação contra
as implementações de referência. DiCE continua no motor legado. Constraints semânticas ≠ causalidade.

## 4. Integração
`service.generate_explanation_from_session` encaminha LORE/CLEAR/CoGS/TREE para cfkit (`pipeline="cfkit"`), mantém o legado (`to_engine_dict`, `status_legacy`).
GUI: «mostrar todas as features» (por omissão só as alteradas), alvo obrigatório em multiclasse, nomes de método honestos.

## 5. Testes
`tests/test_cfkit.py` (49: unidade, validade, hard constraints, leakage treino/teste, m-of-n, OWL, benchmark, guarda anti-nomes-de-dataset),
testes legados/GUI actualizados, e2e da árvore CF. Suite completa: 504 passed, 2 failed (`test_scientific_superiority_pipeline_v7`:
`test_soft_global_tree_selects_without_external_test`, `test_fidelity_hierarchy_c45_original_reloaded` — já falhavam antes, fora do âmbito CF).

## 6. Limitações
Métodos aproximados; plausibilidade = densidade kNN (proxy); OWL só fornece o que tiver ranges/anotações; custos neutros por omissão;
`run_cf_validation.py` usa dados sintéticos (sem dados reais nesta validação).
