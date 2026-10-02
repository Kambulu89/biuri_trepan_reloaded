# IMPLEMENTATION REPORT — Construção de árvores (C4.5 / TREPAN Original / TREPAN Reloaded)

## 1. Arquitectura encontrada (não foi reescrita)

| Papel | Ficheiro / classe |
|---|---|
| C4.5 nativo (gain ratio, poda pessimista) | `core/c45_j48_tree.py` → `C45Classifier` (fachada `C45Tree`) |
| TREPAN Original (motor canónico) | `core/trepan_original.py` → `TrepanOriginalClassifier` (+ `TrepanOriginalExtractor`, adaptador GUI) |
| TREPAN Reloaded | `core/trepan_reloaded_historical.py` → `TrepanReloadedClassifier(TrepanOriginalClassifier)`; adaptador `core/trepan_reloaded_extractor.py` |
| Fluxo | rótulos reais → C4.5; MLP Original → TREPAN Original; oráculo seleccionado → Reloaded |
| Legado CART | `core/trepan.py`, `core/canonical_trepan.py`, `core/trepan_extractor.py` (não usados na produção; mantidos) |

O Reloaded **já herdava** do Original (`BaseTrepan → Original → Reloaded` equivalente): o laço best-first,
as queries e o m-of-n são partilhados; o Reloaded só sobrepõe hooks (`_split_selection_score`, `_priority`,
`_draw_membership_queries`, `_best_mofn`). Por isso toda a instrumentação nova na base serve os dois.

## 2. Diagnóstico: porque é que as árvores ficavam pequenas

Auditoria + reprodução (`scripts/tree_validation_smoke.py`, agnóstico: CSV genérico ou dados sintéticos):

1. **O TREPAN Original é um TREPAN verdadeiro**, não destilação CART: consulta o oráculo, gera queries por nó
   respeitando as restrições do caminho, `min_sample` por nó, best-first por `reach·(1−fidelity)`,
   candidatos simples + m-of-n por beam search, teste de pureza (limite inferior de Wilson).
2. **Um stump pode ser legítimo**: num problema fácil (verificado numa 1.ª execução com um dataset público, já removido do smoke) um m-of-n 2-de-3 na raiz
   deixou os dois filhos com 98,6 % de pureza → `STOP_PURE_NODE`. Num problema sintético mais difícil o limite passa a ser o orçamento de queries (ver `TREE_VALIDATION_REPORT.md`).
3. **Causa de stumps *espúrios* (silenciosos)**: o orçamento de queries é **global** e o extractor usa
   `max_queries = sample_size` (2000 por defeito). A raiz gasta ~600 queries, o 1.º filho ~750, e o 2.º filho
   ficava sem orçamento e era abandonado **sem qualquer registo** (apenas uma entrada interna
   `query_budget_before_min_sample`). Agora é `STOP_QUERY_BUDGET_EXHAUSTED` explícito, com `budget_exhausted=True`.
4. **Bug real no C4.5** (`_numeric_candidate`): os cortes candidatos só eram considerados quando duas linhas
   *adjacentes* ordenadas tinham classes diferentes. Com valores repetidos (binárias, one-hot, ontológicas) isso
   depende da ordem das linhas e podia ignorar o melhor corte (a árvore colapsava). Corrigido com a regra de
   fronteira de Fayyad-Irani ao nível dos blocos de valor.
5. Não foi encontrado bug de renderização nem de armazenamento. Nenhum parâmetro foi alterado para forçar crescimento.

## 3. Alterações

### TREPAN base (`core/trepan_original.py`) — herdadas pelo Reloaded
- `StopReason` (`STOP_PURE_NODE, MAX_DEPTH, MIN_SAMPLES, NO_VALID_SPLIT, MIN_GAIN, QUERY_BUDGET_EXHAUSTED, MAX_NODES, QUERY_GENERATION_FAILURE, NUMERICAL_FAILURE, PRUNED`); cada folha guarda `stop_reason` + `stop_detail` (ex.: `best_candidate_gain` vs `required_min_gain`).
- Novo parâmetro `min_gain` (defeito 0.0 → comportamento anterior inalterado).
- Contabilidade de queries por nó: `queries_requested/generated/valid/rejected/used`, `real_samples`, `synthetic_samples`, `effective_samples`; globais: `query_budget_*`, `budget_starved_nodes_`, `query_budget_exhausted_`. `FeatureDistributionModel.draw` conta candidatos rejeitados pelas restrições do caminho.
- Log best-first `expansion_log_` (ordem, prioridade, componentes, massa, erro estimado, impureza, ganho potencial, e `is_best_first_choice` = prioridade ≥ máximo da fila).
- Contadores de candidatos: simples/m-of-n gerados, avaliados, rejeitados; vitórias simples vs m-of-n.
- Árvore bruta preservada (`tree_raw_root_`, `predict_raw`), poda auditada (`pruning_audit_`, `pruning_summary_`: nós/profundidade antes e depois, fidelidade de treino face ao oráculo antes/depois, `uses_test_data=False`).
- Tempos: treino, queries, procura de splits, m-of-n, poda. Falha de geração de queries num nó passa a `STOP_QUERY_GENERATION_FAILURE` em vez de abortar a árvore.
- Metadados de oráculo (`oracle_info_`), e extractors ligam o relatório a `last_audit` (`build_report`, `stop_reasons`, `stump_diagnostic`).

### C4.5 (`core/c45_j48_tree.py`)
Motivo de paragem por nó, árvore bruta (`root_raw_`) antes da poda pessimista, `pruning_audit_`, correção das fronteiras candidatas (ponto 4 acima).

### Novo `core/tree_build_report.py`
Relatório por árvore (estrutura, construção, poda, motivos de paragem, métricas, complexidade, tempos, reprodutibilidade), `stump_diagnostic` (`TREE_STUMP_DIAGNOSTIC`), `format_report`, `tree_cache_key`.
Fidelity (face ao oráculo) e accuracy (face a y real) são campos distintos.

## 4. Antes / depois

| | Antes | Depois |
|---|---|---|
| TREPAN Original | oráculo+queries+best-first+m-of-n, paragens e orçamento invisíveis | idem + motivos, contadores, log best-first, raw/poda, relatório |
| TREPAN Reloaded | herdava, mesma opacidade | mesmo relatório; com α=β=0 produz árvore idêntica ao Original (testado) |
| C4.5 | gain ratio real, mas cortes dependentes da ordem em empates; poda destruía a árvore | cortes corretos; bruta preservada; poda e paragens auditáveis |

## 5. Testes

`tests/test_tree_construction_audit.py` (24 testes): queries ao oráculo e ignorar y real, orçamento respeitado e reportado,
restrições do caminho, amostra efectiva = real+sintética, best-first real, m-of-n avaliado e seleccionado (oráculo 2-de-3),
predição m-of-n e `<=`/`>`, motivos de paragem (max_depth, max_nodes, min_gain, pura), stump diagnóstico, raw/poda,
fidelity ≠ accuracy, determinismo, serialização, não uso do teste na extracção, Reloaded(α=β=0)=Original sem bónus semântico,
bónus semântico não resgata split sem informação, C4.5 sem `DecisionTreeClassifier` e Gain Ratio vs ganho puro, chave de cache.
Suite completa: 353 passed, 3 failed — **os mesmos 3 já falhavam antes** (ver limitações).

## 6. Limitações / não feito (honestamente)

- **3 falhas pré-existentes não tocadas**: `test_v92_static_guards` (nome de dataset em `gui/ontology_status_presenter.py`), `test_soft_global_tree_selects_without_external_test` (soft tree retirada da produção), `test_fidelity_hierarchy_c45_original_reloaded` (afirma uma hierarquia Reloaded ≥ Original: é um teste que codifica a hipótese da tese e deve ser revisto, não forçado).
- **Orçamento global**: mantido o desenho existente (budget total) para não alterar números; passa a ser explícito. O Craven & Shavlik original não tem orçamento global; recomenda-se decidir por protocolo prévio se o budget deve ser por nó. O default do extractor (`max_queries = sample_size = 2000`) esgota facilmente — escolher o valor a priori, não pelo resultado.
- Reloaded: a separação base/semântico/penalizações já existe em `semantic_split_audit_` (`information_gain`, `selection_score`, `semantic_bonus`), mas não foi acrescentado `complexity_penalty`/`stability_penalty` (não existem no Reloaded actual).
- Não implementados: distribuições gaussiana truncada/bootstrap/condicional (o gerador actual é KDE/frequências com rejeição por restrições), amostragem de querying categórica específica, ablações A–G completas, comparação A/B (Reloaded com MLP Original vs Ontológico), diversidade estrutural entre árvores, guardas NaN/Inf adicionais no splitter, cache em disco de árvores (apenas `tree_cache_key` + teste; a cache existente é de MLPs). Sem ontologia real no ambiente smoke, o efeito semântico do Reloaded não foi exercitado end-to-end.
- Active query generation do Reloaded (EFSR/active sampling) não foi ablacionado nesta entrega.

## 7. Agnosticismo a datasets
Nenhum módulo de `core/` ou `gui/` referencia datasets; os testes novos usam dados sintéticos e o smoke aceita qualquer CSV. Corrigido ainda o falso positivo do guard estático (parâmetro `digits` → `ndigits` em `gui/ontology_status_presenter.py`).
