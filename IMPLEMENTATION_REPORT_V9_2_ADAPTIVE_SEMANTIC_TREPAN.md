# BIURI / TREPAN Reloaded V9.2 — Adaptive Semantic TREPAN

## Objectivo desta revisão

Implementar os mecanismos que faltavam para aumentar legitimamente a capacidade do TREPAN Reloaded sem manipular o teste externo nem alterar a família algorítmica:

1. geração de candidatos `m-of-n` orientada pela ontologia;
2. membership queries activas/semânticas dentro do MESMO orçamento do TREPAN Original;
3. selecção científica train-only da capacidade comum do TREPAN e dos parâmetros semânticos do Reloaded;
4. relatedness OWL tipada, incluindo hierarquia inferida quando o objecto OWL/reasoner a disponibiliza.

Nenhuma regra específica de dataset foi adicionada.

## Alterações implementadas

### `core/trepan_reloaded_historical.py`

- Novos parâmetros científicos:
  - `semantic_candidate_budget`
  - `semantic_relation_threshold`
  - `semantic_active_query_fraction`
  - `semantic_active_pool_multiplier`
- `_ontology_guided_candidates()` constrói candidatos `m-of-n` a partir de grupos/relações OWL, mesmo quando o beam puramente estatístico não os visitaria.
- O audit regista `semantic_candidates_evaluated` e `semantic_candidate_generation_count`.
- Membership queries activas usam um pool sintético sem consultas extra ao oráculo; priorizam proximidade aos limites do nó e eixos semanticamente relevantes.
- O número de membership queries ao MLP continua limitado por `max_queries`.
- Relatedness matrix pode reforçar regras mesmo sem `semantic_feature_groups` explícitos.

### `core/trepan_scientific_tuning.py` (novo)

Selecção em duas fases, exclusivamente no treino:

1. **capacidade comum** — TREPAN Original selecciona entre variantes de nós, `m-of-n`, features candidatas e orçamento de queries;
2. **extensão semântica** — mantendo a capacidade comum fixa, o Reloaded selecciona força semântica, coerência de grupo, candidate budget e política de active query.

Objectivo interno combina fidelidade ao MLP, balanced accuracy, Macro-F1 e penalização de complexidade. O teste externo não entra na API do tuner.

### `core/controlled_trepan_experiment.py`

`ControlledTrepanConfig` passou a incluir os parâmetros semânticos adaptativos. `common_tree_kwargs()` remove-os antes de construir o Original, garantindo a mesma capacidade física nos dois braços.

### `core/ontology_semantic_graph.py`

O grafo continua dataset-agnostic, mas o relatedness deixou de tratar todos os tipos de aresta como equivalentes. Pesos actuais:

- `equivalent`: 1.00
- `subsumption`: 0.88
- `inferred_subsumption`: 0.78
- `inverse`: 0.82
- `domain`/`range`: 0.68

A relatedness usa caminho de produto máximo com decaimento. Quando o objecto OWL disponibiliza `ancestors()` depois do reasoner, esses vínculos são adicionados como `inferred_subsumption`.

### `core/production_training.py`

- tuning científico activado por defeito;
- tuning usa apenas `Z_train/y_train`;
- o `ControlledTrepanConfig` seleccionado é usado pelos dois braços;
- relatório e manifest guardam o histórico do tuning e o config final;
- novo formato de artefacto: `biuri-v9.2-production-3-adaptive-semantic`.

### GUI / extractor histórico

- o modo Científico selecciona a capacidade comum usando somente `_eval_split_original['X_train']` / `y_train`;
- a capacidade seleccionada é propagada ao TREPAN Original e ao TREPAN Reloaded;
- no Reloaded, apenas parâmetros semânticos são afinados depois, também por CV train-only;
- `X_test/y_test` não participam no tuning;
- a projecção semântica final continua disponível, mas não é usada dentro da CV para evitar contaminação entre folds.

## Protocolo preservado

- CART permanece removido do runtime de produção.
- C4.5-Nativo continua baseline supervisionado com rótulos reais.
- TREPAN Original e Reloaded usam o mesmo MLP-oráculo, seed e capacidade final.
- A única diferença permitida no braço Reloaded é a extensão semântica.
- `results/confirmatory_v7/` não foi alterado.
- `run_confirmatory_locked.py --execute` não foi executado.

## Testes executados neste ambiente

Três grupos disjuntos concluídos:

- 26 passed — contrato, preprocessing, MLP factory, avaliação, artefactos e matriz dataset-agnostic;
- 44 passed — TREPAN histórico/Reloaded, tuning adaptativo, ablação controlada, semantic audit, produção, no-CART e modo Científico;
- 30 passed — integração TREPAN, contexto/oráculo Reloaded, mirror sem OWL, ontology gates, static guards e runtime control.

**Total dos grupos disjuntos: 100 passed, 0 failed.**

`python -m compileall -q core gui counterfactuals tests scripts`: OK.

## Confirmatório congelado

Os 5 SHA-256 de `results/confirmatory_v7/` são idênticos ao pacote `production_no_cart` de origem. Ver `V92_ADAPTIVE_SEMANTIC_CONFIRMATORY_HASH_CHECK.json`.

## Limitações do ambiente actual

O readiness global continua `false` neste runtime porque:

- Python: 3.13.5, enquanto o alvo é 3.11/3.12;
- `dtreeviz` não instalado;
- PyQt6 não validado neste runtime;
- Owlready2/HermiT não disponível para uma corrida OWL real desta revisão;
- TensorFlow/CLEAR não validado.

Estas limitações não foram marcadas como aprovadas.

## Interpretação científica

Esta revisão **não programa nem garante** a ordem `C4.5 < TREPAN Original < TREPAN Reloaded`. Ela remove gargalos que impediam o TREPAN de usar plenamente o professor e a ontologia, mantendo a comparação justa. A ordem deve emergir dos dados e ser validada com IC/testes pareados; nunca é imposta ao avaliador.
