# BIURI / TREPAN Reloaded V9.2 — Relatório de integração do núcleo TREPAN histórico

Data da integração: 29/09/2026

## 1. Objectivo desta ronda

Eliminar dos caminhos principais a ambiguidade histórica em que `TREPANExtractor` era um `DecisionTreeClassifier` destilado, passando GUI, contrafactuais e TREPAN Reloaded a consumir o núcleo `TrepanOriginalClassifier` implementado em `core/trepan_original.py`.

A regra aplicada foi: **nenhum caminho que se apresente ao utilizador como “TREPAN Original” pode produzir CART silenciosamente**. O CART legado continua disponível apenas por compatibilidade, com identidade explícita `CART Destilado (legado)` e aviso de deprecação.

## 2. Alterações implementadas

### 2.1 Adaptador público do TREPAN histórico

`core/trepan_original.py`

- adicionado `TrepanOriginalExtractor`, mantendo a API `extract_tree(...)` dos consumidores antigos;
- o modelo entregue é sempre `TrepanOriginalClassifier`, sem atributo sklearn `tree_`;
- contrafactuais fornecidos como `extra_X` são usados como sementes adicionais, mas são novamente rotulados pelo oráculo;
- `extra_y` e `extra_weights` não substituem a autoridade do oráculo e geram aviso explícito;
- auditoria regista `historical_core=True`, `distilled_cart=False`, queries, nós, profundidade, `min_sample`, fidelidade, splits e ordem de expansão;
- exportação DOT preserva testes `m-of-n`.

### 2.2 TREPAN Reloaded sobre o mesmo núcleo

`core/trepan_reloaded_extractor.py`

- sem OWL: delega para `TrepanOriginalExtractor`, mantendo o modo espelhado;
- com OWL aceite: a árvore final é novamente `TrepanOriginalClassifier`, treinada sobre o espaço enriquecido e consultando o oráculo activo;
- deixou de existir redestilação silenciosa do modelo final para `DecisionTreeClassifier` sob o nome TREPAN Reloaded;
- multiobjective pruning baseado em sklearn fica apenas disponível para árvores sklearn legadas e não substitui o núcleo histórico;
- exportação de regras, explicações semânticas, validação de coerência e exportação gráfica reconhecem a estrutura histórica nativa;
- os CARTs que permanecem no ficheiro são auxiliares/diagnósticos legados, não a identidade do modelo final TREPAN.

### 2.3 GUI

`gui/biuri_app_complete.py`

- caminhos principais de treino/extracção passaram a importar `TrepanOriginalExtractor`.

`gui/pyqt_tree_widget.py`

- suporta árvores com `root_`/`MofNTest` sem exigir `tree_`;
- mostra a condição `m-of-n` completa no nó;
- ramos históricos são apresentados como `não` / `sim`;
- suporta labels textuais e distribuições/classes do nó histórico.

`gui/pyqt_tree_controls.py`

- exportação PNG/DOT reconhece TREPAN histórico;
- não converte a árvore histórica para CART para visualização;
- mantém fallback textual/gráfico quando Graphviz não estiver disponível.

A GUI **não foi executada** neste ambiente porque PyQt6 não está instalado. Houve compilação estática e testes de fonte.

### 2.4 Contrafactuais

`counterfactuals/service.py`

- melhoria do TREPAN Original usa `TrepanOriginalExtractor`;
- removido bloco `try/finally` sem efeito;
- a seed é passada directamente ao extractor histórico.

`counterfactuals/pipelines/pipeline_train.py` e `pipeline_improve.py`

- deixam de criar o extractor CART legado para “TREPAN Original”.

`counterfactuals/global_rules.py`

- regras globais suportam estrutura histórica;
- condições `m-of-n` são preservadas como unidade lógica, incluindo `m`, literais e negação;
- não são achatadas para thresholds CART.

`counterfactuals/surrogate_improvement.py`

- métricas de complexidade reconhecem `node_count_` do TREPAN histórico.

Árvores CART locais usadas por algoritmos contrafactuais específicos continuam a existir quando são de facto árvores locais desses métodos; não são rotuladas como TREPAN Original.

### 2.5 Inferência e explicação

`core/inference.py`

- `Predictor.explain(row)` reconhece árvore histórica;
- devolve **apenas o caminho realmente percorrido** pela instância;
- cada passo informa condição `m-of-n`, `m`, `n`, resultado e ramo (`sim`/`não`);
- a folha inclui previsão, probabilidades, suporte e reach;
- não devolve a lista inteira de regras como se fosse a explicação local;
- continua compatível com árvores sklearn.

### 2.6 Identidade científica

`core/algorithm_identity.py`

- `trepan_reloaded` agora é descrito como `núcleo TREPAN histórico + enriquecimento/semântica OWL opcional`;
- criada identidade separada `distilled_cart_legacy` / `CART Destilado (legado)`, marcada como não canónica.

`core/trepan_extractor.py`

- permanece por compatibilidade;
- `TREPANExtractor` é explicitamente legado/deprecated;
- novos consumidores não devem utilizá-lo como TREPAN Original.

### 2.7 Ablação

`core/ablation_study.py`

- TREPAN Original e Reloaded usam `TrepanOriginalClassifier` como família base;
- braço com OWL usa o mesmo núcleo sobre o espaço/oráculo enriquecido;
- campos antigos de distillation/active/Pareto que não pertencem ao núcleo histórico são marcados como desactivados no caminho novo.

`core/confirmatory_benchmark.py` não foi executado nem usado para regenerar o protocolo congelado.

## 3. Testes adicionados/alterados

Novo conjunto principal: `tests/test_true_trepan_integration_v92.py`.

Valida:

1. adaptador público devolve TREPAN histórico, não CART;
2. Reloaded sem OWL usa o mesmo núcleo histórico;
3. regras contrafactuais preservam `m-of-n`;
4. caminhos principais não importam `TREPANExtractor` legado;
5. Reloaded deploya o núcleo histórico;
6. widget GUI contém suporte nativo a TREPAN histórico;
7. inferência devolve caminho `m-of-n` local.

Também foram actualizados testes que pretendem avaliar TREPAN Original para usar `TrepanOriginalExtractor`. O teste que deliberadamente inspeciona o CART legado usa `DistilledCARTExtractor` de forma explícita.

## 4. Comandos e resultados reais

Ambiente disponível:

```text
Python 3.13.5
PyQt6: indisponível
owlready2: indisponível
TensorFlow: indisponível
```

### Núcleo + integração histórica

```text
pytest -q tests/test_true_trepan_integration_v92.py \
          tests/test_trepan_original_historical_v92.py \
          tests/test_trepan_original_v92.py

13 passed in 4.34s
```

### Artefactos + preprocessing

```text
pytest -q tests/test_artifacts_v92.py tests/test_preprocessing_v92.py

9 passed in 2.18s
```

### Matriz dataset-agnostic

```text
pytest -q tests/test_dataset_agnostic_matrix_v92.py

3 passed in 24.64s
```

O cenário largo `120 x 400` demorou aproximadamente 20 segundos e terminou com sucesso.

### Contrafactuais + fidelidade TREPAN

```text
pytest -q tests/test_counterfactual_improvement.py \
          tests/test_counterfactual_service.py \
          tests/test_trepan_original_fidelity.py

12 passed, 10 warnings in 6.02s
```

Os warnings observados foram `ConvergenceWarning` de MLPs de fixtures, aviso intencional de re-rotulagem pelo oráculo para sementes CF e aviso de deprecação do CART legado num teste que o chama explicitamente.

### Routing Reloaded/oráculo

```text
pytest -q tests/test_trepan_reloaded_context.py tests/test_trepan_reloaded_oracle.py

6 passed, 7 warnings in 2.21s
```

### Identidade/protocolo

```text
pytest -q tests/test_protocol_identity_failfast_unittest.py

5 passed in 1.57s
```

### Espelhamento sem ontologia — teste directo

```text
pytest -q tests/test_no_ontology_mirror_original.py::test_mirror_original_tree_same_predictions

1 passed, 1 warning in 3.23s
```

### Compilação

```text
python -m compileall -q core gui counterfactuals tests scripts
```

Resultado: código de saída `0`.

### Suite combinada

Foi tentada uma execução combinada dos grupos acima. O processo ultrapassou o limite de 120 segundos depois de 23 testes apresentados como aprovados pelo pytest e foi interrompido pelo runtime. Portanto **não se declara a suite completa como aprovada**.

## 5. Auditoria de referências legadas

Busca executada nos caminhos `core/`, `gui/`, `counterfactuals/` e `tests/` para:

```text
TREPANExtractor(
from core.trepan_extractor import TREPANExtractor
trepan_original = DecisionTreeClassifier
```

Depois da migração, as únicas ocorrências do texto de import/construção antigo aparecem nas próprias asserções do teste estático que verifica a sua ausência nos caminhos principais.

`DecisionTreeClassifier` ainda existe legitimamente em módulos de baseline/diagnóstico/árvores locais e no `core/confirmatory_benchmark.py` congelado/legado, que não foi executado. Isso não significa que o modelo final entregue como TREPAN Original seja CART.

## 6. Integridade do protocolo congelado

SHA-256 de todos os 5 ficheiros em `results/confirmatory_v7/` foi calculado antes e depois da integração.

```text
CONFIRMATORY_HASHES_IDENTICAL=true
5 ficheiros antes
5 ficheiros depois
```

Nenhum ficheiro do confirmatório V7 foi alterado e `scripts/run_confirmatory_locked.py --execute` não foi executado.

## 7. O que NÃO foi validado neste ambiente

- execução runtime da GUI PyQt6;
- renderização real dos widgets Qt em Windows;
- braço OWL/HermiT, porque `owlready2`/Java reasoning não está disponível neste runtime;
- TensorFlow/CLEAR;
- matriz Python 3.11/3.12 (o runtime disponível é 3.13.5);
- suite pytest completa numa única execução, por timeout do ambiente;
- segundo teste pesado de `test_no_ontology_mirror_original.py`, que entra no comparador completo e ultrapassou o limite temporal em tentativa anterior.

## 8. Estado final desta etapa

A migração pedida nesta ronda está implementada no código-fonte: **GUI principal, contrafactuais, inferência e TREPAN Reloaded já usam/entendem o mesmo núcleo TREPAN histórico**. O antigo CART destilado permanece apenas como componente legado explicitamente identificado e não como “TREPAN Original”.

Para uma release final de produção continuam obrigatórias as validações de ambiente ausentes (PyQt6/Windows, OWL/HermiT, Python 3.11/3.12 e suite integral no CI).
