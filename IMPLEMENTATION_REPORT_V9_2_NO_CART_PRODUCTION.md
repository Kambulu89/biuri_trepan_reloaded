# IMPLEMENTATION REPORT — BIURI / TREPAN Reloaded V9.2 Production No-CART

## Objectivo

Remover definitivamente qualquer participação silenciosa de CART/`DecisionTreeClassifier` na construção do TREPAN Original e do TREPAN Reloaded e endurecer a build para produção.

## Alterações realizadas

### 1. TREPAN Original

`core/trepan_extractor.py` deixou de conter a implementação histórica de árvore destilada. `TREPANExtractor` é agora somente um alias de compatibilidade para `TrepanOriginalExtractor`.

A implementação efectiva continua em:

- `core/trepan_original.py`
- `TrepanOriginalClassifier`
- `TrepanOriginalExtractor`

### 2. TREPAN Reloaded

Foram removidos todos os construtores auxiliares de `DecisionTreeClassifier` de `core/trepan_reloaded_extractor.py`.

O caminho principal com OWL chama exclusivamente:

`core.trepan_reloaded_historical.TrepanReloadedClassifier`.

Métodos privados legados que anteriormente criavam/refinavam outra família de árvore deixaram de o fazer; os caminhos incompatíveis falham explicitamente em vez de recorrer a fallback silencioso.

### 3. C4.5

O baseline permanece `core.c45_j48_tree.C45Classifier`, implementação nativa independente de sklearn trees e treinada com rótulos reais.

### 4. Data Contract

A detecção de possível fuga de alvo deixou de treinar uma árvore sklearn de profundidade 1. Foi substituída por uma avaliação univariada própria, com CV estratificada e stump numérico/categórico interno.

### 5. Contrafactuais

- `counterfactuals/cf_tree.py`: a árvore explicativa local passa a usar `TrepanOriginalClassifier`.
- `counterfactuals/engine.py`: LORE-Local passa a usar TREPAN histórico.
- a geração de CFs a partir de árvore passou a ler regras através da API genérica/histórica e suporta m-of-n sem converter a árvore para sklearn.

### 6. Selecção global e multiobjectivo

`core/soft_global_tree.py` e `core/multiobjective_tree_selector.py` já não constroem novas árvores sklearn. Mantêm apenas compatibilidade de avaliação/seleção de modelos já treinados e bloqueiam redestilação para outra família.

### 7. Confirmatório congelado

`core/confirmatory_benchmark.py` foi transformado em acesso somente-leitura ao resultado V7 congelado. `scripts/run_confirmatory_locked.py --execute` é recusado explicitamente.

`results/confirmatory_v7/` permanece inalterado.

### 8. Artefactos de produção

Novo formato:

`biuri-v9.2-production-2-no-cart`

Manifesto inclui:

`tree_runtime_policy = historical_trepan_only_no_cart`

`core/production_inference.py` valida as classes dos modelos carregados e recusa artefactos de famílias incompatíveis.

### 9. Guard permanente

Criado `tests/test_production_no_cart_v92.py`.

O guard analisa por AST os módulos de runtime e falha se `DecisionTreeClassifier` voltar a ser importado.

`production_readiness_v92.py` inclui agora:

- `confirmatory_v7_unchanged`
- `compileall`
- `no_cart_runtime`
- `scientific_mode_locked`
- `preflight`

## Validação executada neste ambiente

Execução crítica final:

```text
82 passed
0 failed
5 subtests passed
3 warnings de convergência
```

Também foi executado um smoke test completo de produção sem OWL:

```text
Data Contract
→ preprocessing
→ MLP
→ C4.5-Nativo
→ TREPAN Original histórico
→ TREPAN Reloaded histórico
→ avaliação
→ bundle
→ load_production_bundle
→ predict
```

Resultado: concluído sem erro.

A auditoria AST encontrou:

```text
runtime DecisionTreeClassifier imports = 0
```

`compileall` dos módulos de produção: OK.

`results/confirmatory_v7/`: hashes esperados preservados.

## Limitações honestas do ambiente desta execução

O runtime disponível foi Python 3.13.5. O alvo de produção continua Python 3.11/3.12.

Não foi possível validar ao vivo neste runtime:

- PyQt6/GUI;
- Owlready2 + HermiT;
- TensorFlow/CLEAR;
- execução monolítica completa de `pytest` até ao fim, porque ultrapassou o timeout disponível; a execução progrediu sem nova falha visível após a correção das incompatibilidades encontradas.

Por estas razões, a build é uma **production candidate limpa e bloqueada contra CART**, mas a promoção formal a release final deve ocorrer depois do preflight e da suite completa no Windows/Python 3.11 ou 3.12 do utilizador.
