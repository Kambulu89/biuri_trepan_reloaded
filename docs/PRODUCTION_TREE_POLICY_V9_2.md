# Política de árvores — BIURI / TREPAN Reloaded V9.2 Produção

## Regra não negociável

A build de produção não pode construir, seleccionar, refinar, redestilar ou usar silenciosamente `DecisionTreeClassifier` como substituto de TREPAN Original ou TREPAN Reloaded.

Os únicos modelos de árvore suportados no fluxo científico principal são:

- **C4.5-Nativo** — `core.c45_j48_tree.C45Classifier`, treinado directamente com rótulos reais e usado como baseline supervisionado.
- **TREPAN Original** — `core.trepan_original.TrepanOriginalClassifier`, com crescimento best-first, membership queries por nó, `min_sample`, restrições de caminho e testes m-of-n.
- **TREPAN Reloaded** — `core.trepan_reloaded_historical.TrepanReloadedClassifier`, que preserva exactamente a família algorítmica do TREPAN Original e acrescenta apenas conhecimento semântico/OWL.

## O que foi removido

Foram removidos do runtime de produção:

1. construtores auxiliares `DecisionTreeClassifier` no Reloaded;
2. fallback de árvore destilada no antigo `TREPANExtractor`;
3. refinos de dominância/fidelidade/precisão que criavam outra família de árvore;
4. redestilação global para árvore crisp de outra família;
5. poda multiobjectivo que reconstruía novas árvores sklearn;
6. árvore local contrafactual baseada em sklearn — LORE local passa a usar TREPAN histórico;
7. detector de fuga do Data Contract baseado em árvore — substituído por stump univariado próprio;
8. reexecução do benchmark confirmatório congelado V7.

## Compatibilidade

`core.trepan_extractor.TREPANExtractor` permanece apenas como alias de compatibilidade e devolve sempre `TrepanOriginalExtractor` histórico. Não existe fallback para a antiga árvore de destilação.

Artefactos de produção usam:

```text
format_version = biuri-v9.2-production-2-no-cart
tree_runtime_policy = historical_trepan_only_no_cart
```

O loader recusa artefactos em que:

- `trepan_original` não seja `TrepanOriginalClassifier`;
- `trepan_reloaded` não seja `TrepanReloadedClassifier`;
- `c45_native` não seja `C45Classifier`.

## Guard automático

`tests/test_production_no_cart_v92.py` analisa por AST todos os ficheiros Python de:

- `core/`
- `gui/`
- `counterfactuals/`
- `scripts/`

A suite falha se `DecisionTreeClassifier` voltar a ser importado no runtime.

O mesmo controlo foi adicionado a `scripts/production_readiness_v92.py` através do check `no_cart_runtime`.

## Contrato científico

A comparação principal deve manter:

```text
mesmo MLP-oráculo
mesmo split
mesma seed
mesmo min_sample
mesmo max_queries
mesmo max_nodes
mesmo max_depth
mesmo m-of-n
mesma avaliação
```

A única diferença entre TREPAN Original e TREPAN Reloaded deve ser a contribuição semântica/OWL.

Quando a OWL não produz contribuição semântica efectiva, o Reloaded não pode obter vantagem por uma família de árvore diferente ou por um fallback oculto.
