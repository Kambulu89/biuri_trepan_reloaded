# TREPAN Original histórico — BIURI/TREPAN Reloaded V9.2

## Objectivo

Esta implementação separa definitivamente o **TREPAN Original de Craven & Shavlik** do antigo caminho `core/trepan_extractor.py`, que continua a existir apenas por compatibilidade e deve ser tratado como **CART destilado / baseline de destilação**.

O núcleo histórico está em `core/trepan_original.py` e não usa `DecisionTreeClassifier`.

## Mecanismos implementados

- oráculo de caixa-preta (`oracle.predict`);
- membership queries durante a construção da árvore;
- queries geradas por nó, e não apenas globalmente;
- `min_sample` antes da decisão de split;
- frequências empíricas para atributos discretos;
- Kernel Density Estimation para atributos contínuos;
- restrições acumuladas do caminho raiz -> nó;
- `DrawInstance` condicionado às restrições, incluindo testes m-of-n;
- crescimento best-first;
- prioridade `reach * (1 - fidelity)`;
- candidatos binários por Information Gain;
- thresholds entre valores adjacentes cuja classe do oráculo muda;
- pesquisa m-of-n com os operadores `m-of-(n+1)` e `(m+1)-of-(n+1)`;
- beam search (por defeito largura 2);
- teste chi-quadrado para rejeitar alterações m-of-n que não mudam significativamente a partição;
- teste estatístico conservador de pureza;
- limite de profundidade, nós e queries;
- pruning de subárvores funcionalmente redundantes;
- auditoria da ordem de expansão, reach, fidelidade, prioridade e número de queries.

## Presets históricos

A API expõe:

```python
TrepanOriginalClassifier.from_preset("nips_1995")
TrepanOriginalClassifier.from_preset("thesis_1996")
```

O preset `nips_1995` usa `min_sample=1000`, epsilon de pureza `0.05`, alpha `0.05` e beam width 2.

O preset `thesis_1996` usa `min_sample=10000`, alpha de pureza `0.01` e beam width 2. Os limites podem ser sobrepostos explicitamente para experiências controladas.

## Configuração adaptativa vs reprodução histórica

O pipeline `core/pipeline_v92.py` usa o mesmo algoritmo, mas pode reduzir/ajustar o orçamento de queries para uso interactivo em datasets pequenos. Isto **não deve ser descrito como reprodução experimental exacta de 1995/1996**.

Para experiências de tese que pretendem reproduzir o TREPAN histórico, usar um preset explícito e orçamento de queries suficiente.

## Compatibilidade e integração V9.2

`core/trepan_extractor.py` não foi apagado por compatibilidade, mas deixou de ser usado pelos caminhos principais que apresentam o modelo como **TREPAN Original**. O símbolo `TREPANExtractor` é agora explicitamente um alias legado de CART destilado e emite aviso de deprecação; novos caminhos devem usar `TrepanOriginalExtractor`.

O núcleo histórico está integrado em:

- `core/pipeline_v92.py`;
- fluxo principal da GUI (`gui/biuri_app_complete.py`);
- visualização nativa `m-of-n` (`gui/pyqt_tree_widget.py` e `gui/pyqt_tree_controls.py`);
- melhoria e regras contrafactuais (`counterfactuals/service.py`, `counterfactuals/pipelines/*`, `counterfactuals/global_rules.py`);
- `TrepanReloadedExtractor`: sem OWL espelha o Original histórico; com OWL, a árvore final também é `TrepanOriginalClassifier` sobre o espaço/oráculo enriquecido, em vez de redestilar silenciosamente para CART;
- inferência headless: `Predictor.explain(row)` devolve o caminho nativo `m-of-n` realmente percorrido.

Árvores CART continuam a existir em módulos onde são algoritmos genuinamente distintos (por exemplo C4.5 não usa sklearn CART; árvores locais de alguns métodos contrafactuais e diagnósticos legados podem usar CART), mas não são apresentadas como TREPAN Original.

## Validação adicionada

`tests/test_trepan_original_historical_v92.py` verifica:

1. queries respeitam todas as restrições do caminho;
2. `min_sample` é atingido antes de nós expandidos quando há orçamento;
3. prioridade best-first é exactamente `reach * (1 - fidelity)`;
4. uma regra sintética 2-of-3 é recuperada com alta fidelidade;
5. a mesma seed produz a mesma árvore e o mesmo número de queries.

Também permanece `tests/test_trepan_original_v92.py` para compatibilidade com a candidate anterior.

## Validação da integração histórica

Testes específicos adicionais em `tests/test_true_trepan_integration_v92.py` verificam que:

1. o adaptador público devolve `TrepanOriginalClassifier`, sem `tree_`;
2. Reloaded sem ontologia usa o mesmo núcleo histórico;
3. regras contrafactuais preservam condições `m-of-n`;
4. os caminhos principais não importam `TREPANExtractor` legado;
5. o Reloaded deploya o núcleo histórico em vez de `CanonicalTrepanClassifier`/CART;
6. a GUI declara suporte à estrutura histórica nativa;
7. a inferência devolve o caminho `m-of-n` da instância, e não a lista completa de regras.

A execução runtime da GUI PyQt6 não foi validada neste ambiente porque PyQt6 não está instalado; apenas compilação e testes estáticos foram executados. A execução do braço OWL/HermiT também permanece dependente de um ambiente com essas dependências.

## Actualização: Reloaded histórico sem CART intermédio

Na revisão de 29/09/2026, o caminho OWL deixou de terminar num `TrepanOriginalClassifier` genérico e passou a usar `core.trepan_reloaded_historical.TrepanReloadedClassifier`, que **herda directamente** do núcleo histórico.

A equivalência experimental fica:

```text
TREPAN Original  = TrepanOriginalClassifier
TREPAN Reloaded  = TrepanReloadedClassifier(TrepanOriginalClassifier)
```

O Reloaded não troca best-first, membership queries, `min_sample`, constraints, KDE/frequências, Information Gain, `m-of-n`, pureza ou pruning. A contribuição semântica entra apenas como:

- prioridade/score semântico de features/regras durante a procura de split;
- projecção das membership queries para recompor features `onto_*` de forma coerente antes de consultar o oráculo.

Quando os pesos semânticos são todos `1.0` e não existe projector ontológico, os testes verificam igualdade exacta com o TREPAN Original para a mesma seed/configuração.

Os refinadores CART/soft-tree anteriores já não são chamados por `extract_tree_with_ontology()` no caminho principal. O CART restante está explicitamente isolado em `_legacy_cart_baseline()` para compatibilidade com rotinas antigas e não é apresentado nem guardado como TREPAN Reloaded final.
