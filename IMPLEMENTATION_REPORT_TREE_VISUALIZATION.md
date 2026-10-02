# IMPLEMENTATION REPORT — Visualização de árvores (C4.5 / TREPAN Original / TREPAN Reloaded)

Escopo: apenas visualização. Nenhum algoritmo de construção, split, threshold, predição, poda, fidelity ou score foi alterado
(única edição fora da GUI: o export DOT legado de `TrepanOriginalExtractor.export_tree_image` passou de `shape=box, style=rounded`
para `shape=ellipse`; não toca na árvore).

## 1. Auditoria do renderer original

| Pergunta | Resposta |
|---|---|
| Biblioteca | `QPainter` próprio em `gui/pyqt_tree_widget.py` (`InteractiveTreeWidget`); controlos em `gui/pyqt_tree_controls.py`; contentor/seletor em `gui/biuri_app_complete.py` (`TreeVisualizationWidget`). Exportação por Graphviz DOT (TREPAN) e `sklearn.plot_tree` (C4.5): **caixas rectangulares arredondadas**. Sem NetworkX/Matplotlib no ecrã. |
| Posições | `_calculate_positions`: folhas todas com `x=0`, pai = ponto médio, afastamento só entre filhos directos (`min_node_spacing=80`). |
| Texto dos nós | `feature[:10]` + amostras; TREPAN: `condition_text[:19]`. Raio fixo 20. |
| Labels das arestas | C4.5: `feature[:12] ≤ thr` desenhado a meio da aresta; TREPAN: só `não/sim`. |
| Zoom/pan | `screen=(x+pan)*zoom` no desenho, hit-test com `(x-pan)/zoom`; pan em ecrã somado antes do zoom; `zoom_in/out` re-centravam; auto-fit nunca ampliava >1.0. |
| C4.5 | via adaptador sklearn-like `tree_` (categóricos compilados em cadeias binárias com limiares falsos e nós extra). |
| TREPAN | `root_` histórico; m-of-n reduzido ao 1.º literal; amostras = `len(real_y)`. |
| m-of-n | texto completo truncado a 19 chars dentro do círculo. |

### Causas das sobreposições / defeitos (confirmadas no código)
1. **Layout**: todas as folhas a `x=0` + correcção local ⇒ subárvores profundas colidem (nunca se calcula a largura da subárvore).
2. **Raio/fonte fixos** ⇒ texto transborda do círculo; labels de aresta largos cruzam nós e outras arestas; feature repetida no nó e na aresta (C4.5).
3. **Hit-test errado** após pan/zoom (cliques acertavam em nós errados); wheel-zoom e pan inconsistentes.
4. **Filtros avançados eram no-op** (`_filter_tree` terminava em `pass`) e sem aviso de nós escondidos; "Resaltar Camino" era placeholder.
5. **Cores por `hash(str)`** (varia por processo; colisões ⇒ duas classes com a mesma cor; não consistente entre árvores).
6. Árvores pequenas minúsculas (zoom ≤ 1.0), clique abria `QMessageBox` modal; sem painel de detalhes, sem tooltips úteis, exportação em caixas e só do ecrã.
7. C4.5 categórico inflacionava nós (cadeia binária) ⇒ contagem renderizada ≠ lógica.

## 2. Arquitectura nova (`gui/tree_viz/`, Qt-free excepto `render.py`)

```
Tree Model --READ ONLY--> model.py (adapters -> TreeVisualizationModel/VizNode)
   -> layout.py (Reingold-Tilford c/ contornos, nós elípticos, colisões, spacing adaptativo)
   -> render.py (QPainter vectorial; mesmo draw para ecrã e export PNG/SVG/PDF/JSON)
   -> pyqt_tree_widget.py (viewport: zoom/pan/fit/reset/centrar, selecção, colapso, pesquisa, painel)
```
* `model.py`: adaptadores TREPAN (simple / m-of-n, Original+Reloaded), **C4.5 nativo n-ário** (sem cadeias falsas; categórico `= valor`), sklearn-like (converte frações de `tree_.value` em contagens). `tree_signature()` (hash da estrutura científica), `source_node_count()` independente do adaptador, validação (`INVALID_TREE_CYCLE`, `DUPLICATE_NODE_ID`, `MISSING_CHILD`, `INVALID_THRESHOLD`, `UNKNOWN_SPLIT_TYPE`, `MISSING_ROOT`, `UNREACHABLE_NODES`).
* `labels.py`: `make_display_feature_name` (só visual), `format_threshold` (16.794835219 → 16.79; real em `threshold_full`), paleta por **posição estável** da classe (sem hash, daltónicos; classe sempre escrita no nó).
* `layout.py`: largura da subárvore via contornos por nível; dimensão do nó medida pelo texto (elipse/círculo, nunca caixa); espaçamento/fonte por presets (≤7, 8–30, 31–100, >100 nós; só categorias de rendering); deteção node-node, node-label, label-label, edge-label, edge-node com grelha espacial; colisões resolvidas **aumentando espaçamento** (nunca movendo ao acaso nem apagando).
* `details.py`: tooltip, painel de nó, de aresta e **caminho de regra** (m-of-n permanece m-of-n com as condições completas).
* `strings.py`: todos os textos novos centralizados (es por defeito, pt disponível).

## 3. Requisitos principais

* **Nós circulares/elípticos**: interno = círculo/elipse com anel interno subtil e borda grossa; folha = preenchida pela classe. Teste garante `drawEllipse` e ausência de `drawRect`/`shape=box`/`plot_tree`.
* **Política única nó/aresta**: nó = feature compacta + n; aresta = `≤ 16.79` / `> 16.79` (ou `= A` categórico, `não/sí` m-of-n). A feature nunca é repetida na aresta.
* **m-of-n**: nó `(m-of-n)(2/3)`; clique/tooltip lista todas as condições com limiares completos e "VERDADERO cuando al menos m de n". No caminho de regra não é convertido em regra simples.
* **Semântico (Reloaded)**: badge `S` externo só quando `semantic_bonus>0` ou `ontology_influenced`; scores base/bónus/final no painel; nada de campos Reloaded em C4.5/Original. Feature `onto_*` ⇒ `onto:…`, nome completo no painel.
* **Incerteza** definida explicitamente: entropia normalizada do nó (internos `H x.xx`) e confiança (folhas `84%`), ambas no painel.
* **Interação**: zoom (centro e roda no cursor, limites 0.08–6), pan, **Fit** (bounding box real; árvore de 3 nós preenche ≥30 % da largura e fica centrada; árvores grandes têm piso de legibilidade com topo visível + pan), Reset (zoom+pan+selecção, sem reconstruir), Centrar raiz/nó, selecção com realce do caminho, tooltips + clique + painel de detalhes (nó e aresta), colapsar/expandir (botão ou duplo clique), filtros avançados **funcionais e só visuais**, pesquisa por feature/classe/id, vista científica (IDs/amostras), legenda adaptativa às classes reais.
* **Filtros ≠ poda**: nós escondidos indicados (`+ N nodos`, "Mostrando 15 / 27 nodos") e contabilizados: `rendered + hidden == logical`.
* **Troca de árvore** (`update_tree`): só relê a árvore pronta; limpa selecção/colapsos/filtros; relayout + fit; sem retreino. O seletor passa o nome do algoritmo; os controlos mostram "Árbol seleccionado: …".
* **Exportação** (`Exportar`): PNG ≥2400 px (300 dpi), SVG e PDF vectoriais, JSON paralelo (estrutura completa + diagnóstico), sempre com o **layout completo** (ignora filtros), só árvore + legenda + título opcional. Função legada `export_tree_png` mantida (agora círculos).
* **Diagnóstico** (`widget.diagnostic()` / `layout.diagnostics`): algoritmo, nós lógicos/renderizados/escondidos, profundidade, bbox, overlaps, iterações de espaçamento, veredicto — distingue "árvore realmente pequena (3=3)" de "BUG DE VISUALIZAÇÃO (27 lógicos ≠ 3 renderizados)".
* **Estados**: sem árvore/erro mostram motivo (nunca canvas vazio); árvore de 1 nó suportada; erros de estrutura → `TREE_VISUALIZATION_ERROR [CODE]` sem crash.
* **Performance/cache**: layout cacheado por (estrutura, colapsos, filtros, opções de label); zoom/pan não passam pelo layout. Layout de 501 nós: ver testes (<10 s com heurística, ~0,1 s em 111 nós).

## 4. Evidências

Executar `python scripts/export_tree_examples.py` (dados sintéticos 3 classes; também aceita `--csv`). Resultado (`docs/tree_visualization_examples/evidence.json`):

| Árvore | signature antes = depois | predições iguais | nós lógicos | nós renderizados | overlaps | profundidade |
|---|---|---|---|---|---|---|
| C4.5-Nativo | sim | sim | 55 | 55 | 0 | 10 |
| Trepan-Original | sim | sim | 31 | 31 | 0 | 6 |
| Trepan-Reloaded | sim | sim | 31 | 31 | 0 | 6 |

Exemplos exportados em `docs/tree_visualization_examples/{c45_nativo,trepan_original,trepan_reloaded}.{png,svg,pdf,json}`.
(O Reloaded com α=β=0 e sem ontologia é idêntico ao Original; o badge semântico é testado com contribuição injectada no audit.)

## 5. Testes

`tests/test_tree_visualization.py` — 58 testes: não-mutação (assinatura + pickle + predições antes/depois de modelo/layout/detalhes/render/export), rounding só visual, labels (Unicode, notação científica, curto/longo, onto), invariantes de layout (1, 3, balanceada, desbalanceada profundidade 6, 20, 50, 100, 501 nós: sem coordenadas duplicadas, filhos abaixo, esquerdo à esquerda, bbox válida, 0 overlaps), subárvore maior com mais espaço, presets, lógico=renderizado, stump vs bug, C4.5 categórico n-ário, m-of-n (nó, painel, caminho), badge semântico, multiclasse 2/3/6 com cores estáveis, validação de erros (ciclo, id duplicado, NaN, filho em falta, tipo desconhecido), filtros/colapsos só visuais, i18n, ausência de caixas, e Qt offscreen real: fit em 1366×768 / 1920×1080 / 2560×1440, árvore pequena preenche e centra, zoom/pan/reset/centrar, **hit-test coerente com o desenho**, selecção + painel + caminho + aresta, pesquisa, troca de árvore sem tocar nos modelos, estados vazio/erro, exportação PNG/SVG/PDF/JSON (nós `<ellipse>`/`<circle>`, PDF válido, JSON completo), export ignora filtros.
Suite completa: 417 passed, 3 failed — todos pré-existentes (`test_counterfactual_gui` verificado também no código original; soft tree retirada; hierarquia Reloaded ≥ Original, que codifica a hipótese da tese).

## 6. Limitações restantes
* Sem comparação visual lado-a-lado Original vs Reloaded (apenas diagnósticos por árvore); sem destaque de diferenças estruturais.
* Entidade/proveniência ontológica por feature só aparece se existir no audit (hoje: scores, grupos e features semânticas); `Ontology entity` não é inventada.
* Sem teste pixel-perfect nem screenshots comparados (usa-se invariantes). Não testado em monitor HiDPI físico: render vectorial via `QPainter` (nítido por construção).
* Fontes: `QFont("Sans Serif")` do sistema; métricas reais do Qt entram no layout, mas fontes diferentes entre máquinas alteram larguras (o espaçamento adaptativo e os testes de overlap cobrem isso).
* O exportador DOT legado do extractor continua separado (agora elíptico, com texto completo dos nós); o caminho principal de exportação da GUI é o novo.
* Nota de repositório: ficheiros `.pyc` estão versionados; não foram incluídos neste commit.
