# Changelog V9 — melhoria contrafactual da árvore substituta

- Integrada a opção **Melhorar árvore substituta** descrita na tese recebida.
- TREPAN Original e Reloaded recebem lotes CF próprios, com professores,
  matrizes e nomes de features independentes.
- C4.5-Nativo excluído corretamente da melhoria de substitutos.
- Geração CLEAR, COGS ou combinada limitada ao desenvolvimento.
- Implementadas classificação A/B/C, confiança mínima, peso por tipo,
  incerteza de fronteira, densidade e âncoras reais.
- Corrigida a distância de consulta para o vizinho mais próximo e transformado
  o percentil de densidade num gate real de plausibilidade.
- Peso agregado dos CFs limitado a uma fração explícita do orçamento sintético.
- Controlo sem CF e candidata com CF usam a mesma semente.
- Seleção e aceitação separadas: o melhor parâmetro é escolhido num holdout e
  passa por outro holdout independente antes do refit.
- Quality gate exige ganho contra rótulos reais, não inferioridade de balanced
  accuracy, macro-F1, recall minoritário e fidelidade, além de controlar
  complexidade.
- O teste confirmatório não é exposto ao worker, não seleciona candidatos e não
  é avaliado pela opção.
- Corrigida a propagação de `extra_X`, `extra_y` e `extra_weights` nos caminhos
  original/residual do extrator Reloaded.
- O lote CF automático do Reloaded pode ser realmente desligado no controlo
  pareado (`plausible_cf_budget=0`).
- A interface permite escolher alvo, gerador, orçamento, confiança e densidade;
  apresenta métricas do controlo, candidata, A/B/C e blockers.
- Árvores aprovadas aparecem separadamente como “melhorada por CF”; árvores
  rejeitadas não substituem o estado existente.
- “Construir árvore CF” e “Melhorar árvore substituta” permanecem operações
  semanticamente diferentes e com estados separados.
