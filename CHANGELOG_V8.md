# Changelog V8 — desempenho contra rótulos reais

- `SemanticUtilityGate` obrigatório no desenvolvimento: remove constantes e
  duplicações, mede estabilidade, novidade, complexidade e ganho OOF em balanced
  accuracy, macro-F1, accuracy e recall mínimo.
- Rejeições explícitas para leakage, inconsistência, instabilidade, ausência de
  ganho e custo de complexidade.
- MLP Ontológico direto otimizado por CV multiobjetivo e calibrado; professor
  residual preservado como componente independente.
- Professor híbrido Original/Ontológico/Residual com pesos ajustados apenas na
  validação interna e fallback explícito para o Original.
- Probabilidades do professor nas linhas reais de treino são exclusivamente
  OOF; destilação in-sample do professor híbrido é bloqueada.
- C4.5 comparado ao oráculo ontológico nos mesmos folds OOF de desenvolvimento.
- Destilação híbrida reponderada: 60% rótulo real, 30% professor, 10% confiança
  semântica.
- Splits `m`-of-`n`, best-first, soft-tree e poda Pareto reorientados para
  balanced accuracy, macro-F1 e recall minoritário; fidelidade é restrição.
- Consultas ativas incluem erro contra rótulo real aproximado no manifold,
  prioridade minoritária, gap semântico, discordância C4.5 e contrafactuais.
- Plano executável de ablação pareada com 23 variantes e folds idênticos.
- Catálogo de ontologias corrigido: TBoxes internas não são chamadas de
  ontologias externas; ABox Adult contaminada fica proibida.
- Exceções científicas fail-fast específicas adicionadas.
- UI passa a identificar as métricas do professor como validação interna.
- Benchmark confirmatório V8 bloqueado; resultados negativos V7 preservados.
