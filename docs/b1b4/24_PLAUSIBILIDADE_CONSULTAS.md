# Doc. 24 — Plausibilidade estatística das consultas ao oráculo

Estado: **exploratório, DEV (já observado), só treino** (cada unidade: divisão 75/25 própria; metade B do treino = base de amostragem, metade H = holdout de plausibilidade; teste externo/confirmatório não usado). Nada implementado no núcleo; os amostradores vivem em `validation/experimental/query_samplers.py` (flag-less, fora de `core/`). Estados mantidos: B1 e B2-C `TECHNICALLY_VALIDATED / NOT_PROMOTED`; amostragem coerente `PARTIAL_NOT_PROMOTED`; B2-A/B2-B `NOT_IMPLEMENTED`; B3 `DESIGN_ONLY`; B4 `NOT_IMPLEMENTED`.

## 1. Hipóteses registadas antes de correr
- HP1 A amostragem marginal (A/B) gera consultas distinguíveis das linhas reais (AUC real-vs-sintético alto) em todos os datasets.
- HP2 Amostragem conjunta empírica com perturbação controlada (C) reduz AUC e distância ao vizinho mais próximo sem copiar linhas.
- HP3 Em dados mistos (one-hot, binárias, ordinais, nominais), a amostragem marginal ativa categorias mutuamente exclusivas; as variantes com grupos conjuntos (As, C, D) não.
- HP4 Maior plausibilidade pode reduzir cobertura da fronteira de decisão e diversidade (trade-off a medir, não a otimizar isoladamente).
Estratégias: **A** marginal histórica; **As** marginal com grupos conjuntos (one-hot/derivadas recalculadas); **B** marginal coerente (existente); **C** conjunta empírica + perturbação controlada; **D** vizinhança condicional (k-NN, interpolação convexa, categorias copiadas conjuntamente); **E** domínio ontológico (só como *wrapper* `DomainConstrained` com predicados explícitos — ver §6). Métricas conjuntas: plausibilidade, fidelidade a oráculo (substituto), cobertura da fronteira, diversidade, consistência semântica, custo. Sem ajuste de hiperparâmetros dos amostradores após ver resultados (valores por omissão fixados: ruído 0,25, p_resample 0,2, p_flip 0,05, k=5). Nº de comparações: 5 estratégias × 12 métricas × 2 conjuntos; sem testes de significância (descritivo).

## 2. Esquema genérico de atributos (agnóstico)
`infer_schema` classifica cada coluna só a partir dos dados: binária; grupos *one-hot* (componentes do grafo de exclusão mútua, confirmados por soma de linha = 1); discreta inteira nominal/ordinal com vocabulário; contínua. Sem nomes de datasets nem nº fixo de features. Testes (`tests/test_query_samplers.py`, 8 passam): binárias raras independentes **não** são confundidas com one-hot; derivadas recomputadas exactamente; restrições de caminho e de domínio respeitadas; determinismo; C/D não copiam linhas (<2 % de cópias, >450/500 únicas).

## 3. Resultados — 18 unidades DEV (média; iris, wine, breast_cancer ×5, digits ×3)
| Estratégia | AUC real-vs-sint. (orig.) | razão dist. NN | fora de intervalo | viol. algébrica | fidelidade substituto (holdout) | fração perto da fronteira (margem<0,2) | fronteira coberta | diversidade (razão NN) | s/1000 |
|---|---|---|---|---|---|---|---|---|---|
| A marginal histórica | 0,904 | 2,38 | 0,46 | **1,00** | 0,825 | 0,118 | 0,23 | 1,57 | 0,008 |
| As marginal+grupos | 0,904 | 2,38 | 0,46 | 0 | 0,858 | 0,118 | 0,23 | 1,57 | 0,017 |
| B marginal coerente | 0,904 | 2,38 | 0,46 | 0 | 0,858 | 0,118 | 0,23 | 1,57 | 0,016 |
| C conjunta empírica | **0,716** | 1,04 | 0,27 | 0 | **0,906** | 0,029 | **0,42** | 0,87 | 0,015 |
| D vizinhança | **0,683** | 0,38 | 0,00 | 0 | **0,906** | 0,013 | 0,40 | 0,23 | 0,017 |
Violações de one-hot/vocabulário/binárias: 0 em todas (os DEV não têm categóricas). A «violação algébrica 1,0» de A refere-se ao amostrador marginal *independente* no espaço enriquecido (referência Original-OWL, Doc. 21); o Reloaded histórico real já recalcula derivadas (violação 0, Doc. 21) — A,As,B coincidem nas colunas originais (mesma semente) por construção.

## 4. Resultados — dados mistos sintéticos (5 sementes; one-hot 3 e 4 níveis, 2 binárias, ordinal 0–4, nominal, 3 contínuas, 2 derivadas)
| Estratégia | AUC | viol. one-hot | viol. algébrica | fidelidade substituto | fronteira coberta | diversidade |
|---|---|---|---|---|---|---|
| A | 0,850 | **0,815** | 1,0 | 0,782 | 0,31 | 1,47 |
| As | 0,760 | 0 | 0 | 0,778 | 0,77 | 1,01 |
| B | 0,850 | **0,815** | 0 | 0,781 | 0,31 | 1,47 |
| C | 0,594 | 0 | 0 | 0,820 | 0,87 | 0,64 |
| D | 0,640 | 0 | 0 | 0,804 | 0,75 | 0,40 |
Vocabulário e binárias: 0 violações em todas. **Achado**: a amostragem marginal (inclusive a «coerente» existente B, que só recalcula derivadas) ativa em ~81 % das linhas mais de uma (ou nenhuma) categoria num grupo one-hot — a coerência atual só cobre relações algébricas, não exclusão mútua. As/C/D resolvem-no sem conhecer o dataset.

## 5. Leitura
1. **Diagnóstico**: em todos os 4 DEV, A/B ficam a AUC≈0,90 e a razão NN≈2,4 (consultas longe das linhas reais; 46 % fora do intervalo de treino em ≥1 coluna). C reduz AUC para 0,72 e D para 0,68, sem cópias de linhas.
2. **Trade-off medido (não escondido)**: a maior plausibilidade (C/D) acompanha-se de **menor fração de pontos junto à fronteira** (0,03/0,01 vs 0,12) mas **maior cobertura da fronteira** (0,42/0,40 vs 0,23) e fidelidade do substituto no holdout superior (0,906 vs 0,858/0,825). D tem **diversidade baixa (0,23)**: risco de sobreajuste/aglomeração local; C mantém diversidade 0,87.
3. **Custo**: equivalente (≈0,015–0,017 s/1000 consultas; ~2× A, desprezável face ao oráculo MLP).
4. **Aceitação sob restrição de caminho** semelhante (0,43–0,46).
5. Estas métricas medem um **substituto** (modelo simples treinado nas consultas) e não o efeito na árvore TREPAN: **não há evidência de ganho de fidelidade/accuracy do TREPAN**; é um critério de plausibilidade, não de superioridade.

## 6. Decisão de desenho e justificação
- **C** (conjunta empírica + perturbação) é a candidata de menor risco: pouco custo, sem cópia de linhas, preserva categorias/vocabulário/one-hot por construção, melhora plausibilidade e cobertura da fronteira, mantém diversidade. Seria um novo mecanismo **não ontológico** (melhora do amostrador independente da ontologia): só pode entrar no Original **e** no Reloaded de forma idêntica ou como variante do Reloaded atribuível à ontologia; para o contrato EQ v2.0 **não** é permitido apenas no Reloaded. Logo exige decisão do utilizador (§7).
- **D**: justificação fraca (diversidade 0,23; risco de sobreajuste; custo O(n·k)). Não recomendada sem prova de ganho em teste prévio.
- **E**: apenas o *wrapper* de predicados explícitos existe (testado: restrições respeitadas). Não há justificação formal para restrições de domínio derivadas da ontologia nos DEV (a ontologia não fornece exclusão mútua nem intervalos); fica **não proposta**.
- Nada foi ativado por defeito; nada foi ligado ao núcleo.

## 7. Testes propostos e resultados autorizados (pendentes de decisão)
1. Teste TREPAN com amostrador C **no Original e no Reloaded** (par equivalente) em DEV, com réplicas ≥3 (desvio entre réplicas 0,0166, Doc. 23) — para medir ΔF/ΔAcc das árvores, não só plausibilidade.
2. Datasets com categóricas reais (reservas) para validar tratamento de one-hot fora do sintético.
3. Pré-registo e budget: ≈ 1,5 CPU-h (iris/wine/bc ×5 × 3 réplicas × 2 braços + digits reduzido).
Resultados autorizados a reportar: métricas deste documento (exploratórias DEV). Nenhum resultado confirmatório existe.

## 8. Limitações
Sem categóricas reais nos DEV (apenas sintético misto); AUC depende do classificador; substituto ≠ TREPAN; cobertura da fronteira depende do limiar de margem (0,2/0,3 fixados antes); uma única divisão 75/25 por unidade.
