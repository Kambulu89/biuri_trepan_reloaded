# Documento 16 — Preparação de B1 (pontos de código, escalas, instrumentação) e estudo do desempate (plano)

## 1. Contrato EQ v2.0 (resumo e estado)
Ver Doc. 07 §9. S0/S1/S2 ≡ Original (E1–E9 integrais); S3 = porção Original ≡ Original isolado + tentativa semântica auditada (`oracle_accounting_`). **Estado: COMPLIANT** com CI verde em `db417ff` e `0e3cf27`.

## 2. Pontos exatos do código que implementam o bónus semântico (`core/trepan_reloaded_historical.py`)
| Função | Papel | Fórmula |
|---|---|---|
| `_split_selection_score` | score de cada candidato (literais e m-of-n) | `s·w^k + α·OntoDepth + β·ErrorCoverage`, `s = IG/H(nó)` (por omissão), 0 se `IG ≤ 1e-12` |
| `_normalized_information_gain` | componente estatística | `IG/H(y do nó)` ∈ [0,1] |
| `_semantic_weight_for_test` / `_semantic_group_coherence_factor` | peso `w` do teste | média geométrica dos pesos dos literais × `(1 + 0,15·coerência de grupo/relação)` |
| `_onto_depth_for_test`, `_error_coverage_for_test` | termos aditivos | médias por literal de vetores por feature (profundidade OWL; erro local do nó) |
| `_best_mofn` / `_ontology_guided_candidates` | procura e candidatos semânticos (≤ 24) | score acima; candidatos por grupos/relações; gate EFSR compara com o vencedor *data-only* |
| `_evaluate_error_focused_intervention` | gate local/real (não é score) | ganho de fidelidade ≥ 0,002 ou coerência dentro de tolerância 0,015 |
O score histórico é **exatamente** `s + m` com `m = s·(w^k − 1) + α·OntoDepth + β·ErrorCoverage` (verificado por teste, tol 1e-12). A busca *data-only* de referência usa IG cru (bits), o score semântico usa escala normalizada: escalas diferentes por desenho; o gate usa fidelidade local, não scores.

## 3. Escalas reais e relação semântico/estatístico
* Auditoria anterior (Doc. 01 §3.1, 30 nós, 1 ontologia real): razão `α·OD+β·EC` / `s·w^k`: mediana 0,83 (0,51–9,09); o termo aditivo excede o estatístico em 40% dos nós; `OntoDepth` saturado a 1,0; o efeito na **escolha** vem de `ErrorCoverage`, `peso` e coerência de grupo.
* Novo (instrumentação `candidate_log_summary`, agora por candidato, não só vencedor): ver resultados DEV no Doc. 17 (`b1_ratio_m_over_s_median`, `b1_switch_rate`, `b1_regret_*`).
* Propriedades de B1: `bounded`: `|bónus| ≤ λ·s` ⇒ desvio estatístico máximo por nó `2λ/(1+λ)` (0; 9,5%; 18,2%; 33,3%; 57,1%; 88,9% para λ = 0; 0,05; 0,10; 0,20; 0,40; 0,80); `calibrated`: mesma cota com saturação suave; `tiebreak`: `|bónus| ≤ τ = 1/√n_real` (só reordena candidatos indistinguíveis em ±2τ; `n_real` = linhas reais do nó).

## 4. Instrumentação de todos os candidatos
`log_split_candidates` (desligado por omissão; sem efeito em decisões, consultas ou RNG — teste dedicado): por candidato `node_id, m, n, literais, IG cru, componente estatística, peso, OntoDepth, ErrorCoverage, score, variante, λ, fase`. `candidate_log_summary()` agrega por nó: `stat_top`, `winner_stat`, `regret=(stat_top−winner)/stat_top`, `switched`, mediana de `(score−s)/s`.

## 5. Estudo separado de uma política determinística de desempate (plano; Original canónico NÃO modificado)
Decisão: não alterar o TREPAN Original; comportamento dependente de ruído numérico = **limitação de reprodutibilidade** documentada (Doc. 15 §7: `IG(máscara) ≠ IG(~máscara)` ao bit em 56% dos casos; sonda com score exatamente simétrico altera regras em 57/120 ajustes do Original). O protocolo confirmatório usa **ambiente numérico congelado** (`run.py`: `OPENBLAS_CORETYPE`, threads, SIMD) e **não** compara execuções feitas em ambientes numéricos diferentes como se o determinismo estivesse garantido.
Estudo (variante experimental isolada, subclasse/flag fora do Original):
1. Equivalência lógica entre complementares: `m-of-n(L) ≡ (n−m+1)-of-n(¬L)` (provado; 1332 testes aleatórios).
2. Ordenação canónica de candidatos: `(−score_q, nº literais, m, orientação canónica, feature, limiar, greater)`.
3. Tolerâncias: `score_q = round(score, 12)`; sensibilidade a 1e-9…1e-15.
4. Empates estatísticos: tolerância baseada em `n_real` (relação com o `tiebreak` de B1).
5. Preservação das partições (invariante: mesma partição, ramos trocados) e das regras exportadas.
6. Impacto nas árvores: % de execuções com regras/previsões diferentes vs Original canónico (quantifica o custo de qualquer política).
7. Estabilidade entre plataformas: injeção de ruído ±1 ulp nos scores (propriedade) e comparação Linux/Windows em CI.
8. RNG e consultas: a política não pode consumir RNG nem alterar o fluxo ao oráculo (teste de espião).
Condição de avanço: relatório com os 8 pontos antes de qualquer proposta de alteração ao Original (que exigiria autorização e novo manifest).

## 6. Referência pré-B1 (congelamento)
Código: commit `db417ff` (contrato v2.0; B1 inexistente). Unidades DEV iris, wine, breast_cancer × {42, 7, 123, 2024, 11} e digits × {42, 7, 123}, protocolo v1 inalterado, ambiente numérico fixado. Hashes, `oracle_id`, configuração estrutural selecionada e métricas por braço: ver Doc. 17 §1 (preenchido quando a referência terminar).
