# Documento 17 — B1: pré-registo da análise (escrito ANTES de ver os resultados B1) e resultados

## 0. Pré-registo (data de escrita anterior à análise dos resultados das variantes B1; commit registado)
**Âmbito:** só datasets DEV (iris, wine, breast_cancer × seeds {42, 7, 123, 2024, 11}; digits × {42, 7, 123}); sem datasets confirmatórios; protocolo v1, MLP/FrozenOracle, ontologias e tuning estrutural ontology-blind inalterados; ambiente numérico fixado; referência pré-B1 congelada (`docs/evidence/b1b4/pre_b1_reference/`, código `db417ff`).
**Variantes:** `tiebreak` (τ = 1/√n_real); `bounded` e `calibrated` com `lambda_cap ∈ {0, 0.05, 0.10, 0.20, 0.40, 0.80}` (grelha D-8); cada variante com o seu **controlo negativo** (ontologia permutada, mesma calibração). Referência da comparação: (i) **Original** (braço C) — objetivo da hierarquia; (ii) **Reloaded+OWL histórico** (braço E) — para saber se B1 melhora o que já existe; (iii) o controlo permutado da própria variante — ganho atribuível ao *conhecimento* e não à calibração.
**Métricas (por unidade; relatadas por dataset e agregadas por dataset-média; unidade de inferência = dataset):** `Fidelity` ao FrozenOracle e `Accuracy` com rótulos reais no teste do split (relato; nunca para seleção), nós/folhas (complexidade), amostras totais ao oráculo e custo extra face ao Original, `b1_switch_rate`, `b1_regret_median`, `b1_ratio_m_over_s_median`, estado semântico. Margens: `δ_A = δ_F = 0` estritos; **distinguir** `F_R ≥ F_O` (não-inferioridade *observada*) de `F_R > F_O` (superioridade *observada*); empates contam como não-vitórias; reportar fração de empates. Todos os níveis de evidência são `observed_only` (sem inferência estatística formal com N = 4 datasets).
**Critérios de uma variante «a levar para B2» (todos obrigatórios; avaliados nos agregados por dataset, nunca num dataset isolado):**
1. **Não-inferioridade de fidelidade ao Original**: `mean ΔF(variante − Original) ≥ 0` em ≥ 3/4 datasets (se nenhuma o satisfaz, regista-se `NOT_ACHIEVED_ON_DEV`, sem ajustar critérios).
2. **Melhora o histórico**: `mean ΔF(variante − E) > 0` e `mean ΔAcc(variante − E) ≥ 0` em ≥ 3/4 datasets.
3. **Ganho atribuível ao conhecimento**: `mean ΔF(variante − controlo permutado) > 0` e `mean ΔAcc ≥ 0` em ≥ 3/4 datasets.
4. **Complexidade**: nós médios ≤ 1,10 × nós médios de E em cada dataset.
5. **Custo**: amostras totais ao oráculo ≤ as de E (em média por dataset).
6. **Reprodutibilidade/integridade**: com B1 desligado os braços A–F são idênticos à referência (`equivalence_report` PASS); `b1_legacy_logged ≡ E`; resultados repetíveis.
**Regras:** não se escolhe λ nem variante por um único dataset ou seed; se várias variantes satisfazem os critérios, ordena-se por (fidelidade média vs Original, depois accuracy, depois custo) e levam-se **todas as que passam** para B2 como candidatas; uma melhoria só se declara se **o efeito nas árvores e nas métricas** a sustentar — diminuir o bónus semântico (`b1_ratio_m_over_s_median`) não é, por si, evidência.
**Honestidade prévia:** a referência pré-B1 mostra que o braço E (OWL, histórico) tem fidelidade **inferior** à do Original em 4/4 datasets e o controlo permutado iguala ou supera E em fidelidade em 3/4; a hierarquia `F_R > F_O` **não** está satisfeita no ponto de partida. B1 só altera o *peso* do bónus (e não EFSR, consultas ativas, projetor nem representação), pelo que pode não ser suficiente; se assim for, reporta-se.

## 1. Referência pré-B1 (congelada)
Ver `docs/evidence/b1b4/pre_b1_reference/pre_b1_reference.json` (sha256 `75edc11048830f8e6da69cdec5bed7014a2187de7bc6f62633444028cc629ebf`; raw em `raw/`, `UNIT_SHA256.json` por unidade). Médias por dataset (5 seeds; digits 3):

| dataset | braço | Fidelity | Accuracy | nós |
|---|---|---|---|---|
| breast_cancer | Original (C) = Reloaded Core (D) | 0,9371 | 0,9329 | 32,2 |
| | E (OWL) | 0,9259 | 0,9357 | 30,6 |
| | F (OWL permutada) | 0,9273 | 0,9427 | 30,2 |
| | C4.5 | 0,9231 | 0,9301 | 17,0 |
| digits | C = D | 0,8215 | 0,8230 | 229,7 |
| | E | 0,8170 | 0,8230 | 231,7 |
| | F | 0,8348 | 0,8341 | 236,3 |
| | C4.5 | 0,8415 | 0,8489 | 165,0 |
| iris | C = D | 0,9632 | 0,9368 | 35,8 |
| | E | 0,9579 | 0,9316 | 36,2 |
| | F | 0,9526 | 0,9474 | 34,6 |
| | C4.5 | 0,9632 | 0,9579 | 8,6 |
| wine | C = D | 0,9200 | 0,9289 | 38,2 |
| | E | 0,8889 | 0,9022 | 41,0 |
| | F | 0,9244 | 0,9333 | 42,2 |
| | C4.5 | 0,9156 | 0,9289 | 9,8 |
Confirma-se no protocolo real que **D ≡ C** (Reloaded sem ontologia = Original: mesmos valores em todas as 18 unidades; contrato S0–S2).
