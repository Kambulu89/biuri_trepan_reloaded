# Doc. 30 — Resultados dos controlos ontológicos (campanha M29 + G29, DEV)

Estado: **EXECUTADA; análise exploratória; B3-a NÃO iniciado.** Protocolo e amendas: Doc. 29 (§15: A1–A3). Dados DEV (já observados, nunca confirmatórios); **nenhum dado confirmatório usado**; `frozen = false`. Resultados brutos, checkpoints, auditorias de controlos e relatórios em `docs/evidence/b1b4/m29_campaign/` (`main/`, `g29_pilot_patch/`, `report/`); tabelas geradas por `validation/benchmark/m29_report.py`, `m29_tables.py`, `m29_extra_stats.py` (reconstroem-se só dos brutos). Hash do relatório: `m29_report.json` sha256 `7e58b864e9008e5ede9357980f1d19085fcfae6b01e22b6a560a64e855440119`; ledger de CPU `6a9c08fb…14f5`.

## 1. Estado final da CI
Ramo `claude/repository-architecture-analysis-s1vl5a`. `0c4411b` e anteriores: *Dataset agnosticism guard* e *Windows scientific regression* `success`. Os commits do executor M29 (`030b547`, `0e68a00`, `89d57ee`) têm o guard `success`; o estado do Windows e do commit final desta entrega está registado na mensagem final (verificado via API pública do GitHub; não declarado verde antes de todos os fluxos terminarem). Sem merge para `main`.

## 2. Evidência de comparabilidade
- **18/18 unidades** com `check_comparability` estrito (11 propriedades: instâncias e ordem, rótulos, classes, partição, dataset, representação base, oráculo e previsões do oráculo no teste, versão, configuração, orçamento) para os braços A29–F29 e C4.5. Verificado **em tempo de execução** (por unidade, rc≠0 se falhar) e novamente no relatório.
- G29 isenta **apenas** `query_budget` (diferente por desenho, registado); restantes propriedades estritas. Isto falhou uma vez no arranque (wine:2024) por a exigência ser demasiado rígida para G29; corrigido e documentado (Doc. 29 §15), unidades concluídas reverificadas.
- **F29 ≡ A29 em 18/18** (igualdade exata: árvores, previsões, consultas), logo o contrato EQ sob o amostrador C mantém-se.
- Oráculo: `oracle_id`, hash de partição e de representação base por unidade em `manifest_v3.DRAFT.json::controls_m29.oracle_identities`; nenhum baseline alterado.

## 3. Custo efetivo
| registo | execuções | CPU real (s) | relógio (s) | CPU/relógio | CPU-h |
|---|---|---|---|---|---|
| dry2/cpu_ledger.jsonl | 18 | 641 | 565 | 1.13 | 0.178 |
| out/cpu_ledger.jsonl | 18 | 5769 | 5748 | 1.00 | 1.602 |
| out_g/cpu_ledger.jsonl | 2 | 28 | 27 | 1.03 | 0.008 |
| **TOTAL medido** |  | 6437 | 6340 | 1.02 | **1.788** |

Estimado antes: ≤2,5 CPU-h (limite autorizado). **Real: 1,788 CPU-h** (inclui dry-run v2 e piloto). Subestimámos wine/42 (unidade pesada) mas o total ficou abaixo. Rácios: consultas B29/A29 = 2,14×; tempo B29/A29 = 8,7× (o custo semântico é dominante).

## 4. Resultados por braço (todas as 18 unidades; sem arredondar antes de desigualdades estritas)
| braço | unid. | accuracy | fidelidade | precision_macro | macro-F1 | nós | folhas | profund. | m-of-n | splits semânticos | consultas | amostras do oráculo | tempo (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A29 Original+C | 18 | 0.9349 | 0.9327 | 0.9358 | 0.9335 | 69.6 | 35.3 | 8.7 | 28.9 | 0.0 | 42448.9 | — | 5.6 |
| B29 Reloaded OWL real+C | 18 | 0.9287 | 0.9229 | 0.9307 | 0.9268 | 68.8 | 34.9 | 8.8 | 32.1 | 33.9 | 90892.6 | 90892.6 | 49.1 |
| C29 permutada+C | 18 | 0.9232 | 0.9221 | 0.9260 | 0.9221 | 70.7 | 35.9 | 9.4 | 32.4 | 34.9 | 92679.6 | 92679.6 | 39.2 |
| D29 grupos aleatórios+C | 18 | 0.9310 | 0.9244 | 0.9332 | 0.9303 | 71.2 | 36.1 | 9.1 | 32.4 | 35.1 | 93022.6 | 93022.6 | 38.9 |
| E29 convencional+C | 18 | 0.9395 | 0.9397 | 0.9408 | 0.9379 | 68.9 | 35.0 | 9.0 | 28.0 | 0.0 | 88971.6 | 88971.6 | 6.4 |
| F29 Reloaded sem OWL+C | 18 | 0.9349 | 0.9327 | 0.9358 | 0.9335 | 69.6 | 35.3 | 8.7 | 28.9 | 0.0 | 42448.9 | 42448.9 | 5.8 |
| G29 Original+C (orçamento igualado) | 18 | 0.9355 | 0.9329 | 0.9363 | 0.9343 | 70.6 | 35.8 | 8.9 | 28.9 | 0.0 | 95181.8 | — | 6.0 |

Métrica principal: accuracy (`accuracy_real_labels`); fidelidade a `fidelity_to_oracle`; precision_macro e macro-F1 complementares (nunca usadas para declarar a hierarquia). C4.5 (referência): accuracy 0,9239, fidelidade 0,9185, precision_macro 0,9251, macro-F1 0,9223.

## 5. Por dataset e réplica
| dataset | braço | unid. | accuracy | DP acc. | fidelidade | DP fid. | nós | consultas |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | A29 Original+C | 5 | 0.9371 | 0.0068 | 0.9347 | 0.0166 | 35.9 | 17151.5 |
| breast_cancer | B29 Reloaded OWL real+C | 5 | 0.9357 | 0.0094 | 0.9315 | 0.0205 | 33.8 | 40075.3 |
| breast_cancer | C29 permutada+C | 5 | 0.9385 | 0.0093 | 0.9324 | 0.0064 | 34.9 | 40722.1 |
| breast_cancer | D29 grupos aleatórios+C | 5 | 0.9431 | 0.0142 | 0.9352 | 0.0117 | 34.9 | 40172.3 |
| breast_cancer | E29 convencional+C | 5 | 0.9441 | 0.0159 | 0.9380 | 0.0175 | 34.7 | 38026.0 |
| breast_cancer | F29 Reloaded sem OWL+C | 5 | 0.9371 | 0.0068 | 0.9347 | 0.0166 | 35.9 | 17151.5 |
| breast_cancer | G29 Original+C (orçamento igualado) | 5 | 0.9361 | 0.0063 | 0.9385 | 0.0160 | 34.9 | 43797.1 |
| digits | A29 Original+C | 3 | 0.8741 | 0.0139 | 0.8647 | 0.0139 | 251.2 | 190778.8 |
| digits | B29 Reloaded OWL real+C | 3 | 0.8640 | 0.0066 | 0.8612 | 0.0101 | 253.0 | 405871.3 |
| digits | C29 permutada+C | 3 | 0.8694 | 0.0041 | 0.8607 | 0.0045 | 253.2 | 405999.7 |
| digits | D29 grupos aleatórios+C | 3 | 0.8654 | 0.0160 | 0.8620 | 0.0148 | 252.3 | 406952.9 |
| digits | E29 convencional+C | 3 | 0.8820 | 0.0179 | 0.8763 | 0.0172 | 252.3 | 397874.3 |
| digits | F29 Reloaded sem OWL+C | 3 | 0.8741 | 0.0139 | 0.8647 | 0.0139 | 251.2 | 190778.8 |
| digits | G29 Original+C (orçamento igualado) | 3 | 0.8795 | 0.0089 | 0.8669 | 0.0115 | 253.9 | 413776.3 |
| iris | A29 Original+C | 5 | 0.9561 | 0.0206 | 0.9649 | 0.0356 | 35.0 | 10449.2 |
| iris | B29 Reloaded OWL real+C | 5 | 0.9544 | 0.0190 | 0.9526 | 0.0423 | 35.7 | 23132.2 |
| iris | C29 permutada+C | 5 | 0.9421 | 0.0202 | 0.9596 | 0.0288 | 35.3 | 23627.4 |
| iris | D29 grupos aleatórios+C | 5 | 0.9439 | 0.0133 | 0.9526 | 0.0414 | 34.6 | 23803.5 |
| iris | E29 convencional+C | 5 | 0.9491 | 0.0235 | 0.9649 | 0.0310 | 34.2 | 22870.7 |
| iris | F29 Reloaded sem OWL+C | 5 | 0.9561 | 0.0206 | 0.9649 | 0.0356 | 35.0 | 10449.2 |
| iris | G29 Original+C (orçamento igualado) | 5 | 0.9544 | 0.0306 | 0.9667 | 0.0325 | 35.0 | 23564.8 |
| wine | A29 Original+C | 5 | 0.9481 | 0.0181 | 0.9393 | 0.0353 | 29.0 | 10747.9 |
| wine | B29 Reloaded OWL real+C | 5 | 0.9348 | 0.0274 | 0.9215 | 0.0220 | 26.3 | 20483.0 |
| wine | C29 permutada+C | 5 | 0.9215 | 0.0479 | 0.9111 | 0.0505 | 32.6 | 25697.1 |
| wine | D29 grupos aleatórios+C | 5 | 0.9452 | 0.0171 | 0.9230 | 0.0299 | 35.4 | 26734.0 |
| wine | E29 convencional+C | 5 | 0.9600 | 0.0112 | 0.9541 | 0.0242 | 27.8 | 20676.3 |
| wine | F29 Reloaded sem OWL+C | 5 | 0.9481 | 0.0181 | 0.9393 | 0.0353 | 29.0 | 10747.9 |
| wine | G29 Original+C (orçamento igualado) | 5 | 0.9496 | 0.0062 | 0.9333 | 0.0228 | 32.1 | 27026.6 |

| dataset | k | A29 | B29 | C29 | D29 | E29 | F29 | G29 |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0 | 0.9427 / 0.9357 | 0.9315 / 0.9329 | 0.9469 / 0.9399 | 0.9455 / 0.9413 | 0.9497 / 0.9483 | 0.9427 / 0.9357 | 0.9357 / 0.9315 |
| breast_cancer | 1 | 0.9441 / 0.9399 | 0.9385 / 0.9287 | 0.9413 / 0.9315 | 0.9399 / 0.9301 | 0.9343 / 0.9301 | 0.9441 / 0.9399 | 0.9497 / 0.9427 |
| breast_cancer | 2 | 0.9245 / 0.9287 | 0.9371 / 0.9329 | 0.9273 / 0.9259 | 0.9441 / 0.9343 | 0.9483 / 0.9357 | 0.9245 / 0.9287 | 0.9231 / 0.9413 |
| digits | 0 | 0.8785 / 0.8704 | 0.8644 / 0.8652 | 0.8756 / 0.8630 | 0.8815 / 0.8800 | 0.8800 / 0.8741 | 0.8785 / 0.8704 | 0.8807 / 0.8681 |
| digits | 1 | 0.8778 / 0.8704 | 0.8615 / 0.8570 | 0.8681 / 0.8644 | 0.8585 / 0.8526 | 0.8837 / 0.8770 | 0.8778 / 0.8704 | 0.8696 / 0.8526 |
| digits | 2 | 0.8659 / 0.8533 | 0.8659 / 0.8615 | 0.8644 / 0.8548 | 0.8563 / 0.8533 | 0.8822 / 0.8778 | 0.8659 / 0.8533 | 0.8881 / 0.8800 |
| iris | 0 | 0.9632 / 0.9684 | 0.9579 / 0.9526 | 0.9526 / 0.9684 | 0.9526 / 0.9684 | 0.9526 / 0.9579 | 0.9632 / 0.9684 | 0.9526 / 0.9684 |
| iris | 1 | 0.9474 / 0.9632 | 0.9526 / 0.9579 | 0.9263 / 0.9632 | 0.9368 / 0.9421 | 0.9368 / 0.9737 | 0.9474 / 0.9632 | 0.9526 / 0.9684 |
| iris | 2 | 0.9579 / 0.9632 | 0.9526 / 0.9474 | 0.9474 / 0.9474 | 0.9421 / 0.9474 | 0.9579 / 0.9632 | 0.9579 / 0.9632 | 0.9579 / 0.9632 |
| wine | 0 | 0.9511 / 0.9378 | 0.9289 / 0.9156 | 0.9244 / 0.9111 | 0.9600 / 0.9289 | 0.9600 / 0.9600 | 0.9511 / 0.9378 | 0.9600 / 0.9467 |
| wine | 1 | 0.9467 / 0.9378 | 0.9200 / 0.9067 | 0.9244 / 0.9200 | 0.9200 / 0.9067 | 0.9644 / 0.9511 | 0.9467 / 0.9378 | 0.9556 / 0.9333 |
| wine | 2 | 0.9467 / 0.9422 | 0.9556 / 0.9422 | 0.9156 / 0.9022 | 0.9556 / 0.9333 | 0.9556 / 0.9511 | 0.9467 / 0.9422 | 0.9333 / 0.9200 |

## 6. Contrastes G29 e restantes (B29 − braço)
| contraste | Δ accuracy (datasets) | Δ fidelidade (datasets) | Δ accuracy (unidades) | Δ fidelidade (unidades) | p_IUT (datasets) | p_IUT Holm |
|---|---|---|---|---|---|---|
| B29-A29 | -0.0067 (EP 0.0030) | -0.0092 (EP 0.0036) | -0.0063 (EP 0.0053) | -0.0098 (EP 0.0058) | 1.0000 | 1.0000 |
| B29-G29 | -0.0077 (EP 0.0043) | -0.0096 (EP 0.0020) | -0.0068 (EP 0.0051) | -0.0101 (EP 0.0049) | 1.0000 | 1.0000 |
| B29-C29 | +0.0043 (EP 0.0049) | +0.0007 (EP 0.0036) | +0.0054 (EP 0.0069) | +0.0008 (EP 0.0069) | 0.8527 | 1.0000 |
| B29-D29 | -0.0022 (EP 0.0046) | -0.0015 (EP 0.0008) | -0.0023 (EP 0.0052) | -0.0016 (EP 0.0053) | 1.0000 | 1.0000 |
| B29-E29 | -0.0116 (EP 0.0066) | -0.0166 (EP 0.0056) | -0.0109 (EP 0.0056) | -0.0168 (EP 0.0063) | 1.0000 | 1.0000 |

G29 (Original com orçamento de consultas igualado): orçamento igualado em 100% das unidades de Iris/Breast Cancer/Digits e em 53% de Wine (rácio médio de consultas ao B29 em wine 1,29, **confusão assimétrica registada**; wine 1,29 ∉ [0,8; 1,25] ⇒ essas unidades são `confundidas`, não comparáveis). **G29 ≈ A29** (acc 0,9355 vs 0,9349): mais consultas, só por si, não melhoram; logo o défice de B29 vs A29 não se explica por orçamento.

## 7. Controlos de permutação
Validade (portão: dimensão, finitude, escala ≤100×, retenção de pares ≤ acaso): Iris 30/30, Wine 30/30, Digits 18/18 controlos **completos**; reten­ção média vs acaso: iris 0,00 vs 0,33; wine 0,245 vs 0,295; digits 0,327 vs 0,333. **Breast Cancer: 29/30 `structure_only_fallback`** (não satisfazem o controlo exigido; não se apresentam como permutação equivalente). Contraste B29−C29 (nível dataset, n=4): Δacc +0,0043, Δfid +0,0007; B29−D29: −0,0022 / −0,0015. **Real ≈ permutado ≈ aleatório**, dentro do ruído.

## 8. Limites de Breast Cancer
As features derivadas (contrastes e rácios entre famílias) não se podem permutar num controlo comparável: 107 tentativas falharam por dimensão derivada e as restantes por escala até ~4×10¹¹ (vs ≤100× permitido). Apenas 1/30 controlo completo. Atribuição ontológica em Breast Cancer: `NOT_ASSESSABLE`; mantém-se no relatório global (percentagens sobre as 18 unidades) e o fallback é rotulado, nunca tratado como equivalente.

## 9. Limites de E29
E29 (engenharia convencional) tem dimensão derivada muito inferior à OWL real em Breast Cancer (5,6 vs 53), e menor em Iris (1 vs 2), Wine (2,2 vs 3), Digits (2,67 vs 3). Regra pré-registada (diferença >25% ⇒ `confundido`): **E29 confundido em 16/18 unidades** (T9); sem colunas artificiais. Contrastes com E29 ⇒ `NOT_ASSESSABLE`. Correção divulgada: a primeira versão do relatório lia a dimensão efetiva do contexto real (campo errado); corrigida antes da análise final (injeção de `n_derived_effective` a partir das auditorias). Nota descritiva (não conclusiva): E29 é o melhor braço (acc 0,9395, fid 0,9397) — melhor do que Original e Reloaded; sem OWL, apenas features convencionais.
| controlo | comparável | confundido |
|---|---|---|
| C29 permutada+C | 17 | 1 |
| D29 grupos aleatórios+C | 17 | 1 |
| E29 convencional+C | 2 | 16 |
| G29 Original+C (orçamento igualado) | 17 | 1 |

## 10. Ganho ontológico direto (Nível 1)
| Nível 1: ganho direto | n | % |
|---|---|---|
| NO_GAIN | 10 | 55.6% |
| TRADEOFF | 4 | 22.2% |
| ONTOLOGICAL_JOINT_GAIN | 2 | 11.1% |
| EQUIVALENT | 2 | 11.1% |

| atribuição ontológica (completa) | n |
|---|---|
| NOT_ATTRIBUTED | 16 |
| NOT_ASSESSABLE | 2 |

| atribuição sem E29 (secundária, não conclusiva) | n |
|---|---|
| NOT_ATTRIBUTED | 17 |
| NOT_ASSESSABLE | 1 |

| Nível 2: hierarquia global | n | % |
|---|---|---|
| NOT_SATISFIED | 7 | 38.9% |
| HIERARCHY_INFEASIBLE | 9 | 50.0% |
| TRADEOFF | 1 | 5.6% |
| JOINT_HIERARCHY_SATISFIED | 1 | 5.6% |

Duas unidades com ganho conjunto direto vs Original equivalente: **digits:42** (Δacc +0,0096, Δfid +0,0178) e **wine:42** (Δacc +0,0296, Δfid +0,0074). Em ambas a atribuição é `NOT_ASSESSABLE` (E29 confundido). Regressões B29 vs A29 por unidade: accuracy desce em 9/18 (sobe em 4), fidelidade desce em 12/18 (sobe em 4), ambas descem em 7 e sobem em 2. Nível dataset: Δacc −0,0067 (IC95 bootstrap [−0,0117; −0,0016]), Δfid −0,0092 ([−0,015; −0,0034]) ⇒ **B29 pior do que A29** (nível `NOT_SATISFIED`). Não se escondem regressões.

## 11. Evidência de atribuição ontológica
| contraste | nível de evidência | Δ accuracy | IC95 bootstrap | Δ fidelidade | IC95 bootstrap |
|---|---|---|---|---|---|
| B29 vs A29 (Original equivalente) | NOT_SATISFIED | -0.0067 | [-0.0117, -0.0016] | -0.0092 | [-0.015, -0.0034] |
| B29 vs controlo permuted_ontology | OBSERVED_JOINT_GAIN | 0.0091 | [-0.0054, 0.0204] | 0.0034 | [-0.007, 0.0167] |
| B29 vs controlo random_groups | NOT_SATISFIED | -0.0001 | [-0.0093, 0.0105] | -0.0009 | [-0.0019, 0.0] |
| B29 vs controlo budget_matched_original | NOT_SATISFIED | -0.0082 | [-0.0161, -0.0002] | -0.0099 | [-0.0135, -0.0063] |

Atribuição: **NOT_ATTRIBUTED 16, NOT_ASSESSABLE 2** (sem E29: 17/1). Vs permutado, o nível dataset marca `OBSERVED_JOINT_GAIN` (Δacc +0,0091, Δfid +0,0034) mas com ICs que incluem 0 e **sem B29 ≥ A29**, pelo que não constitui ganho ontológico (a cadeia exige superar o Original equivalente). Nenhum `SUPPORTED_*` (4 datasets, DEV).
| braço | splits influenciados | decisões alteradas | regras semânticas | features semânticas usadas | splits semânticos | m-of-n | nós internos |
|---|---|---|---|---|---|---|---|
| A29 Original+C | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28.9 | 34.3 |
| B29 Reloaded OWL real+C | 32.8 | 10.1 | 34.9 | 3.2 | 33.9 | 32.1 | 33.9 |
| C29 permutada+C | 33.0 | 9.7 | 35.9 | 3.2 | 34.9 | 32.4 | 34.9 |
| D29 grupos aleatórios+C | 33.2 | 9.6 | 36.1 | 3.2 | 35.1 | 32.4 | 35.1 |
| E29 convencional+C | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28.0 | 34.0 |
| F29 Reloaded sem OWL+C | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28.9 | 34.3 |
| G29 Original+C (orçamento igualado) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28.9 | 34.8 |

Os mecanismos semânticos executaram (≈34 splits semânticos/árvore; ≈10 decisões alteradas) com a OWL real, permutada e aleatória em paridade — a influência existe mas não depende da verdade da ontologia.

## 12. Hierarquia global (Nível 2, todas as unidades)
Ver tabela de Nível 2 em §10: **JOINT_HIERARCHY_SATISFIED 1/18 (5,6%, digits:42)**, TRADEOFF 1, NOT_SATISFIED 7, **HIERARCHY_INFEASIBLE 9 (50%: Original ≤ C4.5 na unidade; reportado, não corrigido, nenhuma excluída)**. Isto é Nível 2 e não substitui o 1.

## 13. Incerteza e limitações
Só 4 datasets (inferência ao nível dataset, n=4: poder muito baixo; Holm/IUT exploratório); réplicas/draws médios dentro da unidade; DP entre sementes mestras nas tabelas T4 (0,004–0,048 em accuracy), da ordem das diferenças observadas. DEV observado (nunca confirmatório). Ontologias DEV só induzem partições em grupos (relatedness constante 0,7124): pouco conteúdo semântico para além de agrupamento. Wine/G29 parcialmente confundido. Breast Cancer sem controlo permutado válido. E29 confundido. Nenhuma superioridade declarada; estado `observed_only`.

## 14. Recomendação sobre B3-a
**Não implementar B3-a tal como desenhado (restrições sobre grupos).** Os controlos mostram real ≈ permutado ≈ aleatório e B29 < A29; restrições adicionais baseadas nos mesmos grupos não têm fundamento empírico. Falta o controlo "Original sobre a mesma representação enriquecida sob C" (separaria o efeito da representação do do mecanismo semântico). B3-a, B2-A/B e B4 **não** foram implementados.

## 15. Próximas decisões científicas
1. Rever a estratégia de enriquecimento: conhecimento semântico **mais expressivo** do que grupos de atributos (hierarquias reais de conceitos, restrições de domínio/disjunção, relações entre classes e atributos, axiomas que impliquem regras) — cada mecanismo com o mesmo conjunto de controlos (permutado/aleatório/convencional/orçamento).
2. Acrescentar o controlo «Original com a mesma representação enriquecida» e exigir que E29 e a OWL tenham dimensão comparável (ou reportar `NOT_ASSESSABLE`).
3. Plano curto de datasets categóricos (sem confirmatórios): usar German Credit e sintéticos já preparados só como DEV adicional para testar ontologias com relações não triviais; adiado até aprovação (German/42 não autorizado).
4. Decidir se se mantém a hierarquia conjunta como objetivo à luz de 9/18 unidades infeasíveis (Original ≤ C4.5): reportar sempre sobre todas.
5. Aguardar autorização antes de B3-a.

## 16. Errata de leitura (B3.0; números e classificações inalterados)
Registadas na auditoria B3.0 (Doc. 31 §3.4): (1) em breast_cancer a atribuição reportada é `NOT_ATTRIBUTED` (sem ganho direto), não `NOT_ASSESSABLE` (§8 deve ler-se: *seria* `NOT_ASSESSABLE` se houvesse ganho direto, pois o permutado só existe em fallback e E29 está confundido); (2) a confusão de orçamento de G29 aplica-se pela regra assimétrica pré-registada ao nível da média da unidade (1/18 unidades; wine:123), sendo 8/15 as réplicas de wine em [0,8; 1,25]; (3) a cópia de evidência passou a incluir os 18 `split.json`. Integridade: 302/302 hashes e relatório regenerado idêntico (Doc. 31 §1).
