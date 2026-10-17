# Doc. 36 — Identificabilidade ontológica (B3.1)

Estado: **AVALIAÇÃO SINTÉTICA CONCLUÍDA (v1 e v2); `core/` NÃO alterado; nenhuma campanha DEV nem dado confirmatório.** Código: `validation/benchmark/b31_identifiability.py` (problemas, variantes, calibração, congelamento, avaliador), `b31_steering.py` (arnês experimental), `b31_identifiability_report.py`; testes: `tests/test_b31_steering_identifiability.py`. Evidência: `docs/evidence/b1b4/b31/` (`problems_frozen.json`, `problems_frozen_v2.json`, `identifiability_results*.json`, `identifiability_analysis*.json/.md`, `trace_example_v2.json`, `cost_probe.json`, `v1_patch_equivalence_check.json`).

**O que esta fase mede e o que NÃO mede.** Testa se o mecanismo consegue **identificar e usar** uma relação semântica verdadeira (e distingui-la de relações permutadas, aleatórias, irrelevantes, incompletas, ambíguas, inconsistentes ou mal casadas) num problema de verdade conhecida. **Não** demonstra superioridade do Reloaded sobre o Original/C4.5 em datasets reais, nem ganho causal em generalização.

## 1. Desenho
### 1.1 Problemas (congelados)
Dados gaussianos com escalas de atributo heterogéneas (σ_log = 0,7), 32 atributos em 8 conceitos de 4 (nomes descritivos `topico_medida`), **ordem de colunas aleatória por semente**. Oráculo de forma fechada («MLP» substituto): ≥ 2 de 4 atributos da regra positivos.
- **P1 (regra dentro de um conceito):** os 4 atributos da regra são o conceito `g*` (escolhido aleatoriamente por semente).
- **P2 (regra atravessa conceitos):** 2 atributos de cada um de dois conceitos.
Orçamento de pesquisa do Original limitado por `max_features_per_node` (só considera os atributos de maior variância e um conjunto de literais de `3·mf`); **a dificuldade foi calibrada só com o Original**, em 30 sementes de calibração (1000–1029) disjuntas das de avaliação, escolhendo o `mf` cuja taxa de recuperação está mais perto de 0,5 em [0,2; 0,8]:
| `mf` | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|
| P1: recuperação do Original | 0,00 | 0,00 | 0,03 | 0,03 | 0,17 | **0,47** |
| P2: recuperação do Original | 0,00 | 0,00 | 0,00 | 0,07 | 0,33 | **0,53** |
Escolhido `mf = 28` nos dois (o Original **não** recupera a regra em todas as execuções). Recuperação = existe um nó interno cujo teste m-of-n usa **exatamente** os atributos da regra. Fidelidade = concordância com o oráculo em 4000 linhas independentes nunca usadas no ajuste.
### 1.2 Variantes (todas com o MESMO oráculo, dados, orçamento de consultas e semente do Original)
| variante | ontologia / controlo | propósito |
|---|---|---|
| ORIG | TREPAN Original canónico | referência |
| WIDE | Original com `mf = p` (sem ontologia) | efeito do orçamento de atributos (controlo não ontológico) |
| CONV | unidades por agrupamento hierárquico de \|Spearman\| sobre o treino (sem ontologia) | conhecimento vs correlação estatística |
| TRUE | conceitos verdadeiros (8 grupos de 4) | relação útil disponível |
| HIERARCHY | TRUE + pai informativo comum dos conceitos da regra (pares de conceitos sob pais) | hierarquia verdadeira relevante |
| PERM / RAND | partição permutada / aleatória de tamanhos iguais, **garantindo** que a regra não fica num só conceito | destrói a relação, mesma forma |
| EMPTY | ontologia válida sem axiomas úteis (propriedades sem domínio) | abstenção ≡ Original |
| IRRELEVANT | verdadeira mas sobre outros atributos; os da regra em conceitos singulares | verdadeira e irrelevante |
| INCOMPLETE | um atributo da regra omitido do seu conceito | ontologia incompleta |
| NOISY_SUPERSET | conceito da regra + 2 atributos espúrios | verdade com ruído |
| WRONG_SPLIT | atributos da regra divididos 2+2 por dois conceitos com intrusos | relação semanticamente errada |
| AMBIGUOUS | duas propriedades quase homónimas por atributo da regra | matching ambíguo |
| INCONSISTENT | classe insatisfazível | ontologia inconsistente |
| BADMATCH | nomes sem relação lexical com os atributos | matching insuficiente |
Em P2 os mesmos ficheiros servem de «ontologia válida sem vantagem possível» (TRUE: a regra atravessa conceitos) e HIERARCHY (pai comum dos dois conceitos da regra).
### 1.3 Mecanismo (arnês experimental, `SemanticSteeredTrepan`)
Subclasse do Original (não modificado). O Original corre como sempre; para as 3 unidades semânticas mais promissoras (ordenadas pelo melhor ganho de um literal isolado **dentro** da unidade, usando os dados do nó, nunca o teste) corre-se a **mesma** pesquisa restrita aos atributos da unidade; escolhe-se o candidato de maior `_split_selection_score` **original** (estatístico; empate ⇒ Original). A ontologia só decide **onde** procurar: não altera pontuação, não elimina nada. Sem unidades ⇒ idêntico ao Original (testado com oráculo-espião: regras, previsões, consultas, fluxo ao oráculo).
### 1.4 Congelamento
`problems_frozen.json` (v1) guarda problemas, variantes, critérios, sementes (0–39), calibração e SHA-256 do código; o avaliador **recusa** avaliar se o código mudou (`--allow-drift`). Foi *commitado* antes de qualquer avaliação (`425675d`). **Critérios pré-registados:** P1 — C1 TRUE−ORIG ≥ 0,20; C2 TRUE−PERM ≥ 0,20; C3 TRUE−RAND ≥ 0,20; C4 TRUE−CONV ≥ 0,20; C5 EMPTY/INCONSISTENT/BADMATCH = árvore do Original em 100 %; C6 |AMBIGUOUS−ORIG| ≤ 0,10; C7 IRRELEVANT/INCOMPLETE/WRONG_SPLIT − ORIG ≤ +0,10; C8 dano limitado (recuperação ≥ ORIG−0,20; fidelidade ≥ ORIG−0,02); C9 NOISY_SUPERSET reportado. P2 — D1 TRUE−ORIG ≤ 0,10; D2 HIERARCHY−ORIG ≥ 0,20; D3 PERM−ORIG ≤ 0,10. Critérios técnicos do arnês; **não** são evidência de superioridade.

## 2. Resultados v1 (sementes 0–39; congelamento `425675d`)
### P1_within_group (n = 40 sementes de avaliação)

| variante | recuperação [IC95 Wilson] | Δ vs ORIG | McNemar (b,c; p) | fidelidade teste (média ± DP) | Δ fid vs ORIG | decisões alteradas/árvore | árvores idênticas a ORIG | tempo/ORIG |
|---|---|---|---|---|---|---|---|---|
| ORIG | 0.650 [0.50; 0.78] |  |  | 0.9194 ± 0.0867 |  | 0.00 | 40/40 | 1.00 |
| WIDE | 1.000 [0.91; 1.00] | +0.350 | (14,0); 0.000 | 0.9793 ± 0.0123 | +0.0599 | 0.00 | 22/40 | 0.94 |
| CONV | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9657 ± 0.0431 | +0.0463 | 0.42 | 26/40 | 1.26 |
| TRUE | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9749 ± 0.0211 | +0.0555 | 0.55 | 26/40 | 3.15 |
| HIERARCHY | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9749 ± 0.0211 | +0.0555 | 0.55 | 26/40 | 3.32 |
| PERM | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9687 ± 0.0397 | +0.0493 | 0.42 | 26/40 | 3.22 |
| RAND | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9660 ± 0.0495 | +0.0466 | 0.40 | 26/40 | 3.09 |
| EMPTY | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9194 ± 0.0867 | +0.0000 | 0.00 | 40/40 | 2.87 |
| IRRELEVANT | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9194 ± 0.0867 | +0.0000 | 0.00 | 40/40 | 3.23 |
| INCOMPLETE | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9647 ± 0.0465 | +0.0453 | 0.47 | 26/40 | 3.11 |
| NOISY_SUPERSET | 0.725 [0.57; 0.84] | +0.075 | (3,0); 0.250 | 0.9742 ± 0.0223 | +0.0548 | 0.60 | 25/40 | 3.02 |
| WRONG_SPLIT | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9730 ± 0.0264 | +0.0536 | 0.45 | 26/40 | 3.11 |
| AMBIGUOUS | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9194 ± 0.0867 | +0.0000 | 0.00 | 40/40 | 3.33 |
| INCONSISTENT | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9194 ± 0.0867 | +0.0000 | 0.00 | 40/40 | 2.96 |
| BADMATCH | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9194 ± 0.0867 | +0.0000 | 0.00 | 40/40 | 2.90 |

Critérios pré-registados:

- C1_true_beats_orig: **NÃO CUMPRIDO**
- C2_true_beats_perm: **NÃO CUMPRIDO**
- C3_true_beats_rand: **NÃO CUMPRIDO**
- C4_true_beats_conventional: **NÃO CUMPRIDO**
- C5_abstention_equals_original: **CUMPRIDO**
- C6_ambiguous_no_spurious_gain: **CUMPRIDO**
- C7_no_spurious_advantage: **CUMPRIDO**
- C8_bounded_harm: **CUMPRIDO**
- C9_noisy_superset_reported: **{'recovery': 0.725, 'delta_vs_orig': 0.07499999999999996}**

### P2_cross_group (n = 40 sementes de avaliação)

| variante | recuperação [IC95 Wilson] | Δ vs ORIG | McNemar (b,c; p) | fidelidade teste (média ± DP) | Δ fid vs ORIG | decisões alteradas/árvore | árvores idênticas a ORIG | tempo/ORIG |
|---|---|---|---|---|---|---|---|---|
| ORIG | 0.700 [0.55; 0.82] |  |  | 0.9251 ± 0.0851 |  | 0.00 | 40/40 | 1.00 |
| WIDE | 1.000 [0.91; 1.00] | +0.300 | (12,0); 0.000 | 0.9806 ± 0.0076 | +0.0555 | 0.00 | 21/40 | 1.02 |
| CONV | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9707 ± 0.0326 | +0.0456 | 0.40 | 26/40 | 1.40 |
| TRUE | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9747 ± 0.0180 | +0.0496 | 0.40 | 27/40 | 3.20 |
| HIERARCHY | 0.800 [0.65; 0.90] | +0.100 | (4,0); 0.125 | 0.9798 ± 0.0079 | +0.0547 | 0.50 | 26/40 | 3.34 |
| PERM | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9702 ± 0.0358 | +0.0451 | 0.35 | 28/40 | 3.12 |
| RAND | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9700 ± 0.0367 | +0.0449 | 0.35 | 28/40 | 3.10 |
| EMPTY | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9251 ± 0.0851 | +0.0000 | 0.00 | 40/40 | 2.74 |
| IRRELEVANT | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9251 ± 0.0851 | +0.0000 | 0.03 | 40/40 | 3.08 |
| INCOMPLETE | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9588 ± 0.0551 | +0.0337 | 0.28 | 31/40 | 3.16 |
| NOISY_SUPERSET | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9747 ± 0.0180 | +0.0496 | 0.40 | 27/40 | 3.12 |
| WRONG_SPLIT | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9742 ± 0.0194 | +0.0491 | 0.40 | 27/40 | 3.10 |
| AMBIGUOUS | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9251 ± 0.0851 | +0.0000 | 0.00 | 40/40 | 3.30 |
| INCONSISTENT | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9251 ± 0.0851 | +0.0000 | 0.00 | 40/40 | 2.80 |
| BADMATCH | 0.700 [0.55; 0.82] | +0.000 | (0,0); 1.000 | 0.9251 ± 0.0851 | +0.0000 | 0.00 | 40/40 | 2.72 |

Critérios pré-registados:

- D1_true_flat_no_advantage: **CUMPRIDO**
- D2_hierarchy_advantage: **NÃO CUMPRIDO**
- D3_perm_no_advantage: **CUMPRIDO**

**Resumo v1:** C5–C8, D1 e D3 cumpridos; **C1–C4 e D2 NÃO cumpridos** — a recuperação da regra foi **idêntica** ao Original (0,65 em P1; 0,70 em P2) em TRUE, PERM, RAND, CONV, HIERARCHY (P1) e as restantes, enquanto WIDE (sem ontologia) recuperou 1,00 em ambos. A fidelidade melhorou +0,05 em TRUE, PERM, RAND e CONV de forma semelhante (0,975 / 0,969 / 0,966 / 0,966 vs ORIG 0,919): **o benefício de fidelidade não era específico da ontologia**.
### 2.1 Diagnóstico da falha v1 (sementes de diagnóstico 9000–9015, fora de calibração e avaliação)
O conceito com a regra existia, era classificado em 1.º lugar e pesquisado no nó-raiz em 16/16 sementes, mas o candidato da unidade **nunca** era a regra: a pesquisa restrita herdava `max_features_per_node = |unidade| = 4`, e o Original usa esse parâmetro também para o **conjunto de literais do feixe** (`max(8, 3·mf)` = 12 literais, vs 84 no Original); os 12 literais de melhor ganho vinham de 1–2 atributos (limiares quase equivalentes) e o feixe não conseguia juntar os 4 atributos. **Defeito de engenharia do arnês, não da ontologia.**

## 3. Alteração v2 (pré-registada antes de avaliar)
`problems_frozen_v2.json` (`2efe4f8`): **mesmos** problemas e critérios; **única** mudança do arnês = *paridade do orçamento de literais* (a pesquisa dentro da unidade mantém o `max_features_per_node` do Original e filtra os literais à unidade); sementes de avaliação novas **100–139** (disjuntas de calibração, v1 e diagnóstico). Verificação mecânica nas sementes de diagnóstico (sem comparar desempenho): candidato da unidade = regra em 16/16. Em simultâneo, três correções do **registo** não-substantivas, congeladas na v2: proveniência de `MATCH_FAILURE`, rejeição de empate perfeito de matching (§6) e profundidade Wu–Palmer pelo caminho mais longo. **Prova de não-substantividade:** repetiu-se a v1 (sementes 0–9, `pool_parity=False`) com o código corrigido: 300 resultados comparados, **0 diferenças** em árvore e recuperação (`v1_patch_equivalence_check.json`). A v1 fica como resultado válido e reportado; a v2 é uma segunda avaliação, não uma substituição.

## 4. Resultados v2 (sementes 100–139; congelamento `2efe4f8`)
### P1_within_group (n = 40 sementes de avaliação)

| variante | recuperação [IC95 Wilson] | Δ vs ORIG | McNemar (b,c; p) | fidelidade teste (média ± DP) | Δ fid vs ORIG | decisões alteradas/árvore | árvores idênticas a ORIG | tempo/ORIG |
|---|---|---|---|---|---|---|---|---|
| ORIG | 0.625 [0.47; 0.76] |  |  | 0.9203 ± 0.0834 |  | 0.00 | 40/40 | 1.00 |
| WIDE | 1.000 [0.91; 1.00] | +0.375 | (15,0); 0.000 | 0.9802 ± 0.0094 | +0.0598 | 0.00 | 15/40 | 0.92 |
| CONV | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9670 ± 0.0389 | +0.0467 | 0.42 | 23/40 | 2.45 |
| TRUE | 1.000 [0.91; 1.00] | +0.375 | (15,0); 0.000 | 0.9818 ± 0.0093 | +0.0614 | 0.82 | 17/40 | 3.80 |
| HIERARCHY | 1.000 [0.91; 1.00] | +0.375 | (15,0); 0.000 | 0.9818 ± 0.0093 | +0.0614 | 0.82 | 17/40 | 4.08 |
| PERM | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9726 ± 0.0262 | +0.0523 | 0.45 | 23/40 | 4.06 |
| RAND | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9711 ± 0.0325 | +0.0508 | 0.50 | 22/40 | 4.05 |
| EMPTY | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9203 ± 0.0834 | +0.0000 | 0.00 | 40/40 | 2.62 |
| IRRELEVANT | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9203 ± 0.0834 | +0.0000 | 0.03 | 39/40 | 4.41 |
| INCOMPLETE | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9743 ± 0.0296 | +0.0540 | 0.70 | 21/40 | 4.24 |
| NOISY_SUPERSET | 1.000 [0.91; 1.00] | +0.375 | (15,0); 0.000 | 0.9815 ± 0.0094 | +0.0611 | 0.82 | 17/40 | 3.88 |
| WRONG_SPLIT | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9741 ± 0.0195 | +0.0538 | 0.55 | 20/40 | 4.06 |
| AMBIGUOUS | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9203 ± 0.0834 | +0.0000 | 0.03 | 39/40 | 4.44 |
| INCONSISTENT | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9203 ± 0.0834 | +0.0000 | 0.00 | 40/40 | 2.68 |
| BADMATCH | 0.625 [0.47; 0.76] | +0.000 | (0,0); 1.000 | 0.9203 ± 0.0834 | +0.0000 | 0.00 | 40/40 | 2.55 |

Critérios pré-registados:

- C1_true_beats_orig: **CUMPRIDO**
- C2_true_beats_perm: **CUMPRIDO**
- C3_true_beats_rand: **CUMPRIDO**
- C4_true_beats_conventional: **CUMPRIDO**
- C5_abstention_equals_original: **CUMPRIDO**
- C6_ambiguous_no_spurious_gain: **CUMPRIDO**
- C7_no_spurious_advantage: **CUMPRIDO**
- C8_bounded_harm: **CUMPRIDO**
- C9_noisy_superset_reported: **{'recovery': 1.0, 'delta_vs_orig': 0.375}**

### P2_cross_group (n = 40 sementes de avaliação)

| variante | recuperação [IC95 Wilson] | Δ vs ORIG | McNemar (b,c; p) | fidelidade teste (média ± DP) | Δ fid vs ORIG | decisões alteradas/árvore | árvores idênticas a ORIG | tempo/ORIG |
|---|---|---|---|---|---|---|---|---|
| ORIG | 0.650 [0.50; 0.78] |  |  | 0.9149 ± 0.0908 |  | 0.00 | 40/40 | 1.00 |
| WIDE | 1.000 [0.91; 1.00] | +0.350 | (14,0); 0.000 | 0.9797 ± 0.0079 | +0.0648 | 0.00 | 19/40 | 0.90 |
| CONV | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9631 ± 0.0481 | +0.0482 | 0.50 | 23/40 | 2.39 |
| TRUE | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9662 ± 0.0412 | +0.0513 | 0.53 | 22/40 | 3.96 |
| HIERARCHY | 1.000 [0.91; 1.00] | +0.350 | (14,0); 0.000 | 0.9812 ± 0.0075 | +0.0663 | 0.72 | 17/40 | 3.69 |
| PERM | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9632 ± 0.0484 | +0.0483 | 0.57 | 21/40 | 4.13 |
| RAND | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9658 ± 0.0424 | +0.0509 | 0.50 | 23/40 | 4.11 |
| EMPTY | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9149 ± 0.0908 | +0.0000 | 0.00 | 40/40 | 2.62 |
| IRRELEVANT | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9149 ± 0.0908 | +0.0000 | 0.03 | 40/40 | 4.25 |
| INCOMPLETE | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9478 ± 0.0683 | +0.0329 | 0.40 | 26/40 | 3.99 |
| NOISY_SUPERSET | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9662 ± 0.0411 | +0.0513 | 0.55 | 21/40 | 4.00 |
| WRONG_SPLIT | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9669 ± 0.0369 | +0.0520 | 0.60 | 21/40 | 4.05 |
| AMBIGUOUS | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9149 ± 0.0908 | +0.0000 | 0.00 | 40/40 | 4.25 |
| INCONSISTENT | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9149 ± 0.0908 | +0.0000 | 0.00 | 40/40 | 2.74 |
| BADMATCH | 0.650 [0.50; 0.78] | +0.000 | (0,0); 1.000 | 0.9149 ± 0.0908 | +0.0000 | 0.00 | 40/40 | 2.65 |

Critérios pré-registados:

- D1_true_flat_no_advantage: **CUMPRIDO**
- D2_hierarchy_advantage: **CUMPRIDO**
- D3_perm_no_advantage: **CUMPRIDO**

**Contrastes de fidelidade emparelhados (v2):**
| contraste (fidelidade, emparelhado por semente) | Δ média [IC95 bootstrap] | V/D/E |
|---|---|---|
| P1 TRUE-PERM | +0.0091 [+0.0026; +0.0177] | 18/5/17 |
| P1 TRUE-RAND | +0.0106 [+0.0028; +0.0217] | 17/5/18 |
| P1 TRUE-CONV | +0.0147 [+0.0048; +0.0275] | 18/4/18 |
| P1 TRUE-WIDE | +0.0016 [+0.0007; +0.0027] | 13/1/26 |
| P1 TRUE-ORIG | +0.0615 [+0.0373; +0.0866] | 22/1/17 |
| P1 HIERARCHY-ORIG | +0.0615 [+0.0367; +0.0871] | 22/1/17 |
| P1 HIERARCHY-TRUE | +0.0000 [+0.0000; +0.0000] | 0/0/40 |
| P1 PERM-ORIG | +0.0523 [+0.0286; +0.0772] | 16/1/23 |
| P1 RAND-ORIG | +0.0508 [+0.0267; +0.0763] | 15/2/23 |
| P2 TRUE-PERM | +0.0030 [-0.0026; +0.0113] | 12/4/24 |
| P2 TRUE-RAND | +0.0004 [-0.0046; +0.0061] | 12/5/23 |
| P2 TRUE-CONV | +0.0031 [-0.0000; +0.0083] | 10/4/26 |
| P2 TRUE-WIDE | -0.0135 [-0.0276; -0.0030] | 10/10/20 |
| P2 TRUE-ORIG | +0.0513 [+0.0283; +0.0771] | 17/1/22 |
| P2 HIERARCHY-ORIG | +0.0663 [+0.0396; +0.0952] | 22/1/17 |
| P2 HIERARCHY-TRUE | +0.0150 [+0.0036; +0.0298] | 13/9/18 |
| P2 PERM-ORIG | +0.0483 [+0.0266; +0.0722] | 17/2/21 |
| P2 RAND-ORIG | +0.0509 [+0.0287; +0.0757] | 16/0/24 |

**Resumo v2: todos os critérios pré-registados cumpridos (C1–C8, D1–D3).**
1. **O mecanismo identifica e usa a relação verdadeira:** TRUE recupera a regra em 40/40 (1,00 [0,91; 1,00]) contra 0,625 do Original (McNemar 15–0, p < 0,001); PERM, RAND e CONV (relação destruída ou estatística) ficam em 0,625 = Original. Em P2 (regra a atravessar conceitos) a ontologia **plana** verdadeira não dá vantagem (0,65 = ORIG; D1) mas a **hierarquia verdadeira** com pai comum dá 1,00 (D2: +0,35, 14–0), evidência de que a taxonomia (e não só a partição) é aproveitada. NOISY_SUPERSET (verdade + 2 atributos espúrios) mantém 1,00.
2. **Casos negativos sem vantagem espúria nem dano:** IRRELEVANT, INCOMPLETE, WRONG_SPLIT e AMBIGUOUS ficam em 0,625 (P1) / 0,65 (P2) = Original; EMPTY, INCONSISTENT e BADMATCH reproduzem a árvore do Original em 40/40 (AMBIGUOUS e IRRELEVANT em 39–40/40, com ≤ 1 decisão alterada no total). Nenhum dano de recuperação e fidelidade ≥ Original em todas as variantes.
3. **Mas a ontologia não é necessária:** `WIDE` (Original com `mf = p`, sem ontologia) também recupera 1,00 em P1 e P2 (incluindo o caso «plano» em que TRUE falha) a 0,92× o tempo do Original (TRUE: ≈ 3,8×, incluindo compilação). Em P1, TRUE−WIDE: +0,0016 de fidelidade (IC95 [+0,0007; +0,0027]), irrelevante na prática; em P2 plano, TRUE−WIDE = −0,0135 (IC95 [−0,0276; −0,0030]). **Conclusão honesta:** nestes problemas a orientação por ontologia recupera o que um aumento de orçamento de atributos também recupera; o seu valor é pesquisar **seletivamente** um subconjunto relevante, não superar o Original com orçamento maior.
4. **O ganho de fidelidade é em grande parte inespecífico:** PERM, RAND e CONV melhoram +0,047 a +0,052 sobre o Original (TRUE: +0,061) só por pesquisarem mais candidatos pela mesma pontuação estatística. O incremento atribuível à relação verdadeira é TRUE−PERM = +0,009 [+0,003; +0,018] e TRUE−RAND = +0,011 [+0,003; +0,022] (P1) e **não distinguível de zero** em P2 plano (+0,003 [−0,003; +0,011]). Sem estes controlos o ganho seria atribuído erradamente à ontologia.
5. **Limites:** problema sintético com a regra *alinhada por construção* com o conceito (a verdade é conhecida e relevante, como pedido); um único mecanismo (preferência sobre onde procurar); critérios técnicos; 40 sementes por problema (IC de Wilson largos nas taxas intermédias); **nenhuma conclusão sobre generalização, datasets reais ou superioridade sobre Original/C4.5**.

## 4b. Concordância entre v1 e v2
A mesma ontologia e os mesmos problemas dão resultados opostos nos critérios C1–C4/D2 consoante o arnês limite (v1) ou iguale (v2) o orçamento de literais da pesquisa dentro da unidade: o defeito era do arnês, não da relação semântica. Ambas as avaliações ficam reportadas; só a v2 é usada para concluir sobre identificabilidade, e o desvio está declarado (§3 e §8).

## 5. Rastreabilidade (cadeia obrigatória)
Implementada em `SemanticSteeredTrepan.semantic_trace_` / `trace_report()`: **`OWL Axiom → Inferred/Validated Relation → Feature Mapping → Semantic Fact → Candidate → Statistical Decision → Tree Effect`**. Exemplo real (`trace_example_v2.json`, problema P1, semente 100, variante TRUE, nó-raiz da pesquisa nº 1):
| elo | registo |
|---|---|
| axioma OWL | `acoustic_mean Domain AcousticProfile` (… e os 3 restantes `Domain`) |
| relação extraída | `ATTRIBUTE_APPLIES_TO_CONCEPT(acoustic_*, AcousticProfile)`, derivação `ASSERTED` (explícita), estatuto `LOGICALLY_ENTAILED`, `permitted_use=PREFERENCE` |
| mapeamento de atributos | `acoustic_mean → acoustic_mean` (propriedade de dados), confiança do matching 1,0 |
| facto semântico | `F-ATT-1af107de1bd0`, `F-ATT-436e14127ca4`, `F-ATT-be0c628afe26`, `F-ATT-e1265296325f` → unidade `AcousticProfile` (IC 2,08; 1.ª por ganho de literal isolado) |
| candidato | teste 3-de-4 sobre `acoustic_{mean,range,trend,peak}`; **«existiria sem OWL?» sim, em potência**: os 4 atributos estavam no conjunto de literais do Original (`unit_features_all_in_original_pool=true`), mas o feixe do Original não o montou |
| decisão estatística | pontuação **original** (ganho normalizado) 0,145 → 0,254; `score_changed=false`, `scoring_modified=false`; `choice_changed=true` |
| condição logicamente admissível? | n/a — preferência; `hard_constraints_applied=0` |
| efeito na árvore | `in_final_tree=true`; 1 de 2 nós internos influenciado |
Respostas às perguntas do protocolo: *qual axioma?* (`source_axioms` do facto) · *qual relação?* (`kind`) · *explícita ou inferida?* (`derivation`) · *confiança do matching?* (`matching_confidence`) · *existiria sem OWL?* (`candidate_exists_without_owl`, `unit_features_all_in_original_pool`) · *pontuação alterada?* (`score_changed`) · *escolha alterada?* (`choice_changed`) · *admissível?* (`logical_admissibility`) · *efeito?* (`in_final_tree`, `n_influenced_in_final_tree`). **Demonstra influência operacional, não ganho causal.** Teste: `test_traceability_chain_for_a_semantically_influenced_candidate` (16 sementes; exige exemplos com cadeia completa).

## 6. Achados adicionais
1. **Matching — empate perfeito aceite pelo gate do núcleo.** Duas propriedades com o mesmo rótulo do atributo (score 1,0, margem 0) → o gate aceita a 1.ª arbitrariamente (`score ≥ 0,98` dispensa a margem). O núcleo **não foi alterado**; o registo trata o caso como `PERFECT_TIE_AMBIGUOUS` (rejeitado, `MATCH_FAILURE`) — teste e Doc. 34.
2. **Controlos não ontológicos explicam o efeito.** `WIDE` (Original com `mf = p`, sem ontologia) alcança a recuperação máxima; o benefício de fidelidade de qualquer orientação por unidades (mesmo permutada, aleatória ou convencional) ≈ o da ontologia verdadeira na v1. Logo, **a ontologia verdadeira não é, nestes problemas, necessária nem suficiente para o ganho; o que ela faz é permitir pesquisar um subconjunto relevante sem alargar o orçamento geral.** Atribuir o ganho à ontologia exige sempre estes controlos.
3. **Abstenção correta:** EMPTY, INCONSISTENT e BADMATCH reproduzem a árvore do Original em 100 % das sementes, com as mesmas consultas (v1 e v2).
4. **Ambiguidade:** AMBIGUOUS rejeita o matching dos atributos da regra (`REJECTED_AMBIGUOUS`) e não há vantagem espúria.

## 7. Custos
Medição `cost_probe.json` (P1, semente 100; CPU partilhada): compilação do registo **1,14 s** (reasoner HermiT ≈ 0,9 s; **0,25 s sem reasoner**); ajuste do Original 0,67 s; ajuste com orientação (3 unidades) 1,29 s ⇒ ≈ 1,9× o Original no ajuste e ≈ 3× com compilação (v1: razão média de tempo ≈ 3,1×; ver tabelas). Consultas ao oráculo: o mecanismo **não gera consultas adicionais** por nó; o total difere do Original em +0 a +7 % (v2) só porque a árvore difere (EMPTY, INCONSISTENT e BADMATCH: idêntico). Em `core/` o reasoner correria uma vez por ontologia (não por árvore) e a compilação pode ser memoizada por hash da ontologia.

## 8. Desvios e transparência
1. v1 → v2 por defeito de engenharia do arnês, com diagnóstico em sementes fora de calibração/avaliação, nova pré-inscrição e sementes novas; a v1 não foi apagada nem reinterpretada.
2. Três correções do registo após o congelamento da v1: provadas não-substantivas (300 comparações, 0 diferenças).
3. O commit `6dd6019` ficou com um teste do registo desatualizado (proveniência de `MATCH_FAILURE`) e a sintaxe do relatório incompatível com Python 3.11; ambos corrigidos nos commits seguintes (CI desse commit é vermelha por esse motivo).
4. A etiqueta `b30-freeze` não pôde ser enviada ao remoto; o congelamento está provado por `freeze_B30.json` (hashes de árvore git) e pelo commit `9bc7cbb`.

## 9. Critérios de conclusão técnica da B3.1 (pedido §14)
| # | critério | estado |
|---|---|---|
| 1 | compilador de factos funciona em `validation/` | **sim** (4 ontologias DEV compiladas; 12 variantes sintéticas) |
| 2 | tipos e proveniência verificáveis | **sim** (campos obrigatórios, ids determinísticos, hash da ontologia, axiomas de origem; testes) |
| 3 | relatedness distingue relações relevantes de genéricas | **sim** (Doc. 35; histórica constante vs taxonómica graduada; ancestral genérico ignorado) |
| 4 | inferências do reasoner propagadas quando aplicáveis | **sim** (HermiT `C ⊑ A` chega ao atributo e à unidade); DEV: 0 inferências novas ⇒ `NO_NEW_INFERENCES` (não é falha) |
| 5 | mundo aberto respeitado | **sim** (sem factos negativos; disjunção/domínio/facetas sem uso obrigatório) |
| 6 | testes positivos e negativos de identificabilidade | **v2: todos os critérios pré-registados cumpridos**; v1: C1–C4 e D2 não cumpridos (defeito de arnês diagnosticado, reportado) |
| 7 | sem alteração aos baselines | **sim** (`git diff 9bc7cbb HEAD -- core` vazio; Original, C4.5, MLP, resultados brutos e âncoras intactos) |
| 8 | testes de regressão passam | **sim** (428 passed, 40 skipped) |
| 9 | agnosticismo preservado | **sim** (verificador estático e contrato; tudo sem nomes de datasets no núcleo) |
| 10 | CI verde | **parcial**: `9bc7cbb` (B3.0) verde nos 3 fluxos; os commits da B3.1 têm o *guard* verde e o Windows em execução no momento da entrega — **não declarado verde** |
**A conclusão técnica da B3.1 não equivale a superioridade do TREPAN Reloaded.** Não se alterou `core/`; a integração experimental em `core/` e uma campanha DEV exigem nova autorização.
