# Doc. 38 — Comparação OWL versus Original WIDE (B3.2)

Estado: **EXPERIÊNCIAS SINTÉTICAS PRÉ-REGISTADAS EXECUTADAS; `core/` NÃO alterado; sem datasets confirmatórios nem campanha DEV.** Código: `validation/benchmark/b32_comparison.py`, `b32_run.py`, `b32_report.py`; evidência: `docs/evidence/b1b4/b32/` (`b32_frozen.json`, `results_*.json`, `analysis_b32.json/.md`, `matching_diagnosis.json`, `review_package/`). Pré-registo (commit `62e5a00`) anterior a qualquer avaliação nas sementes 200–239; o avaliador recusa-se a correr se o hash do código mudou (nenhum desvio).

## 0. Resposta curta
**Nestes problemas sintéticos a ontologia verdadeira NÃO oferece vantagem própria sobre pesquisa genérica com recursos equivalentes.** Veredicto pré-registado: `NO_ONTOLOGICAL_ADVANTAGE` em P1′ e P2′.
- A orientação pela ontologia verdadeira recupera a regra (P1′: 0,975) muito acima da ontologia permutada, aleatória, neutralizada e do agrupamento por correlação (0,475 = Original) com **o mesmo nº de candidatos avaliados** — a relação é identificável (relação-específica).
- Mas o **rastreio univariado genérico (F2) produz exatamente as mesmas árvores** (40/40 sementes) e **Original WIDE** recupera 1,000 com **~½ do CPU de ajuste e sem compilação**; WIDE com feixe maior (H1/H2) obtém accuracy **superior** à da ontologia (−0,014 [−0,022; −0,007]). Em P2′ (regra que atravessa conceitos) a ontologia plana não ajuda (0,525 = Original); só a hierarquia verdadeira ajuda (0,975), igualada de novo por WIDE/SCREEN.
- Por isso **não se declara superioridade ontológica**; a vantagem face ao Original limitado (+0,50 de recuperação) deve-se a ele estar limitado, não à ontologia.

## 1. Contrato de equivalência de recursos
Igual em TODOS os braços: dados e instâncias de treino/teste (mesma semente), **o mesmo MLP** (oráculo; `StandardScaler`+`MLPClassifier(32,)`, acc. ≈ 0,79, imperfeito de propósito), `min_sample`, `max_queries`, `max_nodes`, `max_depth`, `max_n`, `min_samples_leaf`, rótulos reais e partição de avaliação (4000 linhas independentes).
| dimensão | Comparação 1 — igual nº de candidatos/literais | Comparação 2 — igual consultas / CPU | medido (P1′, C = OWL) |
|---|---|---|---|
| unidades/pesquisas | C, D, E, F1, F2, G e negativos: **9 pesquisas por unidade** (3 por nó, tamanho 4) | — | 9,0 em todos |
| candidatos AVALIADOS (`_split_selection_score`) | **igualados por construção** (D/E/F1/F2: razão C/x = 1,04/1,04/1,02/1,00); H1 calibrado só em candidatos (razão C/H1 = 0,89); G fica a 0,70 (G avalia +43 %) | não igualado (H2: C/H2 = 2,02) | C 16 884 · A 4 803 · B 5 661 |
| candidatos GERADOS / literais listados | reportados, **não** igualados | idem | tabelas |
| consultas ao oráculo / amostras | iguais em ±5 % (só variam com a forma da árvore) | **igualadas por construção** (mesmo `min_sample`) | 645–678 / 945–978 |
| CPU de ajuste | reportado | **H2 calibrado só em CPU** (razão C/H2 = 0,95) | C 1,85 s · B 0,98 s · A 0,82 s |
| compilação da ontologia (reasoner Java) | reportada à parte | idem | 1,25 s (parede) por árvore |
| memória (pico `tracemalloc`, 10 sementes) | reportada | idem | A 2,39 MB · B 2,37 · C 2,47 · H2 2,46 (diferenças ≤ 4 %) |
| complexidade da árvore | **não igualada** (resultado) | idem | nós internos, profundidade, literais/regra nas tabelas |
**Não se afirma que todos os recursos estão igualados:** em cada comparação só as dimensões indicadas o estão; as restantes são reportadas. O braço C **não** é o Reloaded: é o TREPAN Original + orientação por ontologia (arnês `EfficientSteered`, árvore idêntica à do arnês B3.1 `pool_parity`, testado em semente de diagnóstico).

## 2. Limitações dos resultados da B3.1 (reconhecidas)
1. A v1 falhou (C1–C4, D2) por defeito do arnês; a v2 corrigiu a paridade do orçamento de literais.
2. **WIDE recuperou a regra em 100 %** (igual à ontologia verdadeira) com menos recursos.
3. Custo elevado da orientação por OWL (≈ 3,8× em tempo incl. compilação).
4. O incremento de fidelidade específico da relação era pequeno e heterogéneo (TRUE−PERM +0,009 em P1; não distinguível de zero em P2 plano).
5. As ontologias DEV são partições planas; não há ontologias enriquecidas validadas; revisões pendentes (Doc. 39).
6. A regra foi **alinhada por construção** com o conceito (como pedido), com oráculo de forma fechada: a vantagem máxima possível da ontologia, e mesmo assim não excede os controlos genéricos.
7. CI do commit final: ver §10.

## 3. Desenho
### 3.1 Braços
| id | descrição |
|---|---|
| A ORIG | Original com orçamento histórico limitado (`max_features_per_node` = 28 calibrado) |
| B WIDE | Original com `mf = p = 32` (variante experimental; Original canónico intocado) |
| C OWL | orientação pela ontologia verdadeira (8 conceitos de 4 atributos; P2′ também HIERARCHY) |
| D PERM / E RAND | partição permutada / aleatória de tamanhos iguais, com a regra garantidamente dispersa |
| F1 CONV | unidades por agrupamento hierárquico de \|Spearman\| nos dados de treino |
| F2 SCREEN | **rastreio univariado genérico**: ordena atributos por ganho de informação de um limiar face às previsões do MLP e forma 8 unidades de 4 (mesmo nº/tamanho que C; sem ontologia) |
| G NEUTRALIZED | ontologia verdadeira com as relações (`domain`) dos atributos da regra removidas; tudo o resto igual |
| H1 / H2 | WIDE com `beam_width` maior: H1 igualando candidatos avaliados de C, H2 igualando o CPU de ajuste de C (calibrados só em recursos: `beam_width` 32/8 em P1′, 24/6 em P2′) |
| N_* | negativos: IRRELEVANT, INCOMPLETE, WRONG_SPLIT |
| refs | MLP (accuracy) e C4.5 canónico treinado nos rótulos reais |
### 3.2 Problemas, calibração e congelamento
P1′ (regra 2-de-4 dentro de um conceito) e P2′ (2+2 atravessando dois conceitos); 32 atributos em 8 conceitos; MLP treinado nos rótulos reais (n = 300). **Dificuldade calibrada só com o Original** (30 sementes 2000–2029): taxa de recuperação por `mf` {16, 20, 24, 28, 30} = P1′ {0,03; 0,17; 0,27; **0,57**; 0,77}, P2′ {0,10; 0,20; 0,33; **0,57**; 0,80} ⇒ `mf = 28`. `beam_width` de H1/H2 calibrado com 15 sementes **só em recursos** (nunca desempenho). Avaliação: 40 sementes (200–239) por problema; estabilidade: 4 configurações novas × 20 sementes (300–319); memória: 10 sementes. Critérios (`b32_frozen.json`): **O1** recup.(C) − máx(D,E,F1,F2,G) ≥ 0,20; **O2/O3** vs H1/H2: Δfidelidade ≥ +0,005 com IC95 inferior > 0 *ou* Δrecup. ≥ 0,10; **O4** vs WIDE: (a) fidelidade, (b) CPU ≤ 0,8× com recup. ≥ −0,05, (c) nós internos ≤ 0,8× sem perder > 0,005 de fidelidade, (d) DP da fidelidade ≤ 0,8×. Veredicto `ONTOLOGICAL_ADVANTAGE_OVER_GENERIC` só com O1∧O2∧O3∧O4; vantagem sobre A isolada **não** conta se WIDE a iguala com menos recursos.

## 4. Resultados (todas as sementes; sem arredondar antes das comparações)
### P1prime_within_group (n = 40 sementes)

| braço | recup. [IC95] | fidelidade (±DP) | accuracy (±DP) | prec. macro | macro-F1 | nós int. | prof. | lit./regra | literais listados | cand. gerados (simples+m-of-n) | cand. AVALIADOS | consultas | amostras ao oráculo | CPU ajuste (s) | compil. (s) | unid. pesquisadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.475 [0.33; 0.63] | 0.7644 ± 0.0332 | 0.8673 ± 0.0862 | 0.8471 | 0.8450 | 2.05 | 2.05 | 3.68 | 3844 | 4832 | 4803 | 645 | 945 | 0.82 | 0.00 | 0.0 |
| B_WIDE | 1.000 [0.91; 1.00] | 0.7921 ± 0.0134 | 0.9608 ± 0.0224 | 0.9542 | 0.9547 | 1.50 | 1.50 | 3.99 | 4424 | 5693 | 5661 | 671 | 971 | 0.98 | 0.00 | 0.0 |
| C_OWL | 0.975 [0.87; 1.00] | 0.7922 ± 0.0141 | 0.9570 ± 0.0265 | 0.9503 | 0.9505 | 1.62 | 1.62 | 3.95 | 9388 | 17051 | 16884 | 676 | 976 | 1.85 | 1.25 | 9.0 |
| D_PERM | 0.475 [0.33; 0.63] | 0.7783 ± 0.0255 | 0.9162 ± 0.0518 | 0.9058 | 0.9045 | 2.35 | 2.35 | 3.49 | 9347 | 16312 | 16225 | 662 | 962 | 1.62 | 1.25 | 9.0 |
| E_RAND | 0.475 [0.33; 0.63] | 0.7793 ± 0.0219 | 0.9125 ± 0.0541 | 0.9039 | 0.8982 | 2.15 | 2.15 | 3.42 | 9344 | 16274 | 16184 | 668 | 968 | 1.65 | 1.25 | 9.0 |
| F1_CONV | 0.475 [0.33; 0.63] | 0.7796 ± 0.0231 | 0.9150 ± 0.0508 | 0.9061 | 0.9015 | 2.25 | 2.23 | 3.41 | 9679 | 16588 | 16494 | 663 | 963 | 1.66 | 0.00 | 9.0 |
| F2_SCREEN | 0.975 [0.87; 1.00] | 0.7922 ± 0.0141 | 0.9570 ± 0.0265 | 0.9503 | 0.9505 | 1.62 | 1.62 | 3.95 | 9389 | 17055 | 16887 | 676 | 976 | 1.82 | 0.00 | 9.0 |
| G_NEUTRALIZED | 0.475 [0.33; 0.63] | 0.7643 ± 0.0339 | 0.8664 ± 0.0872 | 0.8465 | 0.8432 | 2.05 | 2.05 | 3.71 | 12630 | 24398 | 24110 | 645 | 945 | 2.35 | 1.24 | 9.0 |
| H1_WIDE_BEAM_CAND | 1.000 [0.91; 1.00] | 0.7933 ± 0.0120 | 0.9708 ± 0.0176 | 0.9662 | 0.9662 | 1.40 | 1.40 | 3.98 | 4419 | 19253 | 18915 | 678 | 978 | 5.59 | 0.00 | 0.0 |
| H2_WIDE_BEAM_CPU | 1.000 [0.91; 1.00] | 0.7932 ± 0.0131 | 0.9708 ± 0.0163 | 0.9671 | 0.9660 | 1.23 | 1.23 | 4.00 | 4423 | 8424 | 8359 | 670 | 970 | 1.95 | 0.00 | 0.0 |
| N_IRRELEVANT | 0.475 [0.33; 0.63] | 0.7644 ± 0.0332 | 0.8673 ± 0.0862 | 0.8471 | 0.8450 | 2.05 | 2.05 | 3.68 | 9334 | 16407 | 16122 | 645 | 945 | 1.86 | 1.25 | 9.0 |
| N_INCOMPLETE | 0.475 [0.33; 0.63] | 0.7743 ± 0.0349 | 0.8935 ± 0.0708 | 0.8814 | 0.8762 | 2.12 | 2.12 | 3.57 | 9190 | 16342 | 16182 | 669 | 969 | 1.74 | 1.23 | 9.0 |
| N_WRONG_SPLIT | 0.475 [0.33; 0.63] | 0.7832 ± 0.0192 | 0.9148 ± 0.0481 | 0.9044 | 0.9002 | 2.17 | 2.15 | 3.39 | 9868 | 16889 | 16779 | 663 | 963 | 1.67 | 1.24 | 9.0 |

Referências: MLP accuracy 0.7930; C4.5 canónico accuracy 0.9577, fidelidade ao MLP 0.7905, precision macro 0.9527, macro-F1 0.9506.

| C vs … | Δ recup. | McNemar (b,c; p) | Δ fidelidade [IC95] | Δ accuracy [IC95] | cand. avaliados C/x | CPU C/x | consultas C/x | nós int. C/x |
|---|---|---|---|---|---|---|---|---|
| A_ORIG | +0.500 | (20,0); 0.000 | +0.0278 [+0.0171; +0.0387] | +0.0897 [+0.0621; +0.1179] | 3.52 | 2.27 | 1.05 | 0.79 |
| B_WIDE | -0.025 | (0,1); 1.000 | +0.0000 [-0.0018; +0.0018] | -0.0038 [-0.0101; +0.0006] | 2.98 | 1.90 | 1.01 | 1.08 |
| D_PERM | +0.500 | (20,0); 0.000 | +0.0139 [+0.0068; +0.0217] | +0.0408 [+0.0247; +0.0581] | 1.04 | 1.14 | 1.02 | 0.69 |
| E_RAND | +0.500 | (20,0); 0.000 | +0.0129 [+0.0072; +0.0193] | +0.0445 [+0.0285; +0.0617] | 1.04 | 1.12 | 1.01 | 0.76 |
| F1_CONV | +0.500 | (20,0); 0.000 | +0.0125 [+0.0066; +0.0192] | +0.0420 [+0.0268; +0.0581] | 1.02 | 1.12 | 1.02 | 0.72 |
| F2_SCREEN | +0.000 | (0,0); 1.000 | +0.0000 [+0.0000; +0.0000] | +0.0000 [+0.0000; +0.0000] | 1.00 | 1.02 | 1.00 | 1.00 |
| G_NEUTRALIZED | +0.500 | (20,0); 0.000 | +0.0279 [+0.0169; +0.0389] | +0.0906 [+0.0625; +0.1188] | 0.70 | 0.79 | 1.05 | 0.79 |
| H1_WIDE_BEAM_CAND | -0.025 | (0,1); 1.000 | -0.0012 [-0.0042; +0.0014] | -0.0138 [-0.0223; -0.0073] | 0.89 | 0.33 | 1.00 | 1.16 |
| H2_WIDE_BEAM_CPU | -0.025 | (0,1); 1.000 | -0.0010 [-0.0041; +0.0018] | -0.0138 [-0.0222; -0.0072] | 2.02 | 0.95 | 1.01 | 1.33 |
| N_IRRELEVANT | +0.500 | (20,0); 0.000 | +0.0278 [+0.0171; +0.0387] | +0.0897 [+0.0621; +0.1179] | 1.05 | 1.00 | 1.05 | 0.79 |
| N_INCOMPLETE | +0.500 | (20,0); 0.000 | +0.0179 [+0.0089; +0.0282] | +0.0635 [+0.0418; +0.0863] | 1.04 | 1.07 | 1.01 | 0.76 |
| N_WRONG_SPLIT | +0.500 | (20,0); 0.000 | +0.0089 [+0.0040; +0.0145] | +0.0422 [+0.0283; +0.0565] | 1.01 | 1.11 | 1.02 | 0.75 |

Critérios pré-registados: O1=False; O2(H1)=False; O3(H2)=False; O4(vs WIDE)={'a_fidelity': False, 'b_cost': False, 'c_complexity': False, 'd_stability': False, 'any': False}; **veredicto: NO_ONTOLOGICAL_ADVANTAGE**

Hierarquia exploratória: {'mean_Acc_C>Acc_A': True, 'mean_Acc_A>Acc_C45': False, 'mean_Fid_C>Fid_A': True, 'mean_Fid_C>Fid_C45': True, 'per_seed_rate_Acc_C>A': 0.575, 'per_seed_rate_Fid_C>A': 0.475, 'all_five_conditions_means': False}

### P2prime_cross_group (n = 40 sementes)

| braço | recup. [IC95] | fidelidade (±DP) | accuracy (±DP) | prec. macro | macro-F1 | nós int. | prof. | lit./regra | literais listados | cand. gerados (simples+m-of-n) | cand. AVALIADOS | consultas | amostras ao oráculo | CPU ajuste (s) | compil. (s) | unid. pesquisadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.525 [0.37; 0.67] | 0.7712 ± 0.0377 | 0.8848 ± 0.0899 | 0.8748 | 0.8676 | 1.73 | 1.73 | 3.75 | 3841 | 4792 | 4773 | 671 | 971 | 0.90 | 0.00 | 0.0 |
| B_WIDE | 1.000 [0.91; 1.00] | 0.7949 ± 0.0131 | 0.9661 ± 0.0185 | 0.9617 | 0.9606 | 1.15 | 1.15 | 4.00 | 4410 | 5675 | 5654 | 692 | 992 | 1.00 | 0.00 | 0.0 |
| C_OWL | 0.525 [0.37; 0.67] | 0.7907 ± 0.0161 | 0.9276 ± 0.0450 | 0.9202 | 0.9153 | 1.90 | 1.88 | 3.47 | 9343 | 16188 | 16056 | 681 | 981 | 1.63 | 1.22 | 9.0 |
| D_PERM | 0.525 [0.37; 0.67] | 0.7867 ± 0.0223 | 0.9223 ± 0.0553 | 0.9155 | 0.9103 | 1.98 | 1.98 | 3.53 | 9338 | 16286 | 16184 | 678 | 978 | 1.66 | 1.22 | 9.0 |
| E_RAND | 0.525 [0.37; 0.67] | 0.7871 ± 0.0198 | 0.9211 ± 0.0546 | 0.9132 | 0.9087 | 1.95 | 1.95 | 3.51 | 9338 | 16280 | 16182 | 680 | 980 | 1.66 | 1.22 | 9.0 |
| F1_CONV | 0.525 [0.37; 0.67] | 0.7904 ± 0.0185 | 0.9215 ± 0.0547 | 0.9123 | 0.9089 | 2.00 | 2.00 | 3.57 | 9749 | 16718 | 16608 | 680 | 980 | 1.73 | 0.00 | 9.0 |
| F2_SCREEN | 0.975 [0.87; 1.00] | 0.7950 ± 0.0127 | 0.9623 ± 0.0278 | 0.9574 | 0.9560 | 1.18 | 1.18 | 3.98 | 9368 | 17044 | 16862 | 696 | 996 | 1.93 | 0.00 | 9.0 |
| G_NEUTRALIZED | 0.525 [0.37; 0.67] | 0.7718 ± 0.0367 | 0.8842 ± 0.0910 | 0.8737 | 0.8663 | 1.73 | 1.73 | 3.77 | 12526 | 24127 | 23834 | 672 | 972 | 2.34 | 1.25 | 9.0 |
| H1_WIDE_BEAM_CAND | 1.000 [0.91; 1.00] | 0.7961 ± 0.0114 | 0.9729 ± 0.0128 | 0.9705 | 0.9684 | 1.27 | 1.27 | 4.00 | 4413 | 15457 | 15268 | 689 | 989 | 4.34 | 0.00 | 0.0 |
| H2_WIDE_BEAM_CPU | 1.000 [0.91; 1.00] | 0.7960 ± 0.0104 | 0.9733 ± 0.0133 | 0.9709 | 0.9690 | 1.18 | 1.18 | 4.00 | 4415 | 7477 | 7418 | 684 | 984 | 1.62 | 0.00 | 0.0 |
| N_IRRELEVANT | 0.525 [0.37; 0.67] | 0.7712 ± 0.0377 | 0.8848 ± 0.0899 | 0.8748 | 0.8676 | 1.73 | 1.73 | 3.75 | 9330 | 16393 | 16142 | 671 | 971 | 1.84 | 1.21 | 9.0 |
| N_INCOMPLETE | 0.525 [0.37; 0.67] | 0.7868 ± 0.0246 | 0.9201 ± 0.0584 | 0.9115 | 0.9066 | 1.90 | 1.90 | 3.56 | 9235 | 15985 | 15853 | 680 | 980 | 1.66 | 1.20 | 9.0 |
| N_WRONG_SPLIT | 0.525 [0.37; 0.67] | 0.7910 ± 0.0154 | 0.9265 ± 0.0498 | 0.9195 | 0.9138 | 1.88 | 1.88 | 3.54 | 9794 | 16638 | 16516 | 681 | 981 | 1.67 | 1.23 | 9.0 |
| C_HIERARCHY | 0.975 [0.87; 1.00] | 0.7954 ± 0.0123 | 0.9631 ± 0.0248 | 0.9585 | 0.9569 | 1.27 | 1.27 | 3.98 | 9923 | 21523 | 21432 | 691 | 991 | 1.92 | 1.28 | 9.0 |

Referências: MLP accuracy 0.7947; C4.5 canónico accuracy 0.9541, fidelidade ao MLP 0.7933, precision macro 0.9496, macro-F1 0.9462.

| C vs … | Δ recup. | McNemar (b,c; p) | Δ fidelidade [IC95] | Δ accuracy [IC95] | cand. avaliados C/x | CPU C/x | consultas C/x | nós int. C/x |
|---|---|---|---|---|---|---|---|---|
| A_ORIG | +0.000 | (0,0); 1.000 | +0.0195 [+0.0096; +0.0298] | +0.0428 [+0.0264; +0.0597] | 3.36 | 1.82 | 1.02 | 1.10 |
| B_WIDE | -0.475 | (0,19); 0.000 | -0.0042 [-0.0086; -0.0002] | -0.0385 [-0.0506; -0.0261] | 2.84 | 1.63 | 0.98 | 1.65 |
| D_PERM | +0.000 | (0,0); 1.000 | +0.0041 [+0.0001; +0.0088] | +0.0053 [-0.0008; +0.0134] | 0.99 | 0.98 | 1.00 | 0.96 |
| E_RAND | +0.000 | (0,0); 1.000 | +0.0036 [+0.0004; +0.0075] | +0.0065 [+0.0015; +0.0127] | 0.99 | 0.98 | 1.00 | 0.97 |
| F1_CONV | +0.000 | (0,0); 1.000 | +0.0004 [-0.0036; +0.0040] | +0.0061 [-0.0009; +0.0146] | 0.97 | 0.94 | 1.00 | 0.95 |
| F2_SCREEN | -0.450 | (0,18); 0.000 | -0.0043 [-0.0084; -0.0007] | -0.0347 [-0.0474; -0.0219] | 0.95 | 0.84 | 0.98 | 1.62 |
| G_NEUTRALIZED | +0.000 | (0,0); 1.000 | +0.0189 [+0.0093; +0.0291] | +0.0434 [+0.0267; +0.0609] | 0.67 | 0.70 | 1.01 | 1.10 |
| H1_WIDE_BEAM_CAND | -0.475 | (0,19); 0.000 | -0.0054 [-0.0101; -0.0012] | -0.0453 [-0.0590; -0.0316] | 1.05 | 0.38 | 0.99 | 1.49 |
| H2_WIDE_BEAM_CPU | -0.475 | (0,19); 0.000 | -0.0053 [-0.0099; -0.0013] | -0.0457 [-0.0597; -0.0315] | 2.16 | 1.01 | 1.00 | 1.62 |
| N_IRRELEVANT | +0.000 | (0,0); 1.000 | +0.0195 [+0.0096; +0.0298] | +0.0428 [+0.0264; +0.0597] | 0.99 | 0.89 | 1.02 | 1.10 |
| N_INCOMPLETE | +0.000 | (0,0); 1.000 | +0.0039 [-0.0001; +0.0092] | +0.0076 [-0.0001; +0.0168] | 1.01 | 0.99 | 1.00 | 1.00 |
| N_WRONG_SPLIT | +0.000 | (0,0); 1.000 | -0.0002 [-0.0023; +0.0017] | +0.0011 [-0.0033; +0.0070] | 0.97 | 0.98 | 1.00 | 1.01 |
| C_HIERARCHY | -0.450 | (0,18); 0.000 | -0.0047 [-0.0088; -0.0011] | -0.0355 [-0.0476; -0.0231] | 0.75 | 0.85 | 0.99 | 1.49 |

Critérios pré-registados: O1=False; O2(H1)=False; O3(H2)=False; O4(vs WIDE)={'a_fidelity': False, 'b_cost': False, 'c_complexity': False, 'd_stability': False, 'any': False}; **veredicto: NO_ONTOLOGICAL_ADVANTAGE**

Hierarquia exploratória: {'mean_Acc_C>Acc_A': True, 'mean_Acc_A>Acc_C45': False, 'mean_Fid_C>Fid_A': True, 'mean_Fid_C>Fid_C45': False, 'per_seed_rate_Acc_C>A': 0.45, 'per_seed_rate_Fid_C>A': 0.375, 'all_five_conditions_means': False}


### 4.1 Estabilidade (E3, 4 configurações novas × 20 sementes novas, oráculo de forma fechada, `mf = round(0,875·p)`)
| config (grupos × tamanho, regra m-de-k) | A ORIG | B WIDE | C OWL | D PERM | E RAND | G NEUTRALIZED |
|---|---|---|---|---|---|---|
| 6×4, 2-de-4 (recuperação) | 0.65 | 1.00 | 1.00 | 0.65 | 0.65 | 0.65 |
| &nbsp;&nbsp;fidelidade ao oráculo | 0.9246 | 0.9825 | 0.9844 | 0.9708 | 0.9709 | 0.9246 |
| 8×3, 2-de-3 (recuperação) | 0.75 | 1.00 | 1.00 | 0.75 | 0.75 | 0.75 |
| &nbsp;&nbsp;fidelidade ao oráculo | 0.9350 | 0.9855 | 0.9870 | 0.9838 | 0.9832 | 0.9350 |
| 8×5, 3-de-5 (recuperação) | 0.55 | 1.00 | 1.00 | 0.55 | 0.55 | 0.55 |
| &nbsp;&nbsp;fidelidade ao oráculo | 0.8813 | 0.9646 | 0.9691 | 0.9396 | 0.9426 | 0.8806 |
| 6×6, 3-de-6 (recuperação) | 0.45 | 1.00 | 0.95 | 0.45 | 0.45 | 0.45 |
| &nbsp;&nbsp;fidelidade ao oráculo | 0.8829 | 0.9560 | 0.9581 | 0.9279 | 0.9331 | 0.8822 |

Análise **pós-hoc** (não pré-registada): fidelidade C−WIDE emparelhada = +0,0018 [+0,0002; +0,0038], +0,0015 [+0,0001; +0,0035], +0,0045 [+0,0007; +0,0089], +0,0022 [−0,0009; +0,0059] nas 4 configurações (todas abaixo do limiar pré-registado de +0,005), com CPU de ajuste ≈ 1,9–2,4× o de WIDE.

## 5. Auditoria dos testes de identificabilidade (pedido §6)
| # | pergunta | evidência |
|---|---|---|
| 1 | As mesmas classes de candidatos estão disponíveis? | C executa a pesquisa base do Original (A) **mais** 9 pesquisas por unidade: lista 9 388 literais vs 4 424 (B) e 3 844 (A). WIDE lista os literais dos 32 atributos mas o feixe só usa os `3·mf` melhores; as unidades dão ao feixe acesso a literais dos atributos da regra de que o feixe do Original limitado prescinde. As classes de candidatos **não são um subconjunto uma da outra** |
| 2 | A orientação OWL aumenta implicitamente a pesquisa? | **Sim**: candidatos avaliados C/A = 3,52 e C/B = 2,98; CPU C/A = 2,27, C/B = 1,90 |
| 3 | Os controlos tiveram orçamento equivalente? | D, E, F1, F2: razão de candidatos avaliados 1,00–1,04 e mesmas 9 pesquisas por unidade ⇒ **sim** (Comparação 1); G avalia +43 % (0,70); negativos 1,01–1,05; H1 (WIDE) 1,12×; H2 iguala CPU. O B3.1 não media isto — agora está medido |
| 4 | A regra foi construída ou apenas considerada? | **Construída** pelo feixe estatístico do Original dentro da unidade (a ontologia nunca injeta o teste); a ontologia só determina **onde** o feixe procura |
| 5 | A ontologia verdadeira alterou o ranking dos candidatos? | **Não altera pontuações** (`scoring_modified=false`); altera **quais** candidatos entram: `choice_changed` médio 1,2 por árvore (C) vs 0,65 (PERM); 23/40 árvores diferem de A (F2: idêntico a C) |
| 6 | Recuperação por relações semânticas ou por mais tentativas? | Com **as mesmas tentativas** (D, E, F1, G) a recuperação cai para 0,475 = Original ⇒ não são «mais tentativas» por si; mas **F2 (sem ontologia, mesmas tentativas)** recupera 0,975 com árvores idênticas a C: a relação de que C precisa também é obtida dos dados quando os atributos relevantes têm sinal univariado |
| 7 | Os casos negativos preservam oportunidades equivalentes? | Sim: mesmas 9 pesquisas e candidatos avaliados 1,01–1,05× de C; recuperação = A (0,475/0,525) |
| 8 | O 40/40 mantém-se em análises não usadas para calibrar o arnês? | **Em parte:** sementes novas 200–239: C 39/40 (0,975; falha: semente 214) em P1′; E3 (4 configurações novas): 1,00; 1,00; 1,00; 0,95. O resultado é estável **mas** WIDE tem sempre 1,00 |

## 6. Conclusão sobre a contribuição ontológica
1. **Identificável:** a ontologia verdadeira permite que o mecanismo recupere a regra; ontologias permutada/aleatória/neutralizada/irrelevante/incompleta/errada não (controlos válidos, recursos iguais).
2. **Não superior:** (i) F2 iguala C exatamente em P1′; (ii) WIDE iguala a recuperação com menos recursos e com accuracy não inferior (−0,0038 [−0,0101; +0,0006]); (iii) H1/H2 (generic com recursos iguais a C) têm accuracy **superior** (+0,014) e menos nós; (iv) a fidelidade ao MLP satura em ≈ 0,792 em todos os braços que recuperam a regra (limitada pela imperfeição do MLP, acc. 0,793): ΔFid C−B = 0,0000 [−0,0018; +0,0018].
3. **Critério de ganho ontológico (§5 do pedido) não cumprido:** não há melhor fidelidade, melhor accuracy, menor custo, menor complexidade, maior estabilidade (DP da fidelidade C 0,0141 vs B 0,0134) nem vantagem sobre os controlos genéricos de recursos equivalentes.
4. **Quando a ontologia ajudaria (hipótese, não demonstrada aqui):** quando os atributos relevantes **não** têm sinal univariado e não se descobrem por rastreio, e o orçamento de atributos não pode ser alargado — regime não testado. Em P2′ só a **hierarquia** verdadeira (0,975) ultrapassa a ontologia plana (0,525), mas SCREEN/WIDE também resolvem.
5. **Hierarquia do projeto (exploratória; C é Original+orientação, não o Reloaded):** média P1′: Acc(C) 0,957 > Acc(A) 0,867 ✓; **Acc(A) > Acc(C4.5) 0,958 ✗** (o Original limitado é pior que o C4.5 nos rótulos reais, `HIERARCHY_INFEASIBLE`-like, como em 9/18 unidades M29); Fid(C) 0,7922 > Fid(A) 0,7644 ✓; Fid(C) > Fid(C4.5) 0,7905 ✓ em P1′, ✗ em P2′. As cinco condições em conjunto: **não** cumpridas em P1′ nem em P2′ (e C não supera C4.5 em accuracy: −0,0007).
6. **Decisão científica:** a intervenção **não** pode ser declarada cientificamente bem-sucedida; a ausência de ganho nestes testes não impede novas investigações (regimes onde o rastreio e o alargamento do orçamento falham; ontologias enriquecidas validadas; Doc. 39).

## 7. Limitações
Sintético; oráculo MLP deliberadamente imperfeito (a fidelidade ao MLP é um teto, não o objetivo geométrico); regra alinhada com o conceito; um único mecanismo (preferência sobre onde procurar); `mf` calibrado para que o Original falhe ~50 % (por desenho); 40 sementes (ICs de Wilson largos); o braço C **não** é o Reloaded nem usa as features derivadas; custo medido em CPU de um só processo (`process_time`) com compilação em tempo de parede (Java).

## 8. Reprodução
`python -m validation.benchmark.b32_run freeze` (já feito); `… run --problem P1prime_within_group --out …`; `… run --problem P2prime_cross_group …`; `… stability --out …`; `… run --problem P1prime_within_group --memory …`; `python -m validation.benchmark.b32_report --e … --stability … --memory … --out …`.

## 9. Decisões pedidas
1. Aceitar o veredicto `NO_ONTOLOGICAL_ADVANTAGE` nestes regimes e não integrar M0–M6 em `core/`.
2. Autorizar (ou não) um novo desenho pré-registado em **regime difícil para o genérico** (atributos relevantes sem sinal univariado; orçamento de atributos fixo): é onde a ontologia ainda pode oferecer algo que WIDE/SCREEN não oferecem.
3. Designar os revisores do Doc. 39 antes de qualquer ontologia enriquecida real.

## 10. Testes e CI
Testes novos B3.2: `test_b32_comparison.py` (4: contagem de recursos sem efeito nas decisões, braço eficiente ≡ arnês de paridade, controlos com o mesmo nº de pesquisas e candidatos avaliados comparáveis, integridade do pré-registo e hash do código), `test_b32_matching.py` (6), `test_b32_review_package.py` (3). Regressão local (Python 3.11): EQ Original≡Reloaded sem OWL, agnosticismo (contrato, dataset, guardas estáticas), hierarquia conjunta, M29, B3.0, B3.1, B3.2, amostrador C, controlos e B3-a: **439 passed, 40 skipped, 0 failed** (os 40 *skips* são pré-existentes); verificador estático de agnosticismo OK. CI: ver o relatório de entrega (só declarada verde com todos os fluxos concluídos).

## Registo de congelação da B3.2 (decisão do utilizador após a B3.2)

| Campo | Valor |
|---|---|
| `B32_RESULT` | `NO_ONTOLOGICAL_ADVANTAGE` |
| `B32_SCOPE` | `SYNTHETIC_P1_P2_EVALUATED_REGIMES` (o veredicto aplica-se só aos regimes sintéticos P1/P2 avaliados; não é impossibilidade geral) |
| `M0_M6_CORE_INTEGRATION` | `NOT_AUTHORIZED` |
| CI final confirmada | commits `62e5a00` e `0590fa9`: Windows 3.11, Windows 3.12 e agnosticism-guard em sucesso |
| Evidência congelada | `docs/evidence/b1b4/b32/freeze_B32.json` (SHA-256 com fins de linha normalizados de todas as evidências B3.1/B3.2 e do código b30–b32); `git diff 9bc7cbb HEAD -- core` vazio |
