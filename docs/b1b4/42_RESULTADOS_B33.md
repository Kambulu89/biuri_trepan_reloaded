# Doc. 42 — Resultados B3.3: interações sem sinal univariado

Protocolo pré-registado: Doc. 41 (commit `cabcf4f`, anterior a qualquer avaliação nas sementes 400–429). Dados brutos: `docs/evidence/b1b4/b33/results_*.json` (+ `memory_*.json`, sementes de diagnóstico 500–509); análise reconstruída só deles por `validation/benchmark/b33_report.py` → `analysis_b33.json`. Só `validation/`, sintéticos; sem DEV nem confirmatórios; sementes não repetidas nem escolhidas.

## Veredicto global pré-registado: `B33_NO_ONTOLOGICAL_ADVANTAGE`

Nenhuma das famílias I1–I6 satisfaz o critério de vantagem ontológica (`R1`: recuperação de C acima de **todos** os controlos D, E, F, G, H, I em ≥ 0,20). Em todas as famílias com recuperação possível, o Original alargado (B), o rastreio univariado (G), o rastreio de interações por pares (H) e o `beam` calibrado (J1/J2) igualam a recuperação da orientação por ontologia verdadeira (C); em I2 (sem sinal univariado) o melhor braço é o rastreio genérico por pares (H), **não** a ontologia. O resultado não prova impossibilidade geral: aplica-se às famílias, parâmetros e regimes avaliados.

## Tabela por família (30 sementes; média)

| Família | Veredicto | Ganho conjunto C vs A | Atribuição | rec A | rec B | rec C | rec D/E | rec G | rec H | rec I | Δacc C−B | Δfid C−B | SIMPLER_EQUIVALENT a C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1_AND_PAIR | NO_ONTOLOGICAL_ADVANTAGE | SUPPORTED_JOINT_GAIN | NOT_ATTRIBUTED | 0.37 | 1.00 | 1.00 | 1.00/1.00 | 1.00 | 1.00 | 0.37 | -0.0004 | +0.0000 | B_WIDE, F_CONV, G_UNI_SCREEN, J2_WIDE_BEAM_CPU |
| I2_XOR_NO_SIGNAL | NO_ONTOLOGICAL_ADVANTAGE | SUPPORTED_JOINT_GAIN | NOT_ATTRIBUTED | 0.20 | 0.23 | 0.37 | 0.20/0.20 | 0.23 | 0.57 | 0.13 | +0.0604 | +0.0438 | — |
| I3_XOR_WEAK_SIGNAL | NO_ONTOLOGICAL_ADVANTAGE | SUPPORTED_JOINT_GAIN | NOT_ATTRIBUTED | 0.40 | 1.00 | 1.00 | 1.00/1.00 | 1.00 | 1.00 | 0.43 | +0.0033 | +0.0004 | B_WIDE, G_UNI_SCREEN, H_PAIR_SCREEN, J2_WIDE_BEAM_CPU |
| I4_XOR_WEAK_NOISY | NO_ONTOLOGICAL_ADVANTAGE | SUPPORTED_JOINT_GAIN | NOT_ATTRIBUTED | 0.40 | 1.00 | 1.00 | 1.00/1.00 | 1.00 | 1.00 | 0.40 | +0.0069 | +0.0002 | G_UNI_SCREEN |
| I5_XOR_WEAK_HIERARCHY | NO_ONTOLOGICAL_ADVANTAGE | SUPPORTED_JOINT_GAIN | NOT_ATTRIBUTED | 0.37 | 1.00 | 1.00 | 1.00/1.00 | 1.00 | 1.00 | 0.37 | +0.0041 | +0.0011 | B_WIDE, G_UNI_SCREEN, J2_WIDE_BEAM_CPU |
| I6_XOR_WEAK_LARGE_BUDGET | NO_ONTOLOGICAL_ADVANTAGE | SUPPORTED_JOINT_GAIN | NOT_ATTRIBUTED | 0.60 | 1.00 | 1.00 | 1.00/1.00 | 1.00 | 1.00 | 0.60 | +0.0042 | +0.0009 | B_WIDE, G_UNI_SCREEN, J2_WIDE_BEAM_CPU |

## Accuracy / fidelidade / precision / macro-F1 por braço

### I1_AND_PAIR
Referências: C4.5 canónico acc 0.998, fid 0.875; MLP acc 0.875.

| Braço | rec(par no caminho) | acc | fid | precision_macro | macro_F1 | nós int. | prof. | candidatos avaliados | consultas | CPU ajuste (s) | compilação+rastreio (s) | custo total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.37 | 0.839 | 0.791 | 0.749 | 0.745 | 1.7 | 1.7 | 4231 | 186 | 0.72 | 0.00 | 0.72 |
| B_WIDE | 1.00 | 0.999 | 0.875 | 0.999 | 0.998 | 1.1 | 1.1 | 5357 | 459 | 0.72 | 0.00 | 0.72 |
| C_OWL | 1.00 | 0.998 | 0.875 | 0.999 | 0.998 | 1.1 | 1.1 | 14979 | 471 | 1.51 | 1.45 | 2.96 |
| D_PERM | 1.00 | 0.994 | 0.874 | 0.996 | 0.992 | 1.7 | 1.7 | 14848 | 286 | 1.57 | 1.54 | 3.11 |
| E_RAND | 1.00 | 0.995 | 0.874 | 0.996 | 0.993 | 1.7 | 1.7 | 14873 | 279 | 1.52 | 1.53 | 3.06 |
| F_CONV | 1.00 | 0.995 | 0.874 | 0.997 | 0.994 | 1.6 | 1.6 | 15152 | 295 | 1.56 | 0.00 | 1.56 |
| G_UNI_SCREEN | 1.00 | 0.998 | 0.875 | 0.999 | 0.998 | 1.1 | 1.1 | 15009 | 467 | 1.56 | 0.02 | 1.58 |
| H_PAIR_SCREEN | 1.00 | 0.998 | 0.875 | 0.999 | 0.998 | 1.2 | 1.2 | 14989 | 471 | 1.53 | 0.89 | 2.42 |
| I_NEUTRALIZED | 0.37 | 0.837 | 0.790 | 0.751 | 0.736 | 2.0 | 2.0 | 23086 | 184 | 2.26 | 1.48 | 3.75 |
| J1_WIDE_BEAM_CAND | 1.00 | 0.999 | 0.875 | 0.999 | 0.998 | 1.0 | 1.0 | 15304 | 466 | 4.54 | 0.00 | 4.54 |
| J2_WIDE_BEAM_CPU | 1.00 | 0.999 | 0.875 | 0.999 | 0.998 | 1.0 | 1.0 | 7172 | 464 | 1.45 | 0.00 | 1.45 |
| N_IRRELEVANT | 0.37 | 0.839 | 0.791 | 0.754 | 0.745 | 1.8 | 1.8 | 14754 | 182 | 1.72 | 1.53 | 3.25 |
| N_INCOMPLETE | 0.63 | 0.904 | 0.821 | 0.878 | 0.875 | 1.9 | 1.9 | 14755 | 226 | 1.63 | 1.50 | 3.13 |

### I2_XOR_NO_SIGNAL
Referências: C4.5 canónico acc 0.519, fid 0.526; MLP acc 0.844.

| Braço | rec(par no caminho) | acc | fid | precision_macro | macro_F1 | nós int. | prof. | candidatos avaliados | consultas | CPU ajuste (s) | compilação+rastreio (s) | custo total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.20 | 0.510 | 0.524 | 0.518 | 0.480 | 2.6 | 2.6 | 5773 | 3 | 1.26 | 0.00 | 1.26 |
| B_WIDE | 0.23 | 0.512 | 0.525 | 0.520 | 0.479 | 2.5 | 2.5 | 6181 | 2 | 1.30 | 0.00 | 1.30 |
| C_OWL | 0.37 | 0.573 | 0.569 | 0.586 | 0.544 | 2.7 | 2.7 | 17949 | 3 | 2.54 | 1.36 | 3.90 |
| D_PERM | 0.20 | 0.510 | 0.524 | 0.518 | 0.479 | 2.7 | 2.7 | 18042 | 3 | 2.67 | 1.43 | 4.09 |
| E_RAND | 0.20 | 0.510 | 0.525 | 0.517 | 0.479 | 2.6 | 2.6 | 17963 | 5 | 2.56 | 1.39 | 3.96 |
| F_CONV | 0.27 | 0.517 | 0.529 | 0.526 | 0.482 | 2.7 | 2.7 | 18485 | 2 | 2.71 | 0.00 | 2.71 |
| G_UNI_SCREEN | 0.23 | 0.518 | 0.528 | 0.528 | 0.488 | 2.6 | 2.6 | 18073 | 3 | 2.63 | 0.02 | 2.66 |
| H_PAIR_SCREEN | 0.57 | 0.594 | 0.587 | 0.608 | 0.571 | 2.8 | 2.8 | 17855 | 8 | 2.55 | 0.86 | 3.41 |
| I_NEUTRALIZED | 0.13 | 0.511 | 0.525 | 0.518 | 0.483 | 2.6 | 2.6 | 26202 | 2 | 3.26 | 1.41 | 4.67 |
| J1_WIDE_BEAM_CAND | 0.20 | 0.512 | 0.526 | 0.518 | 0.492 | 2.7 | 2.7 | 16334 | 3 | 6.11 | 0.00 | 6.11 |
| J2_WIDE_BEAM_CPU | 0.20 | 0.515 | 0.529 | 0.522 | 0.484 | 2.6 | 2.6 | 9077 | 3 | 2.70 | 0.00 | 2.70 |
| N_IRRELEVANT | 0.17 | 0.510 | 0.524 | 0.517 | 0.479 | 2.6 | 2.6 | 17575 | 3 | 2.52 | 1.40 | 3.92 |
| N_INCOMPLETE | 0.17 | 0.510 | 0.524 | 0.517 | 0.479 | 2.6 | 2.6 | 17765 | 3 | 2.60 | 1.41 | 4.00 |

### I3_XOR_WEAK_SIGNAL
Referências: C4.5 canónico acc 0.997, fid 0.813; MLP acc 0.813.

| Braço | rec(par no caminho) | acc | fid | precision_macro | macro_F1 | nós int. | prof. | candidatos avaliados | consultas | CPU ajuste (s) | compilação+rastreio (s) | custo total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.40 | 0.734 | 0.675 | 0.732 | 0.716 | 2.3 | 2.2 | 4196 | 40 | 0.77 | 0.00 | 0.77 |
| B_WIDE | 1.00 | 0.977 | 0.812 | 0.977 | 0.976 | 2.4 | 2.4 | 5339 | 138 | 0.76 | 0.00 | 0.76 |
| C_OWL | 1.00 | 0.980 | 0.812 | 0.980 | 0.980 | 2.5 | 2.5 | 14903 | 118 | 1.56 | 1.54 | 3.10 |
| D_PERM | 1.00 | 0.952 | 0.797 | 0.952 | 0.952 | 2.8 | 2.1 | 14678 | 55 | 1.51 | 1.54 | 3.05 |
| E_RAND | 1.00 | 0.957 | 0.801 | 0.957 | 0.957 | 2.8 | 2.1 | 14617 | 57 | 1.49 | 1.47 | 2.96 |
| F_CONV | 1.00 | 0.964 | 0.803 | 0.963 | 0.963 | 2.8 | 2.1 | 15149 | 67 | 1.58 | 0.00 | 1.58 |
| G_UNI_SCREEN | 1.00 | 0.980 | 0.812 | 0.980 | 0.980 | 2.5 | 2.5 | 14952 | 118 | 1.61 | 0.02 | 1.63 |
| H_PAIR_SCREEN | 1.00 | 0.980 | 0.812 | 0.980 | 0.980 | 2.5 | 2.5 | 14959 | 118 | 1.60 | 0.88 | 2.48 |
| I_NEUTRALIZED | 0.43 | 0.737 | 0.676 | 0.737 | 0.725 | 2.4 | 2.3 | 23036 | 45 | 2.41 | 1.52 | 3.93 |
| J1_WIDE_BEAM_CAND | 1.00 | 0.977 | 0.812 | 0.977 | 0.976 | 2.4 | 2.4 | 14584 | 138 | 4.36 | 0.00 | 4.36 |
| J2_WIDE_BEAM_CPU | 1.00 | 0.977 | 0.812 | 0.977 | 0.976 | 2.4 | 2.4 | 7102 | 138 | 1.47 | 0.00 | 1.47 |
| N_IRRELEVANT | 0.40 | 0.734 | 0.675 | 0.732 | 0.716 | 2.2 | 2.2 | 14625 | 44 | 1.69 | 1.50 | 3.19 |
| N_INCOMPLETE | 0.67 | 0.843 | 0.737 | 0.843 | 0.838 | 2.4 | 2.1 | 14592 | 52 | 1.66 | 1.51 | 3.17 |

### I4_XOR_WEAK_NOISY
Referências: C4.5 canónico acc 0.831, fid 0.664; MLP acc 0.646.

| Braço | rec(par no caminho) | acc | fid | precision_macro | macro_F1 | nós int. | prof. | candidatos avaliados | consultas | CPU ajuste (s) | compilação+rastreio (s) | custo total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.40 | 0.676 | 0.607 | 0.676 | 0.662 | 2.3 | 2.2 | 4223 | 7 | 0.75 | 0.00 | 0.75 |
| B_WIDE | 1.00 | 0.868 | 0.681 | 0.868 | 0.866 | 2.3 | 2.1 | 5607 | 8 | 0.87 | 0.00 | 0.87 |
| C_OWL | 1.00 | 0.874 | 0.681 | 0.875 | 0.874 | 2.4 | 2.0 | 15038 | 0 | 1.61 | 1.50 | 3.11 |
| D_PERM | 1.00 | 0.837 | 0.667 | 0.838 | 0.836 | 2.8 | 2.0 | 14782 | 25 | 1.61 | 1.46 | 3.07 |
| E_RAND | 1.00 | 0.841 | 0.668 | 0.842 | 0.841 | 2.8 | 2.0 | 14782 | 21 | 1.54 | 1.45 | 3.00 |
| F_CONV | 1.00 | 0.845 | 0.670 | 0.845 | 0.844 | 2.8 | 2.0 | 15155 | 22 | 1.63 | 0.00 | 1.63 |
| G_UNI_SCREEN | 1.00 | 0.875 | 0.681 | 0.875 | 0.874 | 2.4 | 2.1 | 15105 | 0 | 1.66 | 0.02 | 1.68 |
| H_PAIR_SCREEN | 1.00 | 0.873 | 0.681 | 0.873 | 0.872 | 2.4 | 2.0 | 15086 | 0 | 1.66 | 0.89 | 2.54 |
| I_NEUTRALIZED | 0.40 | 0.675 | 0.607 | 0.676 | 0.672 | 2.4 | 2.3 | 23151 | 5 | 2.33 | 1.44 | 3.78 |
| J1_WIDE_BEAM_CAND | 1.00 | 0.867 | 0.680 | 0.868 | 0.866 | 2.2 | 2.1 | 15155 | 8 | 4.74 | 0.00 | 4.74 |
| J2_WIDE_BEAM_CPU | 1.00 | 0.866 | 0.680 | 0.867 | 0.865 | 2.3 | 2.1 | 7300 | 8 | 1.60 | 0.00 | 1.60 |
| N_IRRELEVANT | 0.40 | 0.676 | 0.607 | 0.676 | 0.662 | 2.4 | 2.2 | 14697 | 7 | 1.80 | 1.50 | 3.29 |
| N_INCOMPLETE | 0.67 | 0.758 | 0.637 | 0.758 | 0.755 | 2.5 | 2.1 | 14670 | 18 | 1.71 | 1.51 | 3.22 |

### I5_XOR_WEAK_HIERARCHY
Referências: C4.5 canónico acc 0.998, fid 0.818; MLP acc 0.818.

| Braço | rec(par no caminho) | acc | fid | precision_macro | macro_F1 | nós int. | prof. | candidatos avaliados | consultas | CPU ajuste (s) | compilação+rastreio (s) | custo total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.37 | 0.746 | 0.688 | 0.743 | 0.732 | 2.3 | 2.2 | 3374 | 33 | 0.78 | 0.00 | 0.78 |
| B_WIDE | 1.00 | 0.973 | 0.814 | 0.973 | 0.973 | 2.5 | 2.5 | 5339 | 91 | 0.91 | 0.00 | 0.91 |
| C_OWL | 1.00 | 0.977 | 0.815 | 0.977 | 0.977 | 2.6 | 2.5 | 17994 | 91 | 1.49 | 1.13 | 2.63 |
| D_PERM | 1.00 | 0.963 | 0.807 | 0.963 | 0.963 | 2.8 | 2.1 | 12897 | 50 | 1.33 | 1.08 | 2.41 |
| E_RAND | 1.00 | 0.963 | 0.808 | 0.963 | 0.963 | 2.8 | 2.1 | 12929 | 56 | 1.36 | 1.07 | 2.43 |
| F_CONV | 0.97 | 0.950 | 0.800 | 0.950 | 0.949 | 2.8 | 2.2 | 13326 | 47 | 1.44 | 0.00 | 1.44 |
| G_UNI_SCREEN | 1.00 | 0.977 | 0.815 | 0.977 | 0.977 | 2.6 | 2.6 | 13316 | 91 | 1.52 | 0.02 | 1.54 |
| H_PAIR_SCREEN | 1.00 | 0.977 | 0.815 | 0.977 | 0.977 | 2.6 | 2.5 | 13304 | 91 | 1.50 | 0.87 | 2.36 |
| I_NEUTRALIZED | 0.37 | 0.744 | 0.687 | 0.743 | 0.737 | 2.2 | 2.1 | 21329 | 35 | 2.18 | 1.07 | 3.26 |
| J1_WIDE_BEAM_CAND | 1.00 | 0.973 | 0.814 | 0.973 | 0.973 | 2.5 | 2.4 | 19796 | 91 | 6.57 | 0.00 | 6.57 |
| J2_WIDE_BEAM_CPU | 1.00 | 0.973 | 0.814 | 0.973 | 0.973 | 2.5 | 2.4 | 6532 | 91 | 1.36 | 0.00 | 1.36 |
| N_IRRELEVANT | 0.37 | 0.746 | 0.688 | 0.744 | 0.736 | 2.1 | 2.0 | 13001 | 37 | 1.57 | 1.07 | 2.64 |
| N_INCOMPLETE | 0.53 | 0.810 | 0.722 | 0.810 | 0.803 | 2.4 | 2.1 | 13026 | 42 | 1.51 | 1.07 | 2.58 |

### I6_XOR_WEAK_LARGE_BUDGET
Referências: C4.5 canónico acc 0.911, fid 0.708; MLP acc 0.698.

| Braço | rec(par no caminho) | acc | fid | precision_macro | macro_F1 | nós int. | prof. | candidatos avaliados | consultas | CPU ajuste (s) | compilação+rastreio (s) | custo total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_ORIG | 0.60 | 0.793 | 0.666 | 0.790 | 0.781 | 2.1 | 2.0 | 6553 | 0 | 1.42 | 0.00 | 1.42 |
| B_WIDE | 1.00 | 0.923 | 0.718 | 0.923 | 0.923 | 2.4 | 2.1 | 8871 | 5 | 1.75 | 0.00 | 1.75 |
| C_OWL | 1.00 | 0.927 | 0.719 | 0.927 | 0.927 | 2.4 | 2.2 | 24170 | 5 | 3.46 | 1.47 | 4.93 |
| D_PERM | 1.00 | 0.920 | 0.716 | 0.920 | 0.920 | 2.6 | 2.1 | 24384 | 0 | 3.62 | 1.44 | 5.05 |
| E_RAND | 1.00 | 0.918 | 0.716 | 0.918 | 0.917 | 2.6 | 2.1 | 24253 | 0 | 3.57 | 1.44 | 5.02 |
| F_CONV | 1.00 | 0.921 | 0.716 | 0.920 | 0.920 | 2.6 | 2.1 | 24717 | 0 | 3.61 | 0.00 | 3.61 |
| G_UNI_SCREEN | 1.00 | 0.928 | 0.719 | 0.928 | 0.927 | 2.4 | 2.2 | 24244 | 5 | 3.51 | 0.04 | 3.55 |
| H_PAIR_SCREEN | 1.00 | 0.927 | 0.719 | 0.927 | 0.927 | 2.4 | 2.2 | 24329 | 5 | 3.53 | 2.23 | 5.76 |
| I_NEUTRALIZED | 0.60 | 0.790 | 0.664 | 0.789 | 0.782 | 2.3 | 2.2 | 36929 | 0 | 4.78 | 1.46 | 6.24 |
| J1_WIDE_BEAM_CAND | 1.00 | 0.923 | 0.718 | 0.923 | 0.923 | 2.4 | 2.2 | 20317 | 5 | 6.92 | 0.00 | 6.92 |
| J2_WIDE_BEAM_CPU | 1.00 | 0.923 | 0.718 | 0.923 | 0.923 | 2.4 | 2.1 | 12127 | 5 | 3.18 | 0.00 | 3.18 |
| N_IRRELEVANT | 0.60 | 0.792 | 0.665 | 0.789 | 0.781 | 2.1 | 2.0 | 22253 | 0 | 3.27 | 1.50 | 4.77 |
| N_INCOMPLETE | 0.77 | 0.850 | 0.690 | 0.849 | 0.846 | 2.3 | 2.0 | 24025 | 0 | 3.64 | 1.46 | 5.10 |

## Leitura

1. **O ganho conjunto de C sobre A (Nível 1 literal) existe nas seis famílias (`SUPPORTED_JOINT_GAIN`) mas não é atribuível à ontologia**: B_WIDE, G e H obtêm o mesmo ganho (|Δacc|,|Δfid| ≤ 0,006 face a C em I1 e I3–I6; só em I4 a diferença de acc para B chega a 0,006), com custo total menor (B custa 0,7–1,7 s vs 2,6–4,9 s de C, incluindo a compilação do reasoner Java). Há, porém, um efeito ontológico mensurável **face às partições arbitrárias**: em I3, I4 e I5 a ontologia verdadeira (C) supera a permutada/aleatória (D/E) em accuracy (+0,014 a +0,037) e fidelidade (+0,007 a +0,015), e a neutralizada (I) coincide com o Original limitado — isto é, **a informação útil é a partição correta dos atributos do par**; mas essa mesma informação é obtida a menor custo por B/G/H a partir dos dados, pelo que a atribuição à ontologia falha (`NOT_ATTRIBUTED`).
2. **I2 (sem sinal univariado)**: o limite indutivo confirma-se — recuperação ≤ 0,57 em todos os braços e acc ≤ 0,594 (C4.5 0,519). C melhora sobre A (+0,062 acc, +0,045 fid, IC95 > 0) mas D_PERM obtém **o mesmo** ganho (+0,0626), logo o ganho vem da pesquisa restrita por unidades arbitrárias, não de conhecimento verdadeiro. O melhor braço é H (rastreio genérico de interações por pares: recuperação 0,57 vs 0,37 de C), a um custo total de 3,4 s vs 3,9 s de C. Para esta família `R1` falha por −0,20 (H > C).
3. **Hierarquia (I5)**: a ontologia com pai comum não acrescenta recuperação nem acc/fid face a G/B; o resultado é o mesmo que o da B3.2 (P2′).
4. **Ontologia incompleta/irrelevante/neutralizada**: sem vantagem (recuperação do Original limitado ou intermédia, como esperado — o conceito do par é necessário para a orientação funcionar).
5. **Orçamentos**: nenhuma igualdade implícita foi declarada. C examina mais candidatos que A/B (ver `eval_calls`) e custa mais CPU total; J1 iguala `eval_calls` e J2 iguala CPU ajuste e ambos igualam a recuperação de C. Memória (tracemalloc, médias de pico por semente, MB, 10 sementes × 6 famílias): A_ORIG 3.7, B_WIDE 3.9, C_OWL 4.0, J2_WIDE_BEAM_CPU 3.9.
6. **Cadeia da hierarquia (exploratória)**: `Acc(C) > Acc(A) > Acc(C4.5)` falha em todas as famílias porque o C4.5 canónico supera o Original limitado (em I1, I3, I5 o C4.5 aprende a regra quase perfeitamente com os rótulos reais: acc ≈ 0,998); em termos de unidade é `HIERARCHY_INFEASIBLE` com os baselines fixados: reporta-se, não se corrige. A fidelidade C > A verifica-se em todas; C > C4.5 em fidelidade verifica-se em I1, I2, I4, I6 e falha em I3 e I5.
7. **Saturação de fidelidade**: a fidelidade é limitada pela exatidão do MLP (0,65–0,88); acc e fid de C, G, H e B coincidem à 3.ª casa quando todos recuperam o par.

## Custo da experiência e desvios

CPU de ajuste+compilação+rastreio dos braços ≈ 2,26 CPU-h nas avaliações e memória (limite duro pré-registado 4 CPU-h, não excedido; a geração de instâncias/MLP e C4.5 não é contabilizada nesse número e é pequena). Desvios: (i) I6 passou de p=64 para p=48 (limite do gerador B3.1; o primeiro arranque falhou antes de produzir resultados); (ii) o veredicto não depende de qualquer decisão posterior aos dados: nenhuma semente, família, critério ou braço foi alterado após ver resultados.

## Pendências e decisões

* `M0_M6_CORE_INTEGRATION` permanece `NOT_AUTHORIZED`; nenhuma campanha DEV foi executada.
* Evidência: o melhor método com custo igual ou menor que a ontologia para estas famílias é o rastreio genérico (G para sinal univariado fraco; H para ausência de sinal univariado), e o Original alargado (B) quando o espaço de atributos é pequeno.
* Questão aberta para decisão do responsável: a ontologia só pode oferecer vantagem em regimes onde o conhecimento semântico contenha informação NÃO recuperável dos dados (p. ex. poucos exemplos, ruído que destrói o sinal estatístico mas não a relação, ou restrições semânticas que eliminem candidatos). Estes regimes não foram avaliados aqui.
