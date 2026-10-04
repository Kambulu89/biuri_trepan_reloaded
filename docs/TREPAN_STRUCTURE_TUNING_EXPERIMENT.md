# Tuning da estrutura do TREPAN — Repeated Stratified K-Fold e seleção lexicográfica

Gerado por `scripts/render_tuning_experiment.py` a partir de `scripts/run_trepan_tuning_experiment.py`. Nenhuma regra por dataset; o teste só é usado uma vez, depois da configuração escolhida.

## bc_user.arff

### Estabilidade da configuração entre seeds mestre

| seed mestre | selecionada | concordância interna entre repetições | tuning | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|---|---|
| 42 | `purity_epsilon=0.05, max_nodes=31` | 33% | **INSTÁVEL** | 25 | 0.9300699300699301 | 0.9090909090909091 | 720 |
| 7 | `purity_epsilon=0.01, max_nodes=31` | 67% | estável | 9 | 0.916083916083916 | 0.9230769230769231 | 619 |
| 123 | `purity_epsilon=0.05, max_nodes=31` | 33% | **INSTÁVEL** | 23 | 0.916083916083916 | 0.9230769230769231 | 518 |
| 2024 | `purity_epsilon=0.02, max_nodes=31` | 33% | **INSTÁVEL** | 21 | 0.9370629370629371 | 0.9370629370629371 | 687 |
| 11 | `purity_epsilon=0.01, max_nodes=31` | 0% | **INSTÁVEL** | 15 | 0.9370629370629371 | 0.951048951048951 | 629 |

Configurações escolhidas entre seeds mestre: {'purity_epsilon=0.05, max_nodes=31': 2, 'purity_epsilon=0.01, max_nodes=31': 2, 'purity_epsilon=0.02, max_nodes=31': 1}. Moda `purity_epsilon=0.05, max_nodes=31` em 2/5 (40%) → seleção entre seeds **INSTÁVEL** (limiar 60%).

#### bc_user.arff — seed mestre 42 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [42, 1051, 2060]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.928 | 0.022 | 0.894 | 23.7 ± 6.2 | 7.2 | 0.007 | 0.024 | 0.26 | 4.2 | 15000 | 78% | **WINNER** — VENCE: fidelity média 0.928; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.008, t=0.84 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 4 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=31 | 0.936 | 0.026 | 0.894 | 24.8 ± 4.9 | 6.4 | 0.012 | 0.026 | 0.20 | 4.1 | 15000 | 33% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.86), mas desvio-padrão 0.026 > 0.018 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=31 | 0.936 | 0.026 | 0.894 | 25.0 ± 4.8 | 6.4 | 0.012 | 0.026 | 0.19 | 3.9 | 15000 | 11% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.86), mas desvio-padrão 0.026 > 0.018 + tolerância 0.005. |
| purity_epsilon=0.05, max_nodes=63 | 0.927 | 0.021 | 0.894 | 54.8 ± 3.8 | 11.3 | 0.007 | 0.022 | 0.07 | 10.1 | 31000 | 22% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 54.8 nós médios vs 23.7 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.02, max_nodes=63 | 0.933 | 0.018 | 0.901 | 56.8 ± 4.3 | 10.0 | 0.006 | 0.020 | 0.08 | 10.4 | 31000 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 56.8 nós médios vs 23.7 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.933 | 0.019 | 0.901 | 57.0 ± 4.6 | 9.8 | 0.007 | 0.020 | 0.08 | 10.3 | 31000 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 57.0 nós médios vs 23.7 de purity_epsilon=0.05, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.937 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 |
| purity_epsilon=0.02, max_nodes=63 | 0.951 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 |
| purity_epsilon=0.01, max_nodes=63 | 0.951 | 0.923 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 13 | 19 | 21 | 21 | 31 | 21 | 27 | 31 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 17 | 23 | 21 | 27 | 31 | 21 | 23 | 31 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 17 | 23 | 23 | 27 | 31 | 21 | 23 | 31 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 55 | 59 | 51 | 55 | 49 | 51 | 55 |
| purity_epsilon=0.02, max_nodes=63 | 61 | 61 | 57 | 55 | 51 | 55 | 63 | 57 | 51 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 61 | 55 | 55 | 51 | 57 | 63 | 57 | 51 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(42, 'purity_epsilon=0.05, max_nodes=63'), (1051, 'purity_epsilon=0.05, max_nodes=31'), (2060, 'purity_epsilon=0.02, max_nodes=63')]; concordância 33% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 25, folhas 13, accuracy 0.9300699300699301, fidelity 0.9090909090909091; MLP accuracy 0.951048951048951.

#### bc_user.arff — seed mestre 7 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [7, 1016, 2025]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.926 | 0.036 | 0.866 | 5.7 ± 8.0 | 1.8 | 0.014 | 0.038 | 1.41 | 0.9 | 15000 | 11% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.029, t=1.26 ≤ 1.86), mas desvio-padrão 0.036 > 0.011 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.951 | 0.018 | 0.915 | 22.6 ± 7.9 | 7.7 | 0.014 | 0.014 | 0.35 | 3.6 | 15000 | 78% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.004, t=0.50 ≤ 1.86), mas desvio-padrão 0.018 > 0.011 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=31 | 0.955 | 0.011 | 0.944 | 26.6 ± 3.6 | 7.9 | 0.008 | 0.009 | 0.13 | 5.9 | 15000 | 44% | **WINNER** — VENCE: fidelity média 0.955 (a melhor). 6 indistinguível(eis) em fidelity -> 2 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.05, max_nodes=63 | 0.926 | 0.036 | 0.866 | 9.7 ± 20.0 | 2.2 | 0.014 | 0.038 | 2.07 | 1.5 | 31000 | 0% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.029, t=1.26 ≤ 1.86), mas desvio-padrão 0.036 > 0.011 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.951 | 0.016 | 0.923 | 46.3 ± 17.7 | 10.6 | 0.012 | 0.014 | 0.38 | 8.4 | 31000 | 11% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.004, t=0.63 ≤ 1.86), mas desvio-padrão 0.016 > 0.011 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=63 | 0.955 | 0.011 | 0.944 | 53.0 ± 7.5 | 11.2 | 0.006 | 0.011 | 0.14 | 11.4 | 31000 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 53.0 nós médios vs 26.6 de purity_epsilon=0.01, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.965 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.944 | 0.915 | 0.972 | 0.951 | 0.951 | 0.972 | 0.944 | 0.965 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.951 | 0.944 | 0.965 | 0.951 | 0.958 | 0.972 | 0.944 | 0.965 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.965 |
| purity_epsilon=0.02, max_nodes=63 | 0.944 | 0.944 | 0.923 | 0.972 | 0.944 | 0.951 | 0.972 | 0.944 | 0.965 |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.958 | 0.944 | 0.965 | 0.944 | 0.958 | 0.972 | 0.944 | 0.965 |

Nós por partição:

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 27 | 27 | 21 | 25 | 3 | 23 | 29 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 25 | 21 | 29 | 31 | 27 | 29 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 63 |
| purity_epsilon=0.02, max_nodes=63 | 45 | 57 | 51 | 49 | 53 | 3 | 39 | 57 | 63 |
| purity_epsilon=0.01, max_nodes=63 | 45 | 59 | 57 | 43 | 57 | 53 | 43 | 57 | 63 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=31`. Vencedora por repetição (seed): [(7, 'purity_epsilon=0.01, max_nodes=31'), (1016, 'purity_epsilon=0.01, max_nodes=31'), (2025, 'purity_epsilon=0.02, max_nodes=31')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 9, folhas 5, accuracy 0.916083916083916, fidelity 0.9230769230769231; MLP accuracy 0.9790209790209791.

#### bc_user.arff — seed mestre 123 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [123, 1132, 2141]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.924 | 0.014 | 0.901 | 16.3 ± 10.8 | 5.6 | 0.012 | 0.010 | 0.66 | 2.8 | 15000 | 56% | **WINNER** — VENCE: fidelity média 0.924; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.019, t=1.29 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 5 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=31 | 0.943 | 0.016 | 0.930 | 24.3 ± 4.8 | 7.2 | 0.012 | 0.010 | 0.20 | 4.1 | 15000 | 56% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 24.3 nós médios vs 16.3 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=31 | 0.943 | 0.017 | 0.923 | 26.1 ± 5.0 | 7.0 | 0.016 | 0.010 | 0.19 | 4.2 | 15000 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 26.1 nós médios vs 16.3 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.05, max_nodes=63 | 0.924 | 0.015 | 0.901 | 37.9 ± 26.5 | 8.2 | 0.012 | 0.011 | 0.70 | 6.7 | 31000 | 11% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 37.9 nós médios vs 16.3 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.02, max_nodes=63 | 0.939 | 0.018 | 0.915 | 53.9 ± 4.9 | 11.2 | 0.016 | 0.012 | 0.09 | 9.3 | 31000 | 22% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 53.9 nós médios vs 16.3 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.941 | 0.020 | 0.908 | 54.1 ± 5.4 | 10.4 | 0.020 | 0.011 | 0.10 | 9.6 | 31000 | 0% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.002, t=0.15 ≤ 1.86), mas desvio-padrão 0.020 > 0.014 + tolerância 0.005. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.915 | 0.908 | 0.915 | 0.937 | 0.901 | 0.930 | 0.937 | 0.944 | 0.930 |
| purity_epsilon=0.02, max_nodes=31 | 0.930 | 0.930 | 0.930 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.979 |
| purity_epsilon=0.01, max_nodes=31 | 0.930 | 0.930 | 0.923 | 0.937 | 0.951 | 0.937 | 0.958 | 0.944 | 0.979 |
| purity_epsilon=0.05, max_nodes=63 | 0.908 | 0.908 | 0.923 | 0.937 | 0.901 | 0.930 | 0.937 | 0.944 | 0.930 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.915 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.972 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.937 | 0.908 | 0.937 | 0.951 | 0.944 | 0.951 | 0.958 | 0.972 |

Nós por partição:

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 19 | 3 | 15 | 29 | 3 | 25 | 25 | 25 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 29 | 23 | 29 | 29 | 21 | 25 | 21 | 15 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 31 | 25 | 31 | 29 | 23 | 25 | 29 | 15 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 3 | 55 | 55 | 3 | 45 | 59 | 57 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 47 | 51 | 49 | 55 | 51 | 53 | 59 | 59 | 61 |
| purity_epsilon=0.01, max_nodes=63 | 47 | 45 | 51 | 59 | 55 | 55 | 57 | 57 | 61 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(123, 'purity_epsilon=0.02, max_nodes=31'), (1132, 'purity_epsilon=0.02, max_nodes=31'), (2141, 'purity_epsilon=0.05, max_nodes=31')]; concordância 33% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 23, folhas 12, accuracy 0.916083916083916, fidelity 0.9230769230769231; MLP accuracy 0.965034965034965.

#### bc_user.arff — seed mestre 2024 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [2024, 3033, 4042]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.933 | 0.024 | 0.894 | 18.8 ± 7.6 | 6.8 | 0.017 | 0.022 | 0.40 | 3.6 | 15000 | 78% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.001, t=0.04 ≤ 1.86), mas desvio-padrão 0.024 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.934 | 0.019 | 0.908 | 21.9 ± 4.7 | 6.3 | 0.006 | 0.019 | 0.21 | 4.3 | 15000 | 44% | **WINNER** — VENCE: fidelity média 0.934 (a melhor). 6 indistinguível(eis) em fidelity -> 2 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.01, max_nodes=31 | 0.927 | 0.026 | 0.887 | 24.3 ± 6.4 | 6.7 | 0.012 | 0.024 | 0.26 | 3.9 | 15000 | 0% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.007, t=0.71 ≤ 1.86), mas desvio-padrão 0.026 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.05, max_nodes=63 | 0.933 | 0.022 | 0.908 | 42.8 ± 16.4 | 9.7 | 0.018 | 0.018 | 0.38 | 8.5 | 31000 | 22% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.002, t=0.08 ≤ 1.86), mas desvio-padrão 0.022 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.928 | 0.016 | 0.908 | 52.8 ± 3.1 | 9.2 | 0.001 | 0.018 | 0.06 | 8.9 | 31000 | 11% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 52.8 nós médios vs 21.9 de purity_epsilon=0.02, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.022 | 0.894 | 55.0 ± 1.0 | 9.7 | 0.002 | 0.024 | 0.02 | 9.2 | 31000 | 0% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.009, t=1.14 ≤ 1.86), mas desvio-padrão 0.022 > 0.016 + tolerância 0.005. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.923 | 0.894 | 0.951 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.915 |
| purity_epsilon=0.02, max_nodes=31 | 0.937 | 0.937 | 0.951 | 0.923 | 0.965 | 0.908 | 0.951 | 0.923 | 0.915 |
| purity_epsilon=0.01, max_nodes=31 | 0.937 | 0.937 | 0.951 | 0.887 | 0.958 | 0.908 | 0.951 | 0.923 | 0.894 |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.908 | 0.937 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.908 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.923 | 0.958 | 0.908 | 0.944 | 0.930 | 0.908 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.901 | 0.965 | 0.908 | 0.944 | 0.930 | 0.894 |

Nós por partição:

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 25 | 25 | 13 | 21 | 13 | 21 | 23 | 3 | 25 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 29 | 13 | 21 | 23 | 23 | 17 | 25 | 25 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 13 | 29 | 31 | 23 | 17 | 25 | 31 |
| purity_epsilon=0.05, max_nodes=63 | 33 | 51 | 49 | 41 | 53 | 55 | 49 | 3 | 51 |
| purity_epsilon=0.02, max_nodes=63 | 55 | 55 | 49 | 47 | 53 | 55 | 55 | 55 | 51 |
| purity_epsilon=0.01, max_nodes=63 | 55 | 55 | 55 | 55 | 53 | 55 | 55 | 55 | 57 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31`. Vencedora por repetição (seed): [(2024, 'purity_epsilon=0.02, max_nodes=31'), (3033, 'purity_epsilon=0.05, max_nodes=31'), (4042, 'purity_epsilon=0.05, max_nodes=31')]; concordância 33% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 21, folhas 11, accuracy 0.9370629370629371, fidelity 0.9370629370629371; MLP accuracy 0.972027972027972.

#### bc_user.arff — seed mestre 11 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [11, 1020, 2029]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.941 | 0.030 | 0.873 | 22.8 ± 11.2 | 6.9 | 0.020 | 0.024 | 0.49 | 3.3 | 15000 | 56% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.006, t=0.45 ≤ 1.86), mas desvio-padrão 0.030 > 0.021 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.948 | 0.023 | 0.901 | 23.7 ± 3.9 | 6.7 | 0.015 | 0.020 | 0.16 | 4.2 | 15000 | 33% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 23.7 nós médios vs 22.8 de purity_epsilon=0.01, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.021 | 0.901 | 22.8 ± 4.4 | 6.2 | 0.010 | 0.021 | 0.19 | 4.3 | 15000 | 11% | **WINNER** — VENCE: fidelity média 0.944; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.003, t=0.56 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 4 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.05, max_nodes=63 | 0.941 | 0.029 | 0.873 | 43.0 ± 23.0 | 9.4 | 0.019 | 0.023 | 0.54 | 7.3 | 31000 | 11% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.007, t=0.51 ≤ 1.86), mas desvio-padrão 0.029 > 0.021 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.944 | 0.025 | 0.887 | 55.2 ± 7.6 | 9.9 | 0.014 | 0.022 | 0.14 | 9.7 | 31000 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 55.2 nós médios vs 22.8 de purity_epsilon=0.01, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.942 | 0.024 | 0.887 | 55.2 ± 6.4 | 9.8 | 0.013 | 0.021 | 0.12 | 8.9 | 31000 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 55.2 nós médios vs 22.8 de purity_epsilon=0.01, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.972 | 0.937 | 0.937 | 0.944 |
| purity_epsilon=0.02, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.965 | 0.972 | 0.965 | 0.937 | 0.923 |
| purity_epsilon=0.01, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.944 | 0.965 | 0.965 | 0.937 | 0.923 |
| purity_epsilon=0.05, max_nodes=63 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 |
| purity_epsilon=0.02, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.965 | 0.951 | 0.965 | 0.944 | 0.923 |
| purity_epsilon=0.01, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.951 | 0.958 | 0.965 | 0.937 | 0.923 |

Nós por partição:

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 29 | 29 | 3 | 29 | 27 | 27 | 29 | 29 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 29 | 21 | 17 | 25 | 23 | 21 | 23 | 25 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 29 | 21 | 17 | 25 | 23 | 21 | 23 | 17 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 59 | 3 | 51 | 51 | 53 | 49 | 61 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 53 | 59 | 61 | 37 | 59 | 61 | 51 | 59 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 53 | 59 | 61 | 47 | 57 | 61 | 43 | 59 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=31`. Vencedora por repetição (seed): [(11, 'purity_epsilon=0.02, max_nodes=31'), (1020, 'purity_epsilon=0.02, max_nodes=31'), (2029, 'purity_epsilon=0.05, max_nodes=31')]; concordância 0% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 15, folhas 8, accuracy 0.9370629370629371, fidelity 0.951048951048951; MLP accuracy 0.972027972027972.

## iris_t.arff

### Estabilidade da configuração entre seeds mestre

| seed mestre | selecionada | concordância interna entre repetições | tuning | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|---|---|
| 42 | `purity_epsilon=0.02, max_nodes=31` | 33% | **INSTÁVEL** | 31 | 0.8421052631578947 | 0.9473684210526315 | 78 |
| 7 | `purity_epsilon=0.05, max_nodes=31` | 67% | estável | 27 | 0.9736842105263158 | 0.9736842105263158 | 69 |
| 123 | `purity_epsilon=0.05, max_nodes=31` | 100% | estável | 31 | 0.9473684210526315 | 1.0 | 79 |
| 2024 | `purity_epsilon=0.05, max_nodes=31` | 67% | estável | 31 | 0.868421052631579 | 0.8947368421052632 | 73 |
| 11 | `purity_epsilon=0.02, max_nodes=31` | 67% | estável | 19 | 0.8947368421052632 | 0.9210526315789473 | 73 |

Configurações escolhidas entre seeds mestre: {'purity_epsilon=0.02, max_nodes=31': 2, 'purity_epsilon=0.05, max_nodes=31': 3}. Moda `purity_epsilon=0.05, max_nodes=31` em 3/5 (60%) → seleção entre seeds **estável** (limiar 60%).

#### iris_t.arff — seed mestre 42 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [42, 1051, 2060]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.919 | 0.041 | 0.865 | 27.7 ± 3.2 | 7.7 | 0.031 | 0.033 | 0.11 | 0.5 | 5040 | 33% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.015, t=0.71 ≤ 1.86), mas desvio-padrão 0.041 > 0.033 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.923 | 0.037 | 0.892 | 29.2 ± 2.9 | 8.0 | 0.026 | 0.028 | 0.10 | 0.5 | 5040 | 0% | **WINNER** — VENCE: fidelity média 0.923; indistinguível da melhor (purity_epsilon=0.02, max_nodes=63, Δ=0.012, t=0.64 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 4 após estabilidade -> 2 após complexidade; empate real resolvido pela configuração mais simples. |
| purity_epsilon=0.01, max_nodes=31 | 0.923 | 0.037 | 0.892 | 29.2 ± 2.9 | 8.0 | 0.026 | 0.028 | 0.10 | 0.5 | 5040 | 0% | **LOST** — PERDE o desempate: empate real em fidelity, estabilidade e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=31). |
| purity_epsilon=0.05, max_nodes=63 | 0.931 | 0.038 | 0.865 | 59.0 ± 3.0 | 12.7 | 0.027 | 0.033 | 0.05 | 1.0 | 10416 | 0% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.003, t=0.25 ≤ 1.86), mas desvio-padrão 0.038 > 0.033 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.935 | 0.033 | 0.892 | 60.1 ± 3.3 | 12.4 | 0.022 | 0.029 | 0.06 | 1.0 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 60.1 nós médios vs 29.2 de purity_epsilon=0.02, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.935 | 0.033 | 0.892 | 60.1 ± 3.3 | 12.4 | 0.022 | 0.029 | 0.06 | 1.0 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 60.1 nós médios vs 29.2 de purity_epsilon=0.02, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.895 | 0.892 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.892 |
| purity_epsilon=0.02, max_nodes=31 | 0.895 | 0.892 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.892 |
| purity_epsilon=0.01, max_nodes=31 | 0.895 | 0.892 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.892 |
| purity_epsilon=0.05, max_nodes=63 | 0.895 | 0.946 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.946 |
| purity_epsilon=0.02, max_nodes=63 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 |
| purity_epsilon=0.01, max_nodes=63 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 27 | 27 | 23 | 23 | 29 | 31 | 27 | 31 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 57 | 57 | 59 | 53 | 59 | 63 | 61 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31`. Vencedora por repetição (seed): [(42, 'purity_epsilon=0.02, max_nodes=31'), (1051, 'purity_epsilon=0.05, max_nodes=31'), (2060, 'purity_epsilon=0.05, max_nodes=63')]; concordância 33% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 31, folhas 16, accuracy 0.8421052631578947, fidelity 0.9473684210526315; MLP accuracy 0.8947368421052632.

#### iris_t.arff — seed mestre 7 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [7, 1016, 2025]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.926 | 0.043 | 0.842 | 28.1 ± 3.0 | 7.6 | 0.018 | 0.045 | 0.11 | 0.4 | 5040 | 0% | **WINNER** — VENCE: fidelity média 0.926 (a melhor). 6 indistinguível(eis) em fidelity -> 6 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=31 | 0.923 | 0.045 | 0.842 | 28.6 ± 3.1 | 7.6 | 0.013 | 0.048 | 0.11 | 0.5 | 5040 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 28.6 nós médios vs 28.1 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=31 | 0.923 | 0.045 | 0.842 | 28.6 ± 3.1 | 7.6 | 0.013 | 0.048 | 0.11 | 0.4 | 5040 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 28.6 nós médios vs 28.1 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.043 | 0.842 | 56.1 ± 4.6 | 11.2 | 0.022 | 0.043 | 0.08 | 0.9 | 10416 | 11% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 56.1 nós médios vs 28.1 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.02, max_nodes=63 | 0.920 | 0.044 | 0.842 | 59.0 ± 3.2 | 11.3 | 0.017 | 0.046 | 0.05 | 0.9 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 59.0 nós médios vs 28.1 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.920 | 0.044 | 0.842 | 59.0 ± 3.2 | 11.3 | 0.017 | 0.046 | 0.05 | 0.9 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 59.0 nós médios vs 28.1 de purity_epsilon=0.05, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 |
| purity_epsilon=0.02, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 |
| purity_epsilon=0.01, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 |
| purity_epsilon=0.05, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 |
| purity_epsilon=0.02, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 |
| purity_epsilon=0.01, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 |

Nós por partição:

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 27 | 31 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 31 | 31 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 31 | 31 |
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 61 | 53 | 59 | 57 | 49 | 59 | 49 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(7, 'purity_epsilon=0.05, max_nodes=63'), (1016, 'purity_epsilon=0.05, max_nodes=31'), (2025, 'purity_epsilon=0.05, max_nodes=31')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 27, folhas 14, accuracy 0.9736842105263158, fidelity 0.9736842105263158; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 123 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [123, 1132, 2141]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.940 | 0.038 | 0.865 | 26.6 ± 2.6 | 8.2 | 0.034 | 0.026 | 0.10 | 0.6 | 5040 | 33% | **WINNER** — VENCE: fidelity média 0.940 (a melhor). 6 indistinguível(eis) em fidelity -> 6 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.038 | 0.865 | 26.8 ± 2.7 | 8.0 | 0.034 | 0.026 | 0.10 | 0.5 | 5040 | 22% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 26.8 nós médios vs 26.6 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=31 | 0.940 | 0.038 | 0.865 | 27.0 ± 2.6 | 8.0 | 0.034 | 0.026 | 0.10 | 0.5 | 5040 | 11% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 27.0 nós médios vs 26.6 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.05, max_nodes=63 | 0.940 | 0.038 | 0.865 | 59.7 ± 4.1 | 12.8 | 0.034 | 0.026 | 0.07 | 1.1 | 10416 | 11% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 59.7 nós médios vs 26.6 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.02, max_nodes=63 | 0.940 | 0.038 | 0.865 | 60.8 ± 3.8 | 12.8 | 0.034 | 0.026 | 0.06 | 1.3 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 60.8 nós médios vs 26.6 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.940 | 0.038 | 0.865 | 60.8 ± 3.8 | 12.7 | 0.034 | 0.026 | 0.06 | 1.0 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 60.8 nós médios vs 26.6 de purity_epsilon=0.05, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 |
| purity_epsilon=0.01, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 |
| purity_epsilon=0.05, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 |
| purity_epsilon=0.02, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 |
| purity_epsilon=0.01, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 |

Nós por partição:

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 27 | 25 | 23 | 27 | 27 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 25 | 23 | 27 | 29 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 27 | 23 | 27 | 29 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 63 | 63 | 61 | 61 | 51 | 59 | 63 | 55 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(123, 'purity_epsilon=0.05, max_nodes=31'), (1132, 'purity_epsilon=0.05, max_nodes=31'), (2141, 'purity_epsilon=0.05, max_nodes=31')]; concordância 100% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 31, folhas 16, accuracy 0.9473684210526315, fidelity 1.0; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 2024 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [2024, 3033, 4042]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.955 | 0.019 | 0.919 | 29.4 ± 1.9 | 8.4 | 0.009 | 0.019 | 0.07 | 0.4 | 5040 | 0% | **WINNER** — VENCE: fidelity média 0.955; indistinguível da melhor (purity_epsilon=0.05, max_nodes=63, Δ=0.003, t=0.43 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 6 após estabilidade -> 3 após complexidade; empate real resolvido pela configuração mais simples. |
| purity_epsilon=0.02, max_nodes=31 | 0.955 | 0.019 | 0.919 | 29.4 ± 1.9 | 8.4 | 0.009 | 0.019 | 0.07 | 0.4 | 5040 | 0% | **LOST** — PERDE o desempate: empate real em fidelity, estabilidade e nós; prevalece a configuração mais simples (purity_epsilon=0.05, max_nodes=31). |
| purity_epsilon=0.01, max_nodes=31 | 0.955 | 0.019 | 0.919 | 29.4 ± 1.9 | 8.2 | 0.009 | 0.019 | 0.07 | 0.4 | 5040 | 0% | **LOST** — PERDE o desempate: empate real em fidelity, estabilidade e nós; prevalece a configuração mais simples (purity_epsilon=0.05, max_nodes=31). |
| purity_epsilon=0.05, max_nodes=63 | 0.958 | 0.014 | 0.946 | 57.9 ± 3.2 | 12.6 | 0.005 | 0.015 | 0.05 | 1.0 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 57.9 nós médios vs 29.4 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.02, max_nodes=63 | 0.958 | 0.014 | 0.946 | 59.7 ± 2.2 | 12.9 | 0.005 | 0.015 | 0.04 | 1.2 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 59.7 nós médios vs 29.4 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.958 | 0.014 | 0.946 | 59.7 ± 2.2 | 12.7 | 0.005 | 0.015 | 0.04 | 1.1 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 59.7 nós médios vs 29.4 de purity_epsilon=0.05, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 |

Nós por partição:

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 55 | 61 | 59 | 59 | 51 | 59 | 57 | 59 |
| purity_epsilon=0.02, max_nodes=63 | 61 | 63 | 63 | 59 | 59 | 57 | 59 | 57 | 59 |
| purity_epsilon=0.01, max_nodes=63 | 61 | 63 | 63 | 59 | 59 | 57 | 59 | 57 | 59 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(2024, 'purity_epsilon=0.05, max_nodes=31'), (3033, 'purity_epsilon=0.05, max_nodes=31'), (4042, 'purity_epsilon=0.05, max_nodes=63')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 31, folhas 16, accuracy 0.868421052631579, fidelity 0.8947368421052632; MLP accuracy 0.9210526315789473.

#### iris_t.arff — seed mestre 11 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [11, 1020, 2029]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.934 | 0.057 | 0.811 | 29.4 ± 1.7 | 8.3 | 0.011 | 0.055 | 0.06 | 0.4 | 5040 | 22% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.012, t=0.78 ≤ 1.86), mas desvio-padrão 0.057 > 0.045 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.946 | 0.045 | 0.865 | 30.3 ± 1.0 | 8.7 | 0.009 | 0.048 | 0.03 | 0.4 | 5040 | 0% | **WINNER** — VENCE: fidelity média 0.946 (a melhor). 6 indistinguível(eis) em fidelity -> 4 após estabilidade -> 2 após complexidade; empate real resolvido pela configuração mais simples. |
| purity_epsilon=0.01, max_nodes=31 | 0.946 | 0.045 | 0.865 | 30.3 ± 1.0 | 8.7 | 0.009 | 0.048 | 0.03 | 0.4 | 5040 | 0% | **LOST** — PERDE o desempate: empate real em fidelity, estabilidade e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=31). |
| purity_epsilon=0.05, max_nodes=63 | 0.934 | 0.057 | 0.811 | 57.9 ± 4.3 | 12.4 | 0.011 | 0.055 | 0.07 | 0.9 | 10416 | 22% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.012, t=0.78 ≤ 1.86), mas desvio-padrão 0.057 > 0.045 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.946 | 0.045 | 0.865 | 57.7 ± 3.3 | 12.4 | 0.009 | 0.048 | 0.06 | 1.0 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 57.7 nós médios vs 30.3 de purity_epsilon=0.02, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.946 | 0.045 | 0.865 | 57.7 ± 3.3 | 12.4 | 0.009 | 0.048 | 0.06 | 1.0 | 10416 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 57.7 nós médios vs 30.3 de purity_epsilon=0.02, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 1.000 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 |
| purity_epsilon=0.02, max_nodes=31 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 |
| purity_epsilon=0.01, max_nodes=31 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 |
| purity_epsilon=0.05, max_nodes=63 | 1.000 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 |
| purity_epsilon=0.02, max_nodes=63 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 |
| purity_epsilon=0.01, max_nodes=63 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 |

Nós por partição:

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 29 | 31 | 27 | 31 | 29 | 31 | 27 | 31 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 57 | 59 | 49 | 59 | 63 | 59 | 55 | 63 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31`. Vencedora por repetição (seed): [(11, 'purity_epsilon=0.02, max_nodes=31'), (1020, 'purity_epsilon=0.05, max_nodes=31'), (2029, 'purity_epsilon=0.02, max_nodes=31')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 19, folhas 10, accuracy 0.8947368421052632, fidelity 0.9210526315789473; MLP accuracy 0.9736842105263158.

---

## Caminho da GUI (o que produziu a árvore de 3 nós)

Mesmo tuning, mas dentro de `BiuriApp._execute_training_pipeline` (MLP ajustado com Optuna, orçamento do preset, seed 42). O teste foi avaliado uma única vez no fim (accuracy e fidelity da árvore final: 0,889 / 0,889; 3 nós, 2 folhas).

#### bc_user.arff (caminho da GUI: MLP com Optuna, orçamento do preset) — seed mestre 42 (treino 398, teste 171)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [42, 1051, 2060]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade (se indistinguíveis) -> complexidade -> mais simples.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | prof. média | entre-seeds dp | dp intra-seed | CV de nós | tempo médio/ajuste (s) | orçamento queries | orçamento esgotado | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.937 | 0.020 | 0.910 | 9.4 ± 9.8 | 3.3 | 0.020 | 0.011 | 1.04 | 1.9 | 15000 | 33% | **WINNER** — VENCE: fidelity média 0.937; indistinguível da melhor (purity_epsilon=0.01, max_nodes=31, Δ=0.006, t=0.61 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 4 após estabilidade -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=31 | 0.942 | 0.023 | 0.887 | 24.1 ± 3.9 | 6.7 | 0.018 | 0.016 | 0.16 | 5.3 | 15000 | 78% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.001, t=0.09 ≤ 1.86), mas desvio-padrão 0.023 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=31 | 0.943 | 0.016 | 0.917 | 26.6 ± 3.6 | 6.8 | 0.011 | 0.014 | 0.13 | 5.2 | 15000 | 33% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 26.6 nós médios vs 9.4 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.05, max_nodes=63 | 0.937 | 0.020 | 0.910 | 19.9 ± 25.4 | 4.7 | 0.020 | 0.011 | 1.28 | 4.6 | 31000 | 22% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 19.9 nós médios vs 9.4 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.02, max_nodes=63 | 0.936 | 0.022 | 0.887 | 55.4 ± 3.6 | 10.6 | 0.017 | 0.018 | 0.06 | 13.3 | 31000 | 11% | **LOST** — PERDE em estabilidade: indistinguível em fidelity (Δ=0.007, t=0.78 ≤ 1.86), mas desvio-padrão 0.022 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=63 | 0.938 | 0.016 | 0.917 | 55.7 ± 3.7 | 10.0 | 0.010 | 0.015 | 0.07 | 13.1 | 31000 | 0% | **LOST** — PERDE em complexidade: indistinguível em fidelity e estabilidade, mas com 55.7 nós médios vs 9.4 de purity_epsilon=0.05, max_nodes=31. |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.940 | 0.955 | 0.939 | 0.910 | 0.910 | 0.924 | 0.940 | 0.970 | 0.947 |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.955 | 0.947 | 0.940 | 0.887 | 0.939 | 0.947 | 0.970 | 0.955 |
| purity_epsilon=0.01, max_nodes=31 | 0.925 | 0.955 | 0.947 | 0.940 | 0.917 | 0.939 | 0.947 | 0.970 | 0.947 |
| purity_epsilon=0.05, max_nodes=63 | 0.940 | 0.955 | 0.939 | 0.910 | 0.910 | 0.924 | 0.940 | 0.970 | 0.947 |
| purity_epsilon=0.02, max_nodes=63 | 0.932 | 0.947 | 0.947 | 0.932 | 0.887 | 0.932 | 0.947 | 0.970 | 0.932 |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.947 | 0.947 | 0.932 | 0.917 | 0.932 | 0.947 | 0.970 | 0.924 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 3 | 3 | 3 | 19 | 3 | 3 | 23 | 25 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 23 | 29 | 15 | 25 | 25 | 25 | 23 | 25 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 25 | 29 | 23 | 25 | 31 | 25 | 29 | 21 | 31 |
| purity_epsilon=0.05, max_nodes=63 | 3 | 3 | 3 | 57 | 3 | 3 | 51 | 53 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 55 | 55 | 57 | 49 | 59 | 57 | 53 | 53 | 61 |
| purity_epsilon=0.01, max_nodes=63 | 55 | 55 | 61 | 49 | 57 | 57 | 51 | 57 | 59 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(42, 'purity_epsilon=0.05, max_nodes=31'), (1051, 'purity_epsilon=0.05, max_nodes=31'), (2060, 'purity_epsilon=0.05, max_nodes=31')]; concordância 100% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 3, folhas 2, accuracy 0.8889, fidelity 0.8889; MLP accuracy None.

Tamanho das árvores ajustadas na CV para `purity_epsilon=0.05, max_nodes=31` (por partição, seed · dobra): s42·f0=3, s42·f1=3, s42·f2=3, s1051·f0=19, s1051·f1=3, s1051·f2=3, s2060·f0=23, s2060·f1=25, s2060·f2=3. Árvores de 3 nós em 6 de 9 partições.
