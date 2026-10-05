# Tuning da estrutura do TREPAN — Repeated Stratified K-Fold, orçamento comum não limitante e seleção lexicográfica com estabilidade estrutural

Gerado por `scripts/render_tuning_experiment.py` a partir de `scripts/run_trepan_tuning_experiment.py`. Nenhuma regra por dataset; o teste só é usado uma vez, depois da configuração escolhida.

## bc_user.arff — oráculo `factory`

### Estabilidade da configuração entre seeds mestre

| seed mestre | oracle_id | selecionada | concordância interna entre repetições | tuning | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|---|---|---|
| 42 | `457012f5f057926b` | `purity_epsilon=0.05, max_nodes=31` | 0% | **INSTÁVEL** | 25 | 0.9300699300699301 | 0.9090909090909091 | 710 |
| 7 | `ea40639ef0e616a1` | `purity_epsilon=0.01, max_nodes=31` | 67% | estável | 27 | 0.9230769230769231 | 0.9440559440559441 | 682 |
| 123 | `d156e8f076a2dfb2` | `purity_epsilon=0.02, max_nodes=31` | 67% | estável | 27 | 0.9090909090909091 | 0.916083916083916 | 645 |
| 2024 | `a2f113e60f488f42` | `purity_epsilon=0.02, max_nodes=63` | 33% | **INSTÁVEL** | 35 | 0.9370629370629371 | 0.9370629370629371 | 1108 |
| 11 | `9aa83790a361ec28` | `purity_epsilon=0.02, max_nodes=63` | 0% | **INSTÁVEL** | 47 | 0.9370629370629371 | 0.951048951048951 | 1199 |

Configurações escolhidas entre seeds mestre: {'purity_epsilon=0.05, max_nodes=31': 1, 'purity_epsilon=0.01, max_nodes=31': 1, 'purity_epsilon=0.02, max_nodes=31': 1, 'purity_epsilon=0.02, max_nodes=63': 2}. Moda `purity_epsilon=0.02, max_nodes=63` em 2/5 (40%) → seleção entre seeds **INSTÁVEL** (limiar 60%).

### Conclusão — bc_user.arff

**A árvore de 3 nós não é suportada como uma estrutura robusta e consistentemente preferível; a sua fidelity pode ser equivalente, mas a sua estrutura apresenta forte dependência da amostragem.**

Evidência (só treino, CV repetida): 10 de 30 (seed mestre × configuração) produziram stumps em parte das partições; destas, 0 foram selecionadas.

| seed mestre | configuração | fração de stumps | fidelity média | instab. estrutural | resultado |
|---|---|---|---|---|---|
| 7 | purity_epsilon=0.05, max_nodes=31 | 89% | 0.925 | 1.47 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=31 | 11% | 0.950 | 0.37 | LOST_STABILITY |
| 7 | purity_epsilon=0.05, max_nodes=63 | 89% | 0.926 | 2.07 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=63 | 11% | 0.951 | 0.38 | LOST_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=31 | 33% | 0.923 | 0.68 | LOST_STRUCTURAL_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=63 | 33% | 0.924 | 0.69 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=31 | 11% | 0.934 | 0.35 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=63 | 11% | 0.933 | 0.39 | LOST_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=31 | 22% | 0.941 | 0.50 | LOST_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=63 | 22% | 0.941 | 0.54 | LOST_STABILITY |

#### bc_user.arff — seed mestre 42 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [42, 1051, 2060]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=457012f5f057926b` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '457012f5f057926b', 'trepan_reloaded': '457012f5f057926b'}); queries por escopo: {'tuning': 2661022, 'trepan_pair': 47711, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 31955; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.927 | 0.022 | 0.894 | 28.6 ± 3.6 | 0.13 | 8.4 (0.19) | 14.8 (0.12) | 0.19 | 100% | 63001 | 15822 | 0/9 | 0% | 4.1 | **WINNER** — VENCE: fidelity média 0.927; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.009, t=0.79 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 4 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.19) -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=31 | 0.936 | 0.026 | 0.894 | 25.7 ± 4.9 | 0.19 | 6.6 (0.17) | 13.3 (0.18) | 0.19 | 100% | 63001 | 14650 | 0/9 | 0% | 4.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.86), mas desvio-padrão 0.026 > 0.018 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=31 | 0.936 | 0.026 | 0.894 | 25.0 ± 4.8 | 0.19 | 6.4 (0.14) | 13.0 (0.18) | 0.19 | 100% | 63001 | 14332 | 0/9 | 0% | 3.9 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.86), mas desvio-padrão 0.026 > 0.018 + tolerância 0.005. |
| purity_epsilon=0.05, max_nodes=63 | 0.927 | 0.021 | 0.894 | 54.8 ± 3.8 | 0.07 | 11.3 (0.20) | 27.9 (0.07) | 0.20 | 100% | 63001 | 30396 | 0/9 | 0% | 9.7 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.8 nós médios vs 28.6 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.02, max_nodes=63 | 0.933 | 0.018 | 0.901 | 56.8 ± 4.3 | 0.08 | 10.0 (0.23) | 28.9 (0.07) | 0.23 | 100% | 63001 | 29641 | 0/9 | 0% | 10.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 56.8 nós médios vs 28.6 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.933 | 0.019 | 0.901 | 57.0 ± 4.6 | 0.08 | 9.8 (0.19) | 29.0 (0.08) | 0.19 | 100% | 63001 | 29270 | 0/9 | 0% | 10.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.0 nós médios vs 28.6 de purity_epsilon=0.05, max_nodes=31. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.927 | 0.022 | 0.024 | 0.007 | 0.920, 0.934, 0.927 |
| purity_epsilon=0.02, max_nodes=31 | 0.936 | 0.026 | 0.026 | 0.012 | 0.925, 0.934, 0.948 |
| purity_epsilon=0.01, max_nodes=31 | 0.936 | 0.026 | 0.026 | 0.012 | 0.925, 0.934, 0.948 |
| purity_epsilon=0.05, max_nodes=63 | 0.927 | 0.021 | 0.022 | 0.007 | 0.920, 0.934, 0.927 |
| purity_epsilon=0.02, max_nodes=63 | 0.933 | 0.018 | 0.020 | 0.006 | 0.927, 0.934, 0.939 |
| purity_epsilon=0.01, max_nodes=63 | 0.933 | 0.019 | 0.020 | 0.007 | 0.925, 0.934, 0.939 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_se 78%, area_worst 44%, concave_points_mean 44% | 78% | 0.65 | 0.24 | 10 | 0.59 | 0.18 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.02, max_nodes=31 | radius_se 78%, area_worst 44%, concave_points_mean 44% | 78% | 0.61 | 0.25 | 3 | 0.68 | 0.20 | 0% | area_worst 100%, concavity_worst 100%, radius_se 100%, symmetry_worst 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_se 78%, area_worst 44%, concave_points_mean 44% | 78% | 0.61 | 0.25 | 4 | 0.71 | 0.22 | 0% | area_worst 100%, concavity_worst 100%, radius_se 100%, symmetry_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_se 78%, area_worst 44%, concave_points_mean 44% | 78% | 0.70 | 0.32 | 7 | 0.77 | 0.33 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.02, max_nodes=63 | radius_se 78%, area_worst 44%, concave_points_mean 44% | 78% | 0.71 | 0.36 | 4 | 0.72 | 0.32 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_se 78%, area_worst 44%, concave_points_mean 44% | 78% | 0.72 | 0.37 | 4 | 0.75 | 0.35 | 0% | area_se 100%, area_worst 100%, compactness_mean 100%, concavity_mean 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 |
| purity_epsilon=0.02, max_nodes=63 | 0.951 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 |
| purity_epsilon=0.01, max_nodes=63 | 0.951 | 0.923 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 27 | 21 | 25 | 31 | 31 | 29 | 31 | 31 | 31 |
| purity_epsilon=0.02, max_nodes=31 | 17 | 23 | 21 | 27 | 31 | 29 | 23 | 31 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 17 | 23 | 23 | 27 | 31 | 21 | 23 | 31 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 55 | 59 | 51 | 55 | 49 | 51 | 55 |
| purity_epsilon=0.02, max_nodes=63 | 61 | 61 | 57 | 55 | 51 | 55 | 63 | 57 | 51 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 61 | 55 | 55 | 51 | 57 | 63 | 57 | 51 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(42, 'purity_epsilon=0.05, max_nodes=63'), (1051, 'purity_epsilon=0.02, max_nodes=31'), (2060, 'purity_epsilon=0.02, max_nodes=63')]; concordância 0% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 25, folhas 13, accuracy 0.9300699300699301, fidelity 0.9090909090909091; MLP accuracy 0.951048951048951.

#### bc_user.arff — seed mestre 7 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [7, 1016, 2025]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=ea40639ef0e616a1` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'ea40639ef0e616a1', 'trepan_reloaded': 'ea40639ef0e616a1'}); queries por escopo: {'tuning': 2245801, 'trepan_pair': 44695, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 31657; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.925 | 0.035 | 0.866 | 5.9 ± 8.7 | 1.47 | 1.8 (1.31) | 3.4 (1.26) | 1.47 | 11% | 63001 | 3865 | 0/9 | 0% | 0.7 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.030, t=1.43 ≤ 1.86), mas desvio-padrão 0.035 > 0.010 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.950 | 0.017 | 0.915 | 23.7 ± 8.7 | 0.37 | 7.9 (0.36) | 12.3 (0.35) | 0.37 | 89% | 63001 | 14008 | 0/9 | 0% | 3.8 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.005, t=0.54 ≤ 1.86), mas desvio-padrão 0.017 > 0.010 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=31 | 0.954 | 0.010 | 0.944 | 27.0 ± 3.9 | 0.14 | 7.9 (0.15) | 14.0 (0.14) | 0.15 | 100% | 63001 | 14376 | 0/9 | 0% | 5.6 | **WINNER** — VENCE: fidelity média 0.954; indistinguível da melhor (purity_epsilon=0.01, max_nodes=63, Δ=0.001, t=0.24 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 2 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.15) -> 1 após complexidade. |
| purity_epsilon=0.05, max_nodes=63 | 0.926 | 0.036 | 0.866 | 9.7 ± 20.0 | 2.07 | 2.2 (1.65) | 5.3 (1.88) | 2.07 | 11% | 63001 | 5418 | 0/9 | 0% | 1.4 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.029, t=1.35 ≤ 1.86), mas desvio-padrão 0.036 > 0.010 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.951 | 0.016 | 0.923 | 46.3 ± 17.7 | 0.38 | 10.6 (0.37) | 23.7 (0.37) | 0.38 | 89% | 63001 | 26937 | 0/9 | 0% | 8.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.004, t=0.57 ≤ 1.86), mas desvio-padrão 0.016 > 0.010 + tolerância 0.005. |
| purity_epsilon=0.01, max_nodes=63 | 0.955 | 0.011 | 0.944 | 53.0 ± 7.5 | 0.14 | 11.2 (0.14) | 27.0 (0.14) | 0.14 | 100% | 63001 | 29148 | 0/9 | 0% | 11.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 53.0 nós médios vs 27.0 de purity_epsilon=0.01, max_nodes=31. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.925 | 0.035 | 0.037 | 0.013 | 0.913, 0.923, 0.939 |
| purity_epsilon=0.02, max_nodes=31 | 0.950 | 0.017 | 0.014 | 0.014 | 0.934, 0.958, 0.958 |
| purity_epsilon=0.01, max_nodes=31 | 0.954 | 0.010 | 0.008 | 0.007 | 0.946, 0.958, 0.958 |
| purity_epsilon=0.05, max_nodes=63 | 0.926 | 0.036 | 0.038 | 0.014 | 0.913, 0.923, 0.941 |
| purity_epsilon=0.02, max_nodes=63 | 0.951 | 0.016 | 0.014 | 0.012 | 0.937, 0.955, 0.960 |
| purity_epsilon=0.01, max_nodes=63 | 0.955 | 0.011 | 0.011 | 0.006 | 0.948, 0.955, 0.960 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67% | 89% | 0.34 | 0.19 | 28 | 0.39 | 0.22 | 89% | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67%, concave_points_worst 33% |
| purity_epsilon=0.02, max_nodes=31 | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67% | 89% | 0.56 | 0.23 | 3 | 0.69 | 0.31 | 11% | concave_points_mean 100%, perimeter_worst 100%, concave_points_worst 89%, concavity_mean 89% |
| purity_epsilon=0.01, max_nodes=31 | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67% | 89% | 0.70 | 0.33 | 5 | 0.70 | 0.33 | 0% | concave_points_mean 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67% | 89% | 0.33 | 0.18 | 28 | 0.39 | 0.22 | 89% | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67%, area_worst 33% |
| purity_epsilon=0.02, max_nodes=63 | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67% | 89% | 0.58 | 0.26 | 1 | 0.81 | 0.44 | 11% | concave_points_mean 100%, concave_points_worst 100%, perimeter_worst 100%, concavity_mean 89% |
| purity_epsilon=0.01, max_nodes=63 | concave_points_mean 89%, perimeter_worst 67%, radius_worst 67% | 89% | 0.75 | 0.36 | 4 | 0.69 | 0.42 | 0% | area_worst 100%, compactness_worst 100%, concave_points_mean 100%, concave_points_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.958 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.944 | 0.915 | 0.972 | 0.951 | 0.951 | 0.972 | 0.944 | 0.958 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.951 | 0.944 | 0.965 | 0.951 | 0.958 | 0.972 | 0.944 | 0.958 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.965 |
| purity_epsilon=0.02, max_nodes=63 | 0.944 | 0.944 | 0.923 | 0.972 | 0.944 | 0.951 | 0.972 | 0.944 | 0.965 |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.958 | 0.944 | 0.965 | 0.944 | 0.958 | 0.972 | 0.944 | 0.965 |

Nós por partição:

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 27 | 27 | 21 | 31 | 3 | 23 | 31 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 25 | 21 | 29 | 31 | 27 | 31 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 63 |
| purity_epsilon=0.02, max_nodes=63 | 45 | 57 | 51 | 49 | 53 | 3 | 39 | 57 | 63 |
| purity_epsilon=0.01, max_nodes=63 | 45 | 59 | 57 | 43 | 57 | 53 | 43 | 57 | 63 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=31`. Vencedora por repetição (seed): [(7, 'purity_epsilon=0.01, max_nodes=31'), (1016, 'purity_epsilon=0.01, max_nodes=63'), (2025, 'purity_epsilon=0.01, max_nodes=31')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 27, folhas 14, accuracy 0.9230769230769231, fidelity 0.9440559440559441; MLP accuracy 0.9790209790209791.

#### bc_user.arff — seed mestre 123 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [123, 1132, 2141]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=d156e8f076a2dfb2` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'd156e8f076a2dfb2', 'trepan_reloaded': 'd156e8f076a2dfb2'}); queries por escopo: {'tuning': 2518647, 'trepan_pair': 47673, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 32586; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.923 | 0.013 | 0.901 | 17.9 ± 12.2 | 0.68 | 5.8 (0.63) | 9.4 (0.64) | 0.68 | 67% | 63001 | 11444 | 0/9 | 0% | 3.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.68 (CV de nós 0.68; nós 3–31) > 0.14 + tolerância 0.05. |
| purity_epsilon=0.02, max_nodes=31 | 0.943 | 0.016 | 0.930 | 25.7 ± 4.5 | 0.17 | 7.6 (0.16) | 13.3 (0.17) | 0.17 | 100% | 63001 | 14983 | 0/9 | 0% | 4.1 | **WINNER** — VENCE: fidelity média 0.943 (a melhor). 6 indistinguível(eis) em fidelity -> 5 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.17) -> 1 após complexidade. |
| purity_epsilon=0.01, max_nodes=31 | 0.943 | 0.017 | 0.923 | 26.1 ± 5.0 | 0.19 | 7.0 (0.20) | 13.6 (0.18) | 0.20 | 100% | 63001 | 14137 | 0/9 | 0% | 3.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.19; nós 15–31) > 0.14 + tolerância 0.05. |
| purity_epsilon=0.05, max_nodes=63 | 0.924 | 0.015 | 0.901 | 38.8 ± 26.9 | 0.69 | 8.2 (0.67) | 19.9 (0.68) | 0.69 | 67% | 63001 | 21200 | 0/9 | 0% | 6.1 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.69 (CV de nós 0.69; nós 3–61) > 0.14 + tolerância 0.05. |
| purity_epsilon=0.02, max_nodes=63 | 0.939 | 0.018 | 0.915 | 54.8 ± 5.4 | 0.10 | 11.2 (0.14) | 27.9 (0.10) | 0.14 | 100% | 63001 | 29965 | 0/9 | 0% | 9.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.8 nós médios vs 25.7 de purity_epsilon=0.02, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.941 | 0.020 | 0.908 | 54.1 ± 5.4 | 0.10 | 10.4 (0.17) | 27.6 (0.10) | 0.17 | 100% | 63001 | 29365 | 0/9 | 0% | 9.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.15 ≤ 1.86), mas desvio-padrão 0.020 > 0.013 + tolerância 0.005. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.923 | 0.013 | 0.009 | 0.011 | 0.913, 0.923, 0.934 |
| purity_epsilon=0.02, max_nodes=31 | 0.943 | 0.016 | 0.010 | 0.012 | 0.930, 0.946, 0.953 |
| purity_epsilon=0.01, max_nodes=31 | 0.943 | 0.017 | 0.010 | 0.016 | 0.927, 0.941, 0.960 |
| purity_epsilon=0.05, max_nodes=63 | 0.924 | 0.015 | 0.011 | 0.012 | 0.913, 0.923, 0.937 |
| purity_epsilon=0.02, max_nodes=63 | 0.939 | 0.018 | 0.012 | 0.016 | 0.920, 0.946, 0.951 |
| purity_epsilon=0.01, max_nodes=63 | 0.941 | 0.020 | 0.011 | 0.020 | 0.920, 0.944, 0.960 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_worst 78%, concavity_worst 67%, symmetry_worst 33% | 78% | 0.43 | 0.12 | 4 | 0.41 | 0.10 | 33% | concavity_worst 100%, radius_worst 89%, concave_points_worst 78%, symmetry_worst 78% |
| purity_epsilon=0.02, max_nodes=31 | radius_worst 78%, concavity_worst 67%, symmetry_worst 33% | 78% | 0.65 | 0.23 | 6 | 0.69 | 0.24 | 0% | concave_points_worst 100%, concavity_worst 100%, radius_worst 100%, symmetry_worst 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_worst 78%, concavity_worst 67%, symmetry_worst 33% | 78% | 0.67 | 0.24 | 3 | 0.64 | 0.28 | 0% | concave_points_worst 100%, concavity_worst 100%, radius_worst 100%, symmetry_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_worst 78%, concavity_worst 67%, symmetry_worst 33% | 78% | 0.42 | 0.16 | 4 | 0.42 | 0.12 | 33% | concavity_worst 100%, radius_worst 89%, concave_points_worst 78%, symmetry_worst 78% |
| purity_epsilon=0.02, max_nodes=63 | radius_worst 78%, concavity_worst 67%, symmetry_worst 33% | 78% | 0.77 | 0.35 | 3 | 0.73 | 0.35 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_worst 78%, concavity_worst 67%, symmetry_worst 33% | 78% | 0.78 | 0.36 | 2 | 0.79 | 0.35 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.915 | 0.908 | 0.915 | 0.937 | 0.901 | 0.930 | 0.937 | 0.937 | 0.930 |
| purity_epsilon=0.02, max_nodes=31 | 0.930 | 0.930 | 0.930 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.979 |
| purity_epsilon=0.01, max_nodes=31 | 0.930 | 0.930 | 0.923 | 0.937 | 0.951 | 0.937 | 0.958 | 0.944 | 0.979 |
| purity_epsilon=0.05, max_nodes=63 | 0.908 | 0.908 | 0.923 | 0.937 | 0.901 | 0.930 | 0.937 | 0.944 | 0.930 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.915 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.972 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.937 | 0.908 | 0.937 | 0.951 | 0.944 | 0.951 | 0.958 | 0.972 |

Nós por partição:

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 21 | 3 | 15 | 29 | 3 | 27 | 31 | 29 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 29 | 23 | 29 | 27 | 25 | 29 | 27 | 15 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 31 | 25 | 31 | 29 | 23 | 25 | 29 | 15 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 3 | 55 | 55 | 3 | 53 | 59 | 57 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 47 | 51 | 49 | 55 | 51 | 61 | 59 | 59 | 61 |
| purity_epsilon=0.01, max_nodes=63 | 47 | 45 | 51 | 59 | 55 | 55 | 57 | 57 | 61 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31`. Vencedora por repetição (seed): [(123, 'purity_epsilon=0.02, max_nodes=31'), (1132, 'purity_epsilon=0.02, max_nodes=31'), (2141, 'purity_epsilon=0.05, max_nodes=31')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 27, folhas 14, accuracy 0.9090909090909091, fidelity 0.916083916083916; MLP accuracy 0.965034965034965.

#### bc_user.arff — seed mestre 2024 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [2024, 3033, 4042]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=a2f113e60f488f42` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'a2f113e60f488f42', 'trepan_reloaded': 'a2f113e60f488f42'}); queries por escopo: {'tuning': 3970231, 'trepan_pair': 92272, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 31874; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.934 | 0.020 | 0.915 | 24.1 ± 8.3 | 0.34 | 7.9 (0.35) | 12.6 (0.33) | 0.35 | 89% | 63001 | 14079 | 0/9 | 0% | 3.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.35 (CV de nós 0.34; nós 3–31) > 0.11 + tolerância 0.05. |
| purity_epsilon=0.02, max_nodes=31 | 0.932 | 0.016 | 0.908 | 24.6 ± 4.6 | 0.19 | 7.0 (0.29) | 12.8 (0.18) | 0.29 | 100% | 63001 | 14562 | 0/9 | 0% | 3.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.29 (CV de nós 0.19; nós 17–31) > 0.11 + tolerância 0.05. |
| purity_epsilon=0.01, max_nodes=31 | 0.927 | 0.026 | 0.887 | 24.3 ± 6.4 | 0.26 | 6.7 (0.21) | 12.7 (0.25) | 0.26 | 100% | 63001 | 14120 | 0/9 | 0% | 3.7 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.007, t=0.33 ≤ 1.86), mas desvio-padrão 0.026 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.05, max_nodes=63 | 0.933 | 0.022 | 0.908 | 43.4 ± 16.8 | 0.39 | 9.7 (0.34) | 22.2 (0.38) | 0.39 | 89% | 63001 | 27240 | 0/9 | 0% | 8.2 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.64 ≤ 1.86), mas desvio-padrão 0.022 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.928 | 0.016 | 0.908 | 53.4 ± 2.8 | 0.05 | 9.2 (0.11) | 27.2 (0.05) | 0.11 | 100% | 63001 | 29713 | 0/9 | 0% | 9.0 | **WINNER** — VENCE: fidelity média 0.928; indistinguível da melhor (purity_epsilon=0.05, max_nodes=31, Δ=0.006, t=0.39 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 3 após estabilidade da fidelity -> 1 após estabilidade estrutural (índice 0.11) -> 1 após complexidade. |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.022 | 0.894 | 55.0 ± 1.0 | 0.02 | 9.7 (0.12) | 28.0 (0.02) | 0.12 | 100% | 63001 | 29053 | 0/9 | 0% | 8.9 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.009, t=0.53 ≤ 1.86), mas desvio-padrão 0.022 > 0.016 + tolerância 0.005. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.934 | 0.020 | 0.016 | 0.016 | 0.925, 0.953, 0.925 |
| purity_epsilon=0.02, max_nodes=31 | 0.932 | 0.016 | 0.015 | 0.004 | 0.937, 0.930, 0.930 |
| purity_epsilon=0.01, max_nodes=31 | 0.927 | 0.026 | 0.024 | 0.012 | 0.941, 0.918, 0.923 |
| purity_epsilon=0.05, max_nodes=63 | 0.933 | 0.022 | 0.018 | 0.018 | 0.923, 0.953, 0.923 |
| purity_epsilon=0.02, max_nodes=63 | 0.928 | 0.016 | 0.018 | 0.001 | 0.927, 0.930, 0.927 |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.022 | 0.024 | 0.002 | 0.927, 0.925, 0.923 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_worst 100%, radius_se 89%, concavity_mean 67% | 100% | 0.56 | 0.19 | 4 | 0.64 | 0.24 | 11% | concavity_mean 100%, radius_se 100%, radius_worst 100%, compactness_mean 89% |
| purity_epsilon=0.02, max_nodes=31 | radius_worst 100%, radius_se 89%, concavity_mean 67% | 100% | 0.69 | 0.26 | 3 | 0.65 | 0.22 | 0% | compactness_mean 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_worst 100%, radius_se 89%, concavity_mean 67% | 100% | 0.67 | 0.24 | 2 | 0.78 | 0.25 | 0% | compactness_mean 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_worst 100%, radius_se 89%, concavity_mean 67% | 100% | 0.58 | 0.24 | 2 | 0.73 | 0.35 | 11% | concavity_mean 100%, radius_se 100%, radius_worst 100%, compactness_mean 89% |
| purity_epsilon=0.02, max_nodes=63 | radius_worst 100%, radius_se 89%, concavity_mean 67% | 100% | 0.75 | 0.36 | 15 | 0.76 | 0.38 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_worst 100%, radius_se 89%, concavity_mean 67% | 100% | 0.75 | 0.36 | 21 | 0.75 | 0.38 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.923 | 0.915 | 0.937 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.915 |
| purity_epsilon=0.02, max_nodes=31 | 0.937 | 0.937 | 0.937 | 0.923 | 0.958 | 0.908 | 0.951 | 0.923 | 0.915 |
| purity_epsilon=0.01, max_nodes=31 | 0.937 | 0.937 | 0.951 | 0.887 | 0.958 | 0.908 | 0.951 | 0.923 | 0.894 |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.908 | 0.937 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.908 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.923 | 0.958 | 0.908 | 0.944 | 0.930 | 0.908 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.901 | 0.965 | 0.908 | 0.944 | 0.930 | 0.894 |

Nós por partição:

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 25 | 29 | 29 | 31 | 25 | 27 | 23 | 3 | 25 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 29 | 29 | 21 | 31 | 23 | 17 | 25 | 25 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 13 | 29 | 31 | 23 | 17 | 25 | 31 |
| purity_epsilon=0.05, max_nodes=63 | 33 | 51 | 55 | 41 | 53 | 55 | 49 | 3 | 51 |
| purity_epsilon=0.02, max_nodes=63 | 55 | 55 | 55 | 47 | 53 | 55 | 55 | 55 | 51 |
| purity_epsilon=0.01, max_nodes=63 | 55 | 55 | 55 | 55 | 53 | 55 | 55 | 55 | 57 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=63`. Vencedora por repetição (seed): [(2024, 'purity_epsilon=0.02, max_nodes=31'), (3033, 'purity_epsilon=0.05, max_nodes=31'), (4042, 'purity_epsilon=0.02, max_nodes=63')]; concordância 33% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 35, folhas 18, accuracy 0.9370629370629371, fidelity 0.9370629370629371; MLP accuracy 0.972027972027972.

#### bc_user.arff — seed mestre 11 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [11, 1020, 2029]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=9aa83790a361ec28` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '9aa83790a361ec28', 'trepan_reloaded': '9aa83790a361ec28'}); queries por escopo: {'tuning': 5078484, 'trepan_pair': 183891, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 31192; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.941 | 0.029 | 0.873 | 23.9 ± 11.9 | 0.50 | 7.1 (0.49) | 12.4 (0.48) | 0.50 | 78% | 63001 | 12457 | 0/9 | 0% | 3.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.007, t=0.51 ≤ 1.86), mas desvio-padrão 0.029 > 0.021 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.948 | 0.023 | 0.901 | 25.4 ± 4.7 | 0.18 | 7.0 (0.21) | 13.2 (0.18) | 0.21 | 100% | 63001 | 14629 | 0/9 | 0% | 3.6 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.21 (CV de nós 0.18; nós 17–31) > 0.14 + tolerância 0.05. |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.021 | 0.901 | 23.0 ± 4.8 | 0.21 | 6.2 (0.22) | 12.0 (0.20) | 0.22 | 100% | 63001 | 14197 | 0/9 | 0% | 3.6 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.22 (CV de nós 0.21; nós 17–31) > 0.14 + tolerância 0.05. |
| purity_epsilon=0.05, max_nodes=63 | 0.941 | 0.029 | 0.873 | 43.4 ± 23.3 | 0.54 | 9.6 (0.53) | 22.2 (0.52) | 0.54 | 78% | 63001 | 23793 | 0/9 | 0% | 6.6 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.007, t=0.51 ≤ 1.86), mas desvio-padrão 0.029 > 0.021 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.944 | 0.025 | 0.887 | 55.2 ± 7.6 | 0.14 | 9.9 (0.14) | 28.1 (0.14) | 0.14 | 100% | 63001 | 29342 | 0/9 | 0% | 9.2 | **WINNER** — VENCE: fidelity média 0.944; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.004, t=0.57 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 4 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.14) -> 2 após complexidade; empate real resolvido pela configuração mais simples. |
| purity_epsilon=0.01, max_nodes=63 | 0.942 | 0.024 | 0.887 | 55.2 ± 6.4 | 0.12 | 9.8 (0.15) | 28.1 (0.11) | 0.15 | 100% | 63001 | 29034 | 0/9 | 0% | 8.8 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=63). |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.941 | 0.029 | 0.023 | 0.019 | 0.923, 0.960, 0.939 |
| purity_epsilon=0.02, max_nodes=31 | 0.948 | 0.023 | 0.020 | 0.015 | 0.937, 0.965, 0.941 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.021 | 0.021 | 0.010 | 0.937, 0.955, 0.941 |
| purity_epsilon=0.05, max_nodes=63 | 0.941 | 0.029 | 0.023 | 0.019 | 0.923, 0.960, 0.939 |
| purity_epsilon=0.02, max_nodes=63 | 0.944 | 0.025 | 0.022 | 0.014 | 0.930, 0.958, 0.944 |
| purity_epsilon=0.01, max_nodes=63 | 0.942 | 0.024 | 0.021 | 0.013 | 0.930, 0.955, 0.941 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | concave_points_worst 100%, area_worst 78%, radius_worst 56% | 100% | 0.50 | 0.16 | 8 | 0.66 | 0.21 | 22% | concave_points_worst 100%, area_worst 89%, concavity_worst 89%, radius_worst 89% |
| purity_epsilon=0.02, max_nodes=31 | concave_points_worst 100%, area_worst 78%, radius_worst 56% | 100% | 0.69 | 0.25 | 2 | 0.74 | 0.26 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=31 | concave_points_worst 100%, area_worst 78%, radius_worst 56% | 100% | 0.65 | 0.21 | 3 | 0.58 | 0.17 | 0% | area_worst 100%, concave_points_worst 100%, concavity_worst 100%, perimeter_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | concave_points_worst 100%, area_worst 78%, radius_worst 56% | 100% | 0.51 | 0.22 | 3 | 0.53 | 0.26 | 22% | concave_points_worst 100%, area_worst 89%, concavity_worst 89%, radius_worst 89% |
| purity_epsilon=0.02, max_nodes=63 | concave_points_worst 100%, area_worst 78%, radius_worst 56% | 100% | 0.81 | 0.34 | 4 | 0.79 | 0.36 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, compactness_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | concave_points_worst 100%, area_worst 78%, radius_worst 56% | 100% | 0.82 | 0.34 | 3 | 0.81 | 0.36 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, concave_points_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 |
| purity_epsilon=0.02, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.965 | 0.972 | 0.965 | 0.937 | 0.923 |
| purity_epsilon=0.01, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.944 | 0.965 | 0.965 | 0.937 | 0.923 |
| purity_epsilon=0.05, max_nodes=63 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 |
| purity_epsilon=0.02, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.965 | 0.951 | 0.965 | 0.944 | 0.923 |
| purity_epsilon=0.01, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.951 | 0.958 | 0.965 | 0.937 | 0.923 |

Nós por partição:

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 31 | 3 | 29 | 27 | 31 | 31 | 29 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 21 | 17 | 25 | 27 | 31 | 23 | 25 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 21 | 17 | 25 | 23 | 21 | 23 | 17 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 59 | 3 | 51 | 51 | 57 | 49 | 61 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 53 | 59 | 61 | 37 | 59 | 61 | 51 | 59 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 53 | 59 | 61 | 47 | 57 | 61 | 43 | 59 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=63`. Vencedora por repetição (seed): [(11, 'purity_epsilon=0.02, max_nodes=31'), (1020, 'purity_epsilon=0.02, max_nodes=31'), (2029, 'purity_epsilon=0.05, max_nodes=31')]; concordância 0% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 47, folhas 24, accuracy 0.9370629370629371, fidelity 0.951048951048951; MLP accuracy 0.972027972027972.

## iris_t.arff — oráculo `factory`

### Estabilidade da configuração entre seeds mestre

| seed mestre | oracle_id | selecionada | concordância interna entre repetições | tuning | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|---|---|---|
| 42 | `fa910f6f25e5c1a3` | `purity_epsilon=0.02, max_nodes=63` | 33% | **INSTÁVEL** | 55 | 0.868421052631579 | 0.9736842105263158 | 114 |
| 7 | `bec1d5a4e626d590` | `purity_epsilon=0.05, max_nodes=63` | 100% | estável | 63 | 0.9736842105263158 | 0.9736842105263158 | 99 |
| 123 | `3c656fb0e0870746` | `purity_epsilon=0.05, max_nodes=63` | 67% | estável | 61 | 0.9473684210526315 | 1.0 | 107 |
| 2024 | `9886716fca9b4dba` | `purity_epsilon=0.05, max_nodes=31` | 33% | **INSTÁVEL** | 31 | 0.9210526315789473 | 0.8947368421052632 | 62 |
| 11 | `3a8a750475e18459` | `purity_epsilon=0.02, max_nodes=31` | 67% | estável | 31 | 0.9473684210526315 | 0.9736842105263158 | 68 |

Configurações escolhidas entre seeds mestre: {'purity_epsilon=0.02, max_nodes=63': 1, 'purity_epsilon=0.05, max_nodes=63': 2, 'purity_epsilon=0.05, max_nodes=31': 1, 'purity_epsilon=0.02, max_nodes=31': 1}. Moda `purity_epsilon=0.05, max_nodes=63` em 2/5 (40%) → seleção entre seeds **INSTÁVEL** (limiar 60%).

### Conclusão — iris_t.arff

Nenhuma configuração produziu árvores de ≤3 nós (stumps) nas partições de CV desta experiência.

#### iris_t.arff — seed mestre 42 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [42, 1051, 2060]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=fa910f6f25e5c1a3` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'fa910f6f25e5c1a3', 'trepan_reloaded': 'fa910f6f25e5c1a3'}); queries por escopo: {'tuning': 1507165, 'trepan_pair': 65896, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10320; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.919 | 0.041 | 0.865 | 28.6 ± 2.6 | 0.09 | 7.9 (0.18) | 14.8 (0.09) | 0.18 | 100% | 21169 | 4816 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.015, t=0.71 ≤ 1.86), mas desvio-padrão 0.041 > 0.033 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.923 | 0.037 | 0.892 | 29.2 ± 2.9 | 0.10 | 8.0 (0.23) | 15.1 (0.10) | 0.23 | 100% | 21169 | 4608 | 0/9 | 0% | 0.6 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.23 (CV de nós 0.10; nós 23–31) > 0.14 + tolerância 0.05. |
| purity_epsilon=0.01, max_nodes=31 | 0.923 | 0.037 | 0.892 | 29.2 ± 2.9 | 0.10 | 8.0 (0.23) | 15.1 (0.10) | 0.23 | 100% | 21169 | 4608 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.23 (CV de nós 0.10; nós 23–31) > 0.14 + tolerância 0.05. |
| purity_epsilon=0.05, max_nodes=63 | 0.931 | 0.038 | 0.865 | 59.0 ± 3.0 | 0.05 | 12.7 (0.14) | 30.0 (0.05) | 0.14 | 100% | 21169 | 9709 | 0/9 | 0% | 1.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.003, t=0.25 ≤ 1.86), mas desvio-padrão 0.038 > 0.033 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.935 | 0.033 | 0.892 | 60.1 ± 3.3 | 0.06 | 12.4 (0.14) | 30.6 (0.05) | 0.14 | 100% | 21169 | 9381 | 0/9 | 0% | 1.0 | **WINNER** — VENCE: fidelity média 0.935 (a melhor). 6 indistinguível(eis) em fidelity -> 4 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.14) -> 2 após complexidade; empate real resolvido pela configuração mais simples. |
| purity_epsilon=0.01, max_nodes=63 | 0.935 | 0.033 | 0.892 | 60.1 ± 3.3 | 0.06 | 12.4 (0.14) | 30.6 (0.05) | 0.14 | 100% | 21169 | 9381 | 0/9 | 0% | 1.1 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=63). |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.919 | 0.041 | 0.033 | 0.031 | 0.884, 0.937, 0.937 |
| purity_epsilon=0.02, max_nodes=31 | 0.923 | 0.037 | 0.028 | 0.026 | 0.893, 0.937, 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.923 | 0.037 | 0.028 | 0.026 | 0.893, 0.937, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.931 | 0.038 | 0.033 | 0.027 | 0.902, 0.937, 0.955 |
| purity_epsilon=0.02, max_nodes=63 | 0.935 | 0.033 | 0.029 | 0.022 | 0.911, 0.937, 0.955 |
| purity_epsilon=0.01, max_nodes=63 | 0.935 | 0.033 | 0.029 | 0.022 | 0.911, 0.937, 0.955 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | sepal_width 100%, petal_length 56% | 100% | 1.00 | 0.41 | 7 | 1.00 | 0.37 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | sepal_width 100%, petal_length 56% | 100% | 1.00 | 0.42 | 16 | 1.00 | 0.39 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | sepal_width 100%, petal_length 56% | 100% | 1.00 | 0.42 | 16 | 1.00 | 0.39 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | sepal_width 100%, petal_length 56% | 100% | 1.00 | 0.46 | 5 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | sepal_width 100%, petal_length 56% | 100% | 1.00 | 0.48 | 6 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | sepal_width 100%, petal_length 56% | 100% | 1.00 | 0.48 | 6 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

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
| purity_epsilon=0.05, max_nodes=31 | 31 | 27 | 29 | 27 | 23 | 29 | 31 | 29 | 31 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 57 | 57 | 59 | 53 | 59 | 63 | 61 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=63`. Vencedora por repetição (seed): [(42, 'purity_epsilon=0.02, max_nodes=31'), (1051, 'purity_epsilon=0.02, max_nodes=63'), (2060, 'purity_epsilon=0.05, max_nodes=63')]; concordância 33% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 55, folhas 28, accuracy 0.868421052631579, fidelity 0.9736842105263158; MLP accuracy 0.8947368421052632.

#### iris_t.arff — seed mestre 7 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [7, 1016, 2025]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=bec1d5a4e626d590` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'bec1d5a4e626d590', 'trepan_reloaded': 'bec1d5a4e626d590'}); queries por escopo: {'tuning': 1156935, 'trepan_pair': 30510, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 11139; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.926 | 0.043 | 0.842 | 28.1 ± 3.0 | 0.11 | 7.6 (0.20) | 14.6 (0.10) | 0.20 | 100% | 21169 | 4806 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.11; nós 23–31) > 0.13 + tolerância 0.05. |
| purity_epsilon=0.02, max_nodes=31 | 0.923 | 0.045 | 0.842 | 28.6 ± 3.1 | 0.11 | 7.6 (0.20) | 14.8 (0.11) | 0.20 | 100% | 21169 | 4734 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.11; nós 23–31) > 0.13 + tolerância 0.05. |
| purity_epsilon=0.01, max_nodes=31 | 0.923 | 0.045 | 0.842 | 28.6 ± 3.1 | 0.11 | 7.6 (0.20) | 14.8 (0.11) | 0.20 | 100% | 21169 | 4734 | 0/9 | 0% | 0.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.11; nós 23–31) > 0.13 + tolerância 0.05. |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.043 | 0.842 | 56.8 ± 3.8 | 0.07 | 11.4 (0.15) | 28.9 (0.07) | 0.15 | 100% | 21169 | 10036 | 0/9 | 0% | 1.0 | **WINNER** — VENCE: fidelity média 0.923; indistinguível da melhor (purity_epsilon=0.05, max_nodes=31, Δ=0.003, t=0.43 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 6 após estabilidade da fidelity -> 3 após estabilidade estrutural (índice 0.15) -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=63 | 0.920 | 0.044 | 0.842 | 59.0 ± 3.2 | 0.05 | 11.3 (0.13) | 30.0 (0.05) | 0.13 | 100% | 21169 | 9779 | 0/9 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.0 nós médios vs 56.8 de purity_epsilon=0.05, max_nodes=63. |
| purity_epsilon=0.01, max_nodes=63 | 0.920 | 0.044 | 0.842 | 59.0 ± 3.2 | 0.05 | 11.3 (0.13) | 30.0 (0.05) | 0.13 | 100% | 21169 | 9779 | 0/9 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.0 nós médios vs 56.8 de purity_epsilon=0.05, max_nodes=63. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.926 | 0.043 | 0.045 | 0.018 | 0.911, 0.920, 0.946 |
| purity_epsilon=0.02, max_nodes=31 | 0.923 | 0.045 | 0.048 | 0.013 | 0.911, 0.920, 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.923 | 0.045 | 0.048 | 0.013 | 0.911, 0.920, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.043 | 0.043 | 0.022 | 0.902, 0.920, 0.946 |
| purity_epsilon=0.02, max_nodes=63 | 0.920 | 0.044 | 0.046 | 0.017 | 0.902, 0.920, 0.937 |
| purity_epsilon=0.01, max_nodes=63 | 0.920 | 0.044 | 0.046 | 0.017 | 0.902, 0.920, 0.937 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.41 | 9 | 1.00 | 0.49 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.39 | 11 | 1.00 | 0.42 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.39 | 11 | 1.00 | 0.42 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.39 | 6 | 1.00 | 0.35 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.39 | 5 | 1.00 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.39 | 5 | 1.00 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

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
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 61 | 53 | 59 | 57 | 49 | 59 | 55 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=63`. Vencedora por repetição (seed): [(7, 'purity_epsilon=0.05, max_nodes=63'), (1016, 'purity_epsilon=0.05, max_nodes=63'), (2025, 'purity_epsilon=0.05, max_nodes=63')]; concordância 100% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 63, folhas 32, accuracy 0.9736842105263158, fidelity 0.9736842105263158; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 123 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [123, 1132, 2141]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=3c656fb0e0870746` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '3c656fb0e0870746', 'trepan_reloaded': '3c656fb0e0870746'}); queries por escopo: {'tuning': 1545914, 'trepan_pair': 61203, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10797; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.940 | 0.038 | 0.865 | 26.8 ± 2.5 | 0.09 | 8.2 (0.22) | 13.9 (0.09) | 0.22 | 100% | 21169 | 4707 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.22 (CV de nós 0.09; nós 23–31) > 0.12 + tolerância 0.05. |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.038 | 0.865 | 27.0 ± 2.6 | 0.10 | 8.0 (0.23) | 14.0 (0.09) | 0.23 | 100% | 21169 | 4593 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.23 (CV de nós 0.10; nós 23–31) > 0.12 + tolerância 0.05. |
| purity_epsilon=0.01, max_nodes=31 | 0.940 | 0.038 | 0.865 | 27.0 ± 2.6 | 0.10 | 8.0 (0.23) | 14.0 (0.09) | 0.23 | 100% | 21169 | 4556 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.23 (CV de nós 0.10; nós 23–31) > 0.12 + tolerância 0.05. |
| purity_epsilon=0.05, max_nodes=63 | 0.940 | 0.038 | 0.865 | 60.3 ± 2.8 | 0.05 | 12.8 (0.12) | 30.7 (0.05) | 0.12 | 100% | 21169 | 9804 | 0/9 | 0% | 1.1 | **WINNER** — VENCE: fidelity média 0.940; indistinguível da melhor (purity_epsilon=0.05, max_nodes=31, Δ=0.000, t=0.00 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 6 após estabilidade da fidelity -> 3 após estabilidade estrutural (índice 0.12) -> 1 após complexidade. |
| purity_epsilon=0.02, max_nodes=63 | 0.940 | 0.038 | 0.865 | 60.8 ± 3.8 | 0.06 | 12.8 (0.12) | 30.9 (0.06) | 0.12 | 100% | 21169 | 9512 | 0/9 | 0% | 1.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 60.8 nós médios vs 60.3 de purity_epsilon=0.05, max_nodes=63. |
| purity_epsilon=0.01, max_nodes=63 | 0.940 | 0.038 | 0.865 | 60.8 ± 3.8 | 0.06 | 12.7 (0.12) | 30.9 (0.06) | 0.12 | 100% | 21169 | 9475 | 0/9 | 0% | 1.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 60.8 nós médios vs 60.3 de purity_epsilon=0.05, max_nodes=63. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.940 | 0.038 | 0.026 | 0.034 | 0.955, 0.902, 0.964 |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.038 | 0.026 | 0.034 | 0.955, 0.902, 0.964 |
| purity_epsilon=0.01, max_nodes=31 | 0.940 | 0.038 | 0.026 | 0.034 | 0.955, 0.902, 0.964 |
| purity_epsilon=0.05, max_nodes=63 | 0.940 | 0.038 | 0.026 | 0.034 | 0.955, 0.902, 0.964 |
| purity_epsilon=0.02, max_nodes=63 | 0.940 | 0.038 | 0.026 | 0.034 | 0.955, 0.902, 0.964 |
| purity_epsilon=0.01, max_nodes=63 | 0.940 | 0.038 | 0.026 | 0.034 | 0.955, 0.902, 0.964 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.40 | 11 | 1.00 | 0.42 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.42 | 8 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.42 | 8 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.49 | 6 | 1.00 | 0.53 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.53 | 12 | 1.00 | 0.52 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.52 | 12 | 1.00 | 0.53 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

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
| purity_epsilon=0.05, max_nodes=31 | 27 | 27 | 23 | 27 | 27 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 27 | 23 | 27 | 29 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 27 | 23 | 27 | 29 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 63 | 63 | 61 | 61 | 57 | 59 | 63 | 55 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=63`. Vencedora por repetição (seed): [(123, 'purity_epsilon=0.05, max_nodes=63'), (1132, 'purity_epsilon=0.05, max_nodes=63'), (2141, 'purity_epsilon=0.02, max_nodes=63')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 61, folhas 31, accuracy 0.9473684210526315, fidelity 1.0; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 2024 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [2024, 3033, 4042]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=9886716fca9b4dba` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '9886716fca9b4dba', 'trepan_reloaded': '9886716fca9b4dba'}); queries por escopo: {'tuning': 748837, 'trepan_pair': 15925, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10138; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.955 | 0.019 | 0.919 | 29.4 ± 1.9 | 0.07 | 8.4 (0.13) | 15.2 (0.06) | 0.13 | 100% | 21169 | 4585 | 0/9 | 0% | 0.4 | **WINNER** — VENCE: fidelity média 0.955; indistinguível da melhor (purity_epsilon=0.05, max_nodes=63, Δ=0.003, t=0.43 ≤ 1.86). 6 indistinguível(eis) em fidelity -> 6 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.13) -> 3 após complexidade; empate real resolvido pela configuração mais simples. |
| purity_epsilon=0.02, max_nodes=31 | 0.955 | 0.019 | 0.919 | 29.4 ± 1.9 | 0.07 | 8.4 (0.13) | 15.2 (0.06) | 0.13 | 100% | 21169 | 4553 | 0/9 | 0% | 0.4 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.05, max_nodes=31). |
| purity_epsilon=0.01, max_nodes=31 | 0.955 | 0.019 | 0.919 | 29.4 ± 1.9 | 0.07 | 8.2 (0.17) | 15.2 (0.06) | 0.17 | 100% | 21169 | 4516 | 0/9 | 0% | 0.4 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.05, max_nodes=31). |
| purity_epsilon=0.05, max_nodes=63 | 0.958 | 0.014 | 0.946 | 57.9 ± 3.2 | 0.05 | 12.6 (0.22) | 29.4 (0.05) | 0.22 | 100% | 21169 | 9807 | 0/9 | 0% | 0.9 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.22 (CV de nós 0.05; nós 51–61) > 0.13 + tolerância 0.05. |
| purity_epsilon=0.02, max_nodes=63 | 0.958 | 0.014 | 0.946 | 59.7 ± 2.2 | 0.04 | 12.9 (0.18) | 30.3 (0.04) | 0.18 | 100% | 21169 | 9598 | 0/9 | 0% | 1.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.7 nós médios vs 29.4 de purity_epsilon=0.05, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.958 | 0.014 | 0.946 | 59.7 ± 2.2 | 0.04 | 12.7 (0.19) | 30.3 (0.04) | 0.19 | 100% | 21169 | 9523 | 0/9 | 0% | 1.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.19 (CV de nós 0.04; nós 57–63) > 0.13 + tolerância 0.05. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.955 | 0.019 | 0.019 | 0.009 | 0.964, 0.955, 0.946 |
| purity_epsilon=0.02, max_nodes=31 | 0.955 | 0.019 | 0.019 | 0.009 | 0.964, 0.955, 0.946 |
| purity_epsilon=0.01, max_nodes=31 | 0.955 | 0.019 | 0.019 | 0.009 | 0.964, 0.955, 0.946 |
| purity_epsilon=0.05, max_nodes=63 | 0.958 | 0.014 | 0.015 | 0.005 | 0.964, 0.955, 0.955 |
| purity_epsilon=0.02, max_nodes=63 | 0.958 | 0.014 | 0.015 | 0.005 | 0.964, 0.955, 0.955 |
| purity_epsilon=0.01, max_nodes=63 | 0.958 | 0.014 | 0.015 | 0.005 | 0.964, 0.955, 0.955 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 13 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 13 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 13 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.49 | 7 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.49 | 8 | 1.00 | 0.50 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.52 | 8 | 1.00 | 0.53 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

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

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31`. Vencedora por repetição (seed): [(2024, 'purity_epsilon=0.05, max_nodes=31'), (3033, 'purity_epsilon=0.05, max_nodes=63'), (4042, 'purity_epsilon=0.05, max_nodes=63')]; concordância 33% (limiar 60%) → tuning **INSTÁVEL**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 31, folhas 16, accuracy 0.9210526315789473, fidelity 0.8947368421052632; MLP accuracy 0.9210526315789473.

#### iris_t.arff — seed mestre 11 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 3 repetições (seeds [11, 1020, 2029]) = 9 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=3a8a750475e18459` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '3a8a750475e18459', 'trepan_reloaded': '3a8a750475e18459'}); queries por escopo: {'tuning': 751623, 'trepan_pair': 14871, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10597; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | no teto de nós | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.934 | 0.057 | 0.811 | 29.9 ± 1.8 | 0.06 | 8.4 (0.12) | 15.4 (0.06) | 0.12 | 100% | 21169 | 4840 | 0/9 | 0% | 0.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.012, t=0.78 ≤ 1.86), mas desvio-padrão 0.057 > 0.045 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.946 | 0.045 | 0.865 | 30.3 ± 1.0 | 0.03 | 8.7 (0.19) | 15.7 (0.03) | 0.19 | 100% | 21169 | 4636 | 0/9 | 0% | 0.4 | **WINNER** — VENCE: fidelity média 0.946 (a melhor). 6 indistinguível(eis) em fidelity -> 4 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.19) -> 2 após complexidade; empate real resolvido pela configuração mais simples. |
| purity_epsilon=0.01, max_nodes=31 | 0.946 | 0.045 | 0.865 | 30.3 ± 1.0 | 0.03 | 8.7 (0.19) | 15.7 (0.03) | 0.19 | 100% | 21169 | 4636 | 0/9 | 0% | 0.4 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=31). |
| purity_epsilon=0.05, max_nodes=63 | 0.934 | 0.057 | 0.811 | 58.8 ± 3.7 | 0.06 | 12.7 (0.21) | 29.9 (0.06) | 0.21 | 100% | 21169 | 9939 | 0/9 | 0% | 1.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.012, t=0.78 ≤ 1.86), mas desvio-padrão 0.057 > 0.045 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.946 | 0.045 | 0.865 | 57.7 ± 3.3 | 0.06 | 12.4 (0.21) | 29.3 (0.06) | 0.21 | 100% | 21169 | 9615 | 0/9 | 0% | 1.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.7 nós médios vs 30.3 de purity_epsilon=0.02, max_nodes=31. |
| purity_epsilon=0.01, max_nodes=63 | 0.946 | 0.045 | 0.865 | 57.7 ± 3.3 | 0.06 | 12.4 (0.21) | 29.3 (0.06) | 0.21 | 100% | 21169 | 9615 | 0/9 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.7 nós médios vs 30.3 de purity_epsilon=0.02, max_nodes=31. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.934 | 0.057 | 0.055 | 0.011 | 0.928, 0.929, 0.947 |
| purity_epsilon=0.02, max_nodes=31 | 0.946 | 0.045 | 0.048 | 0.009 | 0.946, 0.938, 0.955 |
| purity_epsilon=0.01, max_nodes=31 | 0.946 | 0.045 | 0.048 | 0.009 | 0.946, 0.938, 0.955 |
| purity_epsilon=0.05, max_nodes=63 | 0.934 | 0.057 | 0.055 | 0.011 | 0.928, 0.929, 0.947 |
| purity_epsilon=0.02, max_nodes=63 | 0.946 | 0.045 | 0.048 | 0.009 | 0.946, 0.938, 0.955 |
| purity_epsilon=0.01, max_nodes=63 | 0.946 | 0.045 | 0.048 | 0.009 | 0.946, 0.938, 0.955 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100%, petal_width 22% | 100% | 0.94 | 0.35 | 16 | 0.92 | 0.38 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 89% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100%, petal_width 22% | 100% | 0.94 | 0.41 | 18 | 0.93 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 89% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100%, petal_width 22% | 100% | 0.94 | 0.41 | 18 | 0.93 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 89% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100%, petal_width 22% | 100% | 1.00 | 0.36 | 5 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100%, petal_width 22% | 100% | 1.00 | 0.42 | 11 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100%, petal_width 22% | 100% | 1.00 | 0.42 | 11 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

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
| purity_epsilon=0.05, max_nodes=31 | 29 | 31 | 27 | 31 | 31 | 31 | 27 | 31 | 31 |
| purity_epsilon=0.02, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 57 | 59 | 53 | 63 | 63 | 59 | 55 | 63 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31`. Vencedora por repetição (seed): [(11, 'purity_epsilon=0.02, max_nodes=31'), (1020, 'purity_epsilon=0.05, max_nodes=31'), (2029, 'purity_epsilon=0.02, max_nodes=31')]; concordância 67% (limiar 60%) → tuning **estável**.

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 31, folhas 16, accuracy 0.9473684210526315, fidelity 0.9736842105263158; MLP accuracy 0.9736842105263158.
