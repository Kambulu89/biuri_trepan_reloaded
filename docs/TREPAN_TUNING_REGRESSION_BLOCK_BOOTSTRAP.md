> Regressão (CV 5×3, orçamento comum não limitante, estabilidade interna por block bootstrap das repetições). Breast Cancer e Iris são
> **apenas casos de validação**: a mesma política genérica escolheu a configuração em cada um, sem regras por dataset. "Consenso de
> seleção" entre master seeds é evidência empírica externa; só `tuning_stable` pelo bootstrap sustenta "estável" para uma seed.
> Nota sobre saturação: em todas as runs a configuração vencedora atingiu o teto `max_nodes` em 100% das árvores de CV
> (`structural_stability_censored = True`); a estabilidade estrutural das árvores plenas é, portanto, potencialmente censurada pelo limite.

# Tuning da estrutura do TREPAN — Repeated Stratified K-Fold, orçamento comum não limitante e seleção lexicográfica com estabilidade estrutural

Gerado por `scripts/render_tuning_experiment.py` a partir de `scripts/run_trepan_tuning_experiment.py`. Nenhuma regra por dataset; o teste só é usado uma vez, depois da configuração escolhida.

## bc_user.arff — oráculo `factory`

### 1. Estabilidade interna do tuning (block bootstrap, por seed mestre)

| seed mestre | oracle_id | selecionada | selection_probability | 2.º colocado | margem | reamostragens | tuning |
|---|---|---|---|---|---|---|---|
| 42 | `457012f5f057926b` | `purity_epsilon=0.01, max_nodes=31` | 46.0% | `purity_epsilon=0.05, max_nodes=63` 14.5% | 31.5% | 200 | tuning_uncertain |
| 7 | `ea40639ef0e616a1` | `purity_epsilon=0.01, max_nodes=31` | 72.0% | `purity_epsilon=0.02, max_nodes=31` 22.0% | 50.0% | 200 | tuning_stable |
| 123 | `d156e8f076a2dfb2` | `purity_epsilon=0.01, max_nodes=31` | 40.5% | `purity_epsilon=0.02, max_nodes=31` 46.5% | -6.0% | 200 | tuning_uncertain |
| 2024 | `a2f113e60f488f42` | `purity_epsilon=0.01, max_nodes=63` | 12.5% | `purity_epsilon=0.02, max_nodes=63` 65.0% | -52.5% | 200 | tuning_uncertain |
| 11 | `9aa83790a361ec28` | `purity_epsilon=0.01, max_nodes=31` | 65.0% | `purity_epsilon=0.02, max_nodes=31` 19.5% | 45.5% | 200 | tuning_stable |

### 2. Robustez externa entre seeds mestre (evidência empírica; não define `tuning_stable`)

| seed mestre | selecionada | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|
| 42 | `purity_epsilon=0.01, max_nodes=31` | 31 | 0.958041958041958 | 0.9230769230769231 | 1184 |
| 7 | `purity_epsilon=0.01, max_nodes=31` | 27 | 0.9230769230769231 | 0.9440559440559441 | 1133 |
| 123 | `purity_epsilon=0.01, max_nodes=31` | 29 | 0.8951048951048951 | 0.916083916083916 | 1045 |
| 2024 | `purity_epsilon=0.01, max_nodes=63` | 55 | 0.9020979020979021 | 0.916083916083916 | 1899 |
| 11 | `purity_epsilon=0.01, max_nodes=31` | 29 | 0.9370629370629371 | 0.951048951048951 | 1104 |

Configurações escolhidas: {'purity_epsilon=0.01, max_nodes=31': 4, 'purity_epsilon=0.01, max_nodes=63': 1}. **Consenso de seleção = 80% (4/5) nas master seeds avaliadas** (moda `purity_epsilon=0.01, max_nodes=31`); seeds mestre com `tuning_stable` pelo bootstrap: 2/5.

Leitura: a evidência de robustez **não** é suficiente (consenso externo e/ou reamostragem interna não a sustentam); usar *incerto*, não *estável*.

### Conclusão — bc_user.arff

**A árvore de 3 nós não é suportada como uma estrutura robusta e consistentemente preferível; a sua fidelity pode ser equivalente, mas a sua estrutura apresenta forte dependência da amostragem.**

Evidência (só treino, CV repetida): 10 de 30 (seed mestre × configuração) produziram stumps em parte das partições; destas, 0 foram selecionadas.

| seed mestre | configuração | fração de stumps | fidelity média | instab. estrutural | resultado |
|---|---|---|---|---|---|
| 7 | purity_epsilon=0.05, max_nodes=31 | 80% | 0.922 | 1.31 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=31 | 7% | 0.947 | 0.28 | LOST_STRUCTURAL_STABILITY |
| 7 | purity_epsilon=0.05, max_nodes=63 | 80% | 0.923 | 1.64 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=63 | 7% | 0.949 | 0.30 | LOST_STRUCTURAL_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=31 | 47% | 0.925 | 0.82 | LOST_STRUCTURAL_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=63 | 47% | 0.925 | 0.88 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=31 | 13% | 0.931 | 0.38 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=63 | 13% | 0.929 | 0.40 | LOST_STRUCTURAL_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=31 | 20% | 0.942 | 0.47 | LOST_STRUCTURAL_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=63 | 20% | 0.944 | 0.49 | LOST_STABILITY |

#### bc_user.arff — seed mestre 42 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [42, 1051, 2060, 3069, 4078]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=457012f5f057926b` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '457012f5f057926b', 'trepan_reloaded': '457012f5f057926b'}); queries por escopo: {'tuning': 3925933, 'trepan_pair': 44390, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 31970; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.927 | 0.021 | 0.894 | 27.7 ± 3.2 | 0.11 | 8.2 (0.17) | 14.3 (0.11) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 63001 | 15604 | 0/15 | 0% | 4.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 27.7 nós médios vs 24.7 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.931 | 0.022 | 0.894 | 25.0 ± 4.1 | 0.17 | 6.3 (0.15) | 13.0 (0.16) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14577 | 0/15 | 0% | 3.8 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.34 ≤ 1.76), mas desvio-padrão 0.022 > 0.017 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.931 | 0.022 | 0.894 | 24.7 ± 4.0 | 0.16 | 6.1 (0.14) | 12.9 (0.16) | 0.16 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14252 | 0/15 | 0% | 3.8 | **WINNER** — VENCE: fidelity média 0.931 (a melhor). 6 indistinguível(eis) em fidelity -> 5 após estabilidade da fidelity -> 5 após estabilidade estrutural (índice 0.16) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.926 | 0.020 | 0.887 | 53.8 ± 4.9 | 0.09 | 11.3 (0.18) | 27.4 (0.09) | 0.18 | 100% (15/15) | 63 | ⚠ sim | 63001 | 30139 | 0/15 | 0% | 9.9 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 53.8 nós médios vs 24.7 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.931 | 0.017 | 0.901 | 55.3 ± 5.2 | 0.09 | 9.5 (0.21) | 28.1 (0.09) | 0.21 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29876 | 0/15 | 0% | 10.6 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 55.3 nós médios vs 24.7 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.931 | 0.017 | 0.901 | 54.6 ± 5.2 | 0.10 | 9.2 (0.18) | 27.8 (0.09) | 0.18 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29333 | 0/15 | 0% | 10.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.6 nós médios vs 24.7 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.927 | 0.021 | 0.023 | 0.006 | 0.920, 0.934, 0.927, 0.923, 0.930 |
| purity_epsilon=0.02, max_nodes=31 | 0.931 | 0.022 | 0.022 | 0.012 | 0.925, 0.934, 0.948, 0.927, 0.918 |
| purity_epsilon=0.01, max_nodes=31 | 0.931 | 0.022 | 0.022 | 0.011 | 0.925, 0.934, 0.948, 0.927, 0.920 |
| purity_epsilon=0.05, max_nodes=63 | 0.926 | 0.020 | 0.022 | 0.006 | 0.920, 0.934, 0.927, 0.920, 0.930 |
| purity_epsilon=0.02, max_nodes=63 | 0.931 | 0.017 | 0.019 | 0.006 | 0.927, 0.934, 0.939, 0.925, 0.930 |
| purity_epsilon=0.01, max_nodes=63 | 0.931 | 0.017 | 0.019 | 0.006 | 0.925, 0.934, 0.939, 0.925, 0.930 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.66 | 0.25 | 18 | 0.62 | 0.22 | 0% | area_worst 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.02, max_nodes=31 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.63 | 0.24 | 11 | 0.68 | 0.27 | 0% | concavity_worst 100%, radius_se 100%, symmetry_worst 100%, texture_se 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.64 | 0.24 | 11 | 0.74 | 0.26 | 0% | area_worst 100%, concavity_worst 100%, radius_se 100%, symmetry_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.70 | 0.32 | 14 | 0.77 | 0.36 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.02, max_nodes=63 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.74 | 0.36 | 8 | 0.74 | 0.34 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.74 | 0.37 | 11 | 0.76 | 0.37 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 | 0.894 | 0.937 | 0.937 | 0.915 | 0.923 | 0.951 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 | 0.915 | 0.944 | 0.923 | 0.908 | 0.908 | 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 | 0.915 | 0.944 | 0.923 | 0.908 | 0.915 | 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 | 0.887 | 0.944 | 0.930 | 0.915 | 0.930 | 0.944 |
| purity_epsilon=0.02, max_nodes=63 | 0.951 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 | 0.915 | 0.944 | 0.915 | 0.915 | 0.923 | 0.951 |
| purity_epsilon=0.01, max_nodes=63 | 0.951 | 0.923 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 | 0.915 | 0.944 | 0.915 | 0.923 | 0.915 | 0.951 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 27 | 21 | 25 | 31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 25 | 27 | 29 | 23 |
| purity_epsilon=0.02, max_nodes=31 | 17 | 23 | 21 | 27 | 31 | 29 | 23 | 31 | 29 | 21 | 25 | 23 | 27 | 27 | 21 |
| purity_epsilon=0.01, max_nodes=31 | 17 | 23 | 23 | 27 | 31 | 21 | 23 | 31 | 29 | 21 | 25 | 25 | 27 | 27 | 21 |
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 55 | 59 | 51 | 55 | 49 | 51 | 55 | 55 | 43 | 53 | 47 | 55 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 61 | 61 | 57 | 55 | 51 | 55 | 63 | 57 | 51 | 45 | 49 | 53 | 61 | 59 | 51 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 61 | 55 | 55 | 51 | 57 | 63 | 57 | 51 | 45 | 49 | 51 | 57 | 53 | 51 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=31` — estado interno do tuning: **tuning_uncertain**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 7961)
- selection_probability do vencedor: **46.0%** (intervalo Monte-Carlo 95%: 39.2%–52.9%)
- segundo colocado: `purity_epsilon=0.05, max_nodes=63` 14.5%; margem 1.º−2.º: 31.5%
- IC bootstrap 95% da fidelity média do vencedor: [0.923, 0.940]
- distribuição das configurações escolhidas: `purity_epsilon=0.01, max_nodes=31` 46.0%, `purity_epsilon=0.05, max_nodes=63` 14.5%, `purity_epsilon=0.01, max_nodes=63` 12.5%, `purity_epsilon=0.05, max_nodes=31` 12.0%, `purity_epsilon=0.02, max_nodes=31` 10.0%, `purity_epsilon=0.02, max_nodes=63` 5.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.01, max_nodes=63']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 31, folhas 16, accuracy 0.958041958041958, fidelity 0.9230769230769231; MLP accuracy 0.951048951048951.

#### bc_user.arff — seed mestre 7 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [7, 1016, 2025, 3034, 4043]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=ea40639ef0e616a1` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'ea40639ef0e616a1', 'trepan_reloaded': 'ea40639ef0e616a1'}); queries por escopo: {'tuning': 3455424, 'trepan_pair': 44695, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 32286; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.922 | 0.031 | 0.866 | 8.1 ± 10.6 | 1.31 | 2.5 (1.26) | 4.5 (1.16) | 1.31 | 20% (3/15) | 31 | não | 63001 | 5136 | 0/15 | 0% | 1.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.031, t=1.50 ≤ 1.76), mas desvio-padrão 0.031 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.018 | 0.915 | 25.0 ± 7.0 | 0.28 | 8.4 (0.28) | 13.0 (0.27) | 0.28 | 93% (14/15) | 31 | ⚠ sim | 63001 | 14495 | 0/15 | 0% | 3.8 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.28 (CV de nós 0.28; nós 3–31) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.951 | 0.016 | 0.915 | 26.5 ± 4.6 | 0.17 | 7.8 (0.16) | 13.7 (0.17) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14485 | 0/15 | 0% | 5.2 | **WINNER** — VENCE: fidelity média 0.951; indistinguível da melhor (purity_epsilon=0.01, max_nodes=63, Δ=0.002, t=0.40 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 4 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.17) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.031 | 0.866 | 14.2 ± 23.2 | 1.64 | 3.3 (1.45) | 7.6 (1.53) | 1.64 | 20% (3/15) | 63 | não | 63001 | 8131 | 0/15 | 0% | 2.4 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.030, t=1.45 ≤ 1.76), mas desvio-padrão 0.031 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.949 | 0.018 | 0.915 | 47.4 ± 14.2 | 0.30 | 11.1 (0.29) | 24.2 (0.29) | 0.30 | 93% (14/15) | 63 | ⚠ sim | 63001 | 28152 | 0/15 | 0% | 10.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.30 (CV de nós 0.30; nós 3–63) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.953 | 0.016 | 0.915 | 54.2 ± 6.1 | 0.11 | 11.5 (0.16) | 27.6 (0.11) | 0.16 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29364 | 0/15 | 0% | 11.4 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.2 nós médios vs 26.5 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.922 | 0.031 | 0.032 | 0.012 | 0.913, 0.923, 0.939, 0.908, 0.927 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.018 | 0.015 | 0.012 | 0.934, 0.958, 0.958, 0.951, 0.934 |
| purity_epsilon=0.01, max_nodes=31 | 0.951 | 0.016 | 0.015 | 0.008 | 0.946, 0.958, 0.958, 0.953, 0.939 |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.031 | 0.033 | 0.013 | 0.913, 0.923, 0.941, 0.908, 0.927 |
| purity_epsilon=0.02, max_nodes=63 | 0.949 | 0.018 | 0.015 | 0.012 | 0.937, 0.955, 0.960, 0.958, 0.934 |
| purity_epsilon=0.01, max_nodes=63 | 0.953 | 0.016 | 0.015 | 0.009 | 0.948, 0.955, 0.960, 0.960, 0.939 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.32 | 0.14 | 66 | 0.37 | 0.18 | 80% | concave_points_mean 80%, perimeter_worst 80%, radius_worst 60%, area_worst 40% |
| purity_epsilon=0.02, max_nodes=31 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.59 | 0.23 | 12 | 0.68 | 0.27 | 7% | perimeter_worst 100%, concave_points_mean 93%, concave_points_worst 93%, concavity_mean 93% |
| purity_epsilon=0.01, max_nodes=31 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.68 | 0.29 | 16 | 0.71 | 0.30 | 0% | concave_points_worst 100%, concavity_worst 100%, perimeter_worst 100%, radius_se 100% |
| purity_epsilon=0.05, max_nodes=63 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.31 | 0.13 | 66 | 0.37 | 0.18 | 80% | concave_points_mean 80%, perimeter_worst 80%, radius_worst 60%, area_worst 47% |
| purity_epsilon=0.02, max_nodes=63 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.63 | 0.28 | 3 | 0.75 | 0.36 | 7% | concave_points_worst 100%, perimeter_worst 100%, concave_points_mean 93%, concavity_mean 93% |
| purity_epsilon=0.01, max_nodes=63 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.75 | 0.35 | 11 | 0.73 | 0.39 | 0% | area_worst 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.958 | 0.894 | 0.901 | 0.930 | 0.908 | 0.965 | 0.908 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.944 | 0.915 | 0.972 | 0.951 | 0.951 | 0.972 | 0.944 | 0.958 | 0.958 | 0.951 | 0.944 | 0.923 | 0.965 | 0.915 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.951 | 0.944 | 0.965 | 0.951 | 0.958 | 0.972 | 0.944 | 0.958 | 0.979 | 0.944 | 0.937 | 0.937 | 0.965 | 0.915 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.965 | 0.894 | 0.901 | 0.930 | 0.908 | 0.965 | 0.908 |
| purity_epsilon=0.02, max_nodes=63 | 0.944 | 0.944 | 0.923 | 0.972 | 0.944 | 0.951 | 0.972 | 0.944 | 0.965 | 0.958 | 0.951 | 0.965 | 0.923 | 0.965 | 0.915 |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.958 | 0.944 | 0.965 | 0.944 | 0.958 | 0.972 | 0.944 | 0.965 | 0.979 | 0.944 | 0.958 | 0.937 | 0.965 | 0.915 |

Nós por partição:

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 29 | 3 | 3 | 25 | 3 | 31 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 27 | 27 | 21 | 31 | 3 | 23 | 31 | 29 | 23 | 29 | 27 | 25 | 31 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 25 | 21 | 29 | 31 | 27 | 31 | 29 | 17 | 29 | 31 | 25 | 31 | 21 |
| purity_epsilon=0.05, max_nodes=63 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 63 | 3 | 3 | 59 | 3 | 55 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 45 | 57 | 51 | 49 | 53 | 3 | 39 | 57 | 63 | 41 | 59 | 53 | 43 | 55 | 43 |
| purity_epsilon=0.01, max_nodes=63 | 45 | 59 | 57 | 43 | 57 | 53 | 43 | 57 | 63 | 57 | 55 | 55 | 61 | 55 | 53 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=31` — estado interno do tuning: **tuning_stable**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 7926)
- selection_probability do vencedor: **72.0%** (intervalo Monte-Carlo 95%: 65.4%–77.8%)
- segundo colocado: `purity_epsilon=0.02, max_nodes=31` 22.0%; margem 1.º−2.º: 50.0%
- IC bootstrap 95% da fidelity média do vencedor: [0.945, 0.958]
- distribuição das configurações escolhidas: `purity_epsilon=0.01, max_nodes=31` 72.0%, `purity_epsilon=0.02, max_nodes=31` 22.0%, `purity_epsilon=0.01, max_nodes=63` 5.5%, `purity_epsilon=0.02, max_nodes=63` 0.5%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.01, max_nodes=31', 'purity_epsilon=0.01, max_nodes=63', 'purity_epsilon=0.01, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 27, folhas 14, accuracy 0.9230769230769231, fidelity 0.9440559440559441; MLP accuracy 0.9790209790209791.

#### bc_user.arff — seed mestre 123 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [123, 1132, 2141, 3150, 4159]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=d156e8f076a2dfb2` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'd156e8f076a2dfb2', 'trepan_reloaded': 'd156e8f076a2dfb2'}); queries por escopo: {'tuning': 3640332, 'trepan_pair': 44478, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 32586; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.925 | 0.015 | 0.901 | 15.4 ± 12.6 | 0.82 | 4.8 (0.77) | 8.2 (0.77) | 0.82 | 53% (8/15) | 31 | ⚠ sim | 63001 | 9521 | 0/15 | 0% | 2.3 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.82 (CV de nós 0.82; nós 3–31) > 0.15 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.015 | 0.915 | 26.3 ± 3.9 | 0.15 | 7.7 (0.16) | 13.7 (0.14) | 0.16 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14944 | 0/15 | 0% | 3.9 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 26.3 nós médios vs 25.9 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.939 | 0.017 | 0.908 | 25.9 ± 4.3 | 0.17 | 6.9 (0.18) | 13.5 (0.16) | 0.18 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14134 | 0/15 | 0% | 3.9 | **WINNER** — VENCE: fidelity média 0.939; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.001, t=0.17 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 6 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.18) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.925 | 0.016 | 0.901 | 31.1 ± 27.4 | 0.88 | 6.5 (0.83) | 16.1 (0.85) | 0.88 | 53% (8/15) | 63 | ⚠ sim | 63001 | 17318 | 0/15 | 0% | 5.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.88 (CV de nós 0.88; nós 3–61) > 0.15 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.936 | 0.017 | 0.915 | 54.2 ± 4.6 | 0.09 | 10.7 (0.16) | 27.6 (0.08) | 0.16 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29742 | 0/15 | 0% | 10.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.2 nós médios vs 25.9 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.938 | 0.018 | 0.908 | 54.5 ± 5.0 | 0.09 | 9.9 (0.15) | 27.7 (0.09) | 0.15 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29091 | 0/15 | 0% | 9.8 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.5 nós médios vs 25.9 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.925 | 0.015 | 0.013 | 0.008 | 0.913, 0.923, 0.934, 0.927, 0.927 |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.015 | 0.011 | 0.010 | 0.930, 0.946, 0.953, 0.932, 0.939 |
| purity_epsilon=0.01, max_nodes=31 | 0.939 | 0.017 | 0.012 | 0.013 | 0.927, 0.941, 0.960, 0.930, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.925 | 0.016 | 0.015 | 0.009 | 0.913, 0.923, 0.937, 0.927, 0.927 |
| purity_epsilon=0.02, max_nodes=63 | 0.936 | 0.017 | 0.013 | 0.013 | 0.920, 0.946, 0.951, 0.934, 0.927 |
| purity_epsilon=0.01, max_nodes=63 | 0.938 | 0.018 | 0.013 | 0.015 | 0.920, 0.944, 0.960, 0.937, 0.927 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.37 | 0.11 | 27 | 0.43 | 0.12 | 47% | concavity_worst 87%, radius_worst 87%, concave_points_worst 80%, symmetry_worst 73% |
| purity_epsilon=0.02, max_nodes=31 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.67 | 0.24 | 18 | 0.67 | 0.25 | 0% | concavity_worst 100%, radius_worst 100%, symmetry_worst 100%, texture_mean 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.69 | 0.24 | 11 | 0.68 | 0.27 | 0% | concavity_worst 100%, radius_worst 100%, symmetry_worst 100%, texture_mean 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.36 | 0.13 | 23 | 0.38 | 0.11 | 47% | concavity_worst 87%, radius_worst 87%, concave_points_worst 80%, symmetry_worst 73% |
| purity_epsilon=0.02, max_nodes=63 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.76 | 0.35 | 8 | 0.76 | 0.34 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_worst 100%, perimeter_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.79 | 0.35 | 11 | 0.79 | 0.37 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.915 | 0.908 | 0.915 | 0.937 | 0.901 | 0.930 | 0.937 | 0.937 | 0.930 | 0.901 | 0.930 | 0.951 | 0.944 | 0.923 | 0.915 |
| purity_epsilon=0.02, max_nodes=31 | 0.930 | 0.930 | 0.930 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.979 | 0.915 | 0.930 | 0.951 | 0.944 | 0.930 | 0.944 |
| purity_epsilon=0.01, max_nodes=31 | 0.930 | 0.930 | 0.923 | 0.937 | 0.951 | 0.937 | 0.958 | 0.944 | 0.979 | 0.908 | 0.930 | 0.951 | 0.944 | 0.930 | 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.908 | 0.908 | 0.923 | 0.937 | 0.901 | 0.930 | 0.937 | 0.944 | 0.930 | 0.901 | 0.930 | 0.951 | 0.944 | 0.923 | 0.915 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.915 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.972 | 0.915 | 0.937 | 0.951 | 0.944 | 0.915 | 0.923 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.937 | 0.908 | 0.937 | 0.951 | 0.944 | 0.951 | 0.958 | 0.972 | 0.915 | 0.944 | 0.951 | 0.944 | 0.915 | 0.923 |

Nós por partição:

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 21 | 3 | 15 | 29 | 3 | 27 | 31 | 29 | 3 | 3 | 3 | 29 | 29 | 3 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 29 | 23 | 29 | 27 | 25 | 29 | 27 | 15 | 29 | 25 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 31 | 25 | 31 | 29 | 23 | 25 | 29 | 15 | 21 | 25 | 23 | 27 | 31 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 3 | 55 | 55 | 3 | 53 | 59 | 57 | 3 | 3 | 3 | 49 | 57 | 3 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 47 | 51 | 49 | 55 | 51 | 61 | 59 | 59 | 61 | 47 | 55 | 53 | 57 | 53 | 55 |
| purity_epsilon=0.01, max_nodes=63 | 47 | 45 | 51 | 59 | 55 | 55 | 57 | 57 | 61 | 47 | 61 | 53 | 57 | 55 | 57 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=31` — estado interno do tuning: **tuning_uncertain**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 8042)
- selection_probability do vencedor: **40.5%** (intervalo Monte-Carlo 95%: 33.9%–47.4%)
- segundo colocado: `purity_epsilon=0.02, max_nodes=31` 46.5%; margem 1.º−2.º: -6.0%
- IC bootstrap 95% da fidelity média do vencedor: [0.931, 0.949]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 46.5%, `purity_epsilon=0.01, max_nodes=31` 40.5%, `purity_epsilon=0.02, max_nodes=63` 6.5%, `purity_epsilon=0.05, max_nodes=31` 4.0%, `purity_epsilon=0.01, max_nodes=63` 1.5%, `purity_epsilon=0.05, max_nodes=63` 1.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.01, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 29, folhas 15, accuracy 0.8951048951048951, fidelity 0.916083916083916; MLP accuracy 0.965034965034965.

#### bc_user.arff — seed mestre 2024 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [2024, 3033, 4042, 5051, 6060]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=a2f113e60f488f42` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'a2f113e60f488f42', 'trepan_reloaded': 'a2f113e60f488f42'}); queries por escopo: {'tuning': 5803900, 'trepan_pair': 92092, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 32139; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.931 | 0.016 | 0.915 | 22.7 ± 8.5 | 0.37 | 7.3 (0.38) | 11.9 (0.36) | 0.38 | 87% (13/15) | 31 | ⚠ sim | 63001 | 13874 | 0/15 | 0% | 3.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.38 (CV de nós 0.37; nós 3–31) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.933 | 0.015 | 0.908 | 23.5 ± 4.4 | 0.19 | 6.6 (0.25) | 12.3 (0.18) | 0.25 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14770 | 0/15 | 0% | 4.1 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.25 (CV de nós 0.19; nós 17–31) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.929 | 0.021 | 0.887 | 23.7 ± 6.0 | 0.25 | 6.3 (0.19) | 12.3 (0.24) | 0.25 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14326 | 0/15 | 0% | 4.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.004, t=0.49 ≤ 1.76), mas desvio-padrão 0.021 > 0.015 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.929 | 0.018 | 0.908 | 43.4 ± 17.5 | 0.40 | 9.5 (0.38) | 22.2 (0.39) | 0.40 | 87% (13/15) | 63 | ⚠ sim | 63001 | 26669 | 0/15 | 0% | 8.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.40 (CV de nós 0.40; nós 3–57) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.017 | 0.908 | 54.3 ± 3.8 | 0.07 | 9.4 (0.10) | 27.7 (0.07) | 0.10 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29776 | 0/15 | 0% | 9.5 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.3 nós médios vs 53.8 de purity_epsilon=0.01, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.928 | 0.020 | 0.894 | 53.8 ± 5.9 | 0.11 | 9.5 (0.12) | 27.4 (0.11) | 0.12 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29220 | 0/15 | 0% | 9.2 | **WINNER** — VENCE: fidelity média 0.928; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.005, t=0.52 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 5 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.12) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.931 | 0.016 | 0.013 | 0.012 | 0.925, 0.953, 0.925, 0.927, 0.925 |
| purity_epsilon=0.02, max_nodes=31 | 0.933 | 0.015 | 0.013 | 0.007 | 0.937, 0.930, 0.930, 0.944, 0.925 |
| purity_epsilon=0.01, max_nodes=31 | 0.929 | 0.021 | 0.019 | 0.010 | 0.941, 0.918, 0.923, 0.937, 0.925 |
| purity_epsilon=0.05, max_nodes=63 | 0.929 | 0.018 | 0.014 | 0.014 | 0.923, 0.953, 0.923, 0.918, 0.927 |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.017 | 0.019 | 0.003 | 0.927, 0.930, 0.927, 0.934, 0.932 |
| purity_epsilon=0.01, max_nodes=63 | 0.928 | 0.020 | 0.021 | 0.004 | 0.927, 0.925, 0.923, 0.932, 0.932 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.57 | 0.18 | 12 | 0.73 | 0.20 | 13% | concavity_mean 100%, radius_se 100%, radius_worst 100%, compactness_mean 87% |
| purity_epsilon=0.02, max_nodes=31 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.68 | 0.24 | 12 | 0.66 | 0.22 | 0% | compactness_mean 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.65 | 0.23 | 7 | 0.66 | 0.24 | 0% | compactness_mean 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.59 | 0.23 | 8 | 0.75 | 0.27 | 13% | concavity_mean 100%, radius_se 100%, radius_worst 100%, compactness_mean 87% |
| purity_epsilon=0.02, max_nodes=63 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.76 | 0.36 | 18 | 0.76 | 0.38 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.74 | 0.35 | 29 | 0.75 | 0.37 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.923 | 0.915 | 0.937 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.915 | 0.923 | 0.923 | 0.937 | 0.923 | 0.915 | 0.937 |
| purity_epsilon=0.02, max_nodes=31 | 0.937 | 0.937 | 0.937 | 0.923 | 0.958 | 0.908 | 0.951 | 0.923 | 0.915 | 0.951 | 0.944 | 0.937 | 0.915 | 0.915 | 0.944 |
| purity_epsilon=0.01, max_nodes=31 | 0.937 | 0.937 | 0.951 | 0.887 | 0.958 | 0.908 | 0.951 | 0.923 | 0.894 | 0.937 | 0.944 | 0.930 | 0.915 | 0.915 | 0.944 |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.908 | 0.937 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.908 | 0.923 | 0.923 | 0.908 | 0.923 | 0.923 | 0.937 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.923 | 0.958 | 0.908 | 0.944 | 0.930 | 0.908 | 0.958 | 0.937 | 0.908 | 0.937 | 0.915 | 0.944 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.901 | 0.965 | 0.908 | 0.944 | 0.930 | 0.894 | 0.951 | 0.937 | 0.908 | 0.937 | 0.915 | 0.944 |

Nós por partição:

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 25 | 29 | 29 | 31 | 25 | 27 | 23 | 3 | 25 | 21 | 27 | 25 | 3 | 27 | 21 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 29 | 29 | 21 | 31 | 23 | 17 | 25 | 25 | 19 | 29 | 25 | 21 | 19 | 19 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 13 | 29 | 31 | 23 | 17 | 25 | 31 | 27 | 29 | 27 | 21 | 15 | 17 |
| purity_epsilon=0.05, max_nodes=63 | 33 | 51 | 55 | 41 | 53 | 55 | 49 | 3 | 51 | 55 | 47 | 57 | 3 | 51 | 47 |
| purity_epsilon=0.02, max_nodes=63 | 55 | 55 | 55 | 47 | 53 | 55 | 55 | 55 | 51 | 53 | 59 | 57 | 63 | 49 | 53 |
| purity_epsilon=0.01, max_nodes=63 | 55 | 55 | 55 | 55 | 53 | 55 | 55 | 55 | 57 | 53 | 59 | 55 | 63 | 45 | 37 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=63` — estado interno do tuning: **tuning_uncertain**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 9943)
- selection_probability do vencedor: **12.5%** (intervalo Monte-Carlo 95%: 8.6%–17.8%)
- segundo colocado: `purity_epsilon=0.02, max_nodes=63` 65.0%; margem 1.º−2.º: -52.5%
- IC bootstrap 95% da fidelity média do vencedor: [0.925, 0.931]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=63` 65.0%, `purity_epsilon=0.01, max_nodes=63` 12.5%, `purity_epsilon=0.05, max_nodes=31` 10.5%, `purity_epsilon=0.02, max_nodes=31` 9.0%, `purity_epsilon=0.01, max_nodes=31` 3.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=63 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 55, folhas 28, accuracy 0.9020979020979021, fidelity 0.916083916083916; MLP accuracy 0.972027972027972.

#### bc_user.arff — seed mestre 11 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [11, 1020, 2029, 3038, 4047]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=9aa83790a361ec28` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '9aa83790a361ec28', 'trepan_reloaded': '9aa83790a361ec28'}); queries por escopo: {'tuning': 4303461, 'trepan_pair': 88825, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 63001 (mesmo para todos: True); consumo máximo observado: 31192; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.942 | 0.024 | 0.873 | 23.4 ± 10.9 | 0.47 | 7.1 (0.46) | 12.2 (0.45) | 0.47 | 80% (12/15) | 31 | ⚠ sim | 63001 | 12817 | 0/15 | 0% | 3.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.47 (CV de nós 0.47; nós 3–31) > 0.21 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.946 | 0.020 | 0.901 | 24.6 ± 5.1 | 0.21 | 6.9 (0.19) | 12.8 (0.20) | 0.21 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14445 | 0/15 | 0% | 4.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 24.6 nós médios vs 23.1 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.019 | 0.901 | 23.1 ± 5.1 | 0.22 | 6.5 (0.20) | 12.1 (0.21) | 0.22 | 100% (15/15) | 31 | ⚠ sim | 63001 | 14185 | 0/15 | 0% | 3.9 | **WINNER** — VENCE: fidelity média 0.944; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.002, t=0.44 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 3 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.22) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.944 | 0.025 | 0.873 | 44.7 ± 22.0 | 0.49 | 9.2 (0.49) | 22.9 (0.48) | 0.49 | 80% (12/15) | 63 | ⚠ sim | 63001 | 24445 | 0/15 | 0% | 7.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.17 ≤ 1.76), mas desvio-padrão 0.025 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.945 | 0.025 | 0.887 | 53.9 ± 7.4 | 0.14 | 9.5 (0.14) | 27.5 (0.13) | 0.14 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29193 | 0/15 | 0% | 9.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.001, t=0.12 ≤ 1.76), mas desvio-padrão 0.025 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.024 | 0.887 | 53.9 ± 6.7 | 0.12 | 9.5 (0.15) | 27.5 (0.12) | 0.15 | 100% (15/15) | 63 | ⚠ sim | 63001 | 29008 | 0/15 | 0% | 8.9 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.26 ≤ 1.76), mas desvio-padrão 0.024 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.942 | 0.024 | 0.017 | 0.014 | 0.923, 0.960, 0.939, 0.939, 0.951 |
| purity_epsilon=0.02, max_nodes=31 | 0.946 | 0.020 | 0.017 | 0.012 | 0.937, 0.965, 0.941, 0.951, 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.019 | 0.018 | 0.009 | 0.937, 0.955, 0.941, 0.951, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.944 | 0.025 | 0.020 | 0.014 | 0.923, 0.960, 0.939, 0.948, 0.948 |
| purity_epsilon=0.02, max_nodes=63 | 0.945 | 0.025 | 0.024 | 0.012 | 0.930, 0.958, 0.944, 0.955, 0.939 |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.024 | 0.023 | 0.011 | 0.930, 0.955, 0.941, 0.955, 0.939 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.53 | 0.18 | 18 | 0.68 | 0.26 | 20% | concave_points_worst 100%, area_worst 93%, radius_worst 93%, concavity_worst 87% |
| purity_epsilon=0.02, max_nodes=31 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.67 | 0.24 | 11 | 0.64 | 0.25 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.01, max_nodes=31 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.65 | 0.22 | 13 | 0.62 | 0.21 | 0% | concave_points_worst 100%, concavity_worst 100%, radius_se 100%, texture_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.55 | 0.23 | 10 | 0.62 | 0.28 | 20% | concave_points_worst 100%, area_worst 93%, radius_worst 93%, concavity_worst 87% |
| purity_epsilon=0.02, max_nodes=63 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.79 | 0.34 | 8 | 0.80 | 0.37 | 0% | compactness_mean 100%, compactness_worst 100%, concave_points_worst 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.80 | 0.35 | 8 | 0.77 | 0.37 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_worst 100%, perimeter_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 | 0.923 | 0.937 | 0.958 | 0.951 | 0.951 | 0.951 |
| purity_epsilon=0.02, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.965 | 0.972 | 0.965 | 0.937 | 0.923 | 0.944 | 0.951 | 0.958 | 0.937 | 0.915 | 0.958 |
| purity_epsilon=0.01, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.944 | 0.965 | 0.965 | 0.937 | 0.923 | 0.944 | 0.951 | 0.958 | 0.937 | 0.915 | 0.958 |
| purity_epsilon=0.05, max_nodes=63 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 | 0.923 | 0.944 | 0.979 | 0.951 | 0.944 | 0.951 |
| purity_epsilon=0.02, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.965 | 0.951 | 0.965 | 0.944 | 0.923 | 0.937 | 0.958 | 0.972 | 0.944 | 0.901 | 0.972 |
| purity_epsilon=0.01, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.951 | 0.958 | 0.965 | 0.937 | 0.923 | 0.937 | 0.958 | 0.972 | 0.944 | 0.901 | 0.972 |

Nós por partição:

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 31 | 3 | 29 | 27 | 31 | 31 | 29 | 3 | 29 | 31 | 23 | 3 | 23 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 21 | 17 | 25 | 27 | 31 | 23 | 25 | 29 | 17 | 25 | 21 | 31 | 29 | 17 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 21 | 17 | 25 | 23 | 21 | 23 | 17 | 29 | 17 | 25 | 21 | 31 | 29 | 17 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 59 | 3 | 51 | 51 | 57 | 49 | 61 | 3 | 57 | 63 | 51 | 3 | 53 | 53 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 53 | 59 | 61 | 37 | 59 | 61 | 51 | 59 | 57 | 61 | 55 | 47 | 51 | 41 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 53 | 59 | 61 | 47 | 57 | 61 | 43 | 59 | 57 | 61 | 55 | 47 | 51 | 41 |

**Configuração selecionada:** `purity_epsilon=0.01, max_nodes=31` — estado interno do tuning: **tuning_stable**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 7930)
- selection_probability do vencedor: **65.0%** (intervalo Monte-Carlo 95%: 58.2%–71.3%)
- segundo colocado: `purity_epsilon=0.02, max_nodes=31` 19.5%; margem 1.º−2.º: 45.5%
- IC bootstrap 95% da fidelity média do vencedor: [0.938, 0.950]
- distribuição das configurações escolhidas: `purity_epsilon=0.01, max_nodes=31` 65.0%, `purity_epsilon=0.02, max_nodes=31` 19.5%, `purity_epsilon=0.01, max_nodes=63` 7.5%, `purity_epsilon=0.05, max_nodes=31` 6.0%, `purity_epsilon=0.02, max_nodes=63` 2.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 29, folhas 15, accuracy 0.9370629370629371, fidelity 0.951048951048951; MLP accuracy 0.972027972027972.

## iris_t.arff — oráculo `factory`

### 1. Estabilidade interna do tuning (block bootstrap, por seed mestre)

| seed mestre | oracle_id | selecionada | selection_probability | 2.º colocado | margem | reamostragens | tuning |
|---|---|---|---|---|---|---|---|
| 42 | `fa910f6f25e5c1a3` | `purity_epsilon=0.05, max_nodes=31` | 62.0% | `purity_epsilon=0.02, max_nodes=31` 20.0% | 42.0% | 200 | tuning_stable |
| 7 | `bec1d5a4e626d590` | `purity_epsilon=0.05, max_nodes=63` | 92.0% | `purity_epsilon=0.02, max_nodes=31` 4.0% | 88.0% | 200 | tuning_stable |
| 123 | `3c656fb0e0870746` | `purity_epsilon=0.02, max_nodes=31` | 61.5% | `purity_epsilon=0.05, max_nodes=31` 16.5% | 45.0% | 200 | tuning_stable |
| 2024 | `9886716fca9b4dba` | `purity_epsilon=0.02, max_nodes=31` | 50.0% | `purity_epsilon=0.05, max_nodes=63` 24.5% | 25.5% | 200 | tuning_uncertain |
| 11 | `3a8a750475e18459` | `purity_epsilon=0.02, max_nodes=31` | 95.0% | `purity_epsilon=0.01, max_nodes=31` 3.5% | 91.5% | 200 | tuning_stable |

### 2. Robustez externa entre seeds mestre (evidência empírica; não define `tuning_stable`)

| seed mestre | selecionada | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|
| 42 | `purity_epsilon=0.05, max_nodes=31` | 25 | 0.8947368421052632 | 0.8947368421052632 | 121 |
| 7 | `purity_epsilon=0.05, max_nodes=63` | 63 | 0.9736842105263158 | 0.9736842105263158 | 145 |
| 123 | `purity_epsilon=0.02, max_nodes=31` | 29 | 0.9210526315789473 | 0.9736842105263158 | 119 |
| 2024 | `purity_epsilon=0.02, max_nodes=31` | 29 | 0.9473684210526315 | 0.9210526315789473 | 101 |
| 11 | `purity_epsilon=0.02, max_nodes=31` | 23 | 0.8947368421052632 | 0.9210526315789473 | 109 |

Configurações escolhidas: {'purity_epsilon=0.05, max_nodes=31': 1, 'purity_epsilon=0.05, max_nodes=63': 1, 'purity_epsilon=0.02, max_nodes=31': 3}. **Consenso de seleção = 60% (3/5) nas master seeds avaliadas** (moda `purity_epsilon=0.02, max_nodes=31`); seeds mestre com `tuning_stable` pelo bootstrap: 4/5.

Leitura: o consenso externo e a reamostragem interna apontam no mesmo sentido; só neste caso se usa o termo *robusto/estável* (com a ressalva do número reduzido de master seeds).

### Conclusão — iris_t.arff

Nenhuma configuração produziu árvores de ≤3 nós (stumps) nas partições de CV desta experiência.

#### iris_t.arff — seed mestre 42 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [42, 1051, 2060, 3069, 4078]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=fa910f6f25e5c1a3` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'fa910f6f25e5c1a3', 'trepan_reloaded': 'fa910f6f25e5c1a3'}); queries por escopo: {'tuning': 1143614, 'trepan_pair': 14926, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10441; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.921 | 0.033 | 0.865 | 28.3 ± 2.9 | 0.10 | 8.1 (0.16) | 14.7 (0.10) | 0.16 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4870 | 0/15 | 0% | 0.5 | **WINNER** — VENCE: fidelity média 0.921; indistinguível da melhor (purity_epsilon=0.02, max_nodes=63, Δ=0.009, t=0.49 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 6 após estabilidade da fidelity -> 6 após estabilidade estrutural (índice 0.16) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.925 | 0.031 | 0.892 | 29.3 ± 2.4 | 0.08 | 8.0 (0.19) | 15.1 (0.08) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4619 | 0/15 | 0% | 0.6 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.3 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.925 | 0.031 | 0.892 | 29.3 ± 2.4 | 0.08 | 8.0 (0.19) | 15.1 (0.08) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4619 | 0/15 | 0% | 0.6 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.3 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.927 | 0.031 | 0.865 | 57.8 ± 3.3 | 0.06 | 12.1 (0.20) | 29.4 (0.06) | 0.20 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9759 | 0/15 | 0% | 1.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.8 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.028 | 0.892 | 59.4 ± 3.3 | 0.06 | 11.7 (0.19) | 30.2 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9396 | 0/15 | 0% | 1.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.4 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.930 | 0.028 | 0.892 | 59.3 ± 3.2 | 0.05 | 11.5 (0.19) | 30.1 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9371 | 0/15 | 0% | 1.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.3 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.921 | 0.033 | 0.028 | 0.022 | 0.884, 0.937, 0.937, 0.920, 0.929 |
| purity_epsilon=0.02, max_nodes=31 | 0.925 | 0.031 | 0.025 | 0.020 | 0.893, 0.937, 0.937, 0.920, 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.925 | 0.031 | 0.025 | 0.020 | 0.893, 0.937, 0.937, 0.920, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.927 | 0.031 | 0.025 | 0.021 | 0.902, 0.937, 0.955, 0.911, 0.929 |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |
| purity_epsilon=0.01, max_nodes=63 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.40 | 28 | 1.00 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.42 | 34 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.42 | 34 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.45 | 16 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.46 | 17 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.47 | 17 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.895 | 0.892 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.892 | 0.895 | 0.919 | 0.946 | 0.921 | 0.946 | 0.919 |
| purity_epsilon=0.02, max_nodes=31 | 0.895 | 0.892 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.892 | 0.895 | 0.919 | 0.946 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=31 | 0.895 | 0.892 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.892 | 0.895 | 0.919 | 0.946 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=63 | 0.895 | 0.946 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.921 | 0.946 | 0.919 |
| purity_epsilon=0.02, max_nodes=63 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=63 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 27 | 29 | 27 | 23 | 29 | 31 | 29 | 31 | 21 | 31 | 29 | 29 | 29 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 | 27 | 31 | 29 | 31 | 29 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 | 27 | 31 | 29 | 31 | 29 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 57 | 57 | 59 | 53 | 59 | 63 | 61 | 61 | 53 | 55 | 57 | 61 | 53 | 57 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 | 59 | 59 | 57 | 63 | 53 | 59 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 | 59 | 59 | 57 | 61 | 53 | 59 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=31` — estado interno do tuning: **tuning_stable**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 7961)
- selection_probability do vencedor: **62.0%** (intervalo Monte-Carlo 95%: 55.1%–68.4%)
- segundo colocado: `purity_epsilon=0.02, max_nodes=31` 20.0%; margem 1.º−2.º: 42.0%
- IC bootstrap 95% da fidelity média do vencedor: [0.900, 0.936]
- distribuição das configurações escolhidas: `purity_epsilon=0.05, max_nodes=31` 62.0%, `purity_epsilon=0.02, max_nodes=31` 20.0%, `purity_epsilon=0.05, max_nodes=63` 16.5%, `purity_epsilon=0.02, max_nodes=63` 1.5%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 25, folhas 13, accuracy 0.8947368421052632, fidelity 0.8947368421052632; MLP accuracy 0.8947368421052632.

#### iris_t.arff — seed mestre 7 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [7, 1016, 2025, 3034, 4043]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=bec1d5a4e626d590` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'bec1d5a4e626d590', 'trepan_reloaded': 'bec1d5a4e626d590'}); queries por escopo: {'tuning': 2073697, 'trepan_pair': 65492, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 11139; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.047 | 0.838 | 28.6 ± 2.5 | 0.09 | 7.8 (0.24) | 14.8 (0.09) | 0.24 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4821 | 0/15 | 0% | 0.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.24 (CV de nós 0.09; nós 23–31) > 0.12 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.928 | 0.048 | 0.838 | 28.9 ± 2.6 | 0.09 | 7.5 (0.24) | 14.9 (0.09) | 0.24 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4741 | 0/15 | 0% | 0.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.24 (CV de nós 0.09; nós 23–31) > 0.12 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.928 | 0.048 | 0.838 | 28.9 ± 2.6 | 0.09 | 7.5 (0.24) | 14.9 (0.09) | 0.24 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4741 | 0/15 | 0% | 0.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.24 (CV de nós 0.09; nós 23–31) > 0.12 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.928 | 0.048 | 0.838 | 57.8 ± 3.4 | 0.06 | 11.5 (0.13) | 29.4 (0.06) | 0.13 | 100% (15/15) | 63 | ⚠ sim | 21169 | 10033 | 0/15 | 0% | 0.9 | **WINNER** — VENCE: fidelity média 0.928; indistinguível da melhor (purity_epsilon=0.05, max_nodes=31, Δ=0.002, t=0.13 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 6 após estabilidade da fidelity -> 3 após estabilidade estrutural (índice 0.13) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.925 | 0.048 | 0.838 | 59.4 ± 2.6 | 0.04 | 11.5 (0.12) | 30.2 (0.04) | 0.12 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9756 | 0/15 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.4 nós médios vs 57.8 de purity_epsilon=0.05, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.048 | 0.838 | 59.4 ± 2.6 | 0.04 | 11.5 (0.12) | 30.2 (0.04) | 0.12 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9756 | 0/15 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.4 nós médios vs 57.8 de purity_epsilon=0.05, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.047 | 0.043 | 0.029 | 0.911, 0.920, 0.946, 0.902, 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.928 | 0.048 | 0.045 | 0.028 | 0.911, 0.920, 0.937, 0.902, 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 0.928 | 0.048 | 0.045 | 0.028 | 0.911, 0.920, 0.937, 0.902, 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 0.928 | 0.048 | 0.047 | 0.026 | 0.902, 0.920, 0.946, 0.911, 0.964 |
| purity_epsilon=0.02, max_nodes=63 | 0.925 | 0.048 | 0.047 | 0.026 | 0.902, 0.920, 0.937, 0.902, 0.964 |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.048 | 0.047 | 0.026 | 0.902, 0.920, 0.937, 0.902, 0.964 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.43 | 24 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.41 | 27 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.41 | 27 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.44 | 19 | 1.00 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.44 | 20 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.44 | 20 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.973 | 0.946 |
| purity_epsilon=0.02, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.973 | 0.946 |
| purity_epsilon=0.01, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.973 | 0.946 |
| purity_epsilon=0.05, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 | 0.921 | 0.973 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |

Nós por partição:

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 27 | 31 | 29 | 31 | 29 | 31 | 29 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 29 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 29 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 61 | 53 | 59 | 57 | 49 | 59 | 55 | 59 | 57 | 63 | 59 | 57 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 | 59 | 61 | 61 | 61 | 61 | 57 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 | 59 | 61 | 61 | 61 | 61 | 57 |

**Configuração selecionada:** `purity_epsilon=0.05, max_nodes=63` — estado interno do tuning: **tuning_stable**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 7926)
- selection_probability do vencedor: **92.0%** (intervalo Monte-Carlo 95%: 87.4%–95.0%)
- segundo colocado: `purity_epsilon=0.02, max_nodes=31` 4.0%; margem 1.º−2.º: 88.0%
- IC bootstrap 95% da fidelity média do vencedor: [0.907, 0.944]
- distribuição das configurações escolhidas: `purity_epsilon=0.05, max_nodes=63` 92.0%, `purity_epsilon=0.02, max_nodes=31` 4.0%, `purity_epsilon=0.02, max_nodes=63` 3.0%, `purity_epsilon=0.05, max_nodes=31` 1.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.05, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=63 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 63, folhas 32, accuracy 0.9736842105263158, fidelity 0.9736842105263158; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 123 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [123, 1132, 2141, 3150, 4159]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=3c656fb0e0870746` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '3c656fb0e0870746', 'trepan_reloaded': '3c656fb0e0870746'}); queries por escopo: {'tuning': 1115750, 'trepan_pair': 15784, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10797; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.943 | 0.039 | 0.865 | 27.4 ± 2.5 | 0.09 | 7.9 (0.20) | 14.2 (0.09) | 0.20 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4761 | 0/15 | 0% | 0.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.76), mas desvio-padrão 0.039 > 0.031 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.941 | 0.031 | 0.865 | 27.8 ± 2.5 | 0.09 | 7.8 (0.20) | 14.4 (0.09) | 0.20 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4587 | 0/15 | 0% | 0.5 | **WINNER** — VENCE: fidelity média 0.941; indistinguível da melhor (purity_epsilon=0.05, max_nodes=31, Δ=0.002, t=0.11 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 4 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.20) -> 2 após complexidade; empate real resolvido pela configuração mais simples. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.941 | 0.031 | 0.865 | 27.8 ± 2.5 | 0.09 | 7.8 (0.20) | 14.4 (0.09) | 0.20 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4565 | 0/15 | 0% | 0.5 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=31). ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.941 | 0.041 | 0.865 | 59.3 ± 3.2 | 0.05 | 12.1 (0.18) | 30.1 (0.05) | 0.18 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9720 | 0/15 | 0% | 1.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.34 ≤ 1.76), mas desvio-padrão 0.041 > 0.031 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.941 | 0.031 | 0.865 | 60.3 ± 3.4 | 0.06 | 11.7 (0.17) | 30.7 (0.05) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9397 | 0/15 | 0% | 1.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 60.3 nós médios vs 27.8 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.941 | 0.031 | 0.865 | 60.3 ± 3.4 | 0.06 | 11.7 (0.17) | 30.7 (0.05) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9375 | 0/15 | 0% | 1.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 60.3 nós médios vs 27.8 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.943 | 0.039 | 0.030 | 0.031 | 0.955, 0.902, 0.964, 0.920, 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.01, max_nodes=31 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.05, max_nodes=63 | 0.941 | 0.041 | 0.033 | 0.029 | 0.955, 0.902, 0.964, 0.920, 0.964 |
| purity_epsilon=0.02, max_nodes=63 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.01, max_nodes=63 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.40 | 26 | 1.00 | 0.42 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.42 | 25 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.42 | 25 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.48 | 16 | 1.00 | 0.51 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.51 | 26 | 1.00 | 0.52 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.51 | 26 | 1.00 | 0.53 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 1.000 | 0.919 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 1.000 | 0.892 |
| purity_epsilon=0.02, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |

Nós por partição:

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 27 | 27 | 23 | 27 | 27 | 23 | 29 | 31 | 27 | 31 | 31 | 27 | 25 | 27 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 27 | 23 | 27 | 29 | 23 | 29 | 31 | 27 | 31 | 31 | 27 | 29 | 29 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 27 | 23 | 27 | 29 | 23 | 29 | 31 | 27 | 31 | 31 | 27 | 29 | 29 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 63 | 63 | 61 | 61 | 57 | 59 | 63 | 55 | 61 | 61 | 53 | 55 | 59 | 57 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 | 63 | 61 | 55 | 59 | 61 | 59 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 | 63 | 61 | 55 | 59 | 61 | 59 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31` — estado interno do tuning: **tuning_stable**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 8042)
- selection_probability do vencedor: **61.5%** (intervalo Monte-Carlo 95%: 54.6%–68.0%)
- segundo colocado: `purity_epsilon=0.05, max_nodes=31` 16.5%; margem 1.º−2.º: 45.0%
- IC bootstrap 95% da fidelity média do vencedor: [0.919, 0.957]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 61.5%, `purity_epsilon=0.05, max_nodes=31` 16.5%, `purity_epsilon=0.02, max_nodes=63` 13.0%, `purity_epsilon=0.05, max_nodes=63` 9.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 29, folhas 15, accuracy 0.9210526315789473, fidelity 0.9736842105263158; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 2024 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [2024, 3033, 4042, 5051, 6060]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=9886716fca9b4dba` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '9886716fca9b4dba', 'trepan_reloaded': '9886716fca9b4dba'}); queries por escopo: {'tuning': 1116821, 'trepan_pair': 14906, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10446; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.964 | 0.022 | 0.919 | 29.4 ± 1.7 | 0.06 | 8.7 (0.14) | 15.2 (0.06) | 0.14 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4676 | 0/15 | 0% | 0.5 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.4 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.966 | 0.021 | 0.919 | 29.0 ± 2.1 | 0.07 | 8.5 (0.17) | 15.0 (0.07) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4524 | 0/15 | 0% | 0.4 | **WINNER** — VENCE: fidelity média 0.966; indistinguível da melhor (purity_epsilon=0.02, max_nodes=63, Δ=0.002, t=0.34 ≤ 1.76). 6 indistinguível(eis) em fidelity -> 6 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.17) -> 2 após complexidade; empate real resolvido pela configuração mais simples. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.966 | 0.021 | 0.919 | 29.0 ± 2.1 | 0.07 | 8.4 (0.19) | 15.0 (0.07) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4502 | 0/15 | 0% | 0.4 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=31). ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.966 | 0.019 | 0.946 | 58.5 ± 3.2 | 0.06 | 12.9 (0.19) | 29.7 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9855 | 0/15 | 0% | 1.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.19 (CV de nós 0.06; nós 51–63) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.968 | 0.018 | 0.946 | 59.8 ± 2.9 | 0.05 | 12.9 (0.19) | 30.4 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9566 | 0/15 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.8 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.968 | 0.018 | 0.946 | 59.8 ± 2.9 | 0.05 | 12.8 (0.20) | 30.4 (0.05) | 0.20 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9521 | 0/15 | 0% | 1.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.05; nós 55–63) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.964 | 0.022 | 0.018 | 0.014 | 0.964, 0.955, 0.946, 0.982, 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.966 | 0.021 | 0.015 | 0.017 | 0.964, 0.955, 0.946, 0.991, 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 0.966 | 0.021 | 0.015 | 0.017 | 0.964, 0.955, 0.946, 0.991, 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 0.966 | 0.019 | 0.015 | 0.012 | 0.964, 0.955, 0.955, 0.982, 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 0.968 | 0.018 | 0.012 | 0.015 | 0.964, 0.955, 0.955, 0.991, 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 0.968 | 0.018 | 0.012 | 0.015 | 0.964, 0.955, 0.955, 0.991, 0.973 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 33 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 32 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 32 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.47 | 21 | 1.00 | 0.48 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.46 | 19 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.47 | 19 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 | 0.947 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 | 0.947 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |

Nós por partição:

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 | 29 | 31 | 31 | 27 | 29 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 | 27 | 29 | 31 | 27 | 31 | 25 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 | 27 | 29 | 31 | 27 | 31 | 25 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 55 | 61 | 59 | 59 | 51 | 59 | 57 | 59 | 61 | 63 | 61 | 59 | 59 | 53 |
| purity_epsilon=0.02, max_nodes=63 | 61 | 63 | 63 | 59 | 59 | 57 | 59 | 57 | 59 | 63 | 63 | 61 | 63 | 55 | 55 |
| purity_epsilon=0.01, max_nodes=63 | 61 | 63 | 63 | 59 | 59 | 57 | 59 | 57 | 59 | 63 | 63 | 61 | 63 | 55 | 55 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31` — estado interno do tuning: **tuning_uncertain**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 9943)
- selection_probability do vencedor: **50.0%** (intervalo Monte-Carlo 95%: 43.1%–56.9%)
- segundo colocado: `purity_epsilon=0.05, max_nodes=63` 24.5%; margem 1.º−2.º: 25.5%
- IC bootstrap 95% da fidelity média do vencedor: [0.954, 0.982]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 50.0%, `purity_epsilon=0.05, max_nodes=63` 24.5%, `purity_epsilon=0.05, max_nodes=31` 21.0%, `purity_epsilon=0.02, max_nodes=63` 4.5%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 29, folhas 15, accuracy 0.9473684210526315, fidelity 0.9210526315789473; MLP accuracy 0.9210526315789473.

#### iris_t.arff — seed mestre 11 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [11, 1020, 2029, 3038, 4047]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=3a8a750475e18459` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '3a8a750475e18459', 'trepan_reloaded': '3a8a750475e18459'}); queries por escopo: {'tuning': 1318606, 'trepan_pair': 32024, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 21169 (mesmo para todos: True); consumo máximo observado: 10597; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.938 | 0.053 | 0.811 | 28.6 ± 2.4 | 0.08 | 8.7 (0.14) | 14.8 (0.08) | 0.14 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4835 | 0/15 | 0% | 0.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.012, t=0.68 ≤ 1.76), mas desvio-padrão 0.053 > 0.036 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=31 | 0.950 | 0.036 | 0.865 | 29.0 ± 2.7 | 0.09 | 8.7 (0.19) | 15.0 (0.09) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4632 | 0/15 | 0% | 0.5 | **WINNER** — VENCE: fidelity média 0.950 (a melhor). 6 indistinguível(eis) em fidelity -> 4 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.19) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=31 | 0.950 | 0.036 | 0.865 | 29.8 ± 1.7 | 0.06 | 8.9 (0.18) | 15.4 (0.05) | 0.18 | 100% (15/15) | 31 | ⚠ sim | 21169 | 4598 | 0/15 | 0% | 0.4 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.8 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.05, max_nodes=63 | 0.938 | 0.053 | 0.811 | 59.3 ± 3.4 | 0.06 | 13.6 (0.22) | 30.1 (0.06) | 0.22 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9931 | 0/15 | 0% | 1.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.012, t=0.68 ≤ 1.76), mas desvio-padrão 0.053 > 0.036 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.02, max_nodes=63 | 0.950 | 0.036 | 0.865 | 57.7 ± 4.8 | 0.08 | 12.6 (0.17) | 29.3 (0.08) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9610 | 0/15 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.7 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |
| purity_epsilon=0.01, max_nodes=63 | 0.950 | 0.036 | 0.865 | 57.3 ± 4.6 | 0.08 | 12.4 (0.17) | 29.1 (0.08) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 21169 | 9577 | 0/15 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.3 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.938 | 0.053 | 0.051 | 0.013 | 0.928, 0.929, 0.947, 0.929, 0.955 |
| purity_epsilon=0.02, max_nodes=31 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.01, max_nodes=31 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.05, max_nodes=63 | 0.938 | 0.053 | 0.051 | 0.013 | 0.928, 0.929, 0.947, 0.929, 0.955 |
| purity_epsilon=0.02, max_nodes=63 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.01, max_nodes=63 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100%, petal_width 20% | 100% | 0.94 | 0.39 | 31 | 0.95 | 0.37 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 87% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100%, petal_width 20% | 100% | 0.91 | 0.42 | 30 | 0.93 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 80% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100%, petal_width 20% | 100% | 0.94 | 0.42 | 42 | 0.92 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 87% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.41 | 18 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.46 | 17 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.47 | 19 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 1.000 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 | 0.842 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 1.000 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 | 0.842 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |

Nós por partição:

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 29 | 31 | 27 | 31 | 31 | 31 | 27 | 31 | 31 | 27 | 29 | 27 | 23 | 27 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 29 | 27 | 21 | 27 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 31 | 27 | 31 | 27 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 57 | 59 | 53 | 63 | 63 | 59 | 55 | 63 | 63 | 61 | 59 | 59 | 55 | 63 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 | 61 | 61 | 45 | 61 | 55 | 63 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 | 61 | 59 | 45 | 57 | 55 | 63 |

**Configuração selecionada:** `purity_epsilon=0.02, max_nodes=31` — estado interno do tuning: **tuning_stable**.

Robustez interna da política (block bootstrap das repetições; teste não usado):

- reamostragens: 200 (blocos = 5 repetições × 3 dobras; seed 7930)
- selection_probability do vencedor: **95.0%** (intervalo Monte-Carlo 95%: 91.0%–97.3%)
- segundo colocado: `purity_epsilon=0.01, max_nodes=31` 3.5%; margem 1.º−2.º: 91.5%
- IC bootstrap 95% da fidelity média do vencedor: [0.945, 0.955]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 95.0%, `purity_epsilon=0.01, max_nodes=31` 3.5%, `purity_epsilon=0.05, max_nodes=31` 1.0%, `purity_epsilon=0.01, max_nodes=63` 0.5%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.01, max_nodes=63']
- saturação do vencedor: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); estabilidade estrutural potencialmente censurada: **True**

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 23, folhas 12, accuracy 0.8947368421052632, fidelity 0.9210526315789473; MLP accuracy 0.9736842105263158.
