> Regressão com expansão adaptativa da capacidade (grelha inicial 31/63 → 127 → 255; CV 5×3; mesmas dobras; orçamento comum não
> limitante; bootstrap exato de blocos). Breast Cancer e Iris são **apenas casos de validação**: o mesmo processo genérico escolhe a
> configuração em cada um. "Consenso de seleção" entre master seeds é evidência externa; só `tuning_stable` (a escolhida pela CV completa
> é a moda do bootstrap e tem probabilidade suficiente) sustenta "estável".
>
> **Observação sobre a saturação:** nos dois datasets, em todas as seeds, os candidatos competitivos continuam a atingir o teto
> (`fraction_at_node_cap = 100%`) em todas as capacidades testadas até ao limite de segurança (255 nós): o TREPAN continua a dividir
> nós com queries sintéticas até esgotar o teto, enquanto a fidelity da CV estabiliza (ex.: 0,927–0,930 de 63 a 255 nós). Por isso a
> expansão termina com `expansion_stop_reason = safety_limit_reached` e a estabilidade estrutural fica `censored_by_node_cap`.

# Tuning da estrutura do TREPAN — Repeated Stratified K-Fold, orçamento comum não limitante e seleção lexicográfica com estabilidade estrutural

Gerado por `scripts/render_tuning_experiment.py` a partir de `scripts/run_trepan_tuning_experiment.py`. Nenhuma regra por dataset; o teste só é usado uma vez, depois da configuração escolhida.

## bc_user.arff — oráculo `factory`

### 1. Estabilidade interna do tuning (block bootstrap, por seed mestre)

| seed mestre | oracle_id | escolhida (CV completa) | P(escolhida) | moda do bootstrap | P(moda) | bootstrap_runner_up | top1−top2 | método (amostras) | grelha de nós final (rondas) | tuning |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `457012f5f057926b` | `purity_epsilon=0.05, max_nodes=31` | 9.3% | `purity_epsilon=0.01, max_nodes=31` | 29.0% | `purity_epsilon=0.01, max_nodes=63` 20.8% | 8.2% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 7 | `ea40639ef0e616a1` | `purity_epsilon=0.01, max_nodes=31` | 58.9% | `purity_epsilon=0.01, max_nodes=31` | 58.9% | `purity_epsilon=0.02, max_nodes=31` 29.7% | 29.2% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 123 | `d156e8f076a2dfb2` | `purity_epsilon=0.02, max_nodes=127` | 16.8% | `purity_epsilon=0.02, max_nodes=31` | 45.7% | `purity_epsilon=0.02, max_nodes=127` 16.8% | 28.9% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 2024 | `a2f113e60f488f42` | `purity_epsilon=0.01, max_nodes=63` | 10.4% | `purity_epsilon=0.02, max_nodes=63` | 66.9% | `purity_epsilon=0.01, max_nodes=63` 10.4% | 56.5% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 11 | `9aa83790a361ec28` | `purity_epsilon=0.01, max_nodes=31` | 53.0% | `purity_epsilon=0.01, max_nodes=31` | 53.0% | `purity_epsilon=0.02, max_nodes=31` 14.2% | 38.8% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |

### 2. Robustez externa entre seeds mestre (evidência empírica; não define `tuning_stable`)

| seed mestre | selecionada | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|
| 42 | `purity_epsilon=0.05, max_nodes=31` | 3 | 0.951048951048951 | 0.916083916083916 | 4484 |
| 7 | `purity_epsilon=0.01, max_nodes=31` | 27 | 0.9230769230769231 | 0.9440559440559441 | 3605 |
| 123 | `purity_epsilon=0.02, max_nodes=127` | 109 | 0.9020979020979021 | 0.9090909090909091 | 6050 |
| 2024 | `purity_epsilon=0.01, max_nodes=63` | 55 | 0.9020979020979021 | 0.916083916083916 | 4896 |
| 11 | `purity_epsilon=0.01, max_nodes=31` | 29 | 0.9370629370629371 | 0.951048951048951 | 4006 |

Configurações escolhidas: {'purity_epsilon=0.05, max_nodes=31': 1, 'purity_epsilon=0.01, max_nodes=31': 2, 'purity_epsilon=0.02, max_nodes=127': 1, 'purity_epsilon=0.01, max_nodes=63': 1}. **Consenso de seleção = 40% (2/5) nas master seeds avaliadas** (moda `purity_epsilon=0.01, max_nodes=31`); seeds mestre com `tuning_stable` pelo bootstrap: 0/5.

Leitura: a evidência de robustez **não** é suficiente (consenso externo e/ou reamostragem interna não a sustentam); usar *incerto*, não *estável*.

### Conclusão — bc_user.arff

**A árvore de 3 nós não é suportada como uma estrutura robusta e consistentemente preferível; a sua fidelity pode ser equivalente, mas a sua estrutura apresenta forte dependência da amostragem.**

Evidência (só treino, CV repetida): 20 de 60 (seed mestre × configuração) produziram stumps em parte das partições; destas, 0 foram selecionadas.

| seed mestre | configuração | fração de stumps | fidelity média | instab. estrutural | resultado |
|---|---|---|---|---|---|
| 7 | purity_epsilon=0.05, max_nodes=31 | 80% | 0.922 | 1.31 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=31 | 7% | 0.947 | 0.28 | LOST_STRUCTURAL_STABILITY |
| 7 | purity_epsilon=0.05, max_nodes=63 | 80% | 0.923 | 1.64 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=63 | 7% | 0.949 | 0.30 | LOST_STRUCTURAL_STABILITY |
| 7 | purity_epsilon=0.05, max_nodes=127 | 80% | 0.923 | 1.82 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=127 | 7% | 0.949 | 0.29 | LOST_STRUCTURAL_STABILITY |
| 7 | purity_epsilon=0.05, max_nodes=255 | 80% | 0.923 | 1.93 | LOST_STABILITY |
| 7 | purity_epsilon=0.02, max_nodes=255 | 7% | 0.949 | 0.29 | LOST_STRUCTURAL_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=31 | 47% | 0.925 | 0.82 | LOST_STRUCTURAL_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=63 | 47% | 0.925 | 0.88 | LOST_STRUCTURAL_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=127 | 47% | 0.927 | 0.92 | LOST_STRUCTURAL_STABILITY |
| 123 | purity_epsilon=0.05, max_nodes=255 | 47% | 0.927 | 0.95 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=31 | 13% | 0.931 | 0.38 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=63 | 13% | 0.929 | 0.40 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=127 | 13% | 0.930 | 0.41 | LOST_STRUCTURAL_STABILITY |
| 2024 | purity_epsilon=0.05, max_nodes=255 | 13% | 0.930 | 0.40 | LOST_STRUCTURAL_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=31 | 20% | 0.942 | 0.47 | LOST_STRUCTURAL_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=63 | 20% | 0.944 | 0.49 | LOST_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=127 | 20% | 0.942 | 0.51 | LOST_STRUCTURAL_STABILITY |
| 11 | purity_epsilon=0.05, max_nodes=255 | 20% | 0.943 | 0.52 | LOST_STRUCTURAL_STABILITY |

#### bc_user.arff — seed mestre 42 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [42, 1051, 2060, 3069, 4078]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=457012f5f057926b` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '457012f5f057926b', 'trepan_reloaded': '457012f5f057926b'}); queries por escopo: {'tuning': 12038975, 'trepan_pair': 9000, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 255001 (mesmo para todos: True); consumo máximo observado: 126115; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.927 | 0.021 | 0.894 | 27.7 ± 3.2 | 0.11 | 8.2 (0.17) | 14.3 (0.11) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 255001 | 15604 | 0/15 | 0% | 4.1 | **WINNER** — VENCE: fidelity média 0.927; indistinguível da melhor (purity_epsilon=0.01, max_nodes=31, Δ=0.004, t=0.39 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 10 após estabilidade da fidelity -> 9 após estabilidade estrutural (índice 0.17) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.931 | 0.022 | 0.894 | 25.0 ± 4.1 | 0.17 | 6.3 (0.15) | 13.0 (0.16) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14577 | 0/15 | 0% | 3.9 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.34 ≤ 1.76), mas desvio-padrão 0.022 > 0.017 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.931 | 0.022 | 0.894 | 24.7 ± 4.0 | 0.16 | 6.1 (0.14) | 12.9 (0.16) | 0.16 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14252 | 0/15 | 0% | 3.8 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.76), mas desvio-padrão 0.022 > 0.017 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.926 | 0.020 | 0.887 | 53.8 ± 4.9 | 0.09 | 11.3 (0.18) | 27.4 (0.09) | 0.18 | 100% (15/15) | 63 | ⚠ sim | 255001 | 30139 | 0/15 | 0% | 9.7 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 53.8 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.931 | 0.017 | 0.901 | 55.3 ± 5.2 | 0.09 | 9.5 (0.21) | 28.1 (0.09) | 0.21 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29876 | 0/15 | 0% | 9.9 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.21 (CV de nós 0.09; nós 45–63) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.931 | 0.017 | 0.901 | 54.6 ± 5.2 | 0.10 | 9.2 (0.18) | 27.8 (0.09) | 0.18 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29333 | 0/15 | 0% | 9.5 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.6 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.926 | 0.019 | 0.894 | 101.5 ± 7.1 | 0.07 | 13.9 (0.14) | 51.3 (0.07) | 0.14 | 100% (15/15) | 127 | ⚠ sim | 255001 | 59723 | 0/15 | 0% | 22.7 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 101.5 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.929 | 0.018 | 0.901 | 106.7 ± 8.7 | 0.08 | 13.0 (0.15) | 53.9 (0.08) | 0.15 | 100% (15/15) | 127 | ⚠ sim | 255001 | 58236 | 0/15 | 0% | 23.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 106.7 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.930 | 0.017 | 0.901 | 107.5 ± 9.1 | 0.08 | 12.9 (0.17) | 54.3 (0.08) | 0.17 | 100% (15/15) | 127 | ⚠ sim | 255001 | 57869 | 0/15 | 0% | 23.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 107.5 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.926 | 0.019 | 0.894 | 202.6 ± 17.2 | 0.08 | 16.5 (0.15) | 101.8 (0.08) | 0.15 | 100% (15/15) | 255 | ⚠ sim | 255001 | 121217 | 0/15 | 0% | 52.5 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 202.6 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.930 | 0.017 | 0.908 | 207.7 ± 14.7 | 0.07 | 15.7 (0.15) | 104.3 (0.07) | 0.15 | 100% (15/15) | 255 | ⚠ sim | 255001 | 118857 | 0/15 | 0% | 49.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 207.7 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.931 | 0.019 | 0.901 | 206.9 ± 13.5 | 0.07 | 15.6 (0.14) | 103.9 (0.06) | 0.14 | 100% (15/15) | 255 | ⚠ sim | 255001 | 118126 | 0/15 | 0% | 50.8 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 206.9 nós médios vs 27.7 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.927 | 0.021 | 0.023 | 0.006 | 0.920, 0.934, 0.927, 0.923, 0.930 |
| purity_epsilon=0.02, max_nodes=31 | 0.931 | 0.022 | 0.022 | 0.012 | 0.925, 0.934, 0.948, 0.927, 0.918 |
| purity_epsilon=0.01, max_nodes=31 | 0.931 | 0.022 | 0.022 | 0.011 | 0.925, 0.934, 0.948, 0.927, 0.920 |
| purity_epsilon=0.05, max_nodes=63 | 0.926 | 0.020 | 0.022 | 0.006 | 0.920, 0.934, 0.927, 0.920, 0.930 |
| purity_epsilon=0.02, max_nodes=63 | 0.931 | 0.017 | 0.019 | 0.006 | 0.927, 0.934, 0.939, 0.925, 0.930 |
| purity_epsilon=0.01, max_nodes=63 | 0.931 | 0.017 | 0.019 | 0.006 | 0.925, 0.934, 0.939, 0.925, 0.930 |
| purity_epsilon=0.05, max_nodes=127 | 0.926 | 0.019 | 0.021 | 0.006 | 0.920, 0.934, 0.927, 0.920, 0.930 |
| purity_epsilon=0.02, max_nodes=127 | 0.929 | 0.018 | 0.020 | 0.005 | 0.927, 0.934, 0.934, 0.925, 0.925 |
| purity_epsilon=0.01, max_nodes=127 | 0.930 | 0.017 | 0.019 | 0.004 | 0.925, 0.934, 0.934, 0.927, 0.930 |
| purity_epsilon=0.05, max_nodes=255 | 0.926 | 0.019 | 0.020 | 0.007 | 0.920, 0.934, 0.927, 0.918, 0.930 |
| purity_epsilon=0.02, max_nodes=255 | 0.930 | 0.017 | 0.019 | 0.004 | 0.930, 0.934, 0.934, 0.927, 0.925 |
| purity_epsilon=0.01, max_nodes=255 | 0.931 | 0.019 | 0.021 | 0.004 | 0.927, 0.934, 0.934, 0.927, 0.930 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.66 | 0.25 | 18 | 0.62 | 0.22 | 0% | area_worst 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.02, max_nodes=31 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.63 | 0.24 | 11 | 0.68 | 0.27 | 0% | concavity_worst 100%, radius_se 100%, symmetry_worst 100%, texture_se 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.64 | 0.24 | 11 | 0.74 | 0.26 | 0% | area_worst 100%, concavity_worst 100%, radius_se 100%, symmetry_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.70 | 0.32 | 14 | 0.77 | 0.36 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.02, max_nodes=63 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.74 | 0.36 | 8 | 0.74 | 0.34 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.74 | 0.37 | 11 | 0.76 | 0.37 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.05, max_nodes=127 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.76 | 0.37 | 5 | 0.76 | 0.41 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.02, max_nodes=127 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.82 | 0.41 | 12 | 0.83 | 0.44 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=127 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.83 | 0.43 | 10 | 0.84 | 0.45 | 0% | area_worst 100%, compactness_mean 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.05, max_nodes=255 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.85 | 0.43 | 3 | 0.78 | 0.39 | 0% | area_se 100%, area_worst 100%, compactness_mean 100%, compactness_worst 100% |
| purity_epsilon=0.02, max_nodes=255 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.87 | 0.47 | 5 | 0.87 | 0.47 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, compactness_worst 100% |
| purity_epsilon=0.01, max_nodes=255 | radius_se 80%, area_worst 47%, radius_worst 47% | 80% | 0.88 | 0.47 | 3 | 0.88 | 0.47 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, compactness_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 | 0.894 | 0.937 | 0.937 | 0.915 | 0.923 | 0.951 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 | 0.915 | 0.944 | 0.923 | 0.908 | 0.908 | 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.937 | 0.894 | 0.951 | 0.937 | 0.915 | 0.965 | 0.908 | 0.972 | 0.915 | 0.944 | 0.923 | 0.908 | 0.915 | 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 | 0.887 | 0.944 | 0.930 | 0.915 | 0.930 | 0.944 |
| purity_epsilon=0.02, max_nodes=63 | 0.951 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 | 0.915 | 0.944 | 0.915 | 0.915 | 0.923 | 0.951 |
| purity_epsilon=0.01, max_nodes=63 | 0.951 | 0.923 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.923 | 0.958 | 0.915 | 0.944 | 0.915 | 0.923 | 0.915 | 0.951 |
| purity_epsilon=0.05, max_nodes=127 | 0.930 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 | 0.894 | 0.944 | 0.923 | 0.915 | 0.930 | 0.944 |
| purity_epsilon=0.02, max_nodes=127 | 0.951 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.908 | 0.958 | 0.930 | 0.937 | 0.908 | 0.915 | 0.915 | 0.944 |
| purity_epsilon=0.01, max_nodes=127 | 0.951 | 0.923 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.908 | 0.958 | 0.930 | 0.937 | 0.915 | 0.908 | 0.937 | 0.944 |
| purity_epsilon=0.05, max_nodes=255 | 0.930 | 0.930 | 0.901 | 0.951 | 0.937 | 0.915 | 0.958 | 0.894 | 0.930 | 0.894 | 0.937 | 0.923 | 0.915 | 0.930 | 0.944 |
| purity_epsilon=0.02, max_nodes=255 | 0.951 | 0.930 | 0.908 | 0.951 | 0.937 | 0.915 | 0.937 | 0.908 | 0.958 | 0.937 | 0.937 | 0.908 | 0.915 | 0.915 | 0.944 |
| purity_epsilon=0.01, max_nodes=255 | 0.958 | 0.923 | 0.901 | 0.951 | 0.937 | 0.915 | 0.937 | 0.908 | 0.958 | 0.937 | 0.937 | 0.908 | 0.908 | 0.937 | 0.944 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 27 | 21 | 25 | 31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 25 | 27 | 29 | 23 |
| purity_epsilon=0.02, max_nodes=31 | 17 | 23 | 21 | 27 | 31 | 29 | 23 | 31 | 29 | 21 | 25 | 23 | 27 | 27 | 21 |
| purity_epsilon=0.01, max_nodes=31 | 17 | 23 | 23 | 27 | 31 | 21 | 23 | 31 | 29 | 21 | 25 | 25 | 27 | 27 | 21 |
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 55 | 59 | 51 | 55 | 49 | 51 | 55 | 55 | 43 | 53 | 47 | 55 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 61 | 61 | 57 | 55 | 51 | 55 | 63 | 57 | 51 | 45 | 49 | 53 | 61 | 59 | 51 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 61 | 55 | 55 | 51 | 57 | 63 | 57 | 51 | 45 | 49 | 51 | 57 | 53 | 51 |
| purity_epsilon=0.05, max_nodes=127 | 105 | 99 | 97 | 103 | 107 | 115 | 111 | 109 | 93 | 93 | 97 | 91 | 101 | 105 | 97 |
| purity_epsilon=0.02, max_nodes=127 | 109 | 117 | 103 | 113 | 107 | 115 | 109 | 109 | 83 | 95 | 113 | 103 | 113 | 103 | 109 |
| purity_epsilon=0.01, max_nodes=127 | 107 | 117 | 99 | 113 | 107 | 115 | 109 | 109 | 83 | 95 | 113 | 115 | 107 | 115 | 109 |
| purity_epsilon=0.05, max_nodes=255 | 187 | 207 | 177 | 225 | 207 | 243 | 185 | 213 | 207 | 201 | 181 | 199 | 205 | 193 | 209 |
| purity_epsilon=0.02, max_nodes=255 | 207 | 217 | 187 | 217 | 207 | 243 | 207 | 203 | 193 | 179 | 211 | 205 | 209 | 215 | 215 |
| purity_epsilon=0.01, max_nodes=255 | 205 | 215 | 185 | 217 | 207 | 233 | 207 | 203 | 193 | 179 | 211 | 209 | 219 | 205 | 215 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.05, max_nodes=31` — estado interno do tuning: **tuning_uncertain** (full_cv_selection_not_bootstrap_modal).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **9.3%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.01, max_nodes=31` com probabilidade 29.0% — selecionada é a moda: **False** ⚠ **fragilidade FORTE da seleção full-CV**
- bootstrap_runner_up: `purity_epsilon=0.01, max_nodes=63` 20.8%; top1_top2_margin: 8.2%
- IC bootstrap 95% da fidelity média da escolhida: [0.923, 0.931]
- distribuição das configurações escolhidas: `purity_epsilon=0.01, max_nodes=31` 29.0%, `purity_epsilon=0.01, max_nodes=63` 20.8%, `purity_epsilon=0.05, max_nodes=127` 11.1%, `purity_epsilon=0.02, max_nodes=127` 10.4%, `purity_epsilon=0.05, max_nodes=31` 9.3%, `purity_epsilon=0.05, max_nodes=63` 9.0%, `purity_epsilon=0.02, max_nodes=31` 4.4%, `purity_epsilon=0.02, max_nodes=63` 2.4%, `purity_epsilon=0.02, max_nodes=255` 1.6%, `purity_epsilon=0.01, max_nodes=255` 1.4%, `purity_epsilon=0.01, max_nodes=127` 0.6%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.01, max_nodes=63']
- saturação da escolhida: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 3, folhas 2, accuracy 0.951048951048951, fidelity 0.916083916083916; MLP accuracy 0.951048951048951.

#### bc_user.arff — seed mestre 7 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [7, 1016, 2025, 3034, 4043]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=ea40639ef0e616a1` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'ea40639ef0e616a1', 'trepan_reloaded': 'ea40639ef0e616a1'}); queries por escopo: {'tuning': 9319883, 'trepan_pair': 44695, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 255001 (mesmo para todos: True); consumo máximo observado: 125483; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.922 | 0.031 | 0.866 | 8.1 ± 10.6 | 1.31 | 2.5 (1.26) | 4.5 (1.16) | 1.31 | 20% (3/15) | 31 | não | 255001 | 5136 | 0/15 | 0% | 1.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.032, t=1.52 ≤ 1.76), mas desvio-padrão 0.031 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.018 | 0.915 | 25.0 ± 7.0 | 0.28 | 8.4 (0.28) | 13.0 (0.27) | 0.28 | 93% (14/15) | 31 | ⚠ sim | 255001 | 14495 | 0/15 | 0% | 3.9 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.28 (CV de nós 0.28; nós 3–31) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.951 | 0.016 | 0.915 | 26.5 ± 4.6 | 0.17 | 7.8 (0.16) | 13.7 (0.17) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14485 | 0/15 | 0% | 5.1 | **WINNER** — VENCE: fidelity média 0.951; indistinguível da melhor (purity_epsilon=0.01, max_nodes=255, Δ=0.003, t=0.68 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 8 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.17) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.031 | 0.866 | 14.2 ± 23.2 | 1.64 | 3.3 (1.45) | 7.6 (1.53) | 1.64 | 20% (3/15) | 63 | não | 255001 | 8131 | 0/15 | 0% | 2.3 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.031, t=1.47 ≤ 1.76), mas desvio-padrão 0.031 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=63 | 0.949 | 0.018 | 0.915 | 47.4 ± 14.2 | 0.30 | 11.1 (0.29) | 24.2 (0.29) | 0.30 | 93% (14/15) | 63 | ⚠ sim | 255001 | 28152 | 0/15 | 0% | 9.3 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.30 (CV de nós 0.30; nós 3–63) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.953 | 0.016 | 0.915 | 54.2 ± 6.1 | 0.11 | 11.5 (0.16) | 27.6 (0.11) | 0.16 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29364 | 0/15 | 0% | 11.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.2 nós médios vs 26.5 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.923 | 0.031 | 0.866 | 24.5 ± 44.6 | 1.82 | 3.6 (1.51) | 12.7 (1.75) | 1.82 | 20% (3/15) | 127 | não | 255001 | 13883 | 0/15 | 0% | 5.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.031, t=1.47 ≤ 1.76), mas desvio-padrão 0.031 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=127 | 0.949 | 0.018 | 0.915 | 95.8 ± 27.1 | 0.28 | 13.7 (0.29) | 48.4 (0.28) | 0.29 | 93% (14/15) | 127 | ⚠ sim | 255001 | 55947 | 0/15 | 0% | 21.9 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.29 (CV de nós 0.28; nós 3–115) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.954 | 0.017 | 0.915 | 103.4 ± 9.8 | 0.09 | 14.6 (0.20) | 52.2 (0.09) | 0.20 | 100% (15/15) | 127 | ⚠ sim | 255001 | 59390 | 0/15 | 0% | 24.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 103.4 nós médios vs 26.5 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.923 | 0.031 | 0.866 | 41.8 ± 80.5 | 1.93 | 4.4 (1.60) | 21.4 (1.88) | 1.93 | 20% (3/15) | 255 | não | 255001 | 25876 | 0/15 | 0% | 9.6 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.031, t=1.47 ≤ 1.76), mas desvio-padrão 0.031 > 0.016 + tolerância 0.005. |
| purity_epsilon=0.02, max_nodes=255 | 0.949 | 0.018 | 0.915 | 185.7 ± 53.2 | 0.29 | 16.4 (0.29) | 93.3 (0.29) | 0.29 | 93% (14/15) | 255 | ⚠ sim | 255001 | 112710 | 0/15 | 0% | 49.6 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.29 (CV de nós 0.29; nós 3–227) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.954 | 0.017 | 0.915 | 208.2 ± 11.7 | 0.06 | 17.2 (0.16) | 104.6 (0.06) | 0.16 | 100% (15/15) | 255 | ⚠ sim | 255001 | 120602 | 0/15 | 0% | 54.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 208.2 nós médios vs 26.5 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.922 | 0.031 | 0.032 | 0.012 | 0.913, 0.923, 0.939, 0.908, 0.927 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.018 | 0.015 | 0.012 | 0.934, 0.958, 0.958, 0.951, 0.934 |
| purity_epsilon=0.01, max_nodes=31 | 0.951 | 0.016 | 0.015 | 0.008 | 0.946, 0.958, 0.958, 0.953, 0.939 |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.031 | 0.033 | 0.013 | 0.913, 0.923, 0.941, 0.908, 0.927 |
| purity_epsilon=0.02, max_nodes=63 | 0.949 | 0.018 | 0.015 | 0.012 | 0.937, 0.955, 0.960, 0.958, 0.934 |
| purity_epsilon=0.01, max_nodes=63 | 0.953 | 0.016 | 0.015 | 0.009 | 0.948, 0.955, 0.960, 0.960, 0.939 |
| purity_epsilon=0.05, max_nodes=127 | 0.923 | 0.031 | 0.033 | 0.013 | 0.913, 0.923, 0.941, 0.908, 0.927 |
| purity_epsilon=0.02, max_nodes=127 | 0.949 | 0.018 | 0.016 | 0.012 | 0.939, 0.955, 0.960, 0.958, 0.934 |
| purity_epsilon=0.01, max_nodes=127 | 0.954 | 0.017 | 0.016 | 0.009 | 0.948, 0.955, 0.960, 0.962, 0.941 |
| purity_epsilon=0.05, max_nodes=255 | 0.923 | 0.031 | 0.033 | 0.013 | 0.913, 0.923, 0.941, 0.908, 0.927 |
| purity_epsilon=0.02, max_nodes=255 | 0.949 | 0.018 | 0.016 | 0.012 | 0.939, 0.955, 0.960, 0.958, 0.934 |
| purity_epsilon=0.01, max_nodes=255 | 0.954 | 0.017 | 0.017 | 0.009 | 0.948, 0.955, 0.962, 0.962, 0.941 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.32 | 0.14 | 66 | 0.37 | 0.18 | 80% | concave_points_mean 80%, perimeter_worst 80%, radius_worst 60%, area_worst 40% |
| purity_epsilon=0.02, max_nodes=31 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.59 | 0.23 | 12 | 0.68 | 0.27 | 7% | perimeter_worst 100%, concave_points_mean 93%, concave_points_worst 93%, concavity_mean 93% |
| purity_epsilon=0.01, max_nodes=31 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.68 | 0.29 | 16 | 0.71 | 0.30 | 0% | concave_points_worst 100%, concavity_worst 100%, perimeter_worst 100%, radius_se 100% |
| purity_epsilon=0.05, max_nodes=63 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.31 | 0.13 | 66 | 0.37 | 0.18 | 80% | concave_points_mean 80%, perimeter_worst 80%, radius_worst 60%, area_worst 47% |
| purity_epsilon=0.02, max_nodes=63 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.63 | 0.28 | 3 | 0.75 | 0.36 | 7% | concave_points_worst 100%, perimeter_worst 100%, concave_points_mean 93%, concavity_mean 93% |
| purity_epsilon=0.01, max_nodes=63 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.75 | 0.35 | 11 | 0.73 | 0.39 | 0% | area_worst 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.05, max_nodes=127 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.30 | 0.13 | 66 | 0.37 | 0.18 | 80% | concave_points_mean 80%, perimeter_worst 80%, radius_worst 60%, area_worst 47% |
| purity_epsilon=0.02, max_nodes=127 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.70 | 0.32 | 5 | 0.85 | 0.42 | 7% | concave_points_mean 100%, concave_points_worst 100%, perimeter_worst 100%, compactness_se 93% |
| purity_epsilon=0.01, max_nodes=127 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.82 | 0.39 | 6 | 0.81 | 0.42 | 0% | area_se 100%, area_worst 100%, concave_points_worst 100%, concavity_mean 100% |
| purity_epsilon=0.05, max_nodes=255 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.30 | 0.13 | 66 | 0.37 | 0.18 | 80% | concave_points_mean 80%, perimeter_worst 80%, radius_worst 60%, area_worst 47% |
| purity_epsilon=0.02, max_nodes=255 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.76 | 0.36 | 4 | 0.90 | 0.51 | 7% | concave_points_mean 100%, concave_points_worst 100%, perimeter_worst 100%, area_se 93% |
| purity_epsilon=0.01, max_nodes=255 | concave_points_mean 73%, perimeter_worst 73%, radius_worst 53% | 73% | 0.87 | 0.44 | 8 | 0.86 | 0.46 | 0% | area_mean 100%, area_se 100%, area_worst 100%, concave_points_mean 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.958 | 0.894 | 0.901 | 0.930 | 0.908 | 0.965 | 0.908 |
| purity_epsilon=0.02, max_nodes=31 | 0.944 | 0.944 | 0.915 | 0.972 | 0.951 | 0.951 | 0.972 | 0.944 | 0.958 | 0.958 | 0.951 | 0.944 | 0.923 | 0.965 | 0.915 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.951 | 0.944 | 0.965 | 0.951 | 0.958 | 0.972 | 0.944 | 0.958 | 0.979 | 0.944 | 0.937 | 0.937 | 0.965 | 0.915 |
| purity_epsilon=0.05, max_nodes=63 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.965 | 0.894 | 0.901 | 0.930 | 0.908 | 0.965 | 0.908 |
| purity_epsilon=0.02, max_nodes=63 | 0.944 | 0.944 | 0.923 | 0.972 | 0.944 | 0.951 | 0.972 | 0.944 | 0.965 | 0.958 | 0.951 | 0.965 | 0.923 | 0.965 | 0.915 |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.958 | 0.944 | 0.965 | 0.944 | 0.958 | 0.972 | 0.944 | 0.965 | 0.979 | 0.944 | 0.958 | 0.937 | 0.965 | 0.915 |
| purity_epsilon=0.05, max_nodes=127 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.965 | 0.894 | 0.901 | 0.930 | 0.908 | 0.965 | 0.908 |
| purity_epsilon=0.02, max_nodes=127 | 0.944 | 0.951 | 0.923 | 0.972 | 0.944 | 0.951 | 0.972 | 0.944 | 0.965 | 0.965 | 0.951 | 0.958 | 0.923 | 0.965 | 0.915 |
| purity_epsilon=0.01, max_nodes=127 | 0.944 | 0.958 | 0.944 | 0.965 | 0.944 | 0.958 | 0.972 | 0.944 | 0.965 | 0.986 | 0.944 | 0.958 | 0.944 | 0.965 | 0.915 |
| purity_epsilon=0.05, max_nodes=255 | 0.930 | 0.937 | 0.873 | 0.951 | 0.866 | 0.951 | 0.951 | 0.908 | 0.965 | 0.894 | 0.901 | 0.930 | 0.908 | 0.965 | 0.908 |
| purity_epsilon=0.02, max_nodes=255 | 0.944 | 0.951 | 0.923 | 0.972 | 0.944 | 0.951 | 0.972 | 0.944 | 0.965 | 0.965 | 0.951 | 0.958 | 0.923 | 0.965 | 0.915 |
| purity_epsilon=0.01, max_nodes=255 | 0.944 | 0.958 | 0.944 | 0.965 | 0.944 | 0.958 | 0.979 | 0.944 | 0.965 | 0.986 | 0.944 | 0.958 | 0.944 | 0.965 | 0.915 |

Nós por partição:

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 29 | 3 | 3 | 25 | 3 | 31 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 27 | 27 | 21 | 31 | 3 | 23 | 31 | 29 | 23 | 29 | 27 | 25 | 31 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 25 | 21 | 29 | 31 | 27 | 31 | 29 | 17 | 29 | 31 | 25 | 31 | 21 |
| purity_epsilon=0.05, max_nodes=63 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 63 | 3 | 3 | 59 | 3 | 55 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 45 | 57 | 51 | 49 | 53 | 3 | 39 | 57 | 63 | 41 | 59 | 53 | 43 | 55 | 43 |
| purity_epsilon=0.01, max_nodes=63 | 45 | 59 | 57 | 43 | 57 | 53 | 43 | 57 | 63 | 57 | 55 | 55 | 61 | 55 | 53 |
| purity_epsilon=0.05, max_nodes=127 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 115 | 3 | 3 | 117 | 3 | 99 | 3 |
| purity_epsilon=0.02, max_nodes=127 | 89 | 101 | 107 | 103 | 101 | 3 | 113 | 103 | 115 | 83 | 103 | 97 | 115 | 99 | 105 |
| purity_epsilon=0.01, max_nodes=127 | 89 | 99 | 105 | 121 | 113 | 99 | 101 | 103 | 115 | 103 | 89 | 115 | 109 | 99 | 91 |
| purity_epsilon=0.05, max_nodes=255 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 213 | 3 | 3 | 191 | 3 | 187 | 3 |
| purity_epsilon=0.02, max_nodes=255 | 219 | 211 | 193 | 179 | 195 | 3 | 179 | 209 | 213 | 175 | 209 | 209 | 227 | 187 | 177 |
| purity_epsilon=0.01, max_nodes=255 | 209 | 195 | 211 | 211 | 215 | 233 | 209 | 209 | 213 | 195 | 209 | 205 | 197 | 187 | 225 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.01, max_nodes=31` — estado interno do tuning: **tuning_uncertain** (selection_probability_below_threshold).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **58.9%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.01, max_nodes=31` com probabilidade 58.9% — selecionada é a moda: **True**
- bootstrap_runner_up: `purity_epsilon=0.02, max_nodes=31` 29.7%; top1_top2_margin: 29.2%
- IC bootstrap 95% da fidelity média da escolhida: [0.944, 0.957]
- distribuição das configurações escolhidas: `purity_epsilon=0.01, max_nodes=31` 58.9%, `purity_epsilon=0.02, max_nodes=31` 29.7%, `purity_epsilon=0.01, max_nodes=63` 10.1%, `purity_epsilon=0.02, max_nodes=63` 1.1%, `purity_epsilon=0.01, max_nodes=255` 0.2%, `purity_epsilon=0.01, max_nodes=127` 0.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.01, max_nodes=255', 'purity_epsilon=0.01, max_nodes=63', 'purity_epsilon=0.01, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31']
- saturação da escolhida: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 27, folhas 14, accuracy 0.9230769230769231, fidelity 0.9440559440559441; MLP accuracy 0.9790209790209791.

#### bc_user.arff — seed mestre 123 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [123, 1132, 2141, 3150, 4159]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=d156e8f076a2dfb2` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'd156e8f076a2dfb2', 'trepan_reloaded': 'd156e8f076a2dfb2'}); queries por escopo: {'tuning': 16237128, 'trepan_pair': 188320, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 255001 (mesmo para todos: True); consumo máximo observado: 124297; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.925 | 0.015 | 0.901 | 15.4 ± 12.6 | 0.82 | 4.8 (0.77) | 8.2 (0.77) | 0.82 | 53% (8/15) | 31 | ⚠ sim | 255001 | 9521 | 0/15 | 0% | 2.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.82 (CV de nós 0.82; nós 3–31) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.015 | 0.915 | 26.3 ± 3.9 | 0.15 | 7.7 (0.16) | 13.7 (0.14) | 0.16 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14944 | 0/15 | 0% | 3.9 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.16 (CV de nós 0.15; nós 15–31) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.939 | 0.017 | 0.908 | 25.9 ± 4.3 | 0.17 | 6.9 (0.18) | 13.5 (0.16) | 0.18 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14134 | 0/15 | 0% | 3.8 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.18 (CV de nós 0.17; nós 15–31) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.925 | 0.016 | 0.901 | 31.1 ± 27.4 | 0.88 | 6.5 (0.83) | 16.1 (0.85) | 0.88 | 53% (8/15) | 63 | ⚠ sim | 255001 | 17318 | 0/15 | 0% | 5.2 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.88 (CV de nós 0.88; nós 3–61) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.936 | 0.017 | 0.915 | 54.2 ± 4.6 | 0.09 | 10.7 (0.16) | 27.6 (0.08) | 0.16 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29742 | 0/15 | 0% | 9.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.16 (CV de nós 0.09; nós 47–61) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.938 | 0.018 | 0.908 | 54.5 ± 5.0 | 0.09 | 9.9 (0.15) | 27.7 (0.09) | 0.15 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29091 | 0/15 | 0% | 9.6 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.15 (CV de nós 0.09; nós 45–61) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.927 | 0.016 | 0.901 | 54.2 ± 49.9 | 0.92 | 7.9 (0.87) | 27.6 (0.90) | 0.92 | 53% (8/15) | 127 | ⚠ sim | 255001 | 33256 | 0/15 | 0% | 11.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.92 (CV de nós 0.92; nós 3–109) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.938 | 0.016 | 0.915 | 102.6 ± 9.1 | 0.09 | 13.5 (0.11) | 51.8 (0.09) | 0.11 | 100% (15/15) | 127 | ⚠ sim | 255001 | 58822 | 0/15 | 0% | 21.7 | **WINNER** — VENCE: fidelity média 0.938; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.001, t=0.20 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 12 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.11) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.938 | 0.019 | 0.908 | 104.9 ± 9.0 | 0.09 | 13.3 (0.14) | 52.9 (0.09) | 0.14 | 100% (15/15) | 127 | ⚠ sim | 255001 | 58199 | 0/15 | 0% | 23.5 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 104.9 nós médios vs 102.6 de purity_epsilon=0.02, max_nodes=127. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.927 | 0.017 | 0.901 | 105.1 ± 99.9 | 0.95 | 9.4 (0.88) | 53.1 (0.94) | 0.95 | 53% (8/15) | 255 | ⚠ sim | 255001 | 66014 | 0/15 | 0% | 27.1 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.95 (CV de nós 0.95; nós 3–215) > 0.10 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.938 | 0.018 | 0.901 | 199.5 ± 17.2 | 0.09 | 16.1 (0.11) | 100.3 (0.09) | 0.11 | 100% (15/15) | 255 | ⚠ sim | 255001 | 119513 | 0/15 | 0% | 50.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 199.5 nós médios vs 102.6 de purity_epsilon=0.02, max_nodes=127. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.938 | 0.019 | 0.901 | 199.5 ± 15.4 | 0.08 | 15.7 (0.10) | 100.3 (0.08) | 0.10 | 100% (15/15) | 255 | ⚠ sim | 255001 | 118248 | 0/15 | 0% | 50.6 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 199.5 nós médios vs 102.6 de purity_epsilon=0.02, max_nodes=127. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.925 | 0.015 | 0.013 | 0.008 | 0.913, 0.923, 0.934, 0.927, 0.927 |
| purity_epsilon=0.02, max_nodes=31 | 0.940 | 0.015 | 0.011 | 0.010 | 0.930, 0.946, 0.953, 0.932, 0.939 |
| purity_epsilon=0.01, max_nodes=31 | 0.939 | 0.017 | 0.012 | 0.013 | 0.927, 0.941, 0.960, 0.930, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.925 | 0.016 | 0.015 | 0.009 | 0.913, 0.923, 0.937, 0.927, 0.927 |
| purity_epsilon=0.02, max_nodes=63 | 0.936 | 0.017 | 0.013 | 0.013 | 0.920, 0.946, 0.951, 0.934, 0.927 |
| purity_epsilon=0.01, max_nodes=63 | 0.938 | 0.018 | 0.013 | 0.015 | 0.920, 0.944, 0.960, 0.937, 0.927 |
| purity_epsilon=0.05, max_nodes=127 | 0.927 | 0.016 | 0.015 | 0.007 | 0.918, 0.923, 0.937, 0.930, 0.927 |
| purity_epsilon=0.02, max_nodes=127 | 0.938 | 0.016 | 0.014 | 0.011 | 0.927, 0.946, 0.951, 0.941, 0.927 |
| purity_epsilon=0.01, max_nodes=127 | 0.938 | 0.019 | 0.013 | 0.016 | 0.915, 0.944, 0.958, 0.941, 0.930 |
| purity_epsilon=0.05, max_nodes=255 | 0.927 | 0.017 | 0.016 | 0.008 | 0.915, 0.923, 0.937, 0.930, 0.930 |
| purity_epsilon=0.02, max_nodes=255 | 0.938 | 0.018 | 0.017 | 0.011 | 0.925, 0.946, 0.951, 0.941, 0.930 |
| purity_epsilon=0.01, max_nodes=255 | 0.938 | 0.019 | 0.014 | 0.015 | 0.915, 0.944, 0.955, 0.944, 0.930 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.37 | 0.11 | 27 | 0.43 | 0.12 | 47% | concavity_worst 87%, radius_worst 87%, concave_points_worst 80%, symmetry_worst 73% |
| purity_epsilon=0.02, max_nodes=31 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.67 | 0.24 | 18 | 0.67 | 0.25 | 0% | concavity_worst 100%, radius_worst 100%, symmetry_worst 100%, texture_mean 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.69 | 0.24 | 11 | 0.68 | 0.27 | 0% | concavity_worst 100%, radius_worst 100%, symmetry_worst 100%, texture_mean 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.36 | 0.13 | 23 | 0.38 | 0.11 | 47% | concavity_worst 87%, radius_worst 87%, concave_points_worst 80%, symmetry_worst 73% |
| purity_epsilon=0.02, max_nodes=63 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.76 | 0.35 | 8 | 0.76 | 0.34 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_worst 100%, perimeter_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.79 | 0.35 | 11 | 0.79 | 0.37 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |
| purity_epsilon=0.05, max_nodes=127 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.36 | 0.14 | 22 | 0.36 | 0.11 | 47% | concavity_worst 87%, radius_worst 87%, concave_points_worst 80%, symmetry_worst 73% |
| purity_epsilon=0.02, max_nodes=127 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.82 | 0.42 | 11 | 0.84 | 0.43 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |
| purity_epsilon=0.01, max_nodes=127 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.81 | 0.43 | 10 | 0.78 | 0.42 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |
| purity_epsilon=0.05, max_nodes=255 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.36 | 0.15 | 21 | 0.35 | 0.10 | 47% | concavity_worst 87%, radius_worst 87%, concave_points_worst 80%, symmetry_worst 73% |
| purity_epsilon=0.02, max_nodes=255 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.89 | 0.48 | 3 | 0.87 | 0.45 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, concave_points_worst 100% |
| purity_epsilon=0.01, max_nodes=255 | radius_worst 80%, concavity_worst 67%, concave_points_worst 40% | 80% | 0.87 | 0.48 | 6 | 0.85 | 0.49 | 0% | area_se 100%, area_worst 100%, compactness_mean 100%, compactness_se 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.915 | 0.908 | 0.915 | 0.937 | 0.901 | 0.930 | 0.937 | 0.937 | 0.930 | 0.901 | 0.930 | 0.951 | 0.944 | 0.923 | 0.915 |
| purity_epsilon=0.02, max_nodes=31 | 0.930 | 0.930 | 0.930 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.979 | 0.915 | 0.930 | 0.951 | 0.944 | 0.930 | 0.944 |
| purity_epsilon=0.01, max_nodes=31 | 0.930 | 0.930 | 0.923 | 0.937 | 0.951 | 0.937 | 0.958 | 0.944 | 0.979 | 0.908 | 0.930 | 0.951 | 0.944 | 0.930 | 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.908 | 0.908 | 0.923 | 0.937 | 0.901 | 0.930 | 0.937 | 0.944 | 0.930 | 0.901 | 0.930 | 0.951 | 0.944 | 0.923 | 0.915 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.915 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.972 | 0.915 | 0.937 | 0.951 | 0.944 | 0.915 | 0.923 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.937 | 0.908 | 0.937 | 0.951 | 0.944 | 0.951 | 0.958 | 0.972 | 0.915 | 0.944 | 0.951 | 0.944 | 0.915 | 0.923 |
| purity_epsilon=0.05, max_nodes=127 | 0.923 | 0.908 | 0.923 | 0.937 | 0.901 | 0.930 | 0.937 | 0.944 | 0.930 | 0.901 | 0.930 | 0.958 | 0.944 | 0.923 | 0.915 |
| purity_epsilon=0.02, max_nodes=127 | 0.915 | 0.930 | 0.937 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.972 | 0.923 | 0.944 | 0.958 | 0.944 | 0.915 | 0.923 |
| purity_epsilon=0.01, max_nodes=127 | 0.915 | 0.923 | 0.908 | 0.937 | 0.951 | 0.944 | 0.951 | 0.951 | 0.972 | 0.923 | 0.944 | 0.958 | 0.951 | 0.915 | 0.923 |
| purity_epsilon=0.05, max_nodes=255 | 0.915 | 0.908 | 0.923 | 0.937 | 0.901 | 0.930 | 0.937 | 0.944 | 0.930 | 0.901 | 0.930 | 0.958 | 0.951 | 0.923 | 0.915 |
| purity_epsilon=0.02, max_nodes=255 | 0.901 | 0.930 | 0.944 | 0.951 | 0.951 | 0.937 | 0.937 | 0.944 | 0.972 | 0.923 | 0.944 | 0.958 | 0.951 | 0.915 | 0.923 |
| purity_epsilon=0.01, max_nodes=255 | 0.901 | 0.923 | 0.923 | 0.937 | 0.951 | 0.944 | 0.944 | 0.951 | 0.972 | 0.923 | 0.951 | 0.958 | 0.951 | 0.915 | 0.923 |

Nós por partição:

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 21 | 3 | 15 | 29 | 3 | 27 | 31 | 29 | 3 | 3 | 3 | 29 | 29 | 3 | 3 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 29 | 23 | 29 | 27 | 25 | 29 | 27 | 15 | 29 | 25 | 23 | 29 | 31 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 31 | 25 | 31 | 29 | 23 | 25 | 29 | 15 | 21 | 25 | 23 | 27 | 31 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 3 | 55 | 55 | 3 | 53 | 59 | 57 | 3 | 3 | 3 | 49 | 57 | 3 | 3 |
| purity_epsilon=0.02, max_nodes=63 | 47 | 51 | 49 | 55 | 51 | 61 | 59 | 59 | 61 | 47 | 55 | 53 | 57 | 53 | 55 |
| purity_epsilon=0.01, max_nodes=63 | 47 | 45 | 51 | 59 | 55 | 55 | 57 | 57 | 61 | 47 | 61 | 53 | 57 | 55 | 57 |
| purity_epsilon=0.05, max_nodes=127 | 103 | 3 | 89 | 109 | 3 | 85 | 107 | 101 | 3 | 3 | 3 | 101 | 97 | 3 | 3 |
| purity_epsilon=0.02, max_nodes=127 | 121 | 89 | 103 | 111 | 111 | 97 | 95 | 105 | 91 | 111 | 95 | 111 | 97 | 97 | 105 |
| purity_epsilon=0.01, max_nodes=127 | 121 | 103 | 115 | 101 | 113 | 113 | 113 | 101 | 91 | 97 | 107 | 89 | 101 | 107 | 101 |
| purity_epsilon=0.05, max_nodes=255 | 185 | 3 | 163 | 213 | 3 | 195 | 207 | 215 | 3 | 3 | 3 | 169 | 209 | 3 | 3 |
| purity_epsilon=0.02, max_nodes=255 | 217 | 205 | 223 | 229 | 203 | 175 | 195 | 175 | 189 | 193 | 203 | 171 | 209 | 209 | 197 |
| purity_epsilon=0.01, max_nodes=255 | 217 | 195 | 227 | 219 | 219 | 191 | 189 | 201 | 189 | 189 | 211 | 191 | 195 | 187 | 173 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.02, max_nodes=127` — estado interno do tuning: **tuning_uncertain** (full_cv_selection_not_bootstrap_modal).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **16.8%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.02, max_nodes=31` com probabilidade 45.7% — selecionada é a moda: **False** ⚠ **fragilidade FORTE da seleção full-CV**
- bootstrap_runner_up: `purity_epsilon=0.02, max_nodes=127` 16.8%; top1_top2_margin: 28.9%
- IC bootstrap 95% da fidelity média da escolhida: [0.930, 0.947]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 45.7%, `purity_epsilon=0.02, max_nodes=127` 16.8%, `purity_epsilon=0.01, max_nodes=63` 14.1%, `purity_epsilon=0.02, max_nodes=63` 8.5%, `purity_epsilon=0.01, max_nodes=127` 6.3%, `purity_epsilon=0.01, max_nodes=31` 5.5%, `purity_epsilon=0.05, max_nodes=31` 2.4%, `purity_epsilon=0.05, max_nodes=63` 0.8%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=31']
- saturação da escolhida: 15/15 árvores no teto max_nodes=127 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 109, folhas 55, accuracy 0.9020979020979021, fidelity 0.9090909090909091; MLP accuracy 0.965034965034965.

#### bc_user.arff — seed mestre 2024 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [2024, 3033, 4042, 5051, 6060]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=a2f113e60f488f42` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'a2f113e60f488f42', 'trepan_reloaded': 'a2f113e60f488f42'}); queries por escopo: {'tuning': 13453124, 'trepan_pair': 92092, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 255001 (mesmo para todos: True); consumo máximo observado: 126184; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.931 | 0.016 | 0.915 | 22.7 ± 8.5 | 0.37 | 7.3 (0.38) | 11.9 (0.36) | 0.38 | 87% (13/15) | 31 | ⚠ sim | 255001 | 13874 | 0/15 | 0% | 3.6 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.38 (CV de nós 0.37; nós 3–31) > 0.08 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.933 | 0.015 | 0.908 | 23.5 ± 4.4 | 0.19 | 6.6 (0.25) | 12.3 (0.18) | 0.25 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14770 | 0/15 | 0% | 4.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.25 (CV de nós 0.19; nós 17–31) > 0.08 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.929 | 0.021 | 0.887 | 23.7 ± 6.0 | 0.25 | 6.3 (0.19) | 12.3 (0.24) | 0.25 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14326 | 0/15 | 0% | 4.0 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.004, t=0.49 ≤ 1.76), mas desvio-padrão 0.021 > 0.015 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.929 | 0.018 | 0.908 | 43.4 ± 17.5 | 0.40 | 9.5 (0.38) | 22.2 (0.39) | 0.40 | 87% (13/15) | 63 | ⚠ sim | 255001 | 26669 | 0/15 | 0% | 8.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.40 (CV de nós 0.40; nós 3–57) > 0.08 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.017 | 0.908 | 54.3 ± 3.8 | 0.07 | 9.4 (0.10) | 27.7 (0.07) | 0.10 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29776 | 0/15 | 0% | 9.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 54.3 nós médios vs 53.8 de purity_epsilon=0.01, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.928 | 0.020 | 0.894 | 53.8 ± 5.9 | 0.11 | 9.5 (0.12) | 27.4 (0.11) | 0.12 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29220 | 0/15 | 0% | 9.5 | **WINNER** — VENCE: fidelity média 0.928; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.005, t=0.52 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 11 após estabilidade da fidelity -> 6 após estabilidade estrutural (índice 0.12) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.930 | 0.019 | 0.908 | 87.5 ± 35.5 | 0.41 | 12.2 (0.39) | 44.3 (0.40) | 0.41 | 87% (13/15) | 127 | ⚠ sim | 255001 | 51997 | 0/15 | 0% | 18.2 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.41 (CV de nós 0.41; nós 3–119) > 0.08 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.929 | 0.015 | 0.908 | 102.5 ± 9.5 | 0.09 | 12.7 (0.12) | 51.7 (0.09) | 0.12 | 100% (15/15) | 127 | ⚠ sim | 255001 | 57991 | 0/15 | 0% | 21.9 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 102.5 nós médios vs 53.8 de purity_epsilon=0.01, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.927 | 0.018 | 0.894 | 100.1 ± 9.8 | 0.10 | 12.5 (0.13) | 50.5 (0.10) | 0.13 | 100% (15/15) | 127 | ⚠ sim | 255001 | 57319 | 0/15 | 0% | 23.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 100.1 nós médios vs 53.8 de purity_epsilon=0.01, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.930 | 0.018 | 0.908 | 173.9 ± 70.2 | 0.40 | 13.9 (0.39) | 87.5 (0.40) | 0.40 | 87% (13/15) | 255 | ⚠ sim | 255001 | 105090 | 0/15 | 0% | 40.3 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.40 (CV de nós 0.40; nós 3–217) > 0.08 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.929 | 0.015 | 0.908 | 196.6 ± 15.7 | 0.08 | 15.6 (0.07) | 98.8 (0.08) | 0.08 | 100% (15/15) | 255 | ⚠ sim | 255001 | 117765 | 0/15 | 0% | 47.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 196.6 nós médios vs 53.8 de purity_epsilon=0.01, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.928 | 0.019 | 0.894 | 198.2 ± 13.3 | 0.07 | 15.3 (0.08) | 99.6 (0.07) | 0.08 | 100% (15/15) | 255 | ⚠ sim | 255001 | 117077 | 0/15 | 0% | 49.8 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 198.2 nós médios vs 53.8 de purity_epsilon=0.01, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.931 | 0.016 | 0.013 | 0.012 | 0.925, 0.953, 0.925, 0.927, 0.925 |
| purity_epsilon=0.02, max_nodes=31 | 0.933 | 0.015 | 0.013 | 0.007 | 0.937, 0.930, 0.930, 0.944, 0.925 |
| purity_epsilon=0.01, max_nodes=31 | 0.929 | 0.021 | 0.019 | 0.010 | 0.941, 0.918, 0.923, 0.937, 0.925 |
| purity_epsilon=0.05, max_nodes=63 | 0.929 | 0.018 | 0.014 | 0.014 | 0.923, 0.953, 0.923, 0.918, 0.927 |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.017 | 0.019 | 0.003 | 0.927, 0.930, 0.927, 0.934, 0.932 |
| purity_epsilon=0.01, max_nodes=63 | 0.928 | 0.020 | 0.021 | 0.004 | 0.927, 0.925, 0.923, 0.932, 0.932 |
| purity_epsilon=0.05, max_nodes=127 | 0.930 | 0.019 | 0.015 | 0.013 | 0.927, 0.953, 0.925, 0.920, 0.925 |
| purity_epsilon=0.02, max_nodes=127 | 0.929 | 0.015 | 0.017 | 0.001 | 0.930, 0.930, 0.927, 0.927, 0.930 |
| purity_epsilon=0.01, max_nodes=127 | 0.927 | 0.018 | 0.020 | 0.002 | 0.930, 0.925, 0.925, 0.927, 0.930 |
| purity_epsilon=0.05, max_nodes=255 | 0.930 | 0.018 | 0.014 | 0.013 | 0.925, 0.953, 0.925, 0.920, 0.925 |
| purity_epsilon=0.02, max_nodes=255 | 0.929 | 0.015 | 0.017 | 0.001 | 0.927, 0.930, 0.930, 0.927, 0.930 |
| purity_epsilon=0.01, max_nodes=255 | 0.928 | 0.019 | 0.021 | 0.001 | 0.927, 0.927, 0.927, 0.927, 0.930 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.57 | 0.18 | 12 | 0.73 | 0.20 | 13% | concavity_mean 100%, radius_se 100%, radius_worst 100%, compactness_mean 87% |
| purity_epsilon=0.02, max_nodes=31 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.68 | 0.24 | 12 | 0.66 | 0.22 | 0% | compactness_mean 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.01, max_nodes=31 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.65 | 0.23 | 7 | 0.66 | 0.24 | 0% | compactness_mean 100%, concavity_mean 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.05, max_nodes=63 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.59 | 0.23 | 8 | 0.75 | 0.27 | 13% | concavity_mean 100%, radius_se 100%, radius_worst 100%, compactness_mean 87% |
| purity_epsilon=0.02, max_nodes=63 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.76 | 0.36 | 18 | 0.76 | 0.38 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.74 | 0.35 | 29 | 0.75 | 0.37 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_mean 100%, concavity_worst 100% |
| purity_epsilon=0.05, max_nodes=127 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.64 | 0.28 | 3 | 0.81 | 0.26 | 13% | concavity_mean 100%, radius_se 100%, radius_worst 100%, area_worst 87% |
| purity_epsilon=0.02, max_nodes=127 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.79 | 0.43 | 3 | 0.73 | 0.37 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |
| purity_epsilon=0.01, max_nodes=127 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.79 | 0.43 | 4 | 0.76 | 0.42 | 0% | area_worst 100%, compactness_mean 100%, concave_points_worst 100%, concavity_mean 100% |
| purity_epsilon=0.05, max_nodes=255 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.72 | 0.33 | 3 | 0.94 | 0.28 | 13% | concavity_mean 100%, radius_se 100%, radius_worst 100%, area_se 87% |
| purity_epsilon=0.02, max_nodes=255 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.90 | 0.50 | 3 | 0.86 | 0.53 | 0% | area_worst 100%, compactness_mean 100%, compactness_worst 100%, concave_points_worst 100% |
| purity_epsilon=0.01, max_nodes=255 | radius_se 93%, radius_worst 93%, concavity_mean 60% | 93% | 0.89 | 0.50 | 4 | 0.88 | 0.49 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, concave_points_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.923 | 0.915 | 0.937 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.915 | 0.923 | 0.923 | 0.937 | 0.923 | 0.915 | 0.937 |
| purity_epsilon=0.02, max_nodes=31 | 0.937 | 0.937 | 0.937 | 0.923 | 0.958 | 0.908 | 0.951 | 0.923 | 0.915 | 0.951 | 0.944 | 0.937 | 0.915 | 0.915 | 0.944 |
| purity_epsilon=0.01, max_nodes=31 | 0.937 | 0.937 | 0.951 | 0.887 | 0.958 | 0.908 | 0.951 | 0.923 | 0.894 | 0.937 | 0.944 | 0.930 | 0.915 | 0.915 | 0.944 |
| purity_epsilon=0.05, max_nodes=63 | 0.923 | 0.908 | 0.937 | 0.930 | 0.965 | 0.965 | 0.944 | 0.915 | 0.908 | 0.923 | 0.923 | 0.908 | 0.923 | 0.923 | 0.937 |
| purity_epsilon=0.02, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.923 | 0.958 | 0.908 | 0.944 | 0.930 | 0.908 | 0.958 | 0.937 | 0.908 | 0.937 | 0.915 | 0.944 |
| purity_epsilon=0.01, max_nodes=63 | 0.915 | 0.930 | 0.937 | 0.901 | 0.965 | 0.908 | 0.944 | 0.930 | 0.894 | 0.951 | 0.937 | 0.908 | 0.937 | 0.915 | 0.944 |
| purity_epsilon=0.05, max_nodes=127 | 0.923 | 0.908 | 0.951 | 0.930 | 0.965 | 0.965 | 0.951 | 0.915 | 0.908 | 0.923 | 0.923 | 0.915 | 0.923 | 0.923 | 0.930 |
| purity_epsilon=0.02, max_nodes=127 | 0.908 | 0.930 | 0.951 | 0.923 | 0.958 | 0.908 | 0.944 | 0.930 | 0.908 | 0.930 | 0.937 | 0.915 | 0.937 | 0.915 | 0.937 |
| purity_epsilon=0.01, max_nodes=127 | 0.908 | 0.930 | 0.951 | 0.908 | 0.958 | 0.908 | 0.951 | 0.930 | 0.894 | 0.930 | 0.937 | 0.915 | 0.937 | 0.915 | 0.937 |
| purity_epsilon=0.05, max_nodes=255 | 0.923 | 0.908 | 0.944 | 0.930 | 0.965 | 0.965 | 0.951 | 0.915 | 0.908 | 0.923 | 0.923 | 0.915 | 0.923 | 0.923 | 0.930 |
| purity_epsilon=0.02, max_nodes=255 | 0.908 | 0.930 | 0.944 | 0.923 | 0.958 | 0.908 | 0.944 | 0.937 | 0.908 | 0.930 | 0.937 | 0.915 | 0.937 | 0.915 | 0.937 |
| purity_epsilon=0.01, max_nodes=255 | 0.908 | 0.930 | 0.944 | 0.908 | 0.965 | 0.908 | 0.951 | 0.937 | 0.894 | 0.930 | 0.937 | 0.915 | 0.937 | 0.915 | 0.937 |

Nós por partição:

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 25 | 29 | 29 | 31 | 25 | 27 | 23 | 3 | 25 | 21 | 27 | 25 | 3 | 27 | 21 |
| purity_epsilon=0.02, max_nodes=31 | 21 | 29 | 29 | 21 | 31 | 23 | 17 | 25 | 25 | 19 | 29 | 25 | 21 | 19 | 19 |
| purity_epsilon=0.01, max_nodes=31 | 21 | 29 | 13 | 29 | 31 | 23 | 17 | 25 | 31 | 27 | 29 | 27 | 21 | 15 | 17 |
| purity_epsilon=0.05, max_nodes=63 | 33 | 51 | 55 | 41 | 53 | 55 | 49 | 3 | 51 | 55 | 47 | 57 | 3 | 51 | 47 |
| purity_epsilon=0.02, max_nodes=63 | 55 | 55 | 55 | 47 | 53 | 55 | 55 | 55 | 51 | 53 | 59 | 57 | 63 | 49 | 53 |
| purity_epsilon=0.01, max_nodes=63 | 55 | 55 | 55 | 55 | 53 | 55 | 55 | 55 | 57 | 53 | 59 | 55 | 63 | 45 | 37 |
| purity_epsilon=0.05, max_nodes=127 | 103 | 89 | 119 | 97 | 93 | 103 | 107 | 3 | 101 | 83 | 91 | 101 | 3 | 109 | 111 |
| purity_epsilon=0.02, max_nodes=127 | 97 | 85 | 119 | 105 | 89 | 103 | 107 | 95 | 101 | 111 | 95 | 101 | 117 | 103 | 109 |
| purity_epsilon=0.01, max_nodes=127 | 97 | 85 | 109 | 117 | 93 | 103 | 109 | 95 | 91 | 103 | 95 | 101 | 117 | 87 | 99 |
| purity_epsilon=0.05, max_nodes=255 | 189 | 201 | 207 | 203 | 185 | 217 | 211 | 3 | 185 | 187 | 213 | 195 | 3 | 211 | 199 |
| purity_epsilon=0.02, max_nodes=255 | 211 | 189 | 207 | 181 | 169 | 211 | 193 | 177 | 175 | 213 | 201 | 195 | 219 | 197 | 211 |
| purity_epsilon=0.01, max_nodes=255 | 211 | 189 | 193 | 221 | 183 | 211 | 187 | 177 | 203 | 203 | 201 | 199 | 219 | 187 | 189 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.01, max_nodes=63` — estado interno do tuning: **tuning_uncertain** (full_cv_selection_not_bootstrap_modal).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **10.4%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.02, max_nodes=63` com probabilidade 66.9% — selecionada é a moda: **False** ⚠ **fragilidade FORTE da seleção full-CV**
- bootstrap_runner_up: `purity_epsilon=0.01, max_nodes=63` 10.4%; top1_top2_margin: 56.5%
- IC bootstrap 95% da fidelity média da escolhida: [0.924, 0.931]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=63` 66.9%, `purity_epsilon=0.01, max_nodes=63` 10.4%, `purity_epsilon=0.01, max_nodes=127` 5.9%, `purity_epsilon=0.05, max_nodes=31` 5.6%, `purity_epsilon=0.02, max_nodes=127` 5.1%, `purity_epsilon=0.01, max_nodes=255` 3.8%, `purity_epsilon=0.02, max_nodes=255` 1.9%, `purity_epsilon=0.01, max_nodes=31` 0.2%, `purity_epsilon=0.02, max_nodes=31` 0.1%, `purity_epsilon=0.05, max_nodes=63` 0.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=255', 'purity_epsilon=0.01, max_nodes=127', 'purity_epsilon=0.05, max_nodes=63']
- saturação da escolhida: 15/15 árvores no teto max_nodes=63 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 55, folhas 28, accuracy 0.9020979020979021, fidelity 0.916083916083916; MLP accuracy 0.972027972027972.

#### bc_user.arff — seed mestre 11 (treino 426, teste 143)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [11, 1020, 2029, 3038, 4047]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=9aa83790a361ec28` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '9aa83790a361ec28', 'trepan_reloaded': '9aa83790a361ec28'}); queries por escopo: {'tuning': 11768038, 'trepan_pair': 88825, '(sem escopo)': 429}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 255001 (mesmo para todos: True); consumo máximo observado: 126478; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.942 | 0.024 | 0.873 | 23.4 ± 10.9 | 0.47 | 7.1 (0.46) | 12.2 (0.45) | 0.47 | 80% (12/15) | 31 | ⚠ sim | 255001 | 12817 | 0/15 | 0% | 3.8 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.47 (CV de nós 0.47; nós 3–31) > 0.21 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.946 | 0.020 | 0.901 | 24.6 ± 5.1 | 0.21 | 6.9 (0.19) | 12.8 (0.20) | 0.21 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14445 | 0/15 | 0% | 4.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 24.6 nós médios vs 23.1 de purity_epsilon=0.01, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.019 | 0.901 | 23.1 ± 5.1 | 0.22 | 6.5 (0.20) | 12.1 (0.21) | 0.22 | 100% (15/15) | 31 | ⚠ sim | 255001 | 14185 | 0/15 | 0% | 3.7 | **WINNER** — VENCE: fidelity média 0.944; indistinguível da melhor (purity_epsilon=0.02, max_nodes=31, Δ=0.002, t=0.44 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 5 após estabilidade da fidelity -> 2 após estabilidade estrutural (índice 0.22) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.944 | 0.025 | 0.873 | 44.7 ± 22.0 | 0.49 | 9.2 (0.49) | 22.9 (0.48) | 0.49 | 80% (12/15) | 63 | ⚠ sim | 255001 | 24445 | 0/15 | 0% | 7.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.17 ≤ 1.76), mas desvio-padrão 0.025 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.945 | 0.025 | 0.887 | 53.9 ± 7.4 | 0.14 | 9.5 (0.14) | 27.5 (0.13) | 0.14 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29193 | 0/15 | 0% | 8.9 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.001, t=0.12 ≤ 1.76), mas desvio-padrão 0.025 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.024 | 0.887 | 53.9 ± 6.7 | 0.12 | 9.5 (0.15) | 27.5 (0.12) | 0.15 | 100% (15/15) | 63 | ⚠ sim | 255001 | 29008 | 0/15 | 0% | 8.8 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.26 ≤ 1.76), mas desvio-padrão 0.024 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.942 | 0.023 | 0.873 | 80.6 ± 40.8 | 0.51 | 11.0 (0.49) | 40.8 (0.50) | 0.51 | 80% (12/15) | 127 | ⚠ sim | 255001 | 48382 | 0/15 | 0% | 15.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.51 (CV de nós 0.51; nós 3–111) > 0.21 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.945 | 0.025 | 0.887 | 103.1 ± 10.0 | 0.10 | 12.9 (0.15) | 52.1 (0.10) | 0.15 | 100% (15/15) | 127 | ⚠ sim | 255001 | 57655 | 0/15 | 0% | 21.4 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.001, t=0.18 ≤ 1.76), mas desvio-padrão 0.025 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.945 | 0.024 | 0.887 | 103.8 ± 10.5 | 0.10 | 12.5 (0.14) | 52.4 (0.10) | 0.14 | 100% (15/15) | 127 | ⚠ sim | 255001 | 57660 | 0/15 | 0% | 20.8 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.001, t=0.22 ≤ 1.76), mas desvio-padrão 0.024 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.943 | 0.023 | 0.873 | 163.4 ± 84.5 | 0.52 | 13.1 (0.51) | 82.2 (0.51) | 0.52 | 80% (12/15) | 255 | ⚠ sim | 255001 | 97561 | 0/15 | 0% | 38.2 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.52 (CV de nós 0.52; nós 3–239) > 0.21 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.943 | 0.025 | 0.887 | 196.9 ± 11.0 | 0.06 | 15.1 (0.15) | 98.9 (0.06) | 0.15 | 100% (15/15) | 255 | ⚠ sim | 255001 | 116775 | 0/15 | 0% | 48.2 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.003, t=0.40 ≤ 1.76), mas desvio-padrão 0.025 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.943 | 0.024 | 0.887 | 196.5 ± 9.9 | 0.05 | 15.0 (0.16) | 98.7 (0.05) | 0.16 | 100% (15/15) | 255 | ⚠ sim | 255001 | 117049 | 0/15 | 0% | 50.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.003, t=0.48 ≤ 1.76), mas desvio-padrão 0.024 > 0.019 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.942 | 0.024 | 0.017 | 0.014 | 0.923, 0.960, 0.939, 0.939, 0.951 |
| purity_epsilon=0.02, max_nodes=31 | 0.946 | 0.020 | 0.017 | 0.012 | 0.937, 0.965, 0.941, 0.951, 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.944 | 0.019 | 0.018 | 0.009 | 0.937, 0.955, 0.941, 0.951, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.944 | 0.025 | 0.020 | 0.014 | 0.923, 0.960, 0.939, 0.948, 0.948 |
| purity_epsilon=0.02, max_nodes=63 | 0.945 | 0.025 | 0.024 | 0.012 | 0.930, 0.958, 0.944, 0.955, 0.939 |
| purity_epsilon=0.01, max_nodes=63 | 0.944 | 0.024 | 0.023 | 0.011 | 0.930, 0.955, 0.941, 0.955, 0.939 |
| purity_epsilon=0.05, max_nodes=127 | 0.942 | 0.023 | 0.018 | 0.014 | 0.923, 0.960, 0.939, 0.939, 0.948 |
| purity_epsilon=0.02, max_nodes=127 | 0.945 | 0.025 | 0.025 | 0.012 | 0.930, 0.962, 0.939, 0.951, 0.941 |
| purity_epsilon=0.01, max_nodes=127 | 0.945 | 0.024 | 0.023 | 0.011 | 0.930, 0.960, 0.941, 0.951, 0.941 |
| purity_epsilon=0.05, max_nodes=255 | 0.943 | 0.023 | 0.018 | 0.015 | 0.923, 0.962, 0.939, 0.941, 0.948 |
| purity_epsilon=0.02, max_nodes=255 | 0.943 | 0.025 | 0.025 | 0.012 | 0.930, 0.962, 0.939, 0.944, 0.939 |
| purity_epsilon=0.01, max_nodes=255 | 0.943 | 0.024 | 0.022 | 0.011 | 0.930, 0.960, 0.941, 0.944, 0.939 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.53 | 0.18 | 18 | 0.68 | 0.26 | 20% | concave_points_worst 100%, area_worst 93%, radius_worst 93%, concavity_worst 87% |
| purity_epsilon=0.02, max_nodes=31 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.67 | 0.24 | 11 | 0.64 | 0.25 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_worst 100%, radius_se 100% |
| purity_epsilon=0.01, max_nodes=31 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.65 | 0.22 | 13 | 0.62 | 0.21 | 0% | concave_points_worst 100%, concavity_worst 100%, radius_se 100%, texture_worst 100% |
| purity_epsilon=0.05, max_nodes=63 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.55 | 0.23 | 10 | 0.62 | 0.28 | 20% | concave_points_worst 100%, area_worst 93%, radius_worst 93%, concavity_worst 87% |
| purity_epsilon=0.02, max_nodes=63 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.79 | 0.34 | 8 | 0.80 | 0.37 | 0% | compactness_mean 100%, compactness_worst 100%, concave_points_worst 100%, concavity_worst 100% |
| purity_epsilon=0.01, max_nodes=63 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.80 | 0.35 | 8 | 0.77 | 0.37 | 0% | compactness_mean 100%, concave_points_worst 100%, concavity_worst 100%, perimeter_worst 100% |
| purity_epsilon=0.05, max_nodes=127 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.55 | 0.27 | 6 | 0.60 | 0.34 | 20% | concave_points_worst 100%, area_worst 93%, radius_worst 93%, concavity_worst 87% |
| purity_epsilon=0.02, max_nodes=127 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.85 | 0.43 | 5 | 0.84 | 0.44 | 0% | area_worst 100%, compactness_mean 100%, compactness_worst 100%, concave_points_worst 100% |
| purity_epsilon=0.01, max_nodes=127 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.85 | 0.44 | 6 | 0.89 | 0.44 | 0% | area_worst 100%, compactness_mean 100%, compactness_worst 100%, concave_points_worst 100% |
| purity_epsilon=0.05, max_nodes=255 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.60 | 0.30 | 7 | 0.66 | 0.37 | 20% | concave_points_worst 100%, area_worst 93%, radius_worst 93%, concavity_worst 87% |
| purity_epsilon=0.02, max_nodes=255 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.90 | 0.51 | 5 | 0.94 | 0.53 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, compactness_worst 100% |
| purity_epsilon=0.01, max_nodes=255 | concave_points_worst 100%, area_worst 80%, radius_worst 60% | 100% | 0.89 | 0.51 | 7 | 0.89 | 0.51 | 0% | area_worst 100%, compactness_mean 100%, compactness_se 100%, compactness_worst 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 | 0.923 | 0.937 | 0.958 | 0.951 | 0.951 | 0.951 |
| purity_epsilon=0.02, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.965 | 0.972 | 0.965 | 0.937 | 0.923 | 0.944 | 0.951 | 0.958 | 0.937 | 0.915 | 0.958 |
| purity_epsilon=0.01, max_nodes=31 | 0.958 | 0.951 | 0.901 | 0.958 | 0.944 | 0.965 | 0.965 | 0.937 | 0.923 | 0.944 | 0.951 | 0.958 | 0.937 | 0.915 | 0.958 |
| purity_epsilon=0.05, max_nodes=63 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 | 0.923 | 0.944 | 0.979 | 0.951 | 0.944 | 0.951 |
| purity_epsilon=0.02, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.965 | 0.951 | 0.965 | 0.944 | 0.923 | 0.937 | 0.958 | 0.972 | 0.944 | 0.901 | 0.972 |
| purity_epsilon=0.01, max_nodes=63 | 0.958 | 0.944 | 0.887 | 0.958 | 0.951 | 0.958 | 0.965 | 0.937 | 0.923 | 0.937 | 0.958 | 0.972 | 0.944 | 0.901 | 0.972 |
| purity_epsilon=0.05, max_nodes=127 | 0.958 | 0.937 | 0.873 | 0.937 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 | 0.923 | 0.944 | 0.951 | 0.951 | 0.944 | 0.951 |
| purity_epsilon=0.02, max_nodes=127 | 0.958 | 0.944 | 0.887 | 0.958 | 0.979 | 0.951 | 0.965 | 0.930 | 0.923 | 0.930 | 0.958 | 0.965 | 0.944 | 0.908 | 0.972 |
| purity_epsilon=0.01, max_nodes=127 | 0.958 | 0.944 | 0.887 | 0.958 | 0.965 | 0.958 | 0.965 | 0.937 | 0.923 | 0.930 | 0.958 | 0.965 | 0.944 | 0.908 | 0.972 |
| purity_epsilon=0.05, max_nodes=255 | 0.958 | 0.937 | 0.873 | 0.944 | 0.979 | 0.965 | 0.937 | 0.937 | 0.944 | 0.923 | 0.944 | 0.958 | 0.951 | 0.944 | 0.951 |
| purity_epsilon=0.02, max_nodes=255 | 0.958 | 0.944 | 0.887 | 0.958 | 0.979 | 0.951 | 0.965 | 0.930 | 0.923 | 0.930 | 0.958 | 0.944 | 0.944 | 0.901 | 0.972 |
| purity_epsilon=0.01, max_nodes=255 | 0.958 | 0.944 | 0.887 | 0.958 | 0.965 | 0.958 | 0.965 | 0.937 | 0.923 | 0.930 | 0.958 | 0.944 | 0.944 | 0.901 | 0.972 |

Nós por partição:

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 31 | 3 | 29 | 27 | 31 | 31 | 29 | 3 | 29 | 31 | 23 | 3 | 23 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 21 | 17 | 25 | 27 | 31 | 23 | 25 | 29 | 17 | 25 | 21 | 31 | 29 | 17 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 21 | 17 | 25 | 23 | 21 | 23 | 17 | 29 | 17 | 25 | 21 | 31 | 29 | 17 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 59 | 3 | 51 | 51 | 57 | 49 | 61 | 3 | 57 | 63 | 51 | 3 | 53 | 53 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 53 | 59 | 61 | 37 | 59 | 61 | 51 | 59 | 57 | 61 | 55 | 47 | 51 | 41 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 53 | 59 | 61 | 47 | 57 | 61 | 43 | 59 | 57 | 61 | 55 | 47 | 51 | 41 |
| purity_epsilon=0.05, max_nodes=127 | 105 | 111 | 3 | 99 | 89 | 101 | 89 | 103 | 3 | 107 | 97 | 101 | 3 | 87 | 111 |
| purity_epsilon=0.02, max_nodes=127 | 105 | 107 | 95 | 89 | 103 | 111 | 109 | 111 | 107 | 89 | 115 | 91 | 103 | 121 | 91 |
| purity_epsilon=0.01, max_nodes=127 | 105 | 107 | 95 | 89 | 115 | 107 | 109 | 113 | 107 | 89 | 115 | 91 | 103 | 121 | 91 |
| purity_epsilon=0.05, max_nodes=255 | 187 | 209 | 3 | 187 | 191 | 239 | 203 | 207 | 3 | 229 | 175 | 207 | 3 | 207 | 201 |
| purity_epsilon=0.02, max_nodes=255 | 191 | 203 | 187 | 195 | 187 | 215 | 211 | 205 | 203 | 183 | 203 | 197 | 175 | 205 | 193 |
| purity_epsilon=0.01, max_nodes=255 | 191 | 203 | 187 | 195 | 207 | 191 | 211 | 203 | 203 | 183 | 203 | 197 | 175 | 205 | 193 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.01, max_nodes=31` — estado interno do tuning: **tuning_uncertain** (selection_probability_below_threshold).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **53.0%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.01, max_nodes=31` com probabilidade 53.0% — selecionada é a moda: **True**
- bootstrap_runner_up: `purity_epsilon=0.02, max_nodes=31` 14.2%; top1_top2_margin: 38.8%
- IC bootstrap 95% da fidelity média da escolhida: [0.938, 0.951]
- distribuição das configurações escolhidas: `purity_epsilon=0.01, max_nodes=31` 53.0%, `purity_epsilon=0.02, max_nodes=31` 14.2%, `purity_epsilon=0.01, max_nodes=63` 9.1%, `purity_epsilon=0.05, max_nodes=31` 9.0%, `purity_epsilon=0.01, max_nodes=127` 4.8%, `purity_epsilon=0.01, max_nodes=255` 4.6%, `purity_epsilon=0.02, max_nodes=63` 3.5%, `purity_epsilon=0.02, max_nodes=127` 1.0%, `purity_epsilon=0.02, max_nodes=255` 0.8%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.05, max_nodes=31']
- saturação da escolhida: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 29, folhas 15, accuracy 0.9370629370629371, fidelity 0.951048951048951; MLP accuracy 0.972027972027972.

## iris_t.arff — oráculo `factory`

### 1. Estabilidade interna do tuning (block bootstrap, por seed mestre)

| seed mestre | oracle_id | escolhida (CV completa) | P(escolhida) | moda do bootstrap | P(moda) | bootstrap_runner_up | top1−top2 | método (amostras) | grelha de nós final (rondas) | tuning |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `fa910f6f25e5c1a3` | `purity_epsilon=0.05, max_nodes=31` | 58.0% | `purity_epsilon=0.05, max_nodes=31` | 58.0% | `purity_epsilon=0.02, max_nodes=31` 22.3% | 35.7% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 7 | `bec1d5a4e626d590` | `purity_epsilon=0.02, max_nodes=63` | 20.4% | `purity_epsilon=0.05, max_nodes=63` | 42.8% | `purity_epsilon=0.02, max_nodes=127` 35.6% | 7.2% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 123 | `3c656fb0e0870746` | `purity_epsilon=0.02, max_nodes=63` | 28.2% | `purity_epsilon=0.02, max_nodes=31` | 35.7% | `purity_epsilon=0.02, max_nodes=63` 28.2% | 7.6% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 2024 | `9886716fca9b4dba` | `purity_epsilon=0.02, max_nodes=31` | 29.9% | `purity_epsilon=0.02, max_nodes=31` | 29.9% | `purity_epsilon=0.05, max_nodes=63` 26.9% | 3.0% | exact (3125) | [31, 63, 127, 255] (2) | tuning_uncertain |
| 11 | `3a8a750475e18459` | `purity_epsilon=0.02, max_nodes=31` | 82.1% | `purity_epsilon=0.02, max_nodes=31` | 82.1% | `purity_epsilon=0.01, max_nodes=31` 6.7% | 75.4% | exact (3125) | [31, 63, 127, 255] (2) | tuning_stable |

### 2. Robustez externa entre seeds mestre (evidência empírica; não define `tuning_stable`)

| seed mestre | selecionada | nós finais (Original) | accuracy teste (Original) | fidelity teste (Original) | tempo total (s) |
|---|---|---|---|---|---|
| 42 | `purity_epsilon=0.05, max_nodes=31` | 25 | 0.8947368421052632 | 0.8947368421052632 | 448 |
| 7 | `purity_epsilon=0.02, max_nodes=63` | 61 | 0.9736842105263158 | 0.9736842105263158 | 450 |
| 123 | `purity_epsilon=0.02, max_nodes=63` | 61 | 0.9473684210526315 | 1.0 | 487 |
| 2024 | `purity_epsilon=0.02, max_nodes=31` | 29 | 0.9473684210526315 | 0.9210526315789473 | 412 |
| 11 | `purity_epsilon=0.02, max_nodes=31` | 23 | 0.8947368421052632 | 0.9210526315789473 | 423 |

Configurações escolhidas: {'purity_epsilon=0.05, max_nodes=31': 1, 'purity_epsilon=0.02, max_nodes=63': 2, 'purity_epsilon=0.02, max_nodes=31': 2}. **Consenso de seleção = 40% (2/5) nas master seeds avaliadas** (moda `purity_epsilon=0.02, max_nodes=63`); seeds mestre com `tuning_stable` pelo bootstrap: 1/5.

Leitura: a evidência de robustez **não** é suficiente (consenso externo e/ou reamostragem interna não a sustentam); usar *incerto*, não *estável*.

### Conclusão — iris_t.arff

Nenhuma configuração produziu árvores de ≤3 nós (stumps) nas partições de CV desta experiência.

#### iris_t.arff — seed mestre 42 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [42, 1051, 2060, 3069, 4078]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=fa910f6f25e5c1a3` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'fa910f6f25e5c1a3', 'trepan_reloaded': 'fa910f6f25e5c1a3'}); queries por escopo: {'tuning': 3823485, 'trepan_pair': 14926, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 85681 (mesmo para todos: True); consumo máximo observado: 41739; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.921 | 0.033 | 0.865 | 28.3 ± 2.9 | 0.10 | 8.1 (0.16) | 14.7 (0.10) | 0.16 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4870 | 0/15 | 0% | 0.5 | **WINNER** — VENCE: fidelity média 0.921; indistinguível da melhor (purity_epsilon=0.02, max_nodes=63, Δ=0.009, t=0.49 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 12 após estabilidade da fidelity -> 8 após estabilidade estrutural (índice 0.16) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.925 | 0.031 | 0.892 | 29.3 ± 2.4 | 0.08 | 8.0 (0.19) | 15.1 (0.08) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4619 | 0/15 | 0% | 0.7 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.3 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.925 | 0.031 | 0.892 | 29.3 ± 2.4 | 0.08 | 8.0 (0.19) | 15.1 (0.08) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4619 | 0/15 | 0% | 0.6 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.3 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.927 | 0.031 | 0.865 | 57.8 ± 3.3 | 0.06 | 12.1 (0.20) | 29.4 (0.06) | 0.20 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9759 | 0/15 | 0% | 1.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.8 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.028 | 0.892 | 59.4 ± 3.3 | 0.06 | 11.7 (0.19) | 30.2 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9396 | 0/15 | 0% | 1.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.4 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.930 | 0.028 | 0.892 | 59.3 ± 3.2 | 0.05 | 11.5 (0.19) | 30.1 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9371 | 0/15 | 0% | 1.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.3 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.927 | 0.031 | 0.865 | 122.3 ± 4.5 | 0.04 | 16.3 (0.21) | 61.7 (0.04) | 0.21 | 100% (15/15) | 127 | ⚠ sim | 85681 | 20030 | 0/15 | 0% | 2.3 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.21 (CV de nós 0.04; nós 113–127) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.930 | 0.028 | 0.892 | 122.2 ± 3.5 | 0.03 | 15.3 (0.21) | 61.6 (0.03) | 0.21 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19419 | 0/15 | 0% | 2.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 122.2 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.930 | 0.028 | 0.892 | 122.1 ± 3.5 | 0.03 | 15.2 (0.21) | 61.5 (0.03) | 0.21 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19395 | 0/15 | 0% | 2.4 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 122.1 nós médios vs 28.3 de purity_epsilon=0.05, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.927 | 0.031 | 0.865 | 247.5 ± 5.5 | 0.02 | 22.2 (0.30) | 124.3 (0.02) | 0.30 | 100% (15/15) | 255 | ⚠ sim | 85681 | 40294 | 0/15 | 0% | 4.8 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.30 (CV de nós 0.02; nós 231–255) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.930 | 0.028 | 0.892 | 245.8 ± 5.7 | 0.02 | 20.5 (0.23) | 123.4 (0.02) | 0.23 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39458 | 0/15 | 0% | 4.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.23 (CV de nós 0.02; nós 233–253) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.930 | 0.028 | 0.892 | 245.8 ± 5.8 | 0.02 | 20.2 (0.23) | 123.4 (0.02) | 0.23 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39390 | 0/15 | 0% | 4.9 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.23 (CV de nós 0.02; nós 233–253) > 0.16 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.921 | 0.033 | 0.028 | 0.022 | 0.884, 0.937, 0.937, 0.920, 0.929 |
| purity_epsilon=0.02, max_nodes=31 | 0.925 | 0.031 | 0.025 | 0.020 | 0.893, 0.937, 0.937, 0.920, 0.937 |
| purity_epsilon=0.01, max_nodes=31 | 0.925 | 0.031 | 0.025 | 0.020 | 0.893, 0.937, 0.937, 0.920, 0.937 |
| purity_epsilon=0.05, max_nodes=63 | 0.927 | 0.031 | 0.025 | 0.021 | 0.902, 0.937, 0.955, 0.911, 0.929 |
| purity_epsilon=0.02, max_nodes=63 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |
| purity_epsilon=0.01, max_nodes=63 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |
| purity_epsilon=0.05, max_nodes=127 | 0.927 | 0.031 | 0.025 | 0.021 | 0.902, 0.937, 0.955, 0.911, 0.929 |
| purity_epsilon=0.02, max_nodes=127 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |
| purity_epsilon=0.01, max_nodes=127 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |
| purity_epsilon=0.05, max_nodes=255 | 0.927 | 0.031 | 0.025 | 0.021 | 0.902, 0.937, 0.955, 0.911, 0.929 |
| purity_epsilon=0.02, max_nodes=255 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |
| purity_epsilon=0.01, max_nodes=255 | 0.930 | 0.028 | 0.023 | 0.019 | 0.911, 0.937, 0.955, 0.911, 0.937 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.40 | 28 | 1.00 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.42 | 34 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.42 | 34 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.45 | 16 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.46 | 17 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.47 | 17 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=127 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.47 | 12 | 1.00 | 0.48 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=127 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.47 | 19 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=127 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.47 | 16 | 1.00 | 0.50 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=255 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.47 | 19 | 1.00 | 0.42 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=255 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.46 | 12 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=255 | sepal_width 87%, petal_length 67% | 87% | 1.00 | 0.46 | 11 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.895 | 0.892 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.892 | 0.895 | 0.919 | 0.946 | 0.921 | 0.946 | 0.919 |
| purity_epsilon=0.02, max_nodes=31 | 0.895 | 0.892 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.892 | 0.895 | 0.919 | 0.946 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=31 | 0.895 | 0.892 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.892 | 0.895 | 0.919 | 0.946 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=63 | 0.895 | 0.946 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.921 | 0.946 | 0.919 |
| purity_epsilon=0.02, max_nodes=63 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=63 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=127 | 0.895 | 0.946 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.921 | 0.946 | 0.919 |
| purity_epsilon=0.02, max_nodes=127 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=127 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=255 | 0.895 | 0.946 | 0.865 | 0.974 | 0.892 | 0.946 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.921 | 0.946 | 0.919 |
| purity_epsilon=0.02, max_nodes=255 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=255 | 0.895 | 0.946 | 0.892 | 0.947 | 0.892 | 0.973 | 0.947 | 0.973 | 0.946 | 0.895 | 0.919 | 0.919 | 0.947 | 0.946 | 0.919 |

Nós por partição:

| configuração | s42·f0 | s42·f1 | s42·f2 | s1051·f0 | s1051·f1 | s1051·f2 | s2060·f0 | s2060·f1 | s2060·f2 | s3069·f0 | s3069·f1 | s3069·f2 | s4078·f0 | s4078·f1 | s4078·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 27 | 29 | 27 | 23 | 29 | 31 | 29 | 31 | 21 | 31 | 29 | 29 | 29 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 | 27 | 31 | 29 | 31 | 29 | 29 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 27 | 27 | 31 | 23 | 31 | 31 | 31 | 31 | 27 | 31 | 29 | 31 | 29 | 29 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 57 | 57 | 59 | 53 | 59 | 63 | 61 | 61 | 53 | 55 | 57 | 61 | 53 | 57 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 | 59 | 59 | 57 | 63 | 53 | 59 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 57 | 61 | 53 | 63 | 63 | 61 | 61 | 59 | 59 | 57 | 61 | 53 | 59 |
| purity_epsilon=0.05, max_nodes=127 | 123 | 125 | 115 | 113 | 125 | 127 | 127 | 121 | 123 | 125 | 125 | 127 | 123 | 119 | 117 |
| purity_epsilon=0.02, max_nodes=127 | 125 | 123 | 121 | 119 | 125 | 125 | 123 | 123 | 123 | 113 | 125 | 127 | 123 | 119 | 119 |
| purity_epsilon=0.01, max_nodes=127 | 125 | 123 | 121 | 119 | 125 | 125 | 123 | 123 | 123 | 113 | 125 | 127 | 121 | 119 | 119 |
| purity_epsilon=0.05, max_nodes=255 | 249 | 249 | 241 | 247 | 247 | 249 | 249 | 255 | 251 | 249 | 245 | 251 | 249 | 231 | 251 |
| purity_epsilon=0.02, max_nodes=255 | 249 | 247 | 251 | 253 | 239 | 249 | 237 | 247 | 245 | 247 | 247 | 253 | 247 | 233 | 243 |
| purity_epsilon=0.01, max_nodes=255 | 249 | 247 | 251 | 253 | 239 | 249 | 237 | 247 | 249 | 247 | 247 | 253 | 243 | 233 | 243 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.05, max_nodes=31` — estado interno do tuning: **tuning_uncertain** (selection_probability_below_threshold).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **58.0%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.05, max_nodes=31` com probabilidade 58.0% — selecionada é a moda: **True**
- bootstrap_runner_up: `purity_epsilon=0.02, max_nodes=31` 22.3%; top1_top2_margin: 35.7%
- IC bootstrap 95% da fidelity média da escolhida: [0.902, 0.936]
- distribuição das configurações escolhidas: `purity_epsilon=0.05, max_nodes=31` 58.0%, `purity_epsilon=0.02, max_nodes=31` 22.3%, `purity_epsilon=0.05, max_nodes=63` 17.0%, `purity_epsilon=0.02, max_nodes=63` 2.4%, `purity_epsilon=0.01, max_nodes=63` 0.2%, `purity_epsilon=0.05, max_nodes=127` 0.2%, `purity_epsilon=0.02, max_nodes=127` 0.0%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.05, max_nodes=31']
- saturação da escolhida: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 25, folhas 13, accuracy 0.8947368421052632, fidelity 0.8947368421052632; MLP accuracy 0.8947368421052632.

#### iris_t.arff — seed mestre 7 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [7, 1016, 2025, 3034, 4043]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=bec1d5a4e626d590` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': 'bec1d5a4e626d590', 'trepan_reloaded': 'bec1d5a4e626d590'}); queries por escopo: {'tuning': 4727249, 'trepan_pair': 61864, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 85681 (mesmo para todos: True); consumo máximo observado: 41257; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.047 | 0.838 | 28.6 ± 2.5 | 0.09 | 7.8 (0.24) | 14.8 (0.09) | 0.24 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4821 | 0/15 | 0% | 0.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.24 (CV de nós 0.09; nós 23–31) > 0.12 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.928 | 0.048 | 0.838 | 28.9 ± 2.6 | 0.09 | 7.5 (0.24) | 14.9 (0.09) | 0.24 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4741 | 0/15 | 0% | 0.4 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.15 ≤ 1.76), mas desvio-padrão 0.048 > 0.043 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.928 | 0.048 | 0.838 | 28.9 ± 2.6 | 0.09 | 7.5 (0.24) | 14.9 (0.09) | 0.24 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4741 | 0/15 | 0% | 0.4 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.15 ≤ 1.76), mas desvio-padrão 0.048 > 0.043 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.928 | 0.048 | 0.838 | 57.8 ± 3.4 | 0.06 | 11.5 (0.13) | 29.4 (0.06) | 0.13 | 100% (15/15) | 63 | ⚠ sim | 85681 | 10033 | 0/15 | 0% | 0.9 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.34 ≤ 1.76), mas desvio-padrão 0.048 > 0.043 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.925 | 0.048 | 0.838 | 59.4 ± 2.6 | 0.04 | 11.5 (0.12) | 30.2 (0.04) | 0.12 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9756 | 0/15 | 0% | 1.0 | **WINNER** — VENCE: fidelity média 0.925; indistinguível da melhor (purity_epsilon=0.05, max_nodes=127, Δ=0.005, t=0.64 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 7 após estabilidade da fidelity -> 4 após estabilidade estrutural (índice 0.12) -> 2 após complexidade; empate real resolvido pela configuração mais simples. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.048 | 0.838 | 59.4 ± 2.6 | 0.04 | 11.5 (0.12) | 30.2 (0.04) | 0.12 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9756 | 0/15 | 0% | 1.0 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=63). ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.930 | 0.048 | 0.838 | 119.8 ± 5.1 | 0.04 | 16.0 (0.13) | 60.4 (0.04) | 0.13 | 100% (15/15) | 127 | ⚠ sim | 85681 | 20226 | 0/15 | 0% | 2.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.76), mas desvio-padrão 0.048 > 0.043 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.928 | 0.043 | 0.838 | 119.4 ± 4.4 | 0.04 | 15.1 (0.14) | 60.2 (0.04) | 0.14 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19707 | 0/15 | 0% | 2.2 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 119.4 nós médios vs 59.4 de purity_epsilon=0.02, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.928 | 0.043 | 0.838 | 119.4 ± 4.4 | 0.04 | 15.1 (0.14) | 60.2 (0.04) | 0.14 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19707 | 0/15 | 0% | 2.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 119.4 nós médios vs 59.4 de purity_epsilon=0.02, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.930 | 0.048 | 0.838 | 241.5 ± 6.2 | 0.03 | 22.2 (0.14) | 121.3 (0.03) | 0.14 | 100% (15/15) | 255 | ⚠ sim | 85681 | 40425 | 0/15 | 0% | 4.6 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.76), mas desvio-padrão 0.048 > 0.043 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.928 | 0.043 | 0.838 | 246.2 ± 6.6 | 0.03 | 22.8 (0.20) | 123.6 (0.03) | 0.20 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39529 | 0/15 | 0% | 4.6 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.03; nós 231–253) > 0.12 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.928 | 0.043 | 0.838 | 246.2 ± 6.6 | 0.03 | 22.8 (0.20) | 123.6 (0.03) | 0.20 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39529 | 0/15 | 0% | 4.7 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.03; nós 231–253) > 0.12 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.930 | 0.047 | 0.043 | 0.029 | 0.911, 0.920, 0.946, 0.902, 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.928 | 0.048 | 0.045 | 0.028 | 0.911, 0.920, 0.937, 0.902, 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 0.928 | 0.048 | 0.045 | 0.028 | 0.911, 0.920, 0.937, 0.902, 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 0.928 | 0.048 | 0.047 | 0.026 | 0.902, 0.920, 0.946, 0.911, 0.964 |
| purity_epsilon=0.02, max_nodes=63 | 0.925 | 0.048 | 0.047 | 0.026 | 0.902, 0.920, 0.937, 0.902, 0.964 |
| purity_epsilon=0.01, max_nodes=63 | 0.925 | 0.048 | 0.047 | 0.026 | 0.902, 0.920, 0.937, 0.902, 0.964 |
| purity_epsilon=0.05, max_nodes=127 | 0.930 | 0.048 | 0.045 | 0.029 | 0.902, 0.920, 0.946, 0.911, 0.973 |
| purity_epsilon=0.02, max_nodes=127 | 0.928 | 0.043 | 0.041 | 0.023 | 0.920, 0.920, 0.937, 0.902, 0.964 |
| purity_epsilon=0.01, max_nodes=127 | 0.928 | 0.043 | 0.041 | 0.023 | 0.920, 0.920, 0.937, 0.902, 0.964 |
| purity_epsilon=0.05, max_nodes=255 | 0.930 | 0.048 | 0.045 | 0.029 | 0.902, 0.920, 0.946, 0.911, 0.973 |
| purity_epsilon=0.02, max_nodes=255 | 0.928 | 0.043 | 0.041 | 0.023 | 0.920, 0.920, 0.937, 0.902, 0.964 |
| purity_epsilon=0.01, max_nodes=255 | 0.928 | 0.043 | 0.041 | 0.023 | 0.920, 0.920, 0.937, 0.902, 0.964 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.43 | 24 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.41 | 27 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.41 | 27 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.44 | 19 | 1.00 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.44 | 20 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.44 | 20 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.47 | 13 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.50 | 15 | 1.00 | 0.52 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.50 | 15 | 1.00 | 0.52 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.48 | 7 | 1.00 | 0.49 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.51 | 10 | 1.00 | 0.55 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.51 | 10 | 1.00 | 0.55 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.973 | 0.946 |
| purity_epsilon=0.02, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.973 | 0.946 |
| purity_epsilon=0.01, max_nodes=31 | 0.842 | 0.946 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.973 | 0.946 |
| purity_epsilon=0.05, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 | 0.921 | 0.973 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.05, max_nodes=127 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 | 0.921 | 0.973 | 0.838 | 1.000 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=127 | 0.895 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.01, max_nodes=127 | 0.895 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.05, max_nodes=255 | 0.842 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.919 | 0.919 | 0.921 | 0.973 | 0.838 | 1.000 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=255 | 0.895 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |
| purity_epsilon=0.01, max_nodes=255 | 0.895 | 0.919 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 0.892 | 0.919 | 0.921 | 0.946 | 0.838 | 1.000 | 0.919 | 0.973 |

Nós por partição:

| configuração | s7·f0 | s7·f1 | s7·f2 | s1016·f0 | s1016·f1 | s1016·f2 | s2025·f0 | s2025·f1 | s2025·f2 | s3034·f0 | s3034·f1 | s3034·f2 | s4043·f0 | s4043·f1 | s4043·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 27 | 31 | 29 | 31 | 29 | 31 | 29 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 29 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 25 | 31 | 27 | 27 | 23 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 29 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 59 | 59 | 61 | 53 | 59 | 57 | 49 | 59 | 55 | 59 | 57 | 63 | 59 | 57 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 | 59 | 61 | 61 | 61 | 61 | 57 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 59 | 63 | 53 | 59 | 57 | 59 | 61 | 57 | 59 | 61 | 61 | 61 | 61 | 57 |
| purity_epsilon=0.05, max_nodes=127 | 115 | 111 | 115 | 119 | 121 | 121 | 123 | 121 | 109 | 123 | 125 | 121 | 123 | 125 | 125 |
| purity_epsilon=0.02, max_nodes=127 | 117 | 111 | 121 | 119 | 117 | 121 | 125 | 119 | 111 | 127 | 121 | 117 | 121 | 123 | 121 |
| purity_epsilon=0.01, max_nodes=127 | 117 | 111 | 121 | 119 | 117 | 121 | 125 | 119 | 111 | 127 | 121 | 117 | 121 | 123 | 121 |
| purity_epsilon=0.05, max_nodes=255 | 245 | 243 | 227 | 241 | 241 | 239 | 251 | 245 | 237 | 247 | 233 | 241 | 239 | 245 | 249 |
| purity_epsilon=0.02, max_nodes=255 | 249 | 247 | 249 | 245 | 251 | 231 | 247 | 245 | 233 | 249 | 249 | 253 | 253 | 241 | 251 |
| purity_epsilon=0.01, max_nodes=255 | 249 | 247 | 249 | 245 | 251 | 231 | 247 | 245 | 233 | 249 | 249 | 253 | 253 | 241 | 251 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.02, max_nodes=63` — estado interno do tuning: **tuning_uncertain** (full_cv_selection_not_bootstrap_modal).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **20.4%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.05, max_nodes=63` com probabilidade 42.8% — selecionada é a moda: **False** ⚠ **fragilidade FORTE da seleção full-CV**
- bootstrap_runner_up: `purity_epsilon=0.02, max_nodes=127` 35.6%; top1_top2_margin: 7.2%
- IC bootstrap 95% da fidelity média da escolhida: [0.906, 0.946]
- distribuição das configurações escolhidas: `purity_epsilon=0.05, max_nodes=63` 42.8%, `purity_epsilon=0.02, max_nodes=127` 35.6%, `purity_epsilon=0.02, max_nodes=63` 20.4%, `purity_epsilon=0.05, max_nodes=255` 1.0%, `purity_epsilon=0.05, max_nodes=127` 0.3%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.05, max_nodes=255']
- saturação da escolhida: 15/15 árvores no teto max_nodes=63 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 61, folhas 31, accuracy 0.9736842105263158, fidelity 0.9736842105263158; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 123 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [123, 1132, 2141, 3150, 4159]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=3c656fb0e0870746` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '3c656fb0e0870746', 'trepan_reloaded': '3c656fb0e0870746'}); queries por escopo: {'tuning': 4652819, 'trepan_pair': 61203, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 85681 (mesmo para todos: True); consumo máximo observado: 41198; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.943 | 0.039 | 0.865 | 27.4 ± 2.5 | 0.09 | 7.9 (0.20) | 14.2 (0.09) | 0.20 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4761 | 0/15 | 0% | 0.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.000, t=0.00 ≤ 1.76), mas desvio-padrão 0.039 > 0.031 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.941 | 0.031 | 0.865 | 27.8 ± 2.5 | 0.09 | 7.8 (0.20) | 14.4 (0.09) | 0.20 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4587 | 0/15 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.09; nós 23–31) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.941 | 0.031 | 0.865 | 27.8 ± 2.5 | 0.09 | 7.8 (0.20) | 14.4 (0.09) | 0.20 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4565 | 0/15 | 0% | 0.5 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.09; nós 23–31) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.941 | 0.041 | 0.865 | 59.3 ± 3.2 | 0.05 | 12.1 (0.18) | 30.1 (0.05) | 0.18 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9720 | 0/15 | 0% | 1.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.34 ≤ 1.76), mas desvio-padrão 0.041 > 0.031 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.941 | 0.031 | 0.865 | 60.3 ± 3.4 | 0.06 | 11.7 (0.17) | 30.7 (0.05) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9397 | 0/15 | 0% | 1.2 | **WINNER** — VENCE: fidelity média 0.941; indistinguível da melhor (purity_epsilon=0.05, max_nodes=31, Δ=0.002, t=0.11 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 8 após estabilidade da fidelity -> 6 após estabilidade estrutural (índice 0.17) -> 2 após complexidade; empate real resolvido pela configuração mais simples. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.941 | 0.031 | 0.865 | 60.3 ± 3.4 | 0.06 | 11.7 (0.17) | 30.7 (0.05) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9375 | 0/15 | 0% | 1.2 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=63). ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.941 | 0.041 | 0.865 | 121.1 ± 4.2 | 0.03 | 16.7 (0.19) | 61.1 (0.03) | 0.19 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19692 | 0/15 | 0% | 2.3 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.34 ≤ 1.76), mas desvio-padrão 0.041 > 0.031 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.941 | 0.031 | 0.865 | 119.5 ± 5.9 | 0.05 | 15.7 (0.16) | 60.3 (0.05) | 0.16 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19284 | 0/15 | 0% | 2.4 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 119.5 nós médios vs 60.3 de purity_epsilon=0.02, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.941 | 0.031 | 0.865 | 120.1 ± 6.0 | 0.05 | 15.7 (0.16) | 60.5 (0.05) | 0.16 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19267 | 0/15 | 0% | 2.4 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 120.1 nós médios vs 60.3 de purity_epsilon=0.02, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.941 | 0.041 | 0.865 | 243.7 ± 7.5 | 0.03 | 21.9 (0.24) | 122.3 (0.03) | 0.24 | 100% (15/15) | 255 | ⚠ sim | 85681 | 40026 | 0/15 | 0% | 4.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.002, t=0.34 ≤ 1.76), mas desvio-padrão 0.041 > 0.031 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.939 | 0.031 | 0.865 | 246.6 ± 7.1 | 0.03 | 21.0 (0.15) | 123.8 (0.03) | 0.15 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39304 | 0/15 | 0% | 4.8 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 246.6 nós médios vs 60.3 de purity_epsilon=0.02, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.939 | 0.031 | 0.865 | 247.0 ± 6.0 | 0.02 | 21.1 (0.14) | 124.0 (0.02) | 0.14 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39343 | 0/15 | 0% | 4.9 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 247.0 nós médios vs 60.3 de purity_epsilon=0.02, max_nodes=63. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.943 | 0.039 | 0.030 | 0.031 | 0.955, 0.902, 0.964, 0.920, 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.01, max_nodes=31 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.05, max_nodes=63 | 0.941 | 0.041 | 0.033 | 0.029 | 0.955, 0.902, 0.964, 0.920, 0.964 |
| purity_epsilon=0.02, max_nodes=63 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.01, max_nodes=63 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.05, max_nodes=127 | 0.941 | 0.041 | 0.033 | 0.029 | 0.955, 0.902, 0.964, 0.920, 0.964 |
| purity_epsilon=0.02, max_nodes=127 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.01, max_nodes=127 | 0.941 | 0.031 | 0.024 | 0.024 | 0.955, 0.902, 0.964, 0.938, 0.946 |
| purity_epsilon=0.05, max_nodes=255 | 0.941 | 0.041 | 0.033 | 0.029 | 0.955, 0.902, 0.964, 0.920, 0.964 |
| purity_epsilon=0.02, max_nodes=255 | 0.939 | 0.031 | 0.026 | 0.022 | 0.955, 0.902, 0.955, 0.938, 0.946 |
| purity_epsilon=0.01, max_nodes=255 | 0.939 | 0.031 | 0.026 | 0.022 | 0.955, 0.902, 0.955, 0.938, 0.946 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.40 | 26 | 1.00 | 0.42 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.42 | 25 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.42 | 25 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.48 | 16 | 1.00 | 0.51 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.51 | 26 | 1.00 | 0.52 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.51 | 26 | 1.00 | 0.53 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.52 | 11 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.54 | 9 | 1.00 | 0.58 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.55 | 11 | 1.00 | 0.57 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.51 | 7 | 1.00 | 0.50 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.57 | 9 | 1.00 | 0.52 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.56 | 9 | 1.00 | 0.52 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 1.000 | 0.919 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=31 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 1.000 | 0.892 |
| purity_epsilon=0.02, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=63 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=127 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 1.000 | 0.892 |
| purity_epsilon=0.02, max_nodes=127 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=127 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.05, max_nodes=255 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.946 | 0.921 | 0.892 | 0.946 | 1.000 | 1.000 | 0.892 |
| purity_epsilon=0.02, max_nodes=255 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |
| purity_epsilon=0.01, max_nodes=255 | 0.947 | 0.973 | 0.946 | 0.921 | 0.865 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.946 | 0.946 | 0.974 | 0.946 | 0.919 |

Nós por partição:

| configuração | s123·f0 | s123·f1 | s123·f2 | s1132·f0 | s1132·f1 | s1132·f2 | s2141·f0 | s2141·f1 | s2141·f2 | s3150·f0 | s3150·f1 | s3150·f2 | s4159·f0 | s4159·f1 | s4159·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 27 | 27 | 23 | 27 | 27 | 23 | 29 | 31 | 27 | 31 | 31 | 27 | 25 | 27 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 27 | 27 | 23 | 27 | 29 | 23 | 29 | 31 | 27 | 31 | 31 | 27 | 29 | 29 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 27 | 27 | 23 | 27 | 29 | 23 | 29 | 31 | 27 | 31 | 31 | 27 | 29 | 29 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 63 | 63 | 61 | 61 | 57 | 59 | 63 | 55 | 61 | 61 | 53 | 55 | 59 | 57 | 61 |
| purity_epsilon=0.02, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 | 63 | 61 | 55 | 59 | 61 | 59 |
| purity_epsilon=0.01, max_nodes=63 | 63 | 63 | 61 | 61 | 63 | 61 | 63 | 51 | 61 | 63 | 61 | 55 | 59 | 61 | 59 |
| purity_epsilon=0.05, max_nodes=127 | 121 | 117 | 121 | 125 | 125 | 125 | 113 | 125 | 127 | 117 | 121 | 115 | 123 | 123 | 119 |
| purity_epsilon=0.02, max_nodes=127 | 119 | 117 | 121 | 125 | 119 | 127 | 111 | 119 | 125 | 113 | 127 | 107 | 117 | 121 | 125 |
| purity_epsilon=0.01, max_nodes=127 | 119 | 125 | 121 | 125 | 119 | 127 | 111 | 119 | 125 | 113 | 127 | 107 | 117 | 121 | 125 |
| purity_epsilon=0.05, max_nodes=255 | 251 | 225 | 249 | 253 | 239 | 249 | 247 | 247 | 247 | 235 | 243 | 239 | 237 | 251 | 243 |
| purity_epsilon=0.02, max_nodes=255 | 239 | 227 | 249 | 253 | 249 | 253 | 247 | 255 | 251 | 243 | 243 | 245 | 243 | 249 | 253 |
| purity_epsilon=0.01, max_nodes=255 | 239 | 233 | 249 | 253 | 249 | 253 | 247 | 255 | 251 | 243 | 243 | 245 | 243 | 249 | 253 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.02, max_nodes=63` — estado interno do tuning: **tuning_uncertain** (full_cv_selection_not_bootstrap_modal).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **28.2%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.02, max_nodes=31` com probabilidade 35.7% — selecionada é a moda: **False** ⚠ **fragilidade FORTE da seleção full-CV**
- bootstrap_runner_up: `purity_epsilon=0.02, max_nodes=63` 28.2%; top1_top2_margin: 7.6%
- IC bootstrap 95% da fidelity média da escolhida: [0.920, 0.957]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 35.7%, `purity_epsilon=0.02, max_nodes=63` 28.2%, `purity_epsilon=0.05, max_nodes=63` 13.3%, `purity_epsilon=0.02, max_nodes=127` 11.8%, `purity_epsilon=0.05, max_nodes=31` 10.9%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31']
- saturação da escolhida: 15/15 árvores no teto max_nodes=63 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 61, folhas 31, accuracy 0.9473684210526315, fidelity 1.0; MLP accuracy 0.9473684210526315.

#### iris_t.arff — seed mestre 2024 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [2024, 3033, 4042, 5051, 6060]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=9886716fca9b4dba` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '9886716fca9b4dba', 'trepan_reloaded': '9886716fca9b4dba'}); queries por escopo: {'tuning': 3796477, 'trepan_pair': 14906, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 85681 (mesmo para todos: True); consumo máximo observado: 42219; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.964 | 0.022 | 0.919 | 29.4 ± 1.7 | 0.06 | 8.7 (0.14) | 15.2 (0.06) | 0.14 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4676 | 0/15 | 0% | 0.4 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.4 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.966 | 0.021 | 0.919 | 29.0 ± 2.1 | 0.07 | 8.5 (0.17) | 15.0 (0.07) | 0.17 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4524 | 0/15 | 0% | 0.5 | **WINNER** — VENCE: fidelity média 0.966; indistinguível da melhor (purity_epsilon=0.05, max_nodes=127, Δ=0.004, t=0.28 ≤ 1.76). 12 indistinguível(eis) em fidelity -> 12 após estabilidade da fidelity -> 7 após estabilidade estrutural (índice 0.17) -> 2 após complexidade; empate real resolvido pela configuração mais simples. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.966 | 0.021 | 0.919 | 29.0 ± 2.1 | 0.07 | 8.4 (0.19) | 15.0 (0.07) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4502 | 0/15 | 0% | 0.5 | **LOST** — PERDE o desempate: empate real em fidelity, estabilidades e nós; prevalece a configuração mais simples (purity_epsilon=0.02, max_nodes=31). ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.966 | 0.019 | 0.946 | 58.5 ± 3.2 | 0.06 | 12.9 (0.19) | 29.7 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9855 | 0/15 | 0% | 1.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.19 (CV de nós 0.06; nós 51–63) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.968 | 0.018 | 0.946 | 59.8 ± 2.9 | 0.05 | 12.9 (0.19) | 30.4 (0.05) | 0.19 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9566 | 0/15 | 0% | 1.0 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 59.8 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.968 | 0.018 | 0.946 | 59.8 ± 2.9 | 0.05 | 12.8 (0.20) | 30.4 (0.05) | 0.20 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9521 | 0/15 | 0% | 1.0 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.05; nós 55–63) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.970 | 0.017 | 0.946 | 120.6 ± 4.6 | 0.04 | 19.0 (0.20) | 60.8 (0.04) | 0.20 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19873 | 0/15 | 0% | 2.2 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.04; nós 107–125) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.970 | 0.017 | 0.946 | 122.2 ± 3.9 | 0.03 | 18.3 (0.20) | 61.6 (0.03) | 0.20 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19456 | 0/15 | 0% | 2.2 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.20 (CV de nós 0.03; nós 111–127) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.970 | 0.017 | 0.946 | 122.2 ± 3.9 | 0.03 | 18.1 (0.21) | 61.6 (0.03) | 0.21 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19397 | 0/15 | 0% | 2.4 | **LOST** — PERDE em estabilidade estrutural: indistinguível em fidelity e estável nela, mas índice de instabilidade estrutural 0.21 (CV de nós 0.03; nós 111–127) > 0.14 + tolerância 0.05. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.970 | 0.017 | 0.946 | 246.5 ± 4.7 | 0.02 | 23.7 (0.14) | 123.7 (0.02) | 0.14 | 100% (15/15) | 255 | ⚠ sim | 85681 | 40267 | 0/15 | 0% | 4.7 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 246.5 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.970 | 0.017 | 0.946 | 246.3 ± 5.0 | 0.02 | 22.8 (0.14) | 123.7 (0.02) | 0.14 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39502 | 0/15 | 0% | 4.5 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 246.3 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.970 | 0.017 | 0.946 | 246.2 ± 4.9 | 0.02 | 22.5 (0.14) | 123.6 (0.02) | 0.14 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39476 | 0/15 | 0% | 4.6 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 246.2 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.964 | 0.022 | 0.018 | 0.014 | 0.964, 0.955, 0.946, 0.982, 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.966 | 0.021 | 0.015 | 0.017 | 0.964, 0.955, 0.946, 0.991, 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 0.966 | 0.021 | 0.015 | 0.017 | 0.964, 0.955, 0.946, 0.991, 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 0.966 | 0.019 | 0.015 | 0.012 | 0.964, 0.955, 0.955, 0.982, 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 0.968 | 0.018 | 0.012 | 0.015 | 0.964, 0.955, 0.955, 0.991, 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 0.968 | 0.018 | 0.012 | 0.015 | 0.964, 0.955, 0.955, 0.991, 0.973 |
| purity_epsilon=0.05, max_nodes=127 | 0.970 | 0.017 | 0.012 | 0.010 | 0.973, 0.955, 0.964, 0.982, 0.973 |
| purity_epsilon=0.02, max_nodes=127 | 0.970 | 0.017 | 0.012 | 0.014 | 0.964, 0.955, 0.964, 0.991, 0.973 |
| purity_epsilon=0.01, max_nodes=127 | 0.970 | 0.017 | 0.012 | 0.014 | 0.964, 0.955, 0.964, 0.991, 0.973 |
| purity_epsilon=0.05, max_nodes=255 | 0.970 | 0.017 | 0.012 | 0.010 | 0.973, 0.955, 0.964, 0.982, 0.973 |
| purity_epsilon=0.02, max_nodes=255 | 0.970 | 0.017 | 0.012 | 0.014 | 0.964, 0.955, 0.964, 0.991, 0.973 |
| purity_epsilon=0.01, max_nodes=255 | 0.970 | 0.017 | 0.012 | 0.014 | 0.964, 0.955, 0.964, 0.991, 0.973 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 33 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 32 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100% | 100% | 1.00 | 0.44 | 32 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.47 | 21 | 1.00 | 0.48 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.46 | 19 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100% | 100% | 1.00 | 0.47 | 19 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.47 | 14 | 1.00 | 0.49 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.46 | 17 | 1.00 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=127 | petal_length 100% | 100% | 1.00 | 0.47 | 17 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.46 | 17 | 1.00 | 0.53 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.45 | 11 | 1.00 | 0.42 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=255 | petal_length 100% | 100% | 1.00 | 0.47 | 12 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 | 0.947 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.919 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 | 0.947 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.946 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.05, max_nodes=127 | 0.974 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.973 | 0.973 | 0.947 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.02, max_nodes=127 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.973 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.01, max_nodes=127 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.973 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.05, max_nodes=255 | 0.974 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.973 | 0.973 | 0.947 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.02, max_nodes=255 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.973 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |
| purity_epsilon=0.01, max_nodes=255 | 0.947 | 0.973 | 0.973 | 0.974 | 0.946 | 0.946 | 0.947 | 0.973 | 0.973 | 0.974 | 1.000 | 1.000 | 0.974 | 0.973 | 0.973 |

Nós por partição:

| configuração | s2024·f0 | s2024·f1 | s2024·f2 | s3033·f0 | s3033·f1 | s3033·f2 | s4042·f0 | s4042·f1 | s4042·f2 | s5051·f0 | s5051·f1 | s5051·f2 | s6060·f0 | s6060·f1 | s6060·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 | 29 | 31 | 31 | 27 | 29 | 29 |
| purity_epsilon=0.02, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 | 27 | 29 | 31 | 27 | 31 | 25 |
| purity_epsilon=0.01, max_nodes=31 | 31 | 29 | 31 | 31 | 31 | 27 | 27 | 31 | 27 | 27 | 29 | 31 | 27 | 31 | 25 |
| purity_epsilon=0.05, max_nodes=63 | 61 | 55 | 61 | 59 | 59 | 51 | 59 | 57 | 59 | 61 | 63 | 61 | 59 | 59 | 53 |
| purity_epsilon=0.02, max_nodes=63 | 61 | 63 | 63 | 59 | 59 | 57 | 59 | 57 | 59 | 63 | 63 | 61 | 63 | 55 | 55 |
| purity_epsilon=0.01, max_nodes=63 | 61 | 63 | 63 | 59 | 59 | 57 | 59 | 57 | 59 | 63 | 63 | 61 | 63 | 55 | 55 |
| purity_epsilon=0.05, max_nodes=127 | 119 | 123 | 123 | 123 | 123 | 117 | 121 | 121 | 107 | 125 | 117 | 125 | 119 | 125 | 121 |
| purity_epsilon=0.02, max_nodes=127 | 123 | 127 | 125 | 123 | 123 | 119 | 125 | 125 | 111 | 121 | 121 | 119 | 125 | 121 | 125 |
| purity_epsilon=0.01, max_nodes=127 | 123 | 127 | 125 | 123 | 123 | 119 | 125 | 125 | 111 | 121 | 121 | 119 | 125 | 121 | 125 |
| purity_epsilon=0.05, max_nodes=255 | 245 | 235 | 249 | 249 | 249 | 251 | 241 | 249 | 241 | 249 | 247 | 243 | 247 | 253 | 249 |
| purity_epsilon=0.02, max_nodes=255 | 241 | 241 | 247 | 249 | 245 | 249 | 251 | 245 | 247 | 249 | 253 | 251 | 241 | 251 | 235 |
| purity_epsilon=0.01, max_nodes=255 | 241 | 241 | 247 | 249 | 245 | 249 | 249 | 245 | 247 | 249 | 253 | 251 | 241 | 251 | 235 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.02, max_nodes=31` — estado interno do tuning: **tuning_uncertain** (selection_probability_below_threshold).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **29.9%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.02, max_nodes=31` com probabilidade 29.9% — selecionada é a moda: **True**
- bootstrap_runner_up: `purity_epsilon=0.05, max_nodes=63` 26.9%; top1_top2_margin: 3.0%
- IC bootstrap 95% da fidelity média da escolhida: [0.954, 0.980]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 29.9%, `purity_epsilon=0.05, max_nodes=63` 26.9%, `purity_epsilon=0.05, max_nodes=31` 15.7%, `purity_epsilon=0.02, max_nodes=63` 10.4%, `purity_epsilon=0.02, max_nodes=127` 8.7%, `purity_epsilon=0.05, max_nodes=255` 4.8%, `purity_epsilon=0.05, max_nodes=127` 3.4%, `purity_epsilon=0.02, max_nodes=255` 0.2%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.05, max_nodes=255', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.05, max_nodes=31']
- saturação da escolhida: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 29, folhas 15, accuracy 0.9473684210526315, fidelity 0.9210526315789473; MLP accuracy 0.9210526315789473.

#### iris_t.arff — seed mestre 11 (treino 112, teste 38)

CV: RepeatedStratifiedKFold (conjunto de treino apenas); 3 dobras × 5 repetições (seeds [11, 1020, 2029, 3038, 4047]) = 15 partições; teste usado na seleção: **False**. Regra: lexicográfica: fidelity -> estabilidade da fidelity (se indistinguíveis) -> estabilidade estrutural -> complexidade -> desempate canónico (mais simples).

Oráculo congelado: `oracle_id=3a8a750475e18459` (Pipeline, construtor `factory`); Original e Reloaded consultaram o mesmo oráculo: **True** (ids {'trepan_original': '3a8a750475e18459', 'trepan_reloaded': '3a8a750475e18459'}); queries por escopo: {'tuning': 4009379, 'trepan_pair': 32024, '(sem escopo)': 114}; pesos inalterados no fim: True.

Política de orçamento: **non_binding** — orçamento comum a todos os candidatos: 85681 (mesmo para todos: True); consumo máximo observado: 41585; algum ajuste esgotou o orçamento: **False**.

| configuração | fid. média | desvio | pior partição | nós (média ± dp) | CV nós | prof. média (CV) | folhas média (CV) | instab. estrutural | fraction_at_node_cap (node_cap_reached/n) | max_nodes | censura pelo teto | query_budget | queries_used (média) | budget_exhausted | fraction_budget_exhausted | tempo/ajuste (s) | resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.938 | 0.053 | 0.811 | 28.6 ± 2.4 | 0.08 | 8.7 (0.14) | 14.8 (0.08) | 0.14 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4835 | 0/15 | 0% | 0.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.012, t=0.68 ≤ 1.76), mas desvio-padrão 0.053 > 0.034 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=31 | 0.950 | 0.036 | 0.865 | 29.0 ± 2.7 | 0.09 | 8.7 (0.19) | 15.0 (0.09) | 0.19 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4632 | 0/15 | 0% | 0.5 | **WINNER** — VENCE: fidelity média 0.950 (a melhor). 12 indistinguível(eis) em fidelity -> 8 após estabilidade da fidelity -> 8 após estabilidade estrutural (índice 0.19) -> 1 após complexidade. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=31 | 0.950 | 0.036 | 0.865 | 29.8 ± 1.7 | 0.06 | 8.9 (0.18) | 15.4 (0.05) | 0.18 | 100% (15/15) | 31 | ⚠ sim | 85681 | 4598 | 0/15 | 0% | 0.5 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 29.8 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=31): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=63 | 0.938 | 0.053 | 0.811 | 59.3 ± 3.4 | 0.06 | 13.6 (0.22) | 30.1 (0.06) | 0.22 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9931 | 0/15 | 0% | 1.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.012, t=0.68 ≤ 1.76), mas desvio-padrão 0.053 > 0.034 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=63 | 0.950 | 0.036 | 0.865 | 57.7 ± 4.8 | 0.08 | 12.6 (0.17) | 29.3 (0.08) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9610 | 0/15 | 0% | 1.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.7 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=63 | 0.950 | 0.036 | 0.865 | 57.3 ± 4.6 | 0.08 | 12.4 (0.17) | 29.1 (0.08) | 0.17 | 100% (15/15) | 63 | ⚠ sim | 85681 | 9577 | 0/15 | 0% | 1.1 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 57.3 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=63): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=127 | 0.938 | 0.053 | 0.811 | 120.7 ± 5.5 | 0.05 | 18.4 (0.21) | 60.9 (0.05) | 0.21 | 100% (15/15) | 127 | ⚠ sim | 85681 | 20139 | 0/15 | 0% | 2.1 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.012, t=0.68 ≤ 1.76), mas desvio-padrão 0.053 > 0.034 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=127 | 0.950 | 0.036 | 0.865 | 121.7 ± 4.2 | 0.03 | 17.6 (0.19) | 61.3 (0.03) | 0.19 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19574 | 0/15 | 0% | 2.4 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 121.7 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=127 | 0.950 | 0.036 | 0.865 | 121.5 ± 4.0 | 0.03 | 17.3 (0.17) | 61.3 (0.03) | 0.17 | 100% (15/15) | 127 | ⚠ sim | 85681 | 19492 | 0/15 | 0% | 2.3 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 121.5 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=127): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.05, max_nodes=255 | 0.936 | 0.051 | 0.811 | 245.9 ± 6.1 | 0.02 | 24.0 (0.22) | 123.5 (0.02) | 0.22 | 100% (15/15) | 255 | ⚠ sim | 85681 | 40572 | 0/15 | 0% | 4.5 | **LOST** — PERDE em estabilidade da fidelity: indistinguível em fidelity (Δ=0.014, t=0.78 ≤ 1.76), mas desvio-padrão 0.051 > 0.034 + tolerância 0.005. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.02, max_nodes=255 | 0.948 | 0.034 | 0.865 | 247.1 ± 5.4 | 0.02 | 23.3 (0.18) | 124.1 (0.02) | 0.18 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39505 | 0/15 | 0% | 4.6 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 247.1 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |
| purity_epsilon=0.01, max_nodes=255 | 0.948 | 0.034 | 0.865 | 246.2 ± 5.8 | 0.02 | 22.8 (0.18) | 123.6 (0.02) | 0.18 | 100% (15/15) | 255 | ⚠ sim | 85681 | 39430 | 0/15 | 0% | 4.7 | **LOST** — PERDE em complexidade: indistinguível em fidelity e estável (fidelity e estrutura), mas com 246.2 nós médios vs 29.0 de purity_epsilon=0.02, max_nodes=31. ⚠ estabilidade estrutural potencialmente CENSURADA pelo teto de nós (max_nodes=255): a baixa variância pode vir do limite, não dos dados; o CV de nós baixo não é evidência forte de estabilidade estrutural. |

Estabilidade da fidelity (entre dobras e entre seeds):

| configuração | fid. média | desvio (todas as partições) | dp intra-seed (entre dobras) | dp entre-seeds (médias por seed) | médias por seed |
|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 0.938 | 0.053 | 0.051 | 0.013 | 0.928, 0.929, 0.947, 0.929, 0.955 |
| purity_epsilon=0.02, max_nodes=31 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.01, max_nodes=31 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.05, max_nodes=63 | 0.938 | 0.053 | 0.051 | 0.013 | 0.928, 0.929, 0.947, 0.929, 0.955 |
| purity_epsilon=0.02, max_nodes=63 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.01, max_nodes=63 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.05, max_nodes=127 | 0.938 | 0.053 | 0.051 | 0.013 | 0.928, 0.929, 0.947, 0.929, 0.955 |
| purity_epsilon=0.02, max_nodes=127 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.01, max_nodes=127 | 0.950 | 0.036 | 0.038 | 0.008 | 0.946, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.05, max_nodes=255 | 0.936 | 0.051 | 0.049 | 0.015 | 0.919, 0.929, 0.947, 0.929, 0.955 |
| purity_epsilon=0.02, max_nodes=255 | 0.948 | 0.034 | 0.036 | 0.010 | 0.937, 0.938, 0.955, 0.956, 0.955 |
| purity_epsilon=0.01, max_nodes=255 | 0.948 | 0.034 | 0.036 | 0.010 | 0.937, 0.938, 0.955, 0.956, 0.955 |

Diagnóstico semântico/estrutural (NÃO entra na seleção): as árvores com o mesmo tamanho explicam o mesmo?

| configuração | feature da raiz (freq.) | partilha modal da raiz | Jaccard conj. de features | Jaccard splits (feature, limiar) | pares de igual tamanho | Jaccard features (igual tamanho) | Jaccard splits (igual tamanho) | fração de stumps (≤3 nós) | features mais usadas |
|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | petal_length 100%, petal_width 20% | 100% | 0.94 | 0.39 | 31 | 0.95 | 0.37 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 87% |
| purity_epsilon=0.02, max_nodes=31 | petal_length 100%, petal_width 20% | 100% | 0.91 | 0.42 | 30 | 0.93 | 0.41 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 80% |
| purity_epsilon=0.01, max_nodes=31 | petal_length 100%, petal_width 20% | 100% | 0.94 | 0.42 | 42 | 0.92 | 0.43 | 0% | petal_length 100%, petal_width 100%, sepal_width 100%, sepal_length 87% |
| purity_epsilon=0.05, max_nodes=63 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.41 | 18 | 1.00 | 0.44 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=63 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.46 | 17 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=63 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.47 | 19 | 1.00 | 0.45 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=127 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.47 | 11 | 1.00 | 0.46 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=127 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.51 | 13 | 1.00 | 0.55 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=127 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.51 | 12 | 1.00 | 0.56 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.05, max_nodes=255 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.47 | 9 | 1.00 | 0.47 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.02, max_nodes=255 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.52 | 11 | 1.00 | 0.54 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |
| purity_epsilon=0.01, max_nodes=255 | petal_length 100%, petal_width 20% | 100% | 1.00 | 0.52 | 17 | 1.00 | 0.53 | 0% | petal_length 100%, petal_width 100%, sepal_length 100%, sepal_width 100% |

Fidelity por partição (repetição/seed · dobra):

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 1.000 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 | 0.842 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=31 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.01, max_nodes=31 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.05, max_nodes=63 | 1.000 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 | 0.842 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=63 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.01, max_nodes=63 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.05, max_nodes=127 | 1.000 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 | 0.842 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=127 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.01, max_nodes=127 | 1.000 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.05, max_nodes=255 | 0.974 | 0.973 | 0.811 | 0.921 | 0.946 | 0.919 | 0.921 | 1.000 | 0.919 | 0.842 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.02, max_nodes=255 | 0.974 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |
| purity_epsilon=0.01, max_nodes=255 | 0.974 | 0.973 | 0.865 | 0.921 | 0.973 | 0.919 | 0.947 | 1.000 | 0.919 | 0.921 | 0.973 | 0.973 | 0.947 | 0.946 | 0.973 |

Nós por partição:

| configuração | s11·f0 | s11·f1 | s11·f2 | s1020·f0 | s1020·f1 | s1020·f2 | s2029·f0 | s2029·f1 | s2029·f2 | s3038·f0 | s3038·f1 | s3038·f2 | s4047·f0 | s4047·f1 | s4047·f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| purity_epsilon=0.05, max_nodes=31 | 29 | 31 | 27 | 31 | 31 | 31 | 27 | 31 | 31 | 27 | 29 | 27 | 23 | 27 | 27 |
| purity_epsilon=0.02, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 29 | 27 | 21 | 27 | 27 |
| purity_epsilon=0.01, max_nodes=31 | 29 | 31 | 31 | 31 | 31 | 31 | 29 | 31 | 29 | 31 | 31 | 27 | 31 | 27 | 27 |
| purity_epsilon=0.05, max_nodes=63 | 57 | 57 | 59 | 53 | 63 | 63 | 59 | 55 | 63 | 63 | 61 | 59 | 59 | 55 | 63 |
| purity_epsilon=0.02, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 | 61 | 61 | 45 | 61 | 55 | 63 |
| purity_epsilon=0.01, max_nodes=63 | 57 | 57 | 57 | 53 | 63 | 63 | 57 | 55 | 57 | 61 | 59 | 45 | 57 | 55 | 63 |
| purity_epsilon=0.05, max_nodes=127 | 119 | 113 | 123 | 113 | 123 | 123 | 125 | 127 | 121 | 109 | 121 | 127 | 117 | 123 | 127 |
| purity_epsilon=0.02, max_nodes=127 | 119 | 113 | 119 | 115 | 127 | 123 | 121 | 127 | 123 | 123 | 121 | 125 | 127 | 123 | 119 |
| purity_epsilon=0.01, max_nodes=127 | 119 | 113 | 119 | 115 | 127 | 123 | 121 | 127 | 123 | 123 | 121 | 125 | 125 | 123 | 119 |
| purity_epsilon=0.05, max_nodes=255 | 249 | 251 | 241 | 233 | 253 | 237 | 251 | 245 | 245 | 253 | 245 | 243 | 249 | 241 | 253 |
| purity_epsilon=0.02, max_nodes=255 | 251 | 251 | 251 | 245 | 255 | 237 | 247 | 251 | 243 | 247 | 249 | 237 | 243 | 247 | 253 |
| purity_epsilon=0.01, max_nodes=255 | 251 | 251 | 251 | 245 | 251 | 237 | 247 | 251 | 243 | 247 | 247 | 237 | 235 | 247 | 253 |

**Configuração escolhida pela CV completa:** `purity_epsilon=0.02, max_nodes=31` — estado interno do tuning: **tuning_stable** (ok).

Expansão adaptativa da capacidade (só CV do treino; mesmas dobras):

- initial_node_grid: [31, 63]; final_node_grid: [31, 63, 127, 255]; capacity_expansion_rounds: 2; expansion_triggered: True; expansion_stop_reason: `safety_limit_reached`; fraction_at_node_cap (vencedora final): 100%
- ronda 1: 63 → 127 nós (saturadas: ['purity_epsilon=0.05, max_nodes=63', 'purity_epsilon=0.02, max_nodes=63', 'purity_epsilon=0.01, max_nodes=63'])
- ronda 2: 127 → 255 nós (saturadas: ['purity_epsilon=0.05, max_nodes=127', 'purity_epsilon=0.02, max_nodes=127', 'purity_epsilon=0.01, max_nodes=127'])

Robustez interna da política (bootstrap de blocos das repetições; teste não usado):

- bootstrap_method: **exact**; bootstrap_samples: 3125; avaliações distintas: 126 (blocos = 5 repetições × 3 dobras)
- selected_config_probability (escolhida pela CV completa): **82.1%** (exato: sem erro Monte-Carlo)
- bootstrap_modal_config: `purity_epsilon=0.02, max_nodes=31` com probabilidade 82.1% — selecionada é a moda: **True**
- bootstrap_runner_up: `purity_epsilon=0.01, max_nodes=31` 6.7%; top1_top2_margin: 75.4%
- IC bootstrap 95% da fidelity média da escolhida: [0.943, 0.956]
- distribuição das configurações escolhidas: `purity_epsilon=0.02, max_nodes=31` 82.1%, `purity_epsilon=0.01, max_nodes=31` 6.7%, `purity_epsilon=0.01, max_nodes=63` 4.4%, `purity_epsilon=0.01, max_nodes=255` 2.7%, `purity_epsilon=0.05, max_nodes=31` 1.7%, `purity_epsilon=0.02, max_nodes=63` 1.0%, `purity_epsilon=0.01, max_nodes=127` 1.0%, `purity_epsilon=0.05, max_nodes=127` 0.3%, `purity_epsilon=0.02, max_nodes=255` 0.2%
- vencedora por repetição (apenas informativo; 1 repetição = 3 dobras): ['purity_epsilon=0.02, max_nodes=255', 'purity_epsilon=0.05, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.02, max_nodes=31', 'purity_epsilon=0.01, max_nodes=63']
- saturação da escolhida: 15/15 árvores no teto max_nodes=31 (fraction_at_node_cap 100%); structural_stability_censored: **True** (evidência estrutural: censored_by_node_cap)

Teste (usado uma só vez, depois da escolha): TREPAN Original — nós 23, folhas 12, accuracy 0.8947368421052632, fidelity 0.9210526315789473; MLP accuracy 0.9736842105263158.
