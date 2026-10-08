# TREPAN Reloaded — proposta técnica e experimental (fase de diagnóstico e planeamento)

**Estado: proposta.** Nesta fase **não se altera o algoritmo nem hiperparâmetros**; nada aqui foi ajustado com base em dados de teste. O objectivo não
é fazer o Reloaded superar os baselines, mas definir um método cuja melhoria, se existir, seja demonstrável e estatisticamente defensável. TREPAN
Original e C4.5 permanecem baselines independentes (código, tuning e oráculo inalterados). Base empírica: `docs/ONTOLOGY_ISOLATION_AUDIT.md` e
`docs/evidence/ontology_isolation/` (duas ontologias, um dataset, 5 seeds, configuração leve: exploratório, **não** confirmatório).

## 1. Diagnóstico (evidência observada)

| # | Observação | Fonte |
|---|---|---|
| D1 | `semantic_bonus` > `information_gain` em 30/30 nós auditados; mediana do bónus 0,90 vs. ganho de informação 0,21 (razão mediana 3,9; 1,3–9,6). O bónus domina a pontuação `score = IG + bónus` fora da raiz. | `split_audit_allnodes.json` |
| D2 | 10/30 nós têm decisão alterada pela semântica; 3/10 com ganho local de fidelidade **negativo**; 5 aceites por «coerência semântica dentro da tolerância» (0,015), 5 por ganho local+treino. 19/30 rejeitados por falta de ganho local. | idem |
| D3 | O limiar `error_focus_min_local_fidelity_gain = 0,002` está abaixo da resolução (1/143 ≈ 0,007 com 143 amostras): na prática «qualquer ganho > 0». | idem |
| D4 | Efeito agregado em fidelidade: +0,14 pp / −0,42 pp, DP ≈ 2,5–2,9 pp, 5 seeds: indistinguível de 0. | auditoria §7 |
| D5 | Das 60 features derivadas, 20 são combinações lineares das originais (inúteis para MLP), 40 não lineares; **60/60 replicáveis só pelos nomes das colunas**. A ontologia formaliza o que os nomes já dizem. | `derived_features_audit_*.csv` |
| D6 | MLP com features OWL vs. originais: +0,63 / +0,56 pp, IC95% corrigido ⊃ 0; controlo embaralhado −1,6/−2,0 pp. | `paired_cv_predictive.json` |

**Conclusão do diagnóstico:** não há evidência de ganho atribuível ao conhecimento ontológico; há evidência de que o mecanismo de selecção é
dominado por um termo de escala arbitrária e aceita divisões sem ganho de fidelidade mensurável. Isto é uma limitação do desenho de selecção
testável, não prova de que a ontologia seja inútil.

## 2. Hipóteses verificáveis (pré-registadas; cada uma com critério de falsificação)

- **H1 (escala).** O bónus não calibrado distorce a selecção. *Predição:* com bónus normalizado (ver B1), a fração de divisões alteradas com ganho local < 0 cai para ≈ 0 sem perda de fidelidade. *Falsifica:* fidelidade cai > 1 pp (margem TOST) ou nada muda.
- **H2 (ganho negativo).** Exigir ganho local ≥ ruído (limiar acima da resolução, derivado do tamanho da amostra) melhora fidelidade em dados retidos. *Falsifica:* ΔFidelidade IC95% ⊂ [−1, +1] pp (equivalência) ou negativo.
- **H3 (multiobjectivo).** Seleccionar divisões pela fronteira de Pareto (fidelidade, accuracy, complexidade, coerência semântica) com regra de desempate fixada *a priori* reduz complexidade a fidelidade equivalente. *Falsifica:* sem redução de nós a fidelidade não-inferior.
- **H4 (m-of-n semântico).** Candidatos m-of-n gerados a partir de grupos ontológicos têm fidelidade ≥ m-of-n gerados por grupos aleatórios do mesmo tamanho (controlo negativo). *Falsifica:* diferença ≤ 0.
- **H5 (conhecimento vs. engenharia).** Com **nomes opacos** (`f1…fn`), a ontologia é a única fonte do agrupamento; se o ganho existir, aparece só aí (E3−E2 > 0). Com nomes informativos, espera-se E3−E2 ≈ 0 (D5). *Falsifica:* ganho idêntico com nomes opacos e ontologia embaralhada.
- **H6 (nulo).** Sem estrutura relacional/hierárquica nos dados, nenhuma variante deve diferir de Original (o método não pode «inventar» ganho).

## 3. Intervenções candidatas (a implementar só depois de aprovado o plano; cada uma atrás de flag, desligada por omissão)

- **B1 — calibração do bónus:** (a) rank-normalização por nó (bónus e IG para [0,1] sobre o conjunto de candidatos do nó); (b) bónus limitado a ≤ λ·IG com λ escolhido **só por validação interna aninhada** (nunca teste); (c) bónus como desempate lexicográfico apenas entre candidatos com IG dentro de ε estatístico do melhor.
- **B2 — gate de ganho:** limiar = max(1/n_amostra, erro-padrão binomial do ganho local), derivado dos dados, não de uma constante; rejeitar ganho local < 0 sempre.
- **B3 — selecção multiobjectivo:** vector (fidelidade local+real, accuracy, −complexidade, coerência); conjunto não dominado; escolha por regra única congelada (ex. máx. fidelidade, depois mín. complexidade, depois coerência).
- **B4 — m-of-n fundamentado:** geração a partir de grupos OWL (papel/família/hierarquia) com orçamento fixo; proveniência no audit; controlo com grupos permutados.
- Todas genéricas: usam apenas estrutura (grupos, papéis, contagens), nunca identidade de dataset (contrato de agnosticismo mantido).

## 4. Plano de ablação reproduzível

Estende o §8 da auditoria. Splits, oráculo congelado (`FrozenOracle`) e seeds mestras **pareados** em todos os braços.

| Braço | Descrição |
|---|---|
| A0 C4.5 canónico · A1 TREPAN Original | baselines independentes, inalterados |
| E1 Reloaded-core (sem semântica) | efeito do algoritmo |
| E2 Original + engenharia por nomes | engenharia convencional |
| E3 Original + features OWL | OWL vs. convencional (mesmo algoritmo) |
| E4 Reloaded actual (bónus não calibrado) | estado actual |
| E4+B1 / +B2 / +B3 / +B4 / +B1B2B3B4 | efeito marginal de cada intervenção |
| E5 ontologia embaralhada · E5' grupos aleatórios | controlos negativos |
| E8 nomes opacos × {E2,E3,E4,melhor variante} | cenário decisivo (H5) |
| E9 sem estrutura (features independentes) | H6 |

Protocolo: (1) **manifest novo congelado e hasheado antes de correr** (datasets, seeds, braços, métricas, regra de decisão); (2) selecção de
qualquer parâmetro (λ, ε) por **CV interna aninhada no treino**, o teste externo só lido uma vez por unidade; (3) um World OWL isolado por
unidade e processo-por-unidade preferível; (4) resultados brutos imutáveis, tabelas reconstruídas deles; (5) equivalência exacta das optimizações
(`equivalence_report.py`); (6) resultados nulos e negativos publicados.

## 5. Datasets, seeds, métricas e estatística

- **Datasets:** conjunto fixado *antes* de ver resultados, ≥ 8, estruturas diversas (famílias estatísticas, hierarquias, categóricas, sem estrutura, multi-classe, alta dimensão), incluindo datasets desconhecidos do desenvolvimento e ontologias escritas/validadas **antes** dos resultados (portão de qualidade existente). Acrescentar dataset = dados + `DatasetSpec`, nunca algoritmo.
- **Seeds/partições:** ≥ 5 seeds mestras × CV 5×3 conforme manifest vigente; unidade = (dataset × seed × partição). Para detectar 1 pp com DP ≈ 2,7 pp (80%, α=0,05): ≈ 57 unidades independentes — o desenho deve exceder isso por braço de interesse.
- **Métricas:** primária = fidelidade ao oráculo em dados retidos; secundárias = accuracy, balanced accuracy, nº de nós/profundidade, coerência semântica, estabilidade entre seeds.
- **Inferência:** t reamostrado corrigido (Nadeau–Bengio) por dataset; Wilcoxon entre datasets com Holm (Demšar); bootstrap IC95%; TOST ±1 pp para declarar «sem efeito»; tamanhos de efeito reportados; correcção de múltiplas comparações sobre a família completa de braços.
- **Sem contaminação do teste:** nenhuma decisão (limiares, λ, escolha de variante, filtros de features) usa o teste externo; auditorias de nós usam apenas treino/validação; hashes de partição registados; teste de «permutação de rótulos» como controlo.

## 6. Critérios de decisão (go / no-go) congelados

- **Go** para uma variante apenas se: ΔFidelidade vs. Original com IC95% inferior > 0 em agregação por dataset (Holm) **e** não-inferior em accuracy (TOST ±1 pp) **e** controlo embaralhado/aleatório não reproduz o ganho **e** ganho mantido com nomes opacos (para atribuir à ontologia).
- **Sem efeito** se IC ⊂ [−1,+1] pp: reporta-se equivalência; a funcionalidade mantém-se como opção documentada sem alegação de ganho.
- **No-go/rever** se a variante piora fidelidade além da margem, ou se o ganho só existe com nomes informativos (então é engenharia convencional, não conhecimento ontológico).

## 7. Riscos e limites

Potência insuficiente com poucos datasets; ontologias escritas após conhecer os dados (viés) — mitigar com ontologias congeladas antes; interacção bónus×oráculo; custo computacional (World novo por unidade). Os números do §1 são exploratórios (config. leve) e **não** substituem o protocolo congelado.

## 8. Próximos passos (sem tocar no algoritmo)

1. Aprovação desta proposta e da lista de datasets/ontologias; 2. redigir e hashear o manifest; 3. repetir com código corrigido as unidades `NOT_VERIFIED_OWL_ISOLATION` que se queira citar; 4. implementar B1–B4 atrás de flags (fase seguinte, com revisão); 5. executar a ablação; 6. relatório com decisão go/no-go.
