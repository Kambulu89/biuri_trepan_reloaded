# Doc. 61 — Decisões experimentais H2: limiares, contrato de comparação entre pools, revisores, custos e auditoria de CI/ramo

Estado: **proposta para decisão**; nada congelado; campanhas H2-P, H2-R e H1 **não executadas e não autorizadas**; `core/` intocado; sem merge. As análises desta fase são **estruturais** (contagens de candidatos, consultas, nós, ganho de informação sobre as etiquetas do oráculo no treino): **nenhuma accuracy, fidelidade a rótulos reais ou teste foi calculada ou consultada**. Os axiomas German Credit são rascunhos de IA não validados: não são conhecimento de domínio cientificamente confirmado.

## 1. Auditoria de CI, ramo e `main`
* **`main` = `f1b2955`** (confirmado por `git fetch`). Contém: B3.0–B3.3, correção P3, B3-Q, preparação H2/B3.4 v2 (Docs. 29–52; commits até `21aedf2`). **Não contém**: `557c381` (Docs. 53–57, adaptador T1/T2, auditoria, fichas) nem `d1121a4` (Docs. 58–61, arnês) nem o trabalho desta fase — ficam **só no ramo** `claude/repository-architecture-analysis-s1vl5a`. Sem novos merges.
* **CI de `f1b2955` (main)**: agnosticism guard ✔, Windows scientific regression ✔ (3.11 e 3.12). **CI de `557c381`**: guard ✔, Windows ✔. **CI de `d1121a4`**: guard ✔; **Windows 3.11 ✘ e 3.12 ✘ (falha do job)**. Causa no 3.11 (log lido): 1 teste novo falhou em 1619 (`test_h2_sources_inventory::test_german_inventory_is_factual…`) porque comparava o SHA-256 **bruto** de `german.data` (no Windows o checkout converte LF→CRLF e o hash difere); os restantes 1618 passaram (40 ignorados), incluindo equivalência, metamórficos de dataset desconhecido, identidade dos baselines e guard. O ficheiro de dados **não** foi alterado: corrigiu-se o teste para comparar o hash normalizado CRLF→LF (`file_sha256_lf_normalized`). O job 3.12 (log lido) falhou pelo mesmo e único teste (1 falhado, 1618 passaram, 40 ignorados).
* **`core/`**: nenhuma alteração desde `f1b2955`; a única do programa B3 é `core/ontology_quality.py` (`6540bd7`, P3, isolada e autorizada). Âncoras e evidência congelada intactas (manifests v3, freezes B32/B33, ontologia congelada do German, `german.data`).

## 2. Limiares das portas: sensibilidade e critérios de seleção (nada escolhido nem congelado)
Valores candidatos: Δ_o=0,05, κ=0,375, τ_s=0,8, τ_u=0,5. **Dependência do estimador genérico** (taxa de acerto; 20 repetições; sementes 9800; sintético):

| célula | corr | L1_C0.3 | L1_CV | mutual_info | random_forest | best_of_all |
|---|---|---|---|---|---|---|
| s2.0_n60 | 0.57±0.03 | 0.58±0.04 | 0.47±0.05 | 0.25±0.04 | 0.50±0.03 | 0.64±0.03 |
| s2.0_n120 | 0.78±0.03 | 0.88±0.03 | 0.84±0.04 | 0.30±0.04 | 0.73±0.04 | 0.89±0.03 |
| s2.0_n240 | 0.93±0.02 | 0.99±0.01 | 0.99±0.01 | 0.37±0.04 | 0.89±0.02 | 0.99±0.01 |
| s3.0_n60 | 0.43±0.04 | 0.47±0.05 | 0.39±0.05 | 0.22±0.03 | 0.43±0.03 | 0.56±0.03 |
| s3.0_n120 | 0.60±0.03 | 0.64±0.03 | 0.62±0.03 | 0.17±0.04 | 0.59±0.04 | 0.69±0.03 |
| s3.0_n240 | 0.86±0.03 | 0.91±0.02 | 0.91±0.02 | 0.23±0.04 | 0.82±0.04 | 0.93±0.02 |

Tabela de decisões da porta G2s (nº de 6 células que «passam», isto é, o método genérico NÃO resolve a tarefa; regra pontual `média ≤ τ` / conservadora `média + 1,645·EP ≤ τ`; fonte `threshold_sensitivity.json`):

| estimador | τ=0.5: pontual / conservadora | τ=0.6: pontual / conservadora | τ=0.7: pontual / conservadora | τ=0.8: pontual / conservadora | τ=0.9: pontual / conservadora |
|---|---|---|---|---|---|
| corr | 1 / 1 | 3 / 1 | 3 / 3 | 4 / 3 | 5 / 5 |
| L1_C0.3 | 1 / 0 | 2 / 1 | 3 / 3 | 3 / 3 | 4 / 3 |
| L1_CV | 2 / 1 | 2 / 2 | 3 / 3 | 3 / 3 | 4 / 3 |
| mutual_info | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 |
| random_forest | 2 / 1 | 3 / 2 | 3 / 3 | 4 / 4 | 6 / 5 |
| best_of_all | 0 / 0 | 1 / 0 | 3 / 2 | 3 / 3 | 4 / 3 |

Em τ = 0,8 a decisão **muda com o estimador** nas células `s2.0_n120, s2.0_n240, s3.0_n240`. Informação mútua nunca «resolve» (acerto ≤ 0,37) e tornaria a porta vazia; `best_of_all` é o mais exigente.
**Critérios de seleção independentes dos rótulos de teste (proposta):** (i) o estimador de referência é o **melhor estimador genérico razoável** (união pré-declarada: correlação, L1, L1-CV, floresta, informação mútua — o máximo por célula), fixado antes de qualquer avaliação de variantes OWL; (ii) a regra de decisão é a **conservadora** (limite superior ≤ τ), para não depender do ruído de 20 repetições; (iii) toda a calibração usa **dados sintéticos de semente independente** e, em dados reais, apenas treino e validação interna — nunca o teste externo nem desempenho de qualquer ontologia; (iv) cada limiar tem sensibilidade tabulada e é revisto se a decisão mudar com o estimador; (v) τ, κ, Δ_o e τ_u são decididos pelo responsável pela metodologia e registados em commit próprio antes da fase confirmatória.
**Limitações:** 20 repetições (EP ≈ 0,03–0,05); uma só família sintética (R8); os limiares dependem do gerador e não se transferem automaticamente para dados reais. **Falsos positivos** (a porta passa e o genérico resolvia): estimador de referência fraco. **Falsos negativos** (recusa tarefas em que a ontologia ajudaria): estimador demasiado forte/τ baixo.

## 3. Contrato de comparação entre pools
**Estratégias** (mesmas instâncias, **mesmo MLP/oráculo**, mesmos parâmetros exceto as vagas): **A** pool limitado do Original (12 vagas por variância); **B** pool alargado genericamente (`+r` vagas pelas colunas base seguintes na variância, sem OWL; `r`=nº de literais derivados=10 ⇒ 22 vagas); **C** pool com priorização por relações ontológicas verdadeiras (as `r` derivadas ocupam `r` vagas extra: 12 de base por variância + 10 derivadas = 22; critério independente de rótulos); **controlos**: R = `r` colunas base aleatórias (semente fixa) em vagas extra; P = derivadas da ontologia **permutada** (H); I = derivadas de relações **aleatórias** (I); **A+** = representação enriquecida sem vagas extra (isola a representação); **B_MATCHED** = B com `mf` escolhido por **contagem** de candidatos m-of-n (≥ os de C), sem desempenho (mf = 21, 19, 22 nos holdouts 42, 43, 44).
**Igualado por construção:** instâncias, oráculo, `max_queries`, `max_nodes`, `max_depth`, `max_n`, `beam_width`, `min_samples_leaf`, semente, nº de vagas (A↔A+; B, C, R, P, I a 22). **Reportado por braço:** candidatos disponíveis (gerados), examinados (avaliados), consultas ao MLP, nós, profundidade, literais.
**Regra:** um contraste é `COMPARABLE` se as condições estruturais valem e se, nas quantidades igualadas, a diferença relativa ≤ tolerância (**proposta, não congelada**: vagas 0; m-of-n gerados/avaliados, consultas e nós 10 %; profundidade 0); caso contrário `NOT_COMPARABLE` com as dimensões violadas. Como as binárias geram poucos limiares e as numéricas muitos, «vagas iguais» **não** garante candidatos iguais — por isso B_MATCHED.
**Resultados (holdouts 42, 43, 44; estruturais):**

| Braço | vagas de features | s42 (m-of-n gerados/avaliados; consultas; nós internos) | s43 (m-of-n gerados/avaliados; consultas; nós internos) | s44 (m-of-n gerados/avaliados; consultas; nós internos) |
|---|---|---|---|---|
| A | 12 | 408/392; 0; 3 | 412/411; 27; 3 | 336/329; 0; 2 |
| A_PLUS_ENRICHED | 12 | 424/371; 37; 2 | 420/418; 27; 3 | 176/171; 0; 1 |
| B_WIDE_GENERIC | ver texto | 864/821; 0; 3 | 1008/995; 57; 3 | 688/624; 21; 3 |
| C_ONTOLOGY_PRIORITY | ver texto | 800/758; 0; 3 | 840/786; 127; 3 | 688/674; 0; 3 |
| R_RANDOM_BASE | ver texto | 848/811; 0; 3 | 836/813; 16; 3 | 908/879; 0; 2 |
| P_PERMUTED_ONTOLOGY | ver texto | 844/813; 0; 3 | 848/828; 16; 3 | 812/788; 0; 3 |
| I_RANDOM_RELATIONS | ver texto | 964/936; 0; 3 | 824/788; 13; 3 | 768/714; 19; 2 |
| B_MATCHED_CANDIDATES | ver texto | 810/769; 0; 3 | 806/789; 16; 3 | 688/624; 21; 3 |

(* vagas: A, A+ = 12; B, C, R, P, I = 22; B_MATCHED = 19–22.)

| Contraste | s42 | s43 | s44 |
|---|---|---|---|
| C_ONTOLOGY_PRIORITY vs B_WIDE_GENERIC | COMPARABLE | NOT_COMPARABLE mofn_generated, mofn_evaluated, queries | NOT_COMPARABLE queries |
| C_ONTOLOGY_PRIORITY vs B_MATCHED_CANDIDATES | COMPARABLE | NOT_COMPARABLE queries | NOT_COMPARABLE queries |
| C_ONTOLOGY_PRIORITY vs R_RANDOM_BASE | COMPARABLE | NOT_COMPARABLE queries | NOT_COMPARABLE mofn_generated, mofn_evaluated, n_internal_nodes, depth |
| C_ONTOLOGY_PRIORITY vs P_PERMUTED_ONTOLOGY | COMPARABLE | NOT_COMPARABLE queries | NOT_COMPARABLE mofn_generated, mofn_evaluated |
| C_ONTOLOGY_PRIORITY vs I_RANDOM_RELATIONS | NOT_COMPARABLE mofn_generated, mofn_evaluated | NOT_COMPARABLE queries | NOT_COMPARABLE mofn_generated, queries, n_internal_nodes, depth |
| C_ONTOLOGY_PRIORITY vs A | NOT_COMPARABLE mofn_generated, mofn_evaluated | NOT_COMPARABLE mofn_generated, mofn_evaluated, queries | NOT_COMPARABLE mofn_generated, mofn_evaluated, n_internal_nodes, depth |

**Leitura honesta:** a comparabilidade depende do holdout: C vs B_WIDE só é `COMPARABLE` em 1 de 3 holdouts (nos outros as consultas ao MLP ou os candidatos diferem > 10 %); as consultas dependem da forma da árvore e, com 750 linhas de treino e `min_sample`=300, são quase 0 em muitos casos, por isso a tolerância relativa a contagens pequenas é frágil (propõe-se tolerância absoluta+relativa e **tetos de consultas/candidatos idênticos** como parte do desenho, para que a comparabilidade seja por construção e não observada a posteriori). C vs A é sempre `NOT_COMPARABLE` (C tem mais vagas e candidatos: é exatamente o contraste «mais oportunidade»). **Conclusão do contrato:** nenhum ganho de C sobre A pode ser atribuído à ontologia; só C vs B_MATCHED/R/P/I, quando `COMPARABLE`, isola a origem das vagas.
**Formabilidade pelo Original:** o teste equivalente aos literais é formável a partir do pool de literais em A: [0, 0, 0]; B: [1, 1, 1]; B_MATCHED: [1, 0, 1]; C: [0, 0, 0] literais (de 10) por holdout.
**Interpretação do traço C ≡ F:** a igualdade de traços não conclui inutilidade da ontologia: mostra ausência de **oportunidade de pesquisa** (etapa 1, variância) para 8/10 literais (Doc. 59 §5). Com vagas reservadas (C) passam a ser avaliados 4–8 dos 10 literais (consoante o holdout) e **1** (`AtLeast_savings_account_A63`) altera a decisão de split em 3/3 holdouts (efeito de ganho de informação 0,0006–0,005), sem qualquer evidência sobre accuracy ou fidelidade. Não se classifica nenhuma ontologia como benéfica pela geração dos literais.

## 4. Revisores (pendentes)
Três papéis — especialista de domínio, engenheiro de ontologias, responsável pela metodologia — **sem nomes**; formulários em `…/v0_draft/review_forms/` com nome, data, assinatura e decisão vazios. Nada inventado. Enquanto pendentes: as 14 fichas são hipóteses; qualquer execução é rotulada «rascunho de IA não validado» (verificação de mecanismo). Ontologias sintéticas de verdade conhecida são permitidas em testes técnicos.

## 5. Bloqueios pendentes
(1) Ficheiros S1–S3 (rede bloqueada ou fornecer); (2) revisões independentes; (3) decisão sobre estimador de referência e limiares; (4) tolerâncias e tetos do contrato de pools; (5) decisão sobre o desenho de pool para a fase R; (6) CI Windows verde do commit com a correção.

## 6. Plano de execução e custo (CPU-h; nada autorizado)
| Etapa | Estimativa | Condição |
|---|---|---|
| Análise de pool desta fase (3 holdouts, 8 braços + 20 ablações) | ≈ 16 s por holdout (medido), < 0,02 total | feita |
| Fase R (German, A–J) | ≈ 0,7 | bloqueada por S1–S3, revisões |
| Braços de pool (A, A+, B, C, R, P, I, B_MATCHED + ablação, 20 holdouts) | ≈ 0,12 (extrapolado dos 16 s) | só com tolerâncias aprovadas |
| **Fase R com pools** | **≈ 0,85 (limite proposto 1,0)** | não autorizada |
| Fase P (sintético) | ≈ 1,5 | portas aprovadas |
| H1-B3.4 | 7,2 | não autorizada |
Sequência proposta: S1–S3 → reavaliar fichas → revisão humana → decidir limiares/tolerâncias → pré-registo em commit separado (sementes ≥ 8000, braços, contrastes, critérios) → só então executar R. Hierarquia conjunta: `Acc(Reloaded_OWL) > Acc(Original) > Acc(C4.5)` e `Fid(Reloaded_OWL) > Fid(Original)`, `> Fid(C4.5)`, nas mesmas instâncias e contra o mesmo MLP, atribuível a conhecimento ontológico verdadeiro (controlos permutado/aleatório/convencional) e **sem** mais consultas ou baselines enfraquecidos — continua um **objetivo experimental**, não demonstrado.

## 7. Verificações desta fase
Guard de agnosticismo: OK. Testes H2 (harness, fontes/inventário, auditoria, rastreio, contrato, pool) e contrato de agnosticismo: passam localmente. CI do commit com a correção: por confirmar após o push.
