# Documento 6 — Plano de implementação (PROPOSTA)

Esforço em **dias de trabalho de engenharia** (estimativas grosseiras, a refinar após o Estágio 0; não incluem tempo de execução de benchmarks). Cada fase só começa com a anterior
**aprovada**, CI verde no commit exato, `legacy` bit-a-bit equivalente, agnosticismo verde. Nenhum PR/merge sem pedido explícito.

| Fase | Conteúdo | Depende de | Esforço | Critério de conclusão |
|---|---|---|---|---|
| **0** (esta) | Auditoria, protocolo, arquitetura, ablação, contrato, plano | — | feito | **tua aprovação** do protocolo e das decisões D-1…D-8 |
| **0.5** | Preparar CONFIRM: lista de ≥ 12 datasets, ontologias às cegas, manifest v2 (rascunho sem resultados), emendas a CLAUDE.md (D-1/D-2) | 0 | 2–4 (+ tempo humano das ontologias) | manifest v2 com hashes; lista aprovada por ti |
| **1** | Observabilidade (`candidate_table_`, campos estatístico/semântico) + `hierarchy_contract` + `reloaded_policies` (legacy) + **B1** (3 variantes) | 0.5 | 5–7 | testes: propriedades B1 (nunca escolhe candidato `s_i < s*−τ`; limite `λ/(1+λ)`), equivalência `legacy`, métricas sem arredondamento, estados da hierarquia; relatório DEV (não confirmatório) |
| **2** | **B2** (cross-fitting + LCB, `n_eff`, poda 1-SE opcional, registo por split) | 1 | 4–6 | testes: rejeita ganho negativo confirmado; calibração por simulação (taxa de aceitação falsa ≤ α); `insufficient_evidence`; sem fuga de teste |
| **3** | **B3** (`select_reloaded_config`, MLP por dobra interna, regra lexicográfica, `SelectionRecord`, Modo B/γ, `config_frozen_hash`) | 1,2 | 6–8 | testes: seleção determinística, nunca toca o teste, fallback declarado, Modo B rotulado; custo medido (`perf_measure`) |
| **4** | **B4** (`GroupProvider` semantic/random/shuffled/conv, igualação de orçamento de candidatos) | 1 | 3–5 | testes: mesmo nº e distribuição de tamanhos nos controlos; sem nomes de atributos; agnosticismo estendido |
| **5** | Benchmark: braços do Documento 4 no runner, estatística do Documento 2, quadro comparativo, nomes opacos + sidecar (D-6) | 1–4 | 5–7 + execução | Estágios 0→3 do Doc. 4; relatórios reproduzíveis a partir de brutos; `equivalence_report` |
| **6** | Integração GUI (diagnóstico sem alterar geometria), docs, regressão total, agnosticismo, CI | 5 | 3–5 | opções B1–B4 **desligadas por omissão**; testes GUI; PR aprovado por ti |

## Dependências técnicas críticas
* Fase 3 precisa do MLP reajustado por dobra (reutiliza `fit_mlp`, `inner_folds`, `freeze_oracle`); custo ≈ `K × |grelha| × (MLP + árvore)` por unidade.
* Fase 5 precisa dos datasets/ontologias CONFIRM (Fase 0.5). Sem eles só há resultados DEV — **não confirmatórios**.
* B2 e B3 partilham o registo de decisões; B4 é independente de B2/B3 mas os controlos exigem o orçamento igualado.

## Riscos e mitigação
| Risco | Mitigação |
|---|---|
| Nenhuma variante supera o Original (resultado plausível dado o diagnóstico) | declarado desde já; relatório de falha com decomposição por mecanismo; sem retoque pós-resultado |
| Potência insuficiente (MDE ≈ 1,2–2,4 pp com N=12) | declarar; aumentar N de datasets antes de correr, não depois |
| Custo computacional (nested + ~25 braços × 120 unidades) | medir no Estágio 0; reduzir pela ordem de prioridade do Doc. 4 antes de correr |
| «Winner's curse» na seleção por validação | medir `generalization_gap`; teste externo único; relatar |
| Overfitting ao DEV | DEV nunca em afirmações; CONFIRM novo e imutável |
| Ontologias dependentes de quem as escreve | escrita às cegas + hash; reportar `independent` |
| Regressão do `legacy` | teste de equivalência exata antes de qualquer merge |
| Conflito com regras do projeto (CLAUDE.md) | D-1/D-2 resolvidos antes da Fase 1 |

## Testes (inventário a criar; 20 áreas do pedido)
Pontuação estatística; calibração do bónus; rejeição de divisões prejudiciais; limiares adaptativos; seleção multiobjetivo; construção m-of-n; grupos ontológicos; matching de atributos; ausência de
fuga; fidelidade ao oráculo; accuracy real; preservação dos baselines (hashes); agnosticismo (estático, dinâmico, metamórfico); reprodutibilidade; comparação sem arredondamento; estados da hierarquia
(4 cenários: Original ≤ C4.5, Reloaded ≤ Original, cadeia completa, dados em falta); compatibilidade GUI. Datasets sintéticos e casos adversariais (ganhos iguais, ganhos minúsculos, ontologia vazia, grupos
degenerados, `n_real` muito pequeno, classes desequilibradas).

## Critérios de conclusão do programa
Conforme o Documento 5 §3 (portfólio): ou (a) alegação sustentada com controlos e potência declarada, ou (b) relatório honesto «não sustentada», com os candidatos documentados. Nenhum dos dois resultados
bloqueia a integração das melhorias que passem nos testes de segurança (não-regressão, agnosticismo, reprodutibilidade) e permaneçam desligadas por omissão.
