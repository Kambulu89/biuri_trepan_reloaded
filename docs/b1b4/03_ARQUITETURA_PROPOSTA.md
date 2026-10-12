# Documento 3 — Arquitetura proposta para B1–B4 (PROPOSTA, sem código)

Princípios: (1) **tudo atrás de uma configuração explícita, desligada por omissão**; modo `legacy` ≡ comportamento atual, provado por comparação exata
(`validation/benchmark/equivalence_report.py`); (2) nenhum ficheiro congelado (Original, C4.5, tuning, contrato do oráculo) é alterado; (3) só propriedades
gerais (contagens, tipos, estrutura/qualidade da ontologia, orçamento) — nunca identidade de dataset; (4) cada decisão é registada para auditoria.

## 1. Mapa de alterações

| Módulo | Ação | Intervenção |
|---|---|---|
| `core/trepan_reloaded_historical.py` | **alterar** (único motor tocado): `_split_selection_score`, `_best_mofn`, `_evaluate_error_focused_intervention`, `_ontology_guided_candidates` delegam em políticas injetáveis | B1, B2, B4 |
| `core/reloaded_policies.py` (**novo**) | `ScoreCalibrationPolicy` (legacy / tie-break / bounded / normalized), `SplitAcceptancePolicy` (legacy / evidence-based), `GroupProvider` (semantic / random / shuffled) — funções puras, testáveis sem modelo | B1, B2, B4 |
| `core/hierarchy_contract.py` (**novo**) | `HierarchyStatus` + `evaluate_hierarchy(...)`: puro, sem ML | B3 / reporte |
| `core/reloaded_config_selection.py` (**novo**) | `select_reloaded_config(...)`: CV interna com MLP por dobra, regra lexicográfica, registo de todos os candidatos | B3 |
| `core/benchmark/runner.py` | acrescentar braços R1–R7 e controlos como `ArmSpec` (flags), seleção por unidade; `default_arms` inalterado | todos |
| `core/benchmark/analysis.py` / `stats.py` / `report.py` | quadro comparativo, estados da hierarquia, estatística do Documento 2 | reporte |
| `gui/…` | secção de diagnóstico (hierarquia, Δacc, Δfid, variante, contribuição ontológica, razões de aceitação) **sem mexer na geometria das árvores** | UI |
| `validation/benchmark/BENCHMARK_MANIFEST_v2.json` | novo manifest congelado | protocolo |
| **Não tocar** | `trepan_original.py`, `c45_j48_tree.py`, `trepan_scientific_tuning.py`, `scientific_experiment_contract.py`, `FrozenOracle` | — |

## 2. Observabilidade (pré-requisito da Fase 1)
Sem estes campos não se mede B1–B4 (ver Documento 1 §3.1):
* `candidate_table_` por nó: para **cada** candidato avaliado → `stat_component`, `onto_depth_term`, `error_cov_term`, `weight_factor`, `final_score`, `rank`,
  `is_data_only_winner`, `is_selected`. Cabe no orçamento (≤ 12 literais × beam + ≤ 24 semânticos).
* Campos explícitos `statistical_component` e `semantic_additive_term`; `semantic_bonus` legado fica marcado como `score − IG_cru (escalas mistas)`.
* `effective_sample_count`, `n_real`, `n_synthetic` na decisão de cada split; `uncertainty` do ganho; `decision_reason`.
* Custo estimado: baixo (listas pequenas por nó); validado por teste de equivalência (árvore idêntica com e sem registo).

## 3. B1 — Calibração do bónus (`ScoreCalibrationPolicy`)
Seja, para o candidato *i* de um nó, `s_i` = componente estatística (IG normalizado, ver código atual) e `m_i` = componente semântica (α·OntoDepth+β·ErrorCoverage e
fatores de peso/coerência). Em todas as variantes: **`s_i ≤ 1e-12 ⇒ score_i = 0`** (já existe: split sem informação nunca vence só por bónus).

* **B1-A (desempate):** `τ` = tolerância estatística do nó; `T = {i : s* − s_i ≤ τ}`; seleciona-se `argmax_{i∈T} m_i`. τ = erro-padrão da diferença de ganho entre
  candidatos estimado na amostra de decisão (estimador analítico por omissão; bootstrap como sensibilidade), mínimo `1/n_eff`. **Propriedade testável:** a semântica nunca
  escolhe um candidato com `s_i < s* − τ`.
* **B1-B (limitado):** `score_i = s_i + min(m_i, λ·s_i)`. **Propriedade:** nenhum candidato com `s_i < s*/(1+λ)` pode vencer ⇒ perda estatística máxima `λ/(1+λ)·s*`
  por nó, por construção. Grelha λ ∈ {0,10; 0,25; 0,50} (λ=0 ≡ sem semântica). λ escolhido só por CV interna.
* **B1-C (normalização):** `z_stat`, `z_sem` por *ranks* dentro dos candidatos do nó (robusto a escalas); se `max s − min s < τ` (candidatos estatisticamente indistinguíveis) cai
  em B1-A; `score = z_stat + w·z_sem`, `w = 0,5` fixo (não afinado). Evita normalizações min-max instáveis quando todos os ganhos são pequenos.
Registo: para cada split, `statistical_component`, `semantic_additive_term`, `decision_changed_by_ontology` (comparado com o vencedor data-only **do mesmo nó**).

## 4. B2 — Aceitação baseada em ganho real (`SplitAcceptancePolicy`)
* **Cross-fitting no nó:** a amostra de decisão é dividida em duas metades (partição determinística, seed do nó); o split é *escolhido* numa metade e *avaliado* na outra
  (e vice-versa) → remove o otimismo in-sample do gate atual.
* **Ganho:** `Δ̂ = média_i (acerto_semântico_i − acerto_data_only_i)` (diferença pareada de concordância com o oráculo). Incerteza: variância pareada (tipo McNemar) com
  `n_eff`. **Proposta de `n_eff`:** a *aceitação* usa só as **linhas reais** que chegam ao nó (`n_real`); as queries sintéticas servem para **ordenar** candidatos, não para
  aceitar (são amostras do próprio modelo gerador, não evidência independente). Aceita-se se `LCB_95(Δ̂) ≥ 0` (nunca ganho negativo confirmado); se `n_real` for insuficiente → razão
  `insufficient_evidence` ⇒ mantém o vencedor data-only (≡ Original naquele nó). Consequência esperada: menos intervenções em nós fundos; declarada, não escondida.
* **Granularidade:** reporta-se `1/n_eff`; **não** é usado como limiar universal.
* **Paragem/poda (B2-D):** poda posterior opcional por regra 1-SE em fidelidade OOF (sem acesso ao teste) para subárvores sem benefício; registada em `pruning_audit_`.
* **Registo por divisão:** `node_id, sample_count, effective_sample_count, statistical_gain, semantic_bonus, estimated_fidelity_gain, estimated_accuracy_gain,
  uncertainty, accepted_or_rejected, decision_reason` (os campos pedidos). `estimated_accuracy_gain` só existe no Modo B (Mode A não vê rótulos reais).

## 5. B3 — Seleção multiobjetivo hierárquica (`select_reloaded_config`)
**Nível:** *modelo* (escolha entre configurações do Reloaded já treinadas), não por split — é o nível onde accuracy, fidelidade, complexidade e estabilidade são mensuráveis
em dobras internas. Pareto por split multiplicaria o espaço de procura sem forma honesta de medir accuracy local.

Para cada configuração `c` ∈ grelha pré-declarada, em cada dobra interna `k` (MLP_k, O_k reajustados): `A_R,k(c)`, `F_R,k(c)`, `nós_k(c)`, `queries_k(c)`, coerência semântica;
`A_O,k`, `A_C,k`, `F_O,k` dos baselines nas mesmas dobras. Regra **lexicográfica com tolerâncias estatísticas** (justificação: ordem de prioridade científica clara; evita
somas ponderadas arbitrárias; Pareto devolve conjuntos grandes sem critério de escolha):
1. **Validade/viabilidade:** `F_R(c) ≥ F_O − δ_F`; `nós(c) ≤ 1,25·nós_O`; `queries(c) ≤ orçamento`; sem violações de contrato (oráculo único).
2. **Maximizar `A_R(c)` (OOF)**; candidatos dentro de 1 erro-padrão do melhor formam o conjunto elegível.
3. Dentro do elegível: maior `F_R`; depois menor nº de nós; depois menor DP entre dobras (estabilidade); depois maior coerência semântica; depois id da config (determinístico).
4. **Viabilidade da hierarquia (OOF):** `A_R(c*) > A_O` e `A_O > A_C`; o segundo termo **não depende de c** (propriedade dos baselines) ⇒ só serve para classificar `INFEASIBLE`.
5. **Fallback:** se nenhuma `c` cumprir (1)–(2), regista-se `selected="none"` e usa-se a configuração `legacy`; **não** se substitui por um Original disfarçado. (Se o utilizador quiser
   um «fallback ao Original», tem de ser um braço declarado, nunca contado como vitória do Reloaded — decisão D-3.)
* **Modo A (oráculo):** o Reloaded só vê o oráculo (contrato atual). **Modo B (híbrido supervisionado):** o split score passa a `(1−γ)·IG_oráculo + γ·IG_rótulos_reais(linhas reais do nó)`,
  γ ∈ {0; 0,25; 0,50} escolhido na CV interna (γ=0 ≡ Modo A). Rotulado `reloaded_hybrid`, **braço e relatório próprios**; nunca apresentado como imitação pura do oráculo; não é
  comparável com o Original em «fidelidade pura».

## 6. B4 — m-of-n semântico (`GroupProvider`)
* `semantic`: grupos/relações do grafo OWL (como hoje, via `OntologySemanticGraph`).
* `random`: **mesma distribuição de tamanhos de grupo** e mesmo nº de candidatos, atribuição uniforme aleatória (seed do manifest).
* `shuffled`: permutação dos grupos pelas features (como `ShuffledSemanticProvider`, mas preservando a contagem de candidatos).
* `conventional`: sem grupos; **mesmo orçamento de candidatos m-of-n** gerado pelo beam estatístico alargado (isola «mais procura» de «conhecimento»).
Todos partilham orçamento (`semantic_candidate_budget`), `max_n`, limite de queries e validação pelo mesmo score. Nenhum usa nomes de atributos.

## 7. Contrato de superioridade e relatório
`core/hierarchy_contract.py` (Documento 5) é a única fonte dos estados `HIERARCHY_SATISFIED | NOT_SATISFIED | INFEASIBLE | INSUFFICIENT_EVIDENCE`; funções puras, entradas = vetores
de métricas **não arredondadas**; usadas pelo runner (relatório) e pela GUI (diagnóstico). Nunca altera baselines nem filtra unidades.

## 8. Agnosticismo — como se garante
* `scripts/check_dataset_agnosticism.py` e `tests/test_agnosticism_contract.py` continuam obrigatórios; estender com: sem literais de datasets nos novos módulos; sem ramificação por `dataset_id`.
* Testes dinâmicos/metamórficos novos: permutação de colunas; renomeação de colunas (nomes opacos) com ontologia correspondente; duplicação de classes; escala de features; dataset desconhecido gerado.
* Políticas dependem apenas de: `n_eff`, `n_real`, nº de candidatos, nº de classes, tipos de features, qualidade da ontologia, orçamento.

## 9. Compatibilidade
* `legacy` por omissão; GUI e benchmark atuais intactos; contrafactuais, árvores CF, exportações e relatórios não são alterados.
* Visualização de árvores: nenhuma alteração à geometria; só painéis de texto/tabela adicionais.
* Artefactos treinados guardam `reloaded_policy_config` (hash incluído no `config_hash`).

## 10. Decisões que precisam da tua aprovação
| Id | Decisão | Recomendação |
|---|---|---|
| D-1 | CLAUDE.md §2 diz «nunca otimizar nem codificar para produzir a ordem Reloaded > Original > C4.5». O novo requisito torna-a objetivo formal. | Emendar a regra: a hierarquia é **critério pré-registado de seleção em validação interna e de relatório**; continua proibido: alterar baselines, usar o teste, escolher seeds/datasets a posteriori. Só altero CLAUDE.md com a tua confirmação explícita. |
| D-2 | CLAUDE.md §2 «tuning científico e contrato do oráculo congelados» vs. B1–B4 a alterar o Reloaded. | Congelamento mantém-se para Original/C4.5/tuning estrutural/oráculo; Reloaded evolui sob **manifest v2**; manifest v1 permanece como histórico. |
| D-3 | Fallback para Original em B3. | Não permitir como «Reloaded»; só como braço declarado `fallback_original`. |
| D-4 | Modo B (rótulos reais) | Incluir como braço híbrido separado; reivindicações primárias só do Modo A. |
| D-5 | Definição de `n_eff` no B2 (só linhas reais vs. mistura com sintéticas) | Só reais para aceitar; sintéticas para ordenar. |
| D-6 | Mapeamento explícito coluna↔entidade para nomes opacos | Sidecar opcional (tabela índice→IRI) lido pelo provider; sem sidecar a ontologia é rejeitada. |
| D-7 | Lista de ≥ 12 datasets CONFIRM e quem escreve/valida as ontologias | Tu escolhes/aprovas a lista; ontologias escritas às cegas e congeladas por hash. |
| D-8 | Grelhas (λ, γ) e δ_A, δ_F do Documento 5 | Aprovar antes de correr; δ_A=0 para estado por unidade, δ_A=1 pp para alegação de superioridade. |
