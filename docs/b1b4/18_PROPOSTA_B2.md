# Doc. 18 — Proposta científica de B2 (desenho; sem implementação)

Estado: **DESENHO** autorizado. Não há código de B2/B3/B4 no núcleo; o TREPAN Original, o C4.5, o MLP e o manifest congelado não foram alterados; B1 mantém todas as
opções **desligadas por defeito** (nenhuma variante promovida). Nada aqui é avaliação confirmatória: os números de §1 são diagnóstico DEV exploratório (nível `observed_only`).
Reprodução do diagnóstico: `python -m validation.benchmark.run_diag_b2 --unit DATASET SEED --out DIR` e `python -m validation.benchmark.diag_b2_analysis --dir DIR`
(evidência em `docs/evidence/b1b4/b2_diag/`). Estudo complementar do desempate: Doc. 19.

## 0. Síntese
1. A perda residual do Reloaded+OWL face ao Original **não é atribuível apenas ao portão EFSR**. Na decomposição (15 unidades DEV: iris, wine, breast_cancer; digits em §1.4) a **representação enriquecida isolada** e a **amostragem ativa de consultas** explicam, em ordem de grandeza, mais do que o portão; o score semântico nas features originais é ≈ neutro. Nenhuma componente é distinguível de zero individualmente (EP 0,006–0,012; n=15).
2. O portão atual aceita sobretudo por **tolerância de coerência** (≈77 % das aceitações sem ganho local) e em ≈58 % das aceitações o ganho nas linhas reais é ≤ 0; o ganho é medido na mesma amostra em que o candidato foi escolhido.
3. Recomendação: **B2-C (fallback estatístico com preservação da corrente do Original) + B2-A/B (ganho validado com incerteza)**, em fatorial com B1, com B2-C como primeiro passo por atacar a causa mais plausível (H2: contexto de decisão enviesado) e ser a que permite provar igualdade local ao Original. B2-A/B sozinhos são necessários mas provavelmente insuficientes.
4. Custo (§12) viável dentro do limite de 120 min por tarefa se partido por unidade com retoma; a avaliação **confirmatória** exige mais unidades que as 18 DEV (potência, §12.4).

## 1. Diagnóstico da perda residual de B1 (dados DEV exploratórios)

### 1.1 O que B1 deixou por resolver
B1 (13 variantes de calibração do score; Doc. 17) não satisfez C1 em nenhuma. O braço λ=0 (sem contributo semântico no score) continuou abaixo do Original em 3/4 datasets, a taxa de troca de decisão e o «regret» quase não variam com λ e todos os braços semânticos consomem ≈2,1–2,3× as consultas do Original. Logo: o score semântico não é o principal canal de perda.

### 1.2 Decomposição por ablação (ΔFidelidade, ΔAccuracy emparelhados por unidade vs `trepan_original`, teste externo, 15 unidades; macro por dataset)
| Braço (o que fica ativo) | ΔF macro | EP | ΔAcc macro | Δ nós | consultas ×Orig |
|---|---|---|---|---|---|
| `reloaded_core` (≡ Original, controlo EQ) | 0,0000 | – | 0,0000 | 0 | 1,00 |
| `diag_original_enr` — **Original na representação OWL** (só representação) | −0,0096 | 0,0074 | −0,0053 | −2,1 | 1,00 |
| `reloaded_owl_features` — λ=0, representação + projetor, sem score/portão | −0,0085 | 0,0070 | −0,0087 | +0,1 | 2,18 |
| `reloaded_semantic_score` — score semântico nas features **originais** | −0,0007 | 0,0110 | +0,0141 | +1,1 | 2,28 |
| `diag_active_off` — E sem consultas ativas | −0,0044 | 0,0056 | +0,0005 | +0,3 | 2,20 |
| `diag_gate_off` — E sem portão EFSR | −0,0188 | 0,0099 | −0,0143 | −1,2 | 2,16 |
| `reloaded_owl_full` (E, Reloaded OWL histórico) | −0,0159 | 0,0088 | −0,0097 | +0,5 | 2,24 |
| `reloaded_owl_shuffled` (ontologia permutada) | −0,0053 | 0,0095 | +0,0083 | +0,3 | 2,24 |

Contrastes emparelhados dentro do mesmo conjunto de unidades (ΔF; EP): E − gate_off = +0,003 (0,012) [o portão ajuda ligeiramente ou é neutro]; E − active_off = −0,012 (0,008) [as consultas ativas custam ≈1 pp]; E − owl_features = −0,007 (0,012); E − shuffled = −0,011 (0,009) [a ontologia verdadeira não supera a permutada nesta amostra].

Leitura prudente: (i) com n=15 e EP ≈ 0,006–0,012, **nenhum contraste isolado é estatisticamente distinguível de zero**; é evidência para orientar o desenho, não para concluir; (ii) a ordem de grandeza sugere que representação (≈ −0,010) e amostragem ativa (≈ −0,012) dominam a perda, e o portão é neutro-a-ligeiramente-positivo; (iii) logo B2-A/B (melhorar o portão) tem **teto baixo** se atuar isoladamente, e B2-C (restaurar o contexto de decisão do Original) ataca a componente de amostragem; a representação fica para B3 (fora de âmbito aqui); (iv) o ganho de accuracy do score semântico em features originais (+0,014 Acc, ΔF≈0) é compatível com ruído e não é usado para selecionar nada.

### 1.3 Estatísticas do portão (braço E, 15 unidades, 289 nós)
- Nós com decisão proposta ≠ estatística: 278/289 (96 %); tentativas elegíveis: 264; **aceites: 115 (44 %)**, dos quais **89 (77 %) «dentro da tolerância»** (aceites sem ganho local ≥ 0,002, só por ganho de coerência OWL ≤ 0,015 de perda tolerada).
- Razões de rejeição: sem ganho local 110; falha no gate das linhas reais 39. Aceites com ganho local < 0,01: 76/115; aceites com ganho real ≤ 0: 67/115 (de 137 avaliações do gate real).
- Ganho local médio dos aceites +0,020 (rejeitados −0,004); ganho real médio dos aceites +0,058 (rejeitados −0,176).
- Mediana da amostra de decisão: 672 linhas; candidatos semânticos gerados por nó: 24.
Interpretação: o portão filtra casos claramente maus (ganho real médio −0,18 nos rejeitados) mas a tolerância deixa passar muitos candidatos sem ganho demonstrado; o ganho medido na mesma amostra da seleção é otimista (H1). Estes números são descritivos (não há contrafactual por nó).

## 2. Hipóteses causais (a testar; nenhuma é assumida)

| Id | Hipótese | Mecanismo | Evidência atual | Como B2 a trata |
|---|---|---|---|---|
| H1 | **Ganho otimista no portão (winner's curse)** | O candidato semântico é escolhido entre dezenas (pesquisa m-of-n com score semântico + até 24 candidatos guiados) e o ganho local é medido **na mesma amostra** em que foi escolhido (`_evaluate_error_focused_intervention`, linhas ~777–890). A seleção do máximo sobre muitos candidatos infla o ganho medido. | Aceitações com ganho local pequeno (ver §1.3); λ quase não altera o regret. | B2-A (ganho validado em partição separada), B2-B (limite inferior com incerteza). |
| H2 | **Amostra de decisão enviesada** | A amostra do nó (`_decision_sample`, `_draw_membership_queries` ativo/projetado, fração 0,65, pool×4) é desenhada com incerteza do MLP e com projetor de domínio; é **diferente** da que o Original usaria, e consome o RNG de outra forma. O «fallback data-only» é calculado nessa amostra, logo ≠ escolha do Original. | Original ≢ «data-only dentro do Reloaded» mesmo com λ=0. | B2-C (preservar fluxo do Original e acrescentar a camada semântica numa corrente RNG separada). |
| H3 | **Representação** | A árvore Reloaded opera numa representação enriquecida (features `onto_*`); mesmo sem score/portão semântico, a pesquisa de divisões muda. | Braço `diag_original_enr` / `reloaded_owl_features` (§1.2). | Fora de B2-A/B/C; fica para B3 (representação). B2 mede-a como fator, não a corrige. |
| H4 | **Prioridade de expansão e orçamento** | `_priority` reordena a expansão best-first com pesos do perfil de erro; com `max_nodes` fixo, o orçamento é gasto em nós diferentes. | Nós/profundidade diferentes. | Controlado por flag e medido (§1.2); não é alterado em B2. |
| H5 | **Aceitação sem custo de complexidade** | Aceitar um m-of-n maior (n>1) com ganho marginal cresce a regra sem ganho validado. | `tree_complexity_delta` ainda não instrumentado. | B2-B inclui penalização/limite de complexidade pré-registado. |
| H6 | **Ruído de desempate do Original** | Ver Doc. 19: ruído de ulp muda regras/fluxo do Original em 52/60 ajustes sintéticos; |ΔF| até 0,06. | Medido (Doc. 19). | Comparações emparelhadas multi-semente; sem alterar o Original. |

Nenhuma hipótese é motivo para atribuir toda a perda ao portão: a decomposição (§1.2) mostra a contribuição de cada canal antes de se desenhar a correção.

## 3. Mapa do fluxo real de aceitação (código atual, `core/trepan_reloaded_historical.py`)

```
fit → _grow_tree (best-first por _priority:725)
  por nó:
   _decision_sample(745)                      amostra = linhas reais do nó + consultas ao oráculo
     ├─ _draw_membership_queries(1601) / _unfiltered(1632)   amostragem ativa/projetada (fração 0,65; pool×4; usa predict_proba)
     ├─ _build_error_region_profile(610) + _known_row_uncertainty(590)  perfil de erro → _current_error_profile_, eligible_for_semantic_refinement
   _best_mofn(1313)
     1. semantic_result = super()._best_mofn(X,y)        # feixe m-of-n do Original com score Reloaded
          ├─ _candidate_literals ← _feature_priority_scores(892) (variância·peso^k)
          └─ _split_selection_score(1045) → _calibrated_split_score(1080) (B1: legacy/bounded/calibrated/tiebreak) → _log_candidate(1156)
     2. generated = _ontology_guided_candidates(1193)    # até 24 candidatos extra; substitui se score > melhor + 1e-12
     3. data_result = super()._best_mofn(X,y) com _semantic_search_disabled=True (pesos=1, IG puro)   # «alternativa estatística»
     4. intervention = _evaluate_error_focused_intervention(777)   # PORTÃO EFSR
          · elegibilidade: região de erro ≥ min_disagreement (0,05)
          · gate 1: ganho local de fidelidade na MESMA amostra de decisão ≥ 0,002 (ou dentro da tolerância 0,015 se coerência OWL > 0)
          · gate 2: ganho nas linhas reais do nó (real_X/real_y, subconjunto da própria amostra) > 0 (ou tolerância)
     5. aceite → usa semantic_result; não aceite e ≠ data → fallback para data_result; EFSR desligado ⇒ data_result sempre
   splits → filhos → fila de expansão (prioridade com perfil de erro)
```
Pontos do fluxo em que a ontologia pode actuar além de B1/B2: representação enriquecida (H3), `_feature_priority_scores`, `_priority`, projetor/restrições de domínio, consumo de orçamento de consultas (~2,1–2,3× o Original).

Defeitos de desenho relevantes para B2 (observados no código):
1. O **mesmo conjunto** escolhe o candidato e valida o ganho (H1); os gates 1 e 2 não são independentes (as linhas reais fazem parte da amostra de decisão).
2. A comparação é com `data_result` recalculado na amostra enviesada, não com o resultado do Original (H2).
3. O limiar 0,002 é um ganho absoluto sem escala de incerteza; com amostras de 10²–10³ o erro-padrão do ganho é da ordem de 0,01–0,03 (§6), logo o limiar aceita ruído.
4. A tolerância 0,015 «compra coerência» sem custo mensurável: aceita perdas de fidelidade locais.
5. Candidatos descartados e rejeições não ficam registados por candidato (só o par aceite/rejeitado); B2 exige o log completo (§5, `log_split_candidates` de B1 já existe).

## 4. Desenhos de B2 (nenhum implementado)

Notação: nó v com amostra de decisão D_v = R_v ∪ S_v (linhas reais R_v, sintéticas/consultas S_v); candidato semântico c_sem; alternativa estatística c_stat (o que o Original escolheria no mesmo contexto); φ(c; D) = fidelidade ao oráculo do substituto de profundidade 1 definido por c em D (`split_surrogate_fidelity`).

### 4.1 B2-A — ganho validado (separação seleção/validação)
- Parte-se D_v deterministicamente (hash do índice da linha + node_id, **sem consumir RNG** do fluxo principal) em dobras de seleção D_v^sel e validação D_v^val, estratificadas por classe do oráculo; as linhas reais e as sintéticas são repartidas separadamente. As sintéticas do mesmo lote de consulta ficam na mesma dobra (evita fuga por dependência, §6).
- A pesquisa de candidatos (semântica e estatística) usa **só** D_v^sel. O ganho validado é `validated_fidelity_gain = φ(c_sem; D_v^val) − φ(c_stat; D_v^val)`. Aceita-se se `validated_fidelity_gain ≥ g_min` (g_min pré-registado, default 0 com sinal estrito `>` ou ε de escala).
- Troca sel/val (2 dobras cruzadas) opcional: média dos dois ganhos; custo = 2 avaliações de máscara, sem nova pesquisa se os candidatos finais forem os mesmos (re-pesquisa só na variante «cv-lite», §8).
- Custo: uma avaliação de duas partições em D_v^val (vetorizada, O(|D_v|)), **desprezável** face à pesquisa m-of-n (ver §12).
- Risco: reduz o tamanho efetivo de cada dobra → mais incerteza (tratada por B2-B).

### 4.2 B2-B — aceitação com incerteza
Em vez de ponto estimado ≥ limiar, decide-se sobre um **limite inferior** do ganho:
- Estimador: δ̂ = φ(c_sem; D^val) − φ(c_stat; D^val), diferença emparelhada (mesmas linhas), por linha d_i ∈ {−1,0,+1} (acerto do substituto de c_sem menos o de c_stat).
- Incerteza: erro-padrão **por clusters** (não por linha): clusters = lote de consulta/linha-semente real (§6); intervalo por bootstrap de clusters determinístico (semente derivada do node_id, `numpy.random.Generator` próprio — não toca na RNG do fluxo) ou variância robusta por clusters.
- Regra: aceitar se `LB_{1−α}(δ̂) > −margem_nao_inferioridade` **e** `δ̂ ≥ g_min` **e** estabilidade ≥ s_min **e** `tree_complexity_delta ≤ c_max`. Caso contrário, `rejection_reason` ∈ {`lb_below_margin`, `unstable`, `complexity`, `insufficient_effective_sample`}.
- **Estabilidade**: concordância do candidato escolhido entre B réplicas de bootstrap por clusters de D^sel (reavalia-se só o ranking de c_sem vs c_stat sobre a lista de candidatos finalistas já gerados — sem nova pesquisa nem consultas).
- **Efeito no crescimento**: o ganho é medido também ao nível de subárvore (profundidade 1 + 1 nível de lookahead barato sobre os filhos com rótulos de maioria) para evitar aceitar divisões que só adiam o erro; opção desligada por defeito (custo/benefício a medir).
- **Sem CV em todos os nós**: a CV completa multiplicaria por K o custo de `_best_mofn` (§12); só se permite «cv-lite» (K=2–3) em nós onde a incerteza residual classifica o caso como ambíguo (LB < 0 < δ̂+SE) e o orçamento por nó o permite; é um fator experimental de B2-B, não o padrão.
- Amostras sintéticas correlacionadas **não** são tratadas como independentes (tamanho efetivo, §6).

### 4.3 B2-C — fallback estatístico local
- Princípio: quando o candidato semântico não é aceite, o resultado tem de ser a escolha do Original **no mesmo contexto de decisão**, não uma pesquisa «data-only» sobre uma amostra enviesada.
- Mecanismo proposto (a verificar na fase de implementação): **preservação da corrente do Original** — a amostra-base do nó e a pesquisa m-of-n estatística usam exatamente as consultas/RNG que o Original usaria; os mecanismos semânticos (consultas ativas, candidatos guiados) operam sobre **amostras adicionais** de uma corrente RNG separada (`SeedSequence.spawn`), contabilizadas em `oracle_accounting_` (estado S3). Assim, em qualquer nó em que o semântico é rejeitado e o prefixo da árvore é idêntico, o resultado é **bit a bit o do Original** (auditável pelo teste diferencial EQ).
- Preserva o m-of-n do Original (incluindo a limitação de desempate, Doc. 19); sem heurísticas por dataset; sem usar rótulos reais.
- Custo: queries extra do semântico mantêm-se (~2,1–2,3×); a corrente base é a do Original. Risco: o prefixo deixa de coincidir após o primeiro nó aceite (então só se garante a igualdade local, não global) — documentado e medido por `fallback_used` + divergência.
- Alternativa B2-C′ (mais barata, sem separar correntes): manter o fluxo atual mas escolher o candidato estatístico pelo critério validado de B2-A; não restaura a escolha do Original (H2 persiste). Servirá de braço de comparação no desenho fatorial.

## 5. Contrato de avaliação do ganho

- Métrica principal local: `delta_fidelity = fidelity_semantic_split − fidelity_statistical_alternative` (mesmo oráculo, mesmo protocolo, mesma amostra de validação comparável); principal de avaliação global: fidelidade ao MLP no teste externo (apenas relato, nunca seleção).
- `delta_accuracy = accuracy_semantic_tree − accuracy_original_tree`: **reportada/observada**, **não** usada na aceitação local no modo principal.
- Comparabilidade: candidatos comparados partilham o mesmo `decision_context` (nó, amostra D_v, oráculo, parâmetros); quando partilham partição, `delta=0` e o caso é `same_partition`.
- **Instrumentação obrigatória** (observacional; não altera RNG nem a sequência de consultas): `node_id, candidate_id, candidate_type, ontology_state, statistical_gain, semantic_score, local_fidelity_gain, validated_fidelity_gain, uncertainty_estimate, effective_sample_count, oracle_query_count, accepted, rejection_reason, fallback_used, chosen_candidate_id, tree_complexity_delta`. `candidate_type ∈ {statistical, semantic_scored, ontology_guided, fallback}`.
- Prova de não-interferência da instrumentação: teste EQ com/sem instrumentação ⇒ fluxo de consultas e árvores idênticos (como já feito para `log_split_candidates`).

## 6. Definição de amostra efetiva

- Unidade de informação: linha real de treino que chega ao nó (observação independente do processo gerador dos dados) e lote de consulta sintética (dependente do modelo de distribuição, do projetor e da seleção ativa por incerteza).
- Peso e cluster: cada linha real é um cluster; cada lote sintético é um cluster. Tamanho efetivo: `n_eff = (Σ_c w_c)² / Σ_c w_c²` (Kish) com `w_c` = nº de linhas do cluster; e correção por correlação intra-cluster ρ̂ (estimada nos próprios d_i): `n_eff = n/(1+(m̄−1)ρ̂)`.
- Seleção ativa (amostras escolhidas por incerteza) **enviesa** a avaliação para a fronteira: o ganho nas sintéticas ativas não é estimador do ganho na distribuição real; por isso o ganho nas **linhas reais** é reportado separadamente e o limite inferior usa a combinação pré-registada (ponderação por esquema, nunca por resultado).
- Se `n_eff < n_min` ⇒ `insufficient_effective_sample` ⇒ rejeição conservadora (fallback). `n_min` vem da potência para detetar a margem escolhida (nunca ajustado a resultados).

## 7. Política de validação interna

1. Só dados de treino e consultas ao oráculo congelado; **nunca** o teste externo nem rótulos reais fora do que o Original já usa (R_v existentes).
2. Dobras determinísticas por hash(node_id, índice, semente do protocolo); sem consumir RNG do fluxo principal.
3. Seleção de configuração B2 (limiares, α, margens) **só** em DEV por validação interna do treino; DEV não é validação confirmatória; as sementes confirmatórias permanecem reservadas.
4. Pré-registo antes de executar: matriz (§9), orçamento (§12), limiares, critérios (§10).

## 8. Mecanismo de fallback (resumo)
Árvore de decisão por nó: (1) pré-condições (ontologia efetiva, região elegível, `n_eff ≥ n_min`) → senão estado S2-no-nó: escolha do Original; (2) avaliar candidatos em D^val; (3) aceitar semântico só se critérios de §10 → senão `fallback_used=True`, `chosen_candidate_id` = alternativa estatística (B2-C: a do Original). Todas as decisões registadas por candidato; o estado EQ do nó (S2/S3) fica auditável.

## 9. Matriz B1 × B2 (definida antes da implementação)

B1 (3 eixos já medidos, todos OFF por defeito): `historical` (legado), `tiebreak` (τ=c/√n), `calibrated λ=0,2`, `calibrated λ=0,4`; exploratório separado: `bounded λ=0,8`.
B2: `B2-0` (sem B2 = referência), `B2-A`, `B2-B` (inclui A), `B2-C` (fallback preservador), `B2-B+C`, e controlos B2-A/B/C com ontologia permutada.

| Bloco | Configurações | Papel |
|---|---|---|
| M0 | Original, C4.5, Reloaded histórico (E), E permutado | referências congeladas |
| M1 | {historical, tiebreak, cal 0,2, cal 0,4} × {B2-0, A, B, C, B+C} = 20 | matriz principal |
| M2 | M1 com ontologia permutada (controlo negativo) em 4 configs selecionadas por validação interna | atribuição à ontologia |
| M3 | engenharia convencional de features (controlo) + Original equivalente na mesma representação | atribuição |
| X1 | bounded λ=0,8 × {B2-0, B+C} = 2 (exploratório separado) | exploratório |

Reduções (sem encolher a avaliação estatística): M1 partilha árvores até ao primeiro nó onde diferem apenas se a execução permitir *reuso de prefixo exato* — a decidir por equivalência bit a bit com `equivalence_report.py`; nunca por aproximação.

## 10. Critérios de seleção e de aceitação (pré-registados)

Seleção de configuração (DEV, interna): lexicográfica — (i) fidelidade interna OOF vs Original equivalente, (ii) acurácia interna, (iii) estabilidade por semente, (iv) menor complexidade, (v) menor custo; sem usar teste externo.
Aceitação experimental (relato): `HIERARCHY_SATISFIED | NOT_SATISFIED | INFEASIBLE | INSUFFICIENT_EVIDENCE` para `Acc_R > Acc_O > Acc_C45` e `F_R > F_O` (distinguindo `F_R ≥ F_O` de `F_R > F_O`); avaliação por: fidelidade, acurácia, diferenças vs Original e vs Reloaded OWL histórico, vs ontologia permutada, nós, profundidade, refinamentos aceites/rejeitados, taxa de fallback, consultas ao oráculo, custo computacional, estabilidade por semente.
Critérios de passagem de B2 (a rever com o utilizador): (a) EQ S0–S2 intacto (testes diferenciais); (b) com B2 ligado, nenhum nó aceite com `validated_fidelity_gain` negativo; (c) fidelidade macro-DEV de B2 ≥ Original equivalente com limite inferior do IC emparelhado por unidade > −margem pré-registada; (d) controlo permutado não explica o ganho; (e) custo dentro do orçamento.

## 11. Controlos ontológicos
Mantêm-se: ontologia permutada/aleatória (já `_shuf`), controlo de engenharia convencional (features derivadas sem OWL), Original equivalente na mesma representação (`diag_original_enr`). Para B2 acrescenta-se: **B2 com ontologia permutada** (o portão validado deve rejeitar mais) e **portão com rótulos de classe do oráculo permutados** (sanidade: aceitação deve colapsar para ~0).

## 12. Estimativa de custo (antes de qualquer implementação)

### 12.1 Medidas de base (DEV, Linux, 3 processos em paralelo; tempos de treino da árvore por ajuste, média das unidades)
| Dataset | Original (s) | Reloaded E (s) | pesquisa m-of-n do Original (s) | pesquisa m-of-n em E (s) | nós (auditados) | consultas Orig |
|---|---|---|---|---|---|---|
| iris | 0,3 | 3,1 | 0,05 | 0,57 | ≈18 | ≈9,9 k |
| wine | 1,1 | 19,8 | 0,77 | 15,8 | ≈21 | ≈10,0 k |
| breast_cancer | 2,5 | 27,4 | 1,79 | 10,6 | ≈18 | ≈17,5 k |
| digits | ≈13 | ≈115 (ref. pré-B1) | – | – | – | – |
O custo do Reloaded é dominado pela **pesquisa m-of-n semântica** (10–20× a do Original), não pelas consultas ao oráculo nem pelo portão (portão ≪ 0,1 s por árvore nas auditorias).

### 12.2 Custo por nó dos componentes de B2 (microbenchmark vetorizado, 50–100 candidatos, 80–300 clusters)
| Componente | Custo por nó | Nota |
|---|---|---|
| B2-A (partição determinística + 2 avaliações de máscara) | ≈1–4 ms | sem nova pesquisa nem consultas |
| B2-B (bootstrap por clusters B=200–500, LB, estabilidade) | 3–70 ms (n=400…3000) | sem consultas |
| B2-B «cv-lite» K=3 (re-pesquisa m-of-n por dobra) | ≈ +2× a pesquisa m-of-n semântica no nó | só em nós ambíguos; fator experimental, **não** por defeito |
| B2-C (corrente base do Original + camada semântica) | pesquisa estatística ≈ Original (ver 12.1) | consultas semânticas iguais às atuais (≈2,2×) |
| CV completa em todos os nós (rejeitada) | ×K a pesquisa m-of-n | iris ×3, wine ×(3×15,8 s), digits proibitivo |
Overhead A/B face à pesquisa semântica por nó: wine/bc ≲ 10 %; iris (nó pequeno) pode chegar a ≈ +100 % mas em valor absoluto < 0,3 s por árvore.

### 12.3 Orçamento da matriz (18 unidades DEV: 5 iris, 5 wine, 5 breast_cancer, 3 digits)
Custo de uma configuração Reloaded sobre as 18 unidades: 5·3 + 5·20 + 5·27 + 3·115 ≈ 595 s ≈ **10 min** (×1,1 com A/B ⇒ ≈ 11 min).
| Bloco | Nº configs | Tempo de CPU sequencial |
|---|---|---|
| M1 (4 B1 × 5 B2) | 20 | ≈ 220 min |
| M2 (controlo permutado, 4 configs selecionadas por validação interna) | 4 | ≈ 44 min |
| M3 (Original equivalente na representação + engenharia convencional de features) | ≈ 4 | ≈ 10 min |
| X1 (bounded λ=0,8; exploratório separado) | 2 | ≈ 22 min |
| Subtotal | 30 | **≈ 300 min (5 h CPU)**; 4 núcleos ⇒ ≈ 1,3–1,7 h de relógio |
| Margem de segurança ×2 (contenção, cv-lite em subamostra) | – | ≈ 10 h CPU |
**Ajuste estrutural (ontology-blind, partilhado por todas as configurações da unidade) — custo à parte e dominante em Digits:** iris ≈ 36 s, breast_cancer ≈ 300 s, digits ≈ 2 600 s (≈ 43 min; medido na referência pré-B1). No diagnóstico DEV o *checkpoint* de tuning **não foi reaproveitado** (`tuning_fits_resumed = 0`; o tuning foi recalculado por unidade), pelo que, antes de qualquer execução da matriz, é item de infraestrutura verificar/corrigir o reuso (sem alterar a configuração científica; resultados idênticos por construção) e calcular o tuning **uma vez por unidade**.
Restrição do executor (~120 min por tarefa): a unidade de trabalho é (dataset, semente); para Digits separa-se o tuning (≈ 43 min, 1 tarefa, grava *checkpoint*) das configurações (30 × ≈ 125 s ≈ 63 min, repartidas em ≥ 2 tarefas por unidade para folga); as restantes unidades cabem numa só tarefa; cada braço grava JSON próprio e a retoma salta o que já está completo; o ajuste estrutural reutiliza o *checkpoint* (`BIURI_TUNING_CHECKPOINT`). **Redundância:** reutilizar prefixos de árvores só será feito com prova de equivalência exata (`equivalence_report.py`); não é assumido na estimativa. Não se reduz a avaliação estatística para caber: se não couber, particiona-se por mais tarefas.

### 12.4 Potência e dimensão da avaliação confirmatória
No diagnóstico, o desvio-padrão de ΔF por unidade é ≈ 0,034 (EP 0,0088 com n=15). Para detetar ΔF = 0,01 com potência 80 % (unilateral, α=0,05) são necessárias ≈ ((1,645+0,84)·0,034/0,01)² ≈ **70 unidades**; com 18 unidades DEV só se distinguem efeitos ≳ 0,02. Portanto: (i) DEV serve para **selecionar** configurações (com este limite explícito); (ii) a avaliação confirmatória terá de usar mais unidades (mais sementes e/ou mais datasets, sementes reservadas) — dimensão e custo a definir em novo pré-registo (decisão D6). Custo confirmatório ≈ 70 unidades × 1 configuração selecionada (+ Original, C4.5, controlos) ≈ 70 × (≈ 33 s média ponderada) ≪ matriz DEV, mas depende do mix de datasets (digits domina).

## 13. Testes planeados (a implementar só com aprovação)
1. EQ: `B2=off` ≡ estado atual bit a bit (arms A–F, S0–S2); instrumentação não altera RNG/consultas.
2. Unitários: partição determinística sem consumo de RNG; dobras estratificadas por classe e por lote; clusters; `n_eff` (Kish) em casos conhecidos; limite inferior bootstrap determinístico por node_id; não-inferioridade.
3. Propriedades: invariância à permutação das linhas; invariância ao complemento m-of-n; monotonia (mais ganho ⇒ não rejeita); aceitação=0 sob ontologia permutada sintética sem sinal.
4. Fallback: B2-C ⇒ nó rejeitado devolve exatamente a escolha do Original (teste diferencial vs `trepan_original`) enquanto o prefixo coincide.
5. Contabilidade: consultas extra do semântico em `oracle_accounting_` e soma consistente.
6. Agnosticismo: verificador estático e testes de dataset desconhecido/dimensionalidade/classes/tipos.
7. Custo: teste de limite do overhead por nó (orçamento).

## 14. Riscos de fuga e de sobreajuste
| Risco | Origem | Mitigação |
|---|---|---|
| Fuga teste→seleção | uso de resultados de teste para escolher λ, α, margens | proibido; pré-registo; só validação interna |
| Dupla utilização da amostra | selecionar e validar na mesma D_v | B2-A (dobras separadas) |
| Dependência sintética | lotes de consulta correlacionados | clusters, `n_eff`, dobras por lote |
| Winner's curse | muitos candidatos | validação em dobra independente + LB |
| Sobreajuste de limiares | ajustar margens após ver resultados | limiares fixados antes; DEV ≠ confirmatório |
| Multiplicidade (13+ variantes, 20 configs) | matriz grande | correção/controlo de FDR no relato, sementes confirmatórias reservadas |
| Fuga via rótulos reais | R_v | só as linhas reais que o Original já usa; teste de não-uso de y real fora de R_v |
| Instrumentação altera o fluxo | logging | teste EQ de não-interferência |
| Reuso de prefixo inválido | cache/atalhos | só com equivalência bit a bit |

## 15. Compatibilidade com o contrato EQ v2.0
- S0/S1/S2 inalterados: B2 só actua em nós com ontologia efetiva e tentativa semântica (S3); B2 desligado ≡ comportamento atual (bit a bit, testado).
- S3: B2-C separa a corrente do Original da semântica ⇒ a porção Original ≡ Original isolado e a tentativa semântica fica auditada em `oracle_accounting_`; sem melhoria ⇒ `SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN`.
- Cada mecanismo de B2 é ontológico (decide aceitação de divisões influenciadas pela semântica), com flag, auditoria e controlo negativo; **não** introduz melhoria independente da ontologia (a validação interna só actua sobre candidatos semânticos; o ramo estatístico é o do Original).
- Baselines congelados por hash; um novo manifest versionado (`BENCHMARK_MANIFEST_v3` rascunho) só é criado com aprovação.

## 16. Alternativa recomendada, justificação, riscos e decisões

**Recomendação:** B2 fatorial com ordem de implementação **B2-C → B2-A → B2-B**, tudo atrás de flags **desligadas por defeito**, com instrumentação (§5) como primeiro entregável (observacional, EQ-testada).
Justificação científica: (1) a decomposição sugere que o contexto de decisão enviesado (amostragem ativa/representação) pesa mais do que o portão; só B2-C permite **provar** (teste diferencial) que um nó rejeitado devolve a escolha do Original; (2) B2-A/B respondem ao defeito estatístico verificável do portão (seleção e validação na mesma amostra; 77 % das aceitações sem ganho local; tolerância sem custo) e ao requisito de não tratar amostras sintéticas correlacionadas como independentes; (3) todos são ontológicos (só governam divisões influenciadas pela semântica), auditáveis, com controlo negativo (ontologia permutada) e compatíveis com EQ v2.0.

**Riscos principais:** (R1) B2-A/B podem apenas recuperar o nível do Original (não superá-lo): resultado legítimo (`SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN`/`INSUFFICIENT_EVIDENCE`), nunca imposto; (R2) B2-C duplica complexidade de RNG (correntes separadas) e o prefixo deixa de coincidir após o primeiro nó aceite; (R3) potência insuficiente em DEV (§12.4); (R4) multiplicidade (30 configurações); (R5) a representação enriquecida (H3) pode ser o canal dominante e fica fora de B2 (B3); (R6) limitação de desempate do Original (Doc. 19) gera ruído de medição até ±0,06 de fidelidade por unidade; (R7) custo da pesquisa m-of-n semântica (10–20× o Original).

**Critérios de aceitação do desenho (para passar à implementação):** aprovação da matriz e orçamento (§9, §12), dos critérios (§10) e das decisões abaixo; pré-registo versionado antes de qualquer execução.

**Decisões que necessitam de aprovação do utilizador**
- D1. Aprovar a ordem B2-C → B2-A → B2-B e o fatorial B1×B2 de §9 (30 configurações DEV, ≈5 h CPU, ≈1,5 h de relógio com 4 núcleos).
- D2. Autorizar implementação de B2-C com **corrente RNG separada** para os mecanismos semânticos (muda o consumo de RNG do Reloaded com ontologia, não do Original; afeta reprodutibilidade histórica do Reloaded E — mantida como braço de referência por hash).
- D3. Margem de não-inferioridade, α, g_min, n_min de B2-B: propor valores só por validação interna DEV (pré-registados antes de correr) — aprovar o procedimento.
- D4. Autorizar (ou não) a análise de sensibilidade «Original-det vs Reloaded-det» com a variante de desempate (Doc. 19 §6) como estudo separado.
- D5. Autorizar `cv-lite` como fator experimental (não padrão).
- D6. Definir a dimensão da avaliação confirmatória (nº de unidades ≳ 70 para ΔF=0,01) e reservar sementes/datasets.
- D7. Autorizar um *manifest* v3 em rascunho (sem execução) para B2.
Nenhuma destas ações foi iniciada.
