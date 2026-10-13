# Documento 15 — Fase 1b: V2 (consultas no `mirror`) — análise de hipóteses, causa, intervenção e resultados

## Veredito
| Item | Estado |
|---|---|
| **V2 — parte redundante** (chamadas byte-idênticas repetidas) | **FIXED / VERIFIED** (reutilização exata) |
| **V2 — remanescente** (tentativa semântica + Original no caso 4 sem efeito mensurável) | **NECESSÁRIO POR CONSTRUÇÃO — documentado, não eliminado**; aguarda decisão sobre o contrato do caso 4 |
| Fluxo do Original dentro do `mirror` | **≡ Original isolado** (provado, §3) |
| Comportamento ontológico ativo | **preservado** (âncoras; única diferença legítima: §4.3) |
| `GLOBAL_EQ_CONTRACT` | **NOT_YET_FULLY_COMPLIANT** (identidade do fluxo *completo* no caso 4 não alcançável; ver §6) |
| Desempate m-of-n entre plataformas | investigado, **proposta sem aplicar** (§7; implica alterar o Original) |

## 1. Onde e porquê há duplicação (instrumentação do fluxo completo)
`Reloaded.fit` com `mirror_when_no_semantic_effect=True` e ontologia com estrutura ativa: (1) **passagem semântica** (`super().fit`, o motor do Original com os ganchos ontológicos); (2) se não houve efeito mensurável (`ontology_influenced_splits==0` e `error_focused_interventions_accepted==0`), (3) **reexecução** de `TrepanOriginalClassifier(...).fit` no mesmo oráculo e adoção do seu estado. Consultas ao oráculo (`docs/evidence/b1b4/phase1b/trace_v2.py`, cada chamada com método, forma, hash da entrada e da resposta, fase e cadeia de chamadores):

| origem | método | consumidor | resultado usado para | necessária? |
|---|---|---|---|---|
| `Original.fit` (início) | `predict(X_treino)` | rotulagem inicial | raiz, classes, fidelidade | **idêntica nas duas passagens** (mesmos bytes, mesma resposta) → **redundante** |
| `_decision_sample` → `_draw_membership_queries` (Original) | `predict(amostras sintéticas)` | consultas de pertença do nó | conjunto de decisão do split | necessária à passagem em causa |
| `_draw_membership_queries_unfiltered` (Reloaded, projeção/amostragem ativa/EFSR) | `predict(amostras projetadas)` | idem | splits semânticos, auditoria, decisão do mirror | **necessária** à tentativa; amostras **diferentes** das do Original (projeção/RNG) |
| `_known_row_uncertainty` (EFSR) | `predict_proba(linhas conhecidas)` | perfil de erro/incerteza | prioridade e portas EFSR | **necessária** à tentativa; só existe no caminho semântico |
| reexecução (mirror) | `predict` ×4 | Original | árvore final | fluxo **exatamente igual** ao do Original isolado (`rerun_stream_equals_original_alone = True` em 4/4 casos) |

Conclusões demonstradas: (i) **só 1 chamada** do rerun é byte-idêntica a uma da passagem semântica (a rotulagem inicial; resposta igual); (ii) as consultas de pertença do rerun **parecem** equivalentes às da tentativa (mesma distribuição, mesma semente) mas **não o são** (projeção, amostragem ativa e consumo diferente do RNG) — não são reutilizáveis sem alterar o algoritmo; (iii) os resultados da tentativa decidem se o mirror se aplica, logo **são necessários** ao desenho atual.

## 2. Comparação das hipóteses
| Critério | **H-A** decisão antecipada | **H-B** reutilização total do fluxo | **H-B′** reutilização exata só de chamadas byte-idênticas (**escolhida**) | **H-C** só contabilidade | **H-D** `mirror` desligado por omissão |
|---|---|---|---|---|---|
| 1 Mecanismo | pré-voo estrutural decide «sem efeito» e delega | reaproveitar as consultas da tentativa no Original | memo exato (método+forma+dtype+SHA-256 dos bytes) por *fit*, só na reexecução | contadores «tentativa» vs «Original» | configuração por omissão |
| 2 Causa que ataca | a tentativa inteira | rerun completo | chamadas repetidas idênticas | opacidade do custo | o rerun |
| 3 Código | decisor novo sobre estrutura (não existe predicado exato) | substituir o rerun por replay | 2 invólucros + ligação em `fit` (~110 linhas) | contadores | 1 default (+ GUI/configs) |
| 4 Consultas | −100% da tentativa se acertar | rerun → ~0, mas impossível (streams diferem) | −`n_treino` amostras (−0,4% a −3,2% medido) | 0 | −tentativa (se mirror off) |
| 5 Enriquecimento | **muda** (tentativa omitida/ajustada) | muda | **inalterado** | inalterado | **muda** (mirror é salvaguarda) |
| 6 Árvores/regras | pode mudar | pode mudar | **não** (rerun intocado, mesma resposta) | não | **sim** (árvore semântica sem efeito pode diferir) |
| 7 Fidelidade | em risco | em risco | não | não | em risco |
| 8 Seleção de splits | altera | altera | não | não | altera |
| 9 Fuga | possível se o pré-voo usar labels | possível (reutilizar amostras do teste não, mas misturar estados) | não: mesmo oráculo, mesmo *fit*, sem teste | não | não |
| 10 E1–E9 | quebra se prever mal | quebra E4 | **preserva** (Original-portion exata) | preserva | quebra a salvaguarda |
| 11 Complexidade | alta | alta | baixa | muito baixa | muito baixa |
| 12 Custo exec./validação | validar pré-voo em muitos regimes | validar equivalência de streams (falha) | baixo (testes + DEV) | baixo | baixo, mas semântico |
| **Veredito** | **rejeitada**: «sem efeito» só se conhece depois de construir a árvore | **rejeitada**: provado que os streams diferem | **adotada** (única parte provadamente redundante) | **adotada em complemento** (rastreabilidade) | **rejeitada**: mudança científica, não otimização |
Critério de escolha: a intervenção mais localizada que elimina o provadamente redundante sem alterar o comportamento científico; **não** se escolheu por tempo (a variação de tempo é ruído).

## 3. Implementação (único ficheiro de núcleo: `core/trepan_reloaded_historical.py`)
* `_ExactCallLedger` (passagem semântica): **só observa**; regista contagens por método e as respostas de `predict` indexadas pelo conteúdo exato; `predict_proba` só existe se o oráculo o tiver (preserva `hasattr`); reencaminha atributos. Instalado **apenas** quando `mirror_when_no_semantic_effect` e ontologia efetiva (nunca no caminho ≡ Original nem com `mirror=False`).
* `_ExactReplayOracle` (reexecução do Original): serve de cópia a resposta **só** de chamadas `predict` byte-idênticas (método+forma+dtype+SHA-256) feitas ao **mesmo oráculo no mesmo fit**; tudo o resto (outras ordens, 1 ulp, subconjuntos, dtype diferente, `predict_proba`) vai ao oráculo. A cache é por-*fit* (estado do modelo único), por-oráculo (invólucro dedicado), devolve cópias, não depende do dataset. Pressupõe oráculo determinístico (contrato `FrozenOracle`). `REUSE_IDENTICAL_ORACLE_CALLS=False` (atributo de classe, sem alterar a assinatura) restaura o comportamento anterior.
* `oracle_accounting_` (só relato, fora da allowlist e fora de E1–E9): `semantic_attempt`, `original_rerun`, `reused_identical`, política. `oracle_info_["oracle_type"]` repõe o tipo real do oráculo.

## 4. Resultados (DEV; sem avaliação confirmatória)
### 4.1 Antes/depois (MLP real, projetor genérico = limites do treino; 11 unidades; `v2_before_after.jsonl`)
| dataset (seeds) | amostras Original | Reloaded+mirror antes | depois | retiradas | árvore/regras/previsões = Original | fidelidade/accuracy/nós antes=depois |
|---|---|---|---|---|---|---|
| iris (42,7,123) | 10 112 | 31 050–31 330 | −112 | 0,36% | sim (3/3) | idênticos |
| wine (42,7,123) | 10 133 | 31 099–31 585 | −133 | 0,43% | sim (3/3) | idênticos |
| breast_cancer (42,7,123) | 10 426 | 16 852–34 190 | −426 | 1,25–2,53% | sim (3/3) | idênticos |
| digits (42,7) | 10 999–11 160 | 41 693–41 854 | −1 347 | 3,2% | sim (2/2) | idênticos |
Em todas: `rerun + reutilizado = Original isolado` (amostras e chamadas); tempo e memória sem variação distinguível do ruído (pico +0 MB; o aumento face ao Original, p.ex. 18→30 MB em Digits, é a tentativa semântica, igual antes e depois). **O custo remanescente (~2–3× o Original) é a tentativa semântica**, necessária para decidir «sem efeito».
### 4.2 Testes
`tests/test_eq_reloaded_original.py`: **171 passed + 1 xfailed** (ver CI no §8). Novos: modelo final ≡ Original no mirror; parte do Original do fluxo **exatamente** o fluxo do Original (sem reutilização: sufixo idêntico; com: + chamadas reutilizadas); total = tentativa + Original − reutilizado; reutilização só byte-idêntica (1 ulp, ordem, subconjunto, dtype, `predict_proba`, oráculo diferente: não reutilizam); superfície do invólucro; ausência de invólucro sem mirror/sem ontologia efetiva; reprodutibilidade; outro oráculo ainda ≡ Original; delta V2 explícito na comparação com as âncoras.
### 4.3 Âncoras
`golden_effective.json` **não foi alterado**. Com `mirror` aplicado a única diferença permitida (e verificada contra `oracle_accounting_`) é a retirada das chamadas byte-idênticas (`compare_golden`); sem mirror, bit-a-bit. Descoberta de ambiente: a comparação **exata** depende do kernel BLAS do anfitrião (o contentor foi reiniciado com outro dispatch e o *baseline* sem alterações falhava a âncora exata); passou a correr num subprocesso com ambiente numérico fixado (`tests/golden/check_exact.py`, `OPENBLAS_CORETYPE=Haswell`, como o `run.py`); a comparação portável corre em todas as plataformas.

## 5. Por que o `xfail` forte se mantém
O teste exige o fluxo **completo** do Reloaded (tentativa + Original) igual ao do Original. No caso 4 sem efeito mensurável isto é **inalcançável por construção**: o efeito só se conhece depois de executar a tentativa. Está reformulado como `xfail` estrito com a razão exata; os testes positivos acima provam a correção possível. **Não removo o `xfail` sem a tua decisão** (§6).

## 6. Decisão necessária (contrato do caso 4)
Opções: **(a)** aceitar «Original-portion ≡ Original (provado) + tentativa semântica auditada e contabilizada separadamente» como o contrato do caso 4 (e então o `xfail` passa a ser substituído pelo teste positivo já existente); **(b)** exigir identidade do fluxo completo → obriga a mudar o desenho (ex.: `mirror` desligado e aceitar que a árvore semântica sem efeito possa diferir do Original, ou decidir «sem efeito» sem tentativa, o que não é exato); **(c)** manter aberto. Recomendo (a). Nos benchmarks (`runner.py`, `production_training.py`) o `mirror` já está **desligado** para os braços Reloaded; a V2 afeta só os caminhos que mantêm o default (extractor/GUI/`training_config`).

## 7. Desempates m-of-n entre plataformas (tarefa associada)
Evidência (`tiebreak_probe.py`, `tiebreak_probe_result.json`):
* **Origem do ruído:** `IG(mask) ≠ IG(~mask)` bit-a-bit em 1685/2992 casos aleatórios no **Original**; no Reloaded efetivo `score(teste) ≠ score(complemento)` em 554/1332 (|Δ|≤1,1e-16). Os candidatos complementares (`m-of-n(L)` e `(n−m+1)-of-n(¬L)`, mesma partição) têm pontuações iguais em aritmética exata mas diferentes por ulp; a ordenação `(-score, …)` usa o ruído para os separar, e plataformas/instruções (SIMD, libm) podem desempatar de forma diferente.
* **Complementos corretos?** Sim: `m-of-n(L)` é verdadeiro sse `t ≥ m` (t = nº de literais verdadeiros); o complemento é verdadeiro sse `(n−t) ≥ n−m+1`, i.e. `(n−m+1)-of-n(¬L)`; verificado em 1332 testes aleatórios (`~mask` coincide). A forma canónica dos testes (a menor das duas) está matematicamente correta e **só normaliza a etiqueta**, não resolve a causa.
* **Afeta a árvore?** Nas falhas do Windows afetou **só** o campo de auditoria `semantic_test` (regras, previsões, árvore iguais). Mas a sonda mostra sensibilidade real: substituindo o score por uma versão **exatamente simétrica** (mesma matemática, outro ruído), em 120 ajustes sintéticos do **Original** as regras diferem em 57 (previsões em 27; testes diferentes mesmo a menos de complemento em 28; 124/336 nós com orientação diferente). Logo, empates exatos entre candidatos (complementares *e* distintos) resolvidos por ruído tornam a árvore **dependente da plataforma** — propriedade do Original congelado, não introduzida pela Fase 1.
* **Proposta (NÃO aplicada; exige alterar o Original ⇒ nova autorização e manifest):** desempate determinístico e agnóstico: quantizar a pontuação (`round(score, 12)`) e ordenar por `(-score_q, len(literals), m, orientação canónica, feature, threshold, greater)`, com orientação canónica = menor representação entre teste e complemento; idem em `_candidate_literals`. Testes propostos: invariância a ruído (perturbar scores ±1 ulp ⇒ mesma árvore), invariância à orientação (candidato e complemento ⇒ mesma árvore), propriedade em muitos problemas sintéticos, e comparação entre plataformas na CI.
* No caminho ontológico ativo **não** alterei nada (instrução).

## 8. Ficheiros, hashes e estado
Modificados: `core/trepan_reloaded_historical.py`; `tests/eq_support.py`, `tests/test_eq_reloaded_original.py`, `tests/golden/check_exact.py`; novos: `validation/benchmark/v2_before_after.py`, `docs/evidence/b1b4/phase1b/*`. `core/trepan_original.py` e C4.5/MLP **inalterados**. CI e hashes finais: ver atualização no fim.
