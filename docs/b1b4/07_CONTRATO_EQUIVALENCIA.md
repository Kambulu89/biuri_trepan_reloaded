# Documento 7 — Contrato de equivalência e arquitetura «Reloaded = Original + extensão ontológica» (PROPOSTA a aprovar)

Este documento **prevalece** sobre qualquer passagem dos Documentos 1–6 que o contradiga. Origem: correção arquitetural do utilizador.

## 1. Princípio arquitetural
O TREPAN Reloaded **não é um algoritmo diferente** do TREPAN Original: é o Original **acrescido de mecanismos de enriquecimento semântico derivados de uma ontologia OWL válida**.
O efeito adicional da ontologia é aquilo que se investiga. Qualquer melhoria que exista **sem** ontologia (arquitetura, hiperparâmetros, pré-processamento, oráculo, procura, poda, uso de
rótulos reais) é **proibida no Reloaded**: tornaria impossível atribuir o ganho ao conhecimento.

## 2. Contrato de equivalência (EQ) — definição formal
Seja `θ = (D, O, P, s, E)`: dados de treino `D`, oráculo congelado `O` (mesmo `oracle_id`), parâmetros `P` (orçamento, pureza, m-of-n, beam…), seed `s`, ambiente `E`. Seja `Ω` a informação
ontológica disponível ao Reloaded; `Ω = ∅` se `ontology=None` **ou** se a ontologia foi rejeitada pelo quality gate **ou** não produz estrutura semântica ativa.

```
EQ:   Reloaded(θ, Ω=∅)  ≡  Original(θ)
```
«≡» significa igualdade em **todas** as saídas observáveis abaixo (com tolerância zero em inteiros/strings/bits; floats **bit-a-bit**):

| Id | Saída | Critério |
|---|---|---|
| E1 | construção da árvore | mesma estrutura nó-a-nó: ids, literais, limiares, ordem, folhas, previsões das folhas |
| E2 | geração de regras | `export_rules()` idêntico |
| E3 | condições m-of-n | mesmos `m`, `n`, conjuntos de literais e ordem |
| E4 | consultas ao oráculo | **mesmo fluxo de entradas ao oráculo** (hash da sequência de arrays de `predict`/`predict_proba`), mesmo nº de amostras e de chamadas, mesmo `membership_queries_` |
| E5 | previsões | `predict` (e `predict_proba` se existir) bit-a-bit em qualquer X |
| E6 | fidelidade e accuracy | iguais (derivam de E5) |
| E7 | complexidade | nós, folhas, profundidade, comprimento médio das regras, nº de m-of-n iguais |
| E8 | consumo de aleatoriedade | mesmas contagens de candidatos/splits avaliados (contadores algorítmicos) |
| E9 | seleção de divisões | mesmos scores e vencedor em cada nó |

**Exclusões autorizadas (allowlist):** apenas campos de *log/proveniência* que não influenciem E1–E9 (p.ex. `semantic_split_audit_` vazio/neutro, timestamps, tempos). A allowlist vive num ficheiro
versionado e está **vazia por omissão**; qualquer entrada exige autorização explícita prévia do utilizador. Qualquer outra diferença = **violação do contrato**.

## 3. Contrato de enriquecimento (EN)
Com `Ω ≠ ∅` (ontologia aceite pelo quality gate **e** estrutura semântica ativa) o Reloaded pode acrescentar, **cada mecanismo controlável, auditável e justificável**, apenas:
correspondência semântica de atributos · relações de hierarquia/domínio · enriquecimento de representações · restrições e relações ontológicas · orientação semântica da seleção de divisões ·
calibração do contributo semântico (B1) · avaliação/controlo de divisões influenciadas pela semântica (B2) · seleção de configurações do Reloaded enriquecido (B3) · m-of-n guiado por relações
ontológicas (B4) · mecanismos de aceitação/seleção associados ao enriquecimento.
Cada mecanismo tem uma flag, um registo de auditoria (`mechanism_id`, entradas, decisão, razão) e um controlo negativo (ontologia permutada/aleatória).

## 4. Como a equivalência passa a ser **estrutural** (e não coincidência)
1. **Despacho no topo:** cada método sobreposto em `TrepanReloadedClassifier` começa por
   `if not self._ontology_effective(): return super().<método>(...)` — delega no código do Original sem tocar em nenhuma política B1–B4, EFSR, candidatos semânticos, perfil de erro ou
   consulta extra ao oráculo. `_ontology_effective()` = ontologia aceite ∧ estrutura semântica ativa (`_semantic_structure_active()` + features/pesos não neutros).
2. **Sem efeitos laterais fora do ramo semântico:** nada que consuma oráculo, RNG ou tempo algorítmico é executado no ramo `Ω=∅`.
3. **Mesma via para «sem ontologia», «ontologia rejeitada» e «ontologia neutra»:** todas chegam ao ramo 1.
4. **Teste de contrato** (§6) corre em CI para todas as variantes B1–B4 *com as flags ligadas* e `Ω=∅`: as flags têm de ser inertes.
5. Proibido: constantes «melhoradas» no Reloaded que difiram das do Original (p.ex. `min_samples_leaf` por omissão, `purity_*`): o Reloaded herda os parâmetros do Original; só o que é semântico é novo.

## 5. Estado atual do contrato (auditoria empírica, só leitura)
Sonda `docs/evidence/b1b4/equivalence_probe.py` (3 datasets × 2 seeds; MLP e oráculo congelado partilhados; Reloaded configurado **como o braço D do benchmark**, sem informação ontológica):

| Saída | Resultado |
|---|---|
| E1/E2 regras | **iguais** em 6/6 |
| E5 previsões (teste) | **iguais** em 6/6 |
| E7 nº de nós | **iguais** em 6/6 |
| `membership_queries_` | **iguais** em 6/6 |
| **E4 fluxo de entradas ao oráculo / amostras / chamadas** | **DIFERENTE em 6/6**: Reloaded consulta ≈ 2,6–3,1× mais amostras (p.ex. iris/42: 652 vs 1712; breast_cancer/7: 1109 vs 3408) e ≈ 2,6–2,8× mais chamadas (5→13; 4→11) |

Causa (código): `TrepanReloadedClassifier._decision_sample` → `_build_error_region_profile(include_uncertainty=True)` → `_known_row_uncertainty` chama `oracle.predict_proba` em **todos** os nós, mesmo
sem estrutura semântica (`core/trepan_reloaded_historical.py` l. 391–409, 544–560). O resultado não altera a árvore quando `Ω=∅`, mas **viola E4** e contamina métricas de custo
(`oracle_queries` do `FrozenOracle`) e qualquer comparação de «eficiência» Original vs Reloaded Core.
⇒ **Não-conformidade V1** (corrigir em Fase 1a, antes de qualquer B1–B4, com despacho no topo).
Limitação: a sonda cobre 3 datasets/2 seeds/um orçamento; a conformidade geral exige o teste de propriedade do §6.

## 6. Testes de regressão do contrato (a criar na Fase 1a; nenhum implementado ainda)
1. `test_eq_no_ontology`: para N datasets sintéticos (numéricos, mistos, desequilibrados, binários/multiclasse, p∈[3,60]) × seeds × orçamentos, `Reloaded(Ω=∅)` vs `Original`: E1–E9 (oracle wrapper
   com hash do fluxo de entradas).
2. `test_eq_ontology_rejected`: ontologia presente mas rejeitada pelo gate ⇒ mesmas igualdades.
3. `test_eq_ontology_neutral`: ontologia aceite sem features/grupos/pesos ⇒ mesmas igualdades.
4. `test_eq_all_flags_on`: B1–B4 e EFSR ligados com `Ω=∅` ⇒ mesmas igualdades (flags inertes).
5. `test_eq_oracle_accounting`: contagem de amostras/chamadas do `FrozenOracle` por escopo iguais.
6. `test_eq_allowlist_empty` : a allowlist está vazia; qualquer adição falha o teste até haver autorização registada no manifest.
7. Metamórficos: permutar colunas/linhas, renomear colunas, escalar features monotonicamente ⇒ equivalência preservada; dataset desconhecido gerado.
8. `test_enrichment_changes_only_when_effective`: com `Ω≠∅` e estrutura ativa, pelo menos um mecanismo regista intervenção **ou** a árvore é idêntica ao Original (nunca «diferente sem registo»).
9. Verificação no benchmark: braço **EQ** (Reloaded sem ontologia, com todas as flags) comparado ao Original em **cada unidade**; violação ⇒ unidade marcada `EQ_VIOLATION`, execução bloqueada.

## 7. Atribuição de ganho (o que «medir contra o Original equivalente» significa)
O Original de comparação partilha **oráculo, dados, parâmetros, seed e condições**. Como a ontologia pode enriquecer a *representação*, definem-se contrastes pareados:
| Contraste | Definição | Mede |
|---|---|---|
| Δ_total | R_OWL − O (features originais) | efeito conjunto da ontologia |
| Δ_repr | O(repr. OWL) − O | enriquecimento da representação (Original a correr no espaço enriquecido, mesmo oráculo via projeção) |
| Δ_guia | R_OWL − O(repr. OWL) | orientação semântica (score, m-of-n, aceitação, projetor de queries) para a **mesma** representação |
| Δ_conhecimento | R_OWL − R_OWL~perm / ~rand | conhecimento específico da OWL vs estrutura arbitrária |
| Δ_convencional | O(repr. OWL) − O(repr. engenharia por nomes) | OWL vs engenharia convencional equivalente |
`Δ_total = Δ_repr + Δ_guia`. Nenhum ganho é atribuído à ontologia se `Δ_conhecimento` ou `Δ_convencional` o explicarem.
Duas pistas separadas: **T (árvore)** — oráculo = MLP original; **M (MLP)** — oráculo = MLP ontológico (gate interno), comparações de fidelidade só entre árvores que consultam o **mesmo** oráculo.

## 8. O que sai do âmbito do Reloaded
* **Modo B / híbrido com rótulos reais** (antiga proposta B3): independente da ontologia ⇒ **removido** de B1–B4. Se desejado, é outra linha de investigação (`hybrid`), com nome, braço e relatório
  próprios, **nunca** chamado Reloaded, e só com autorização expressa.
* **Poda 1-SE genérica (B2-D):** só admissível restrita a subárvores que contêm divisões influenciadas pela ontologia; poda geral ⇒ removida.
* **Ganhos «do Core»:** não há objetivo de o Reloaded sem ontologia superar o Original (substitui a antiga linha de investigação de ganho arquitetural independente — a «D-9» que referes não existe nos
  meus documentos com esse número; interpretei-a como esta e fica substituída por D-9 abaixo).


## 9. Contrato de equivalência **v2.0 — estados S0–S3** (decisão de 2026-10-13; substitui a exigência de identidade do fluxo completo no caso 4)
`Reloaded_OWL = Original + tentativa semântica auditada`. Versão do contrato no código: `TrepanReloadedClassifier.EQ_CONTRACT_VERSION = "2.0-S0S3"`; cada ajuste regista-a em `oracle_accounting_`.

| Estado | Definição | Exigência |
|---|---|---|
| **S0** | ontologia ausente | equivalência **integral** E1–E9 (inclui fluxo de consultas) |
| **S1** | ontologia rejeitada (quality gate / grafo inativo) | equivalência **integral** E1–E9 |
| **S2** | ontologia válida **sem mecanismo semântico executado** (nenhuma estrutura/hook ativo) | equivalência **integral** E1–E9 |
| **S3** | ontologia válida **com tentativa semântica executada** | **porção Original ≡ Original isolado** (regras/previsões/fluxo de pedidos) **+ tentativa semântica auditada**; todas as consultas adicionais atribuídas à tentativa semântica; sem alterações ocultas ao oráculo |

Tentativa executada **não** é melhoria observada. Sem alteração de divisões, previsões ou métricas o estado é `SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN`, que **não autoriza** declarar ganho ontológico. Em S3 com `mirror` a porção Original é a reexecução (fluxo exatamente o do Original isolado; chamadas byte-idênticas reutilizadas contabilizadas à parte); em S3 sem `mirror` a passagem completa é a tentativa e a porção Original isolada é o braço de referência medido à parte (`extra_oracle_* vs Original`).

**Contabilidade obrigatória** (`oracle_accounting_`; `validation/benchmark/semantic_cost.py`): `original_oracle_calls/samples`, `semantic_oracle_calls/samples`, `reused_oracle_calls/samples`, `total_oracle_calls/samples` (= original + semantic − reused = recebido pelo oráculo, verificado contra espião), `semantic_attempt_executed`, `semantic_effect_observed`, `semantic_attempt_status`; fora do ajuste (conjuntos de avaliação explícitos, nunca o teste na otimização): `semantic_gain_accuracy`, `semantic_gain_fidelity`, ganho por 1 000 amostras adicionais, diferença de complexidade, taxa de aceitação de candidatos semânticos e proporção de tentativas sem efeito. Distinguir `F_R ≥ F_O` (não-inferioridade observada) de `F_R > F_O` (superioridade estrita observada).

**Testes positivos (substituem o `xfail` da V2):** `tests/test_eq_reloaded_original.py` — `test_contract_state_classification_S0_S1_S2_S3`, `test_contract_S0_S1_S2_full_equivalence_and_zero_semantic_cost`, `test_contract_S3_original_portion_equivalent_and_attempt_fully_accounted`, `test_contract_S3_without_mirror_attempt_is_whole_pass_and_effect_flagged`, `test_contract_gain_is_not_inferred_from_attempt`, mais o diferencial bit-a-bit contra a cópia congelada de `main@1789c17` (comportamento com ontologia efetiva preservado) e os testes V2 (reutilização só byte-idêntica). A alteração das **âncoras originais** continua proibida.

**Estado global:** `GLOBAL_EQ_CONTRACT v2.0: COMPLIANT` — verificado com CI verde no commit exato `db417ff` (Windows 3.11/3.12 run 143 ✔, `agnosticism-guard` run 62 ✔) e no seguinte `0e3cf27` (runs 144 e 63 ✔); suite local 0 falhas (4 shards). Reservas: desempate m-of-n dependente de plataforma e âncora exata guardada dependente do anfitrião (limitações de reprodutibilidade registadas). Desempate m-of-n dependente de plataforma: **limitação de reprodutibilidade registada** (Original canónico não modificado); estudo de política determinística separado (Doc. 16 §4).
