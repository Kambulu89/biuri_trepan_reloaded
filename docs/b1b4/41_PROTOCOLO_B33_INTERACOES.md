# Doc. 41 — Protocolo B3.3: interações sem sinal univariado (pré-registo)

Estado: **PRÉ-REGISTADO** em commit próprio, antes de qualquer avaliação nas sementes 400–429. Só `validation/`, dados sintéticos. Nenhum dataset real, DEV ou confirmatório.
Contexto: `B32_RESULT = NO_ONTOLOGICAL_ADVANTAGE`, `B32_SCOPE = SYNTHETIC_P1_P2_EVALUATED_REGIMES`, `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED` (Doc. 38). A B3.2 mostrou que o rastreio univariado
genérico (F2) reproduz as árvores da ontologia quando o conhecimento só ajuda a *escolher atributos*. A B3.3 pergunta: **quando o par relevante não tem sinal univariado (ou só um sinal fraco),
a ontologia dá uma vantagem que nem o rastreio univariado, nem o rastreio genérico de INTERAÇÕES por pares, nem o Original alargado igualam?**

## 1. Famílias de problemas

Gerador comum (B3.1/B3.2): atributos gaussianos com escalas aleatórias, ordem de colunas aleatória, conceitos verdadeiros = grupos de atributos; **par de atributos relevantes** `(a,b)`; oráculo = MLP
(sklearn, `StandardScaler`+`MLPClassifier(32)`) treinado nos rótulos da família; rótulos reais = regra com ruído de rótulo `eta` (treino e teste). Limiar de cada atributo no quantil `q` da sua distribuição.

| Família | Regra | `q` | `eta` | Layout / ontologia verdadeira | p | Notas |
|---|---|---|---|---|---|---|
| I1_AND_PAIR | `a ∧ b` | 0,5 | 0 | par num conceito (TRUE) | 32 | controlo **com** sinal univariado; não conta para o veredicto global |
| I2_XOR_NO_SIGNAL | `a ⊕ b` | 0,5 | 0 | par num conceito | 32 | **sem sinal univariado**; limite indutivo esperado |
| I3_XOR_WEAK_SIGNAL | `a ⊕ b` | 0,65 | 0 | par num conceito | 32 | sinal univariado fraco |
| I4_XOR_WEAK_NOISY | `a ⊕ b` | 0,65 | 0,10 | idem | 32 | ruído de dificuldade variável |
| I5_XOR_WEAK_HIERARCHY | `a ⊕ b` | 0,65 | 0 | `a`,`b` em conceitos **irmãos** (HIERARCHY: pai comum «relevante») | 32 | relações hierárquicas entre conceitos |
| I6_XOR_WEAK_LARGE_BUDGET | `a ⊕ b` | 0,65 | 0,05 | par num conceito | 48 (8×6) | espaço maior, `max_queries` 2500: limites reais de orçamento |

Atributos irrelevantes: todos os restantes (p−2). Ontologia **incompleta** (N_INCOMPLETE: um atributo do par fora do conceito) e **irrelevante** (N_IRRELEVANT: conceitos verdadeiros sobre outros atributos);
**permutada** (D), **aleatória** (E), **sem ontologia** (A, B, F–H). Desvio registado: o desenho previa I6 com 16 conceitos (p=64); o gerador B3.1 só tem 8 temas × 6 medidas, o primeiro arranque da calibração de I6
terminou com erro antes de produzir resultados (`IndexError`) e I6 passou a 8×6 (p=48) sem alterar qualquer outro ficheiro; a calibração de I6 é a única que se executou para esta família.

**Expressividade.** Um único teste m-of-n não representa a XOR de um par. A XOR é representável por uma árvore de **2 níveis** (os limites `max_depth=3`, `max_nodes=7` chegam). Define-se
`rule_recovery` (= `recovered_path`) como «os dois atributos do par no mesmo teste OU no mesmo caminho raiz→folha»; reporta-se também `pair_same_test`. Quando a regra não é representável/aprendível por
nenhum braço (recuperação < 0,10 em todos), a família é declarada `UNINFORMATIVE_NO_ARM_RECOVERS` (limite indutivo do algoritmo, não conta a favor nem contra a ontologia).

**Dificuldade** definida **sem** consultar variantes ontológicas nas sementes de avaliação: (i) força do sinal univariado = parâmetro `q` do gerador (decidido a priori); (ii) orçamento de pesquisa `max_features_per_node` calibrado **só com o Original**
(recuperação mais próxima de 0,5 em [0,2; 0,8]) em sementes de calibração 3000–3019 — **uma única calibração técnica**.

## 2. Braços (A–I obrigatórios + referências de orçamento + negativos)

| Braço | Descrição |
|---|---|
| A_ORIG | TREPAN Original canónico, orçamento limitado calibrado |
| B_WIDE | Original com `max_features_per_node = p` |
| C_OWL | orientação por ontologia verdadeira da família (harness B3.1/B3.2, preferência pura) |
| D_PERM / E_RAND | ontologia permutada / aleatória (mesmos tamanhos de grupos) |
| F_CONV | engenharia convencional: agrupamento por |Spearman| (mesmo nº/tamanho de unidades) |
| G_UNI_SCREEN | rastreio genérico univariado |
| H_PAIR_SCREEN | rastreio genérico de **interações por pares** (melhor ganho de `(x_i⋚t_i)∧(x_j⋚t_j)`, 3×3 limiares × 4 sentidos, face às previsões do oráculo); 36 candidatos por par; mesmo nº/tamanho de unidades |
| I_NEUTRALIZED | ontologia verdadeira com as relações do par neutralizadas |
| J1 / J2 | WIDE com `beam_width` calibrado (só recursos) para igualar candidatos avaliados / CPU de C |
| N_IRRELEVANT, N_INCOMPLETE | negativos |
Referências: C4.5 canónico (rótulos reais) e MLP. O Original não é alterado; a orientação é preferência estatística (não altera pontuação nem elimina candidatos).

## 3. Orçamentos (reportados em separado; nenhuma igualdade implícita)

Candidatos disponíveis (literais listados, gerados simples/m-of-n), candidatos examinados (`eval_calls`), literais, consultas e amostras ao oráculo, CPU de ajuste, CPU de compilação (reasoner) e de rastreio
(`screen_wall_s`), **custo total** (ajuste+compilação+rastreio), memória (sementes 500–509, `tracemalloc`), nós, profundidade, condições m-of-n. Comparação 1 = igual nº de candidatos examinados (C vs J1);
comparação 2 = igual CPU de ajuste (C vs J2); a compilação/rastreio é somada ao custo total. Não se alarga o espaço de candidatos para a ontologia (mesmas unidades/pesquisas ≤ 3 em C–I).
Custo de H (rastreio por pares) contabilizado no custo total; não se afirma igualdade quando só uma dimensão é igual.

## 4. Métricas

`rule_recovery`, `accuracy` (rótulos reais, métrica principal), `fidelity` (ao mesmo MLP), `precision_macro` e `macro_f1` (complementares), `oracle_queries`, `candidate_count`, `tree_complexity` (nós, profundidade, m-of-n),
`cpu_time`, `ontology_attribution`, `joint_gain_status`. A recuperação sozinha não basta; o critério central é o **ganho conjunto accuracy+fidelidade** (Nível 1: C vs A equivalente).

## 5. Critérios pré-registados (executáveis em `b33_run.CRITERIA33` / `b33_report.py`)

* **Ganho conjunto** (por família): `OBSERVED_JOINT_GAIN` se Δacc(C−A)>0 e Δfid(C−A)>0; `SUPPORTED_JOINT_GAIN` se ambos com IC95 inferior > 0; `TRADEOFF` se sinais opostos; senão `NO_JOINT_GAIN`.
* **Atribuição**: `ATTRIBUTED` se `SUPPORTED_JOINT_GAIN` e, contra **cada** D, E, I (ontologia) e B, F, G, H, J1, J2 (genéricos), Δacc>0 e Δfid>0 com IC95 inferior > 0; senão `NOT_ATTRIBUTED`.
* **Vantagem de recuperação**: R1 = rec(C) − max(rec D,E,F,G,H,I) ≥ 0,20; R2 = rec(C) − rec(B) ≥ 0,10 e rec(C) − max(rec J1, J2) ≥ 0,10.
* **Veredicto por família**: `ONTOLOGICAL_ADVANTAGE` (R1 e R2 e ATTRIBUTED); `RELATION_SPECIFIC_ONLY` (R1 sem o resto); `NO_ONTOLOGICAL_ADVANTAGE` (R1 falha); `UNINFORMATIVE_NO_ARM_RECOVERS`.
* **Veredicto global**: `B33_ADVANTAGE_ESTABLISHED` (≥ 2 de I3–I6 com `ONTOLOGICAL_ADVANTAGE`), `B33_PARTIAL` (1), `B33_NO_ONTOLOGICAL_ADVANTAGE` (0).
* **Método mais simples equivalente** (`SIMPLER_EQUIVALENT`): controlo com |Δacc|,|Δfid| ≤ 0,005 face a C e custo total ≤ 0,8·custo total de C; reporta-se sempre.
* Cadeia da hierarquia (Nível 2) apenas exploratória (C não é o Reloaded integrado).
* Unidade de inferência = família; 30 sementes por família (400–429), emparelhadas; bootstrap 4000; sementes não repetidas nem escolhidas; sem DEV nem confirmatórios.

## 6. Orçamento computacional (limite duro) e execução

Estimativa e limite: ver §7 (preenchido no congelamento). Ordem: testes de funcionamento (`tests/test_b33_interactions.py`, `tests/test_b33_report.py`) → calibração única → congelamento (este commit) →
estimativa de orçamento → execução das 6 famílias × 13 braços × 30 sementes → memória em sementes de diagnóstico 500–509 → análise (Doc. 42).

## 7. Calibração congelada e orçamento

Calibração técnica única (sementes 3000–3019, só Original e recursos), `docs/evidence/b1b4/b33/{cal_*.json,b33_frozen.json}` (hash do código no JSON; o avaliador recusa-se a correr se o código mudar).

| Família | `max_features_per_node` | taxa de recuperação do Original por `mf` (caminho) | `beam_width` J1 / J2 |
|---|---|---|---|
| I1_AND_PAIR | 24 | 16:0.20, 20:0.30, 24:0.55, 28:0.85, 30:1.00 | 32 / 8 |
| I2_XOR_NO_SIGNAL | 30 | 16:0.05, 20:0.05, 24:0.00, 28:0.10, 30:0.20 | 16 / 6 |
| I3_XOR_WEAK_SIGNAL | 24 | 16:0.20, 20:0.35, 24:0.60, 28:0.80, 30:1.00 | 32 / 8 |
| I4_XOR_WEAK_NOISY | 24 | 16:0.20, 20:0.30, 24:0.55, 28:0.80, 30:1.00 | 24 / 6 |
| I5_XOR_WEAK_HIERARCHY | 20 | 16:0.35, 20:0.40, 24:0.65, 28:0.85, 30:0.95 | 48 / 6 |
| I6_XOR_WEAK_LARGE_BUDGET | 36 | 24:0.25, 30:0.35, 36:0.50, 40:0.70, 44:0.80 | 16 / 6 |

I2 (sem sinal univariado) fica abaixo da banda-alvo [0,2; 0,8] em quase toda a grelha (recuperação ≤ 0,20 mesmo com `mf` próximo de p): foi escolhido o `mf` com taxa mais próxima de 0,5 dentro da banda (30, taxa 0,20), a
única grelha de I2 que satisfaz a regra de calibração; o resultado confirma o limite indutivo esperado.

**Estimativa de orçamento** (medida em 3 sementes por família durante os testes de funcionamento; ≈25–35 s de CPU por semente para os 13 braços, mais para I2/I6): 6 famílias × 30 sementes ≈ 180 ajustes completos ≈
1,6–2,2 CPU-h; memória (10 sementes × 4 braços × 6 famílias) ≈ 0,3 CPU-h; análise desprezável. **Total estimado ≈ 2–2,5 CPU-h**, em 4 processos paralelos ≈ 0,6 h de relógio.
**Limite duro: 4 CPU-h** de execução de avaliação (se ultrapassado, a execução pára e reporta-se; nada se repete nem se reescolhe). Sem DEV; sem confirmatórios.
