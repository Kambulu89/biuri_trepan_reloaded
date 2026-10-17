# Doc. 37 — Protocolo de enriquecimento validado de ontologias (B3.1)

Estado: **PROTOCOLO APROVADO PELO UTILIZADOR, SUJEITO A REVISÃO INDEPENDENTE; NENHUMA ONTOLOGIA ENRIQUECIDA FOI VALIDADA.** Todas as propostas abaixo estão em **`DOMAIN_REVIEW_PENDING`**. Código de verificação: `validation/benchmark/b31_protocol.py`; geração das propostas: `b31_make_proposals.py`; testes: `tests/test_b31_protocol.py`. Pacotes: `docs/evidence/b1b4/b31/proposals/`.

## 1. Papéis
| papel | responsabilidade | quem |
|---|---|---|
| **Especialista do domínio** | verdade e condições de aplicação de cada axioma; confirma as citações exatas | **humano — por designar** (`DOMAIN_REVIEW_PENDING`) |
| **Engenheiro de ontologias** | modelação OWL 2 DL, comportamento do reasoner, adequação do mapeamento a atributos | **humano — por designar** |
| **Responsável pela metodologia** | independência dos dados de teste, rastreabilidade, pré-registo, ausência de *leakage* | **humano — por designar** |
| **Claude (assistente)** | prepara propostas, evidências e verificações automáticas; **não valida** | — |
O verificador rejeita revisores com nome «claude/assistant/ai/…»: uma aprovação não humana fica `REJECTED_NON_HUMAN_REVIEWER` (testado).

## 2. Procedimento (10 passos) e o que é automático
| # | passo | automático (máquina) | humano |
|---|---|---|---|
| 1 | Identificar fontes documentais | verifica que cada axioma tem fonte | escolher/citar a fonte |
| 2 | Propor axiomas justificados | verifica texto, justificação e `relation_class` | decidir o que é conhecimento do domínio |
| 3 | Registar a proveniência | exige `citation` + `locator` por fonte; `verification_required` até confirmação | confirmar a citação exata |
| 4 | Revisão de domínio | — | **aprovar/rejeitar** (`APPROVED` com nome, data e referência de assinatura) |
| 5 | Validar OWL 2 DL | **proxy**: ausência de *punning* classe/propriedade + classificação HermiT sem erro | revisão formal com *profile checker* |
| 6 | Consistência e inferências | HermiT: consistência, insatisfazíveis, nº de axiomas inferidos | interpretar inferências |
| 7 | Matching aos atributos | cobertura e estado do gate (`compile_registry`) | rever ambiguidades |
| 8 | Congelar antes da avaliação | exige `frozen_before_evaluation` + `freeze_commit` | decidir o congelamento |
| 9 | Hashes e versões | SHA-256 do ficheiro vs declarado | arquivo/versão |
| 10 | Comparar original/enriquecida/controlos | exige braços `original, enriched, permuted, random, empty` | aprovar o desenho |
**Declaração de validação independente** (`ontology_declared_independently_validated`): só `True` se as 3 revisões forem `APPROVED` por humanos nomeados **e** todos os passos automáticos passarem. Hoje: **`False`** para todos os pacotes.

## 3. Regras de aprovação de axiomas
1. **Relações formais** (`MATHEMATICAL_IDENTITY`, `FORMAL_DOMAIN_CONSTRAINT`): precisam de fonte, das 5 condições (§4) e de revisão `APPROVED` de domínio **e** metodologia para poderem chegar a `HARD_CONSTRAINT` no registo (Doc. 34 §3). Sem isso: no máximo `PREFERENCE`.
2. **Regularidades empíricas** e **hipóteses**: nunca eliminam candidatos/consultas; podem orientar a pesquisa como preferência *se* documentadas; observação em 100 % das linhas de treino **não** as promove.
3. Axiomas de esquema (hierarquia, domínio, equivalência): aplicam-se a atributos como **preferência**; não são validadores fechados de linhas (Doc. 34 §4).
4. Nenhum axioma é acrescentado para «aumentar a expressividade»; nenhum deriva de rótulos, de resultados ou de teste.

## 4. Relações numéricas — condições obrigatórias
Para tratar «a ≤ b» como restrição formal: (1) **mesma grandeza**; (2) **mesmo conjunto de observações**; (3) **definições estatísticas compatíveis**; (4) **o pré-processamento preserva a relação**; (5) **valores ausentes e transformações tratados**. Classificação obrigatória: identidade matemática · restrição formal do domínio · regularidade empírica · hipótese.

## 5. Propostas preparadas (todas `DOMAIN_REVIEW_PENDING`)
### 5.1 `wdbc_order_relations` — «worst ≥ mean» em 10 famílias de medições (Breast Cancer Wisconsin)
- **Justificação proposta** (a confirmar): na documentação do conjunto, «worst» é a média dos três maiores valores dos núcleos da imagem e «mean» a média de todos; para ≥ 3 núcleos, a média dos 3 maiores ≥ a média de todos (identidade decorrente das definições). **Citação de memória: `verification_required`** (o revisor tem de conferir o texto exato de `wdbc.names`).
- **Diagnóstico empírico (só treino, 5 partições, 10 famílias):** taxa máxima de violação **0,0**. **Isto não prova universalidade** (pedido §10); serve apenas de verificação de consistência.
- **Condições:** grandeza, conjunto de observações e definições: *sim, sujeito à fonte*; **pré-processamento: só no espaço bruto** — padronizar cada coluna com média/DP diferentes **não preserva** a ordem (logo qualquer uso em consultas tem de ocorrer na escala original); valores em falta: nenhum no conjunto, relação só para linhas completas.
- **Estatuto atual:** `HYPOTHESIS_UNTIL_REVIEW` → `permitted_use` ≤ `PREFERENCE`; **não** é restrição obrigatória.
- **Motivação medida (B3.0):** 92–93 % das consultas do amostrador marginal do Original e 43–45 % das do amostrador C violam a ordem; o efeito sobre a fidelidade **não foi avaliado**.
### 5.2 `wine_phenol_subset_rejected` — «flavonoides ≤ fenóis totais» (exemplo negativo)
Violação em **36,0 %** das linhas ⇒ **rejeitado como axioma**; a inclusão conceptual não implica ordem entre medidas (métodos analíticos diferentes). Fica registado para mostrar o que a verificação trava.
### 5.3 Outras candidatas (por propor, sem pacote)
Digits: topologia espacial da grelha de píxeis (conhecimento estrutural verdadeiro) → unidades por proximidade; Breast Cancer: relações geométricas **aproximadas** (perímetro ≈ 2π·raio, razão mediana 1,03; área ≈ π·raio², 0,98) → no máximo regularidade/preferência, nunca restrição; Iris: sem relações adicionais conhecidas a priori — o mecanismo deve abster-se.

## 6. Congelamento e versões
- Uma ontologia enriquecida só é avaliada depois de: 3 revisões humanas, `sha256` registado, `freeze_commit` no *git*, problemas/critérios/braços pré-registados (`preregistration`), e comparação com ontologia original, permutada ao nível da relação, aleatória com o mesmo histograma de tipos de axioma, vazia (válida sem axiomas úteis) e controlo convencional de dimensão igualada.
- O avaliador de identificabilidade (Doc. 36) já aplica este princípio: `problems_frozen.json` com hash do código; recusa avaliar se o código mudou (`--allow-drift` obriga a registar o desvio).
- Mudanças posteriores à aprovação ⇒ nova versão e nova revisão; nunca se editam ontologias congeladas.

## 7. Breast Cancer: controlos
Mantêm-se os controlos históricos e os estados de atribuição publicados (Docs. 29–31); a campanha M29 não é reescrita. A padronização equivalente das derivadas em **todos** os braços é autorizada **apenas como nova experiência** com identificação própria, com critérios de comparabilidade explícitos (mesma representação, candidatos, distribuição de consultas e oráculo; verificação de que o MLP não muda); onde a atribuição não possa ser avaliada, `NOT_ASSESSABLE`; `NOT_ATTRIBUTED` mantém-se para «sem ganho direto» (distinto da validade de cada controlo). **Nada disto foi executado.**

## 8. Lacunas e pedidos
Designar os três revisores; fornecer/confirmar as fontes; indicar quais propostas (§5) seguem para revisão; decidir se se constrói uma ontologia enriquecida real para Breast Cancer/Digits a submeter ao protocolo.
