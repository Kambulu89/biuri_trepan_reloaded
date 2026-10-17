# Doc. 50 — Seleção de ontologia expressiva de domínio para o estudo H2

Estado: **investigação documental e proposta** (`B3Q = PROTOCOL_AUTHORIZED`). Nenhuma ontologia expressiva foi aprovada; nada foi validado por humanos; as fontes estão **citadas de memória e não foram verificadas** (sem acesso de rede a `archive.ics.uci.edu`; a tentativa de leitura nesta sessão falhou por resolução de DNS). Só `validation/` e documentação; nenhum benchmark executado.

## 1. Regra de seleção
A escolha do dataset e dos axiomas **não depende** de accuracy ou fidelidade históricas do Reloaded nem de qualquer resultado de teste. Os critérios são os 12 pedidos, avaliados por **documentação, estrutura lógica e viabilidade técnica**, a priori. Para evitar a suspeita de seleção favorável, regista-se que o candidato proposto (German Credit) tem **historial desfavorável** ao Reloaded (fidelidade 0,927 vs 0,994 do Original no estudo contrafactual anterior, com uma implementação anterior do Reloaded, Doc. 12): a escolha vem da estrutura documentada, não do desempenho.

## 2. Critérios (os 12) e escala
Escala descritiva `FORTE / PARCIAL / FRACO / NÃO VERIFICADO` (sem pontuação agregada): D1 documentação independente; D2 conceitos de domínio conhecidos; D3 propriedades e relações verdadeiras; D4 hierarquias substantivas; D5 restrições documentadas; D6 possibilidade de inferência; D7 compatibilidade OWL 2 DL; D8 cobertura dos atributos; D9 qualidade do matching; D10 utilidade potencial para decisões m-of-n; D11 disponibilidade de revisão humana; mais **D12 estatuto no protocolo** (exposição/confirmatório) e **D13 disponibilidade offline dos dados**.

## 3. Matriz de candidatos
| Candidato | D1 | D2 | D3 | D4 | D5 | D6 (inferência OWL) | D7 | D8 | D9 | D10 (m-of-n) | D11 | D12 estatuto | D13 dados |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **German Credit (Statlog)** | PARCIAL (dicionário de dados oficial; **não verificado**) | FORTE (conta, poupança, emprego, finalidade, habitação…) | PARCIAL | **PARCIAL** (escalas ordinais; taxonomia de finalidades) | PARCIAL (limites de intervalos) | **FORTE** (classes cumulativas ⇒ subsunções inferidas, verificado: 55 pelo HermiT) | FORTE (verificado) | **FORTE** (61/61) | FORTE (rótulos exatos) | **PARCIAL–FORTE**: literais `1-de-n` ordenados para variáveis ordinais one-hot | PENDENTE | exposto (EC1) ⇒ **só DEV exploratório** | local (`german.data`) |
| WDBC (breast_cancer) | FORTE (descrição de Street et al. 1993; não verificado) | FORTE (grandeza × estatística) | PARCIAL (`worst ≥ mean` por definição; razões só por núcleo) | PARCIAL | PARCIAL (ordem) | FRACO (OWL 2 DL não exprime ordem entre valores) | PARCIAL | FORTE | FORTE | PARCIAL (restrição de consultas; Doc. 24) | PENDENTE | DEV (exposto) | local |
| Wine | PARCIAL | PARCIAL (fenóis ⊇ flavonoides, proantocianidinas) | PARCIAL (ordem numérica **rejeitada**) | PARCIAL | FRACO | PARCIAL | FORTE | FORTE | FORTE | FRACO–PARCIAL | PENDENTE | DEV (exposto) | local |
| Digits | PARCIAL | FRACO (pixeis) | PARCIAL (adjacência, limites 0–16) | FRACO | PARCIAL (inteiros 0–16) | PARCIAL | FORTE | FORTE | FORTE (nomes `pixel_r_c`) | PARCIAL (vizinhança, mas **recuperável por correlação** com n grande) | PENDENTE | DEV (exposto) | local |
| Iris | PARCIAL | PARCIAL | FRACO | FRACO | FRACO | FRACO | FORTE | FORTE | FORTE | FRACO (4 atributos) | PENDENTE | DEV (exposto) | local |
| Car Evaluation / Nursery | FORTE (modelo DEX documentado) | FORTE | FORTE | FORTE | FORTE | FORTE | FORTE | FORTE | FORTE | **a ontologia espelha o gerador do rótulo** (fuga) | PENDENTE | **candidato confirmatório (Doc. 8): excluído** | indisponível |
| Heart Disease, Pima, Adult e restantes | variável | — | — | — | — | — | — | — | — | — | — | Heart = reserva R6 do confirmatório (**excluído**); Pima: sem dados offline, nunca verificado; Adult: ficheiro OWL com ABox (97 MB) ⇒ rejeitado pelo gate de TBox | **indisponível ou não elegível** |
Nenhum candidato fora do repositório pôde ser verificado (sem rede). Os datasets das listas confirmatória e de reservas (Doc. 8) **não são usados**.

## 4. Propostas
**P1 — principal: German Credit, estudo DEV exploratório.** Razões estruturais (não de desempenho): (i) atributos categóricos one-hot cuja **ordem** (conta corrente, poupança, antiguidade no emprego, qualificação profissional) é informação documentada **ausente da representação one-hot**; (ii) taxonomia das finalidades (veículos; bens domésticos); (iii) representável em **OWL 2 DL** com classes definidas (união), com **inferência real** (verificada: 55 subsunções inferidas, 10 literais derivados que dependem do reasoner); (iv) os controlos são **limpos** neste dataset (ordens permutadas e uniões aleatórias, ao contrário de WDBC onde a permutação é impossível, Doc. 29); (v) cobertura 61/61.
Limitações: **exposto** (EC1) ⇒ nunca confirmatório; um único dataset ⇒ unidade de inferência = 1, qualquer afirmação é `observed_only`; os níveis ordinais e a taxonomia são **rascunho de IA** de memória do dicionário de dados, **sem fonte verificada nem revisão**; atributos sensíveis (`personal_status_sex`, `foreign_worker`, `age`) não são modelados como conhecimento; só 4 variáveis ordinais + 1 taxonomia (14 fichas); o ganho potencial reside em **orientar** a pesquisa, não em alargar o espaço de hipóteses (cada literal é um `1-de-n` que o Original já pode formar).
**P2 — secundária: WDBC com relação de ordem `worst ≥ mean` (T5/T4)** para restringir a plausibilidade das consultas (violações medidas no Doc. 24). Depende de revisão de domínio (`wdbc_order_relations`), de um facto `ORDER_RELATION` utilizável e de um mecanismo de reparação de consultas; a atribuição por permutação é `NOT_ASSESSABLE` (Doc. 29), pelo que só admite o controlo «relação neutralizada» e um controlo de ordens falsas **declarado como negativo**.
**P3 — terciária: Digits com adjacência 4-vizinhança (T9)** — baixa informação externa (a adjacência é recuperável dos dados com n=1347) e exige novo tipo de facto. Wine e Iris: sem estrutura expressiva suficiente para H2. `wine_phenol_subset_rejected` permanece **rejeitada**.
**P0 — validação de mecanismo em sintético:** um gerador com escalas ordinais latentes, em que a verdade é conhecida, para verificar que o mecanismo T1/T2 faz o que o contrato diz (Doc. 52); não substitui conhecimento de domínio real.

## 5. Fontes a localizar e confirmar pelos revisores (todas `NOT_VERIFIED`)
* German Credit: dicionário de dados do repositório UCI (Statlog), descrição dos atributos 1 (conta corrente), 4 (finalidade), 6 (poupança), 7 (emprego atual), 17 (profissão); níveis `A11–A14`, `A61–A65`, `A71–A75`, `A171–A174`, `A40–A410`.
* WDBC: Street, Wolberg & Mangasarian (1993) e descrição UCI (estatísticas média / erro padrão / «pior»).
* Digits (optdigits): descrição UCI do pré-processamento (bitmaps 32×32 → blocos 4×4 → contagens 0–16).
* Wine: descrição UCI das 13 constituintes.
Para cada fonte, os revisores confirmam: texto exato, numeração, ordem dos níveis, âmbito e licença. Até lá todas as fichas ficam `HYPOTHESIS` e `*_REVIEW_PENDING`.

## 6. Pacote de revisão (preparado, a preencher por pessoas reais)
`validation/benchmark/ontologies/proposals/german_credit/v0_draft/`: `README.md` (estado), `sources.json` (`NOT_VERIFIED`), `axioms/*.json` (14 fichas validadas por `h2_contract.validate_card`: IA-rascunho, `HYPOTHESIS`, fonte não verificada, revisões pendentes), OWL O1/O2/O3/H/I com SHA-256 em `docs/evidence/b1b4/b3q/h2_german_proposal_check.json`, relatório do reasoner. Os formulários de revisão são os do Doc. 39. **Não existe revisor designado, assinatura nem aprovação; nenhum axioma de IA é apresentado como conhecimento de domínio independentemente validado.**

## 7. Dependências e limites
Designação dos três revisores; acesso à documentação original; decisão sobre aceitar um estudo H2 de um único dataset como exploratório; ligação do mecanismo ao arnês (Doc. 51 §5); ontologia de domínio para um **segundo** dataset real não confirmatório se se pretender qualquer inferência entre datasets (não existe candidato verificável hoje).
