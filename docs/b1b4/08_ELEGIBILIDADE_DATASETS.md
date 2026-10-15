# Documento 8 — Elegibilidade dos datasets CONFIRM (Fase 0.5) — D-7

**Estado: PRÉ-VERIFICAÇÃO DOCUMENTAL, não verificação.** O ambiente desta sessão **não tem acesso** a `archive.ics.uci.edu` nem a `openml.org` (o proxy devolve 403 / ligação recusada). Por isso:
* nada abaixo foi descarregado, inspecionado ou medido; as colunas «esperado» vêm do **meu conhecimento prévio**, podem estar erradas e estão **assinaladas como não verificadas**;
* só a **exposição prévia no repositório** (EC1) foi verificada por pesquisa no código/resultados/docs deste projeto;
* a verificação objetiva (proveniência, licença, dimensões, tipos, classes, ausentes, duplicados, dependências) fica a cargo de `validation/confirm/inspect_candidates.py` (§5), que corres tu **ou** eu, depois de
  libertares o acesso de rede (host `archive.ics.uci.edu`, ver §7). **Nenhum modelo é treinado nessa inspeção.**
Não foi observado nenhum resultado confirmatório. Estas decisões são tomadas **antes** de qualquer avaliação e não dependem de desempenho.

## 1. Critérios de elegibilidade e exclusão (CONGELADOS aqui; só objetivos e independentes de resultados)
| Id | Critério | Regra |
|---|---|---|
| EC1 | Exposição prévia | Inelegível se o dataset (ou ontologia/derivado) aparece em código, resultados, docs, ontologias ou análises do BIURI relacionadas com mecanismos/hiperparâmetros/avaliação de Original/Reloaded. Não se pode provar a ausência de influência histórica ⇒ regra conservadora. |
| EC2 | Proveniência/licença | Fonte pública versionada com DOI/citação e licença que permita uso científico (esperado: CC BY 4.0); hash do ficheiro no manifest. |
| EC3 | Dimensão | 150 ≤ n ≤ 50 000; 4 ≤ p_original ≤ 200; 2 ≤ classes ≤ 10; **mínimo de 30 exemplos por classe** (5×2 CV estratificada). |
| EC4 | Independência das linhas | Séries temporais (medições consecutivas do mesmo processo) ou dependência espacial/de grupo não controlável ⇒ inelegível. Datasets com carimbo temporal mas uma linha por entidade independente são elegíveis **com risco declarado**. |
| EC5 | Fuga de informação | Atributos documentados como consequência do alvo são **removidos antes dos splits** conforme a documentação oficial (decisão tomada aqui, não por resultados); se não for possível remover ⇒ inelegível. |
| EC6 | Ausentes | ≤ 30 % em qualquer atributo; tratamento determinístico e declarado (categoria «ausente» + indicador), sem estatísticas aprendidas dos dados. |
| EC7 | Duplicados | Linhas **exatamente** duplicadas (features+rótulo) removidas antes dos splits e contadas; duplicados com rótulos conflituantes mantidos e contados. |
| EC8 | Documentação para ontologia | Dicionário de dados oficial que descreva ≥ 70 % dos atributos + ≥ 1 fonte de domínio independente (norma, artigo, manual). Atributos anonimizados ⇒ inelegível. |
| EC9 | Heterogeneidade do conjunto | O conjunto final inclui ≥ 3 domínios, ≥ 2 binários e ≥ 2 multiclasse, ≥ 3 só numéricos e ≥ 3 com atributos categóricos, e **≥ 2 datasets com estrutura semântica fraca** (cenários nulos). |
| EC10 | Sem exclusão por desempenho | Nenhum dataset é retirado ou substituído por causa de resultados. Substituição só por falha verificada de EC1–EC8, por ordem fixa de reservas, com registo. |
| EC11 | Pré-processamento sem aprendizagem | Codificação por **dicionário de dados publicado** (one-hot com vocabulário do dicionário; ordinais só se declarados), nunca estatísticas ajustadas à totalidade dos dados; imputação como categoria/indicador. Mesmo pré-processamento para O, R∅, R_OWL e baselines. |
O benchmark atual (`BenchmarkRunner`) exige matriz numérica finita (`preprocessing_record`: `encoding: none, imputation: none`). Os datasets categóricos/com ausentes precisam do estágio EC11 **fora** do runner (loader),
determinístico e sem treino — dependência técnica D-12 (Doc. 11 e Doc. 12).

## 2. Verificação de exposição prévia no repositório (feita, EC1)
Pesquisa em `*.py *.md *.json *.csv *.owl *.arff *.txt` (excluindo `docs/b1b4`):
| Dataset | Achado | EC1 |
|---|---|---|
| Car Evaluation, Banknote, Spambase, Online Shoppers, Dry Bean, Steel Plates, Occupancy, Bank Marketing, Rice (Cammeo/Osmancik) | sem ocorrências | ✔ |
| Statlog Vehicle Silhouettes | «vehicle» só em listas de nomes proibidos do verificador de agnosticismo e num teste com classes sintéticas (`VehicleChoice`) — **não** o dataset | ✔ |
| Mushroom | «mushroom» só na lista de nomes proibidos (`scripts/check_dataset_agnosticism.py`, `tests/test_dataset_agnosticism.py`) | ✔ (sem uso) |
| **Statlog German Credit** | **usado** em `counterfactuals/experimentos/german_credit/` (`training_summary.json` contém `trepan_fidelity=0,994` e `trepan_reloaded_fidelity=0,927`, `mlp_accuracy_test=0,757`) e em `validation/counterfactual_research/dataset_registry.py` / `evaluar_arboles_mejorados.py`, desde o commit inicial | **✘ EXPOSTO** |
Notas: o histórico anterior ao commit inicial é desconhecido; para os restantes a ausência no repositório não prova que nunca foram vistos fora dele — declaração do autor necessária (§6, Doc. 9).

## 3. Matriz de elegibilidade dos 12 candidatos
`n/p/classes/tipos/ausentes/duplicados/licença` = **esperado de memória, NÃO verificado** (assinalado «~» / «?»). Veredito preliminar = com base em EC1 (verificado) e EC4/EC5 (conhecimento prévio da documentação); EC2/EC3/EC6/EC7/EC8 só se confirmam na inspeção.
| # | Dataset (UCI id?) | n / p / classes (esperado) | Tipos | Riscos conhecidos (a confirmar) | Doc. p/ ontologia (A forte · B média) | Veredito preliminar |
|---|---|---|---|---|---|---|
| 1 | Car Evaluation (19?) | ~1728 / 6 / 4 | nominais (ordinais) | produto cartesiano completo (sem duplicados); classe = função determinística de modelo hierárquico DEX ⇒ ontologia «espelha» o gerador do rótulo; classes minoritárias pequenas (verificar EC3) | A (modelo de decisão publicado) | ELEGÍVEL, risco **R-gerador** |
| 2 | Banknote Authentication (267?) | ~1372 / 4 / 2 | numéricos | ~vários duplicados; poucas features | B (estatísticas de imagem wavelet) | ELEGÍVEL; **estrutura fraca** (cenário nulo, EC9) |
| 3 | Spambase (94?) | ~4601 / 57 / 2 | numéricos | duplicados conhecidos; famílias `word_freq_*`, `char_freq_*`, `capital_run_*` replicáveis pelos nomes | B | ELEGÍVEL; bom para controlo de engenharia convencional |
| 4 | Statlog Vehicle Silhouettes (149?) | ~846 / 18 / 4 | numéricos | n pequeno; classes opel/saab quase indistinguíveis | B | ELEGÍVEL |
| 5 | Online Shoppers Purchasing Intention (468?) | ~12330 / 17 / 2 | 10 num + 7 cat | desequilíbrio (~15 %); duplicados ~; `Month` (sazonalidade, mas sessões de utilizadores distintos) | A (métricas de analítica web documentadas) | ELEGÍVEL, risco **R-temporal-leve** |
| 6 | Dry Bean (602?) | ~13611 / 16 / 7 | numéricos | duplicados ~; custo (n) | A (definições geométricas) | ELEGÍVEL (custo alto) |
| 7 | Steel Plates Faults (198?) | ~1941 / 27 / 7 | num + binários | alvo vem como 7 colunas binárias ⇒ derivar rótulo único documentado; classe «Other_Faults» heterogénea; mínimo por classe (EC3) | B | ELEGÍVEL (verificar EC3/rótulo) |
| 8 | Occupancy Detection (357?) | ~20 560 / 5 / 2 | numéricos + data/hora | **série temporal** medida de minuto a minuto (autocorrelação forte) | A (sensores) | **INELEGÍVEL (EC4)** — substituído por reserva |
| 9 | Mushroom (73?) | ~8124 / 22 / 2 | nominais | atributo com ausentes (~25–30 %, `stalk-root`); separável por regras simples ⇒ **efeito teto** previsível | A (guia de campo) | ELEGÍVEL (verificar EC6), risco **R-teto** (sensibilidade baixa) |
| 10 | Statlog German Credit (144?) | ~1000 / 20 / 2 | 7 num + 13 nom | **EXPOSTO (EC1)**; matriz de custo; atributos sensíveis | B | **INELEGÍVEL (EC1)** — substituído por reserva |
| 11 | Bank Marketing (222?) | ~45 211 / 16 / 2 (versão a fixar) | mistos; «unknown» | `duration` conhecido como fuga ⇒ remover (EC5); ordenado no tempo; possível reincidência do mesmo cliente (não verificável); n grande ⇒ custo | A/B | ELEGÍVEL **com risco** (R-temporal, R-cliente, custo) |
| 12 | Rice (Cammeo/Osmancik) (545?) | ~3810 / 7 / 2 | numéricos | geometria; duplicados ? | B | ELEGÍVEL |
Resumo preliminar: **10 elegíveis a verificar**, **2 inelegíveis** (Occupancy por EC4; German Credit por EC1). Todos os veredictos dependem de a inspeção confirmar EC2/EC3/EC6/EC7/EC8.
Composição prevista (EC9) com os 10 + 2 reservas: numéricos puros: Banknote, Spambase, Vehicle, Dry Bean, Rice (+ Steel com binários) ⇒ ≥ 3 ✔; categóricos: Car, Shoppers, Mushroom, Bank ⇒ ≥ 3 ✔; multiclasse: Car, Vehicle, Dry Bean, Steel ⇒ ≥ 2 ✔; estrutura fraca: Banknote, Rice, Vehicle ⇒ ≥ 2 ✔.

## 4. Reservas (ordem fixa, declarada antes de qualquer resultado)
Também de memória, **por verificar** (EC1 verificado: sem ocorrências no repositório). Entram por ordem se um primário falhar EC1–EC8 na inspeção; substituição documentada.
| Ordem | Dataset (UCI id?) | n / p / classes (esperado) | Nota |
|---|---|---|---|
| R1 | Phishing Websites (327?) | ~11 055 / 30 / 2 | atributos ternários agrupados na documentação (endereço, HTML/JS, domínio): estrutura documentada (A) |
| R2 | MAGIC Gamma Telescope (159?) | ~19 020 / 10 / 2 | parâmetros de Hillas (física) documentados (A); dados simulados |
| R3 | Image Segmentation (50?) | ~2310 / 19 / 7 | regiões de 7 imagens (risco de grupo por imagem; declarar) |
| R4 | Page Blocks Classification (78?) | ~5473 / 10 / 5 | classes muito desequilibradas — verificar EC3 |
| R5 | Contraceptive Method Choice (30?) | ~1473 / 9 / 3 | misto, estrutura fraca (útil para EC9) |
| R6 | Heart Disease (Cleveland) (45?) | ~303 / 13 / 2–5 | ausentes ligeiros; «heart» aparece em testes do repositório com outro sentido — confirmar EC1 |
| R7 | Statlog Landsat Satellite (146?) | ~6435 / 36 / 6 | forte estrutura (bandas × vizinhança) mas **dependência espacial** (risco EC4) |
| R8 | Seeds (236?) | ~210 / 7 / 3 | pequeno; geometria |
Excluídos à partida por EC: Nursery (classe com 2 exemplos, EC3), Wine Quality (confusão com o dataset Wine já exposto), Obesity (maioria sintética), Wall-Following Robot/EEG (séries temporais).

## 5. Instrumento de inspeção (documental, sem modelos)
`validation/confirm/inspect_candidates.py` (a criar na Fase 0.5 final, só leitura de dados) — para cada candidato e reserva gera `docs/b1b4/confirm/<id>_inspection.json` com: URL e id UCI, DOI, licença, data de
descarregamento, `sha256` do ficheiro, n, p, tipos, classes e contagens, ausentes por atributo, duplicados exatos e conflituantes, colunas candidatas a fuga (nomes documentados), presença de colunas temporais/ID/grupo,
e as flags EC1–EC8 por regra automática (+ campo livre para a leitura humana do dicionário de dados). **Não calcula nenhuma métrica de modelo.** Estado: especificado; implementação e execução dependem de acesso de rede.

## 6. Declarações necessárias de ti (autor do protocolo)
(1) Nenhum dos 12 + 8 foi usado por ti para escolher mecanismos/hiperparâmetros do BIURI fora do repositório. (2) Quem escreve as ontologias (Doc. 9) e que não viu resultados de modelos nestes datasets.

## 7. Como desbloquear a verificação
Em *Environment settings → Network access*, acrescentar `archive.ics.uci.edu` (e, se quiseres confirmação cruzada, `www.openml.org`) a *Allowed domains* com «Allow package managers» ligado
(https://code.claude.com/docs/en/cloud-environments#network-access). Alternativa: corres tu `inspect_candidates.py` localmente e partilhas os JSON (sem dados brutos).
