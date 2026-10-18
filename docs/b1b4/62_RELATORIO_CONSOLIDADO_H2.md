# Doc. 62 — Relatório consolidado H2 (após `ca6bf44`): CI, integridade de hashes, fontes, fichas, literais, pools, agnosticismo, limiares e bloqueios

Estado: **preparação técnica; nada autorizado a executar** (H1, H2-P, H2-R suspensas); `core/`, TREPAN Original, C4.5 e MLP intocados; sem merge; sem datasets confirmatórios. As análises são **estruturais**: nenhuma accuracy, fidelidade a rótulos reais ou teste foi calculada ou consultada. **Os axiomas German Credit são rascunhos de IA não validados e as fontes não foram consultadas: nada aqui é avaliação causal do benefício ontológico nem conhecimento de domínio confirmado.**

## 1. Estado final da CI
| Commit | Agnosticism guard | Windows 3.11 | Windows 3.12 |
|---|---|---|---|
| `f1b2955` (main) | ✔ | ✔ | ✔ |
| `557c381` | ✔ | ✔ | ✔ |
| `d1121a4` | ✔ | ✘ | ✘ — causa única: teste que comparava o hash **bruto** de `german.data` (CRLF no Windows); 1 falhado, 1618 passaram |
| `ca6bf44` (corrige a causa) | ✔ | ✔ (≈ 28 min) | ✔ (≈ 40 min) |
Os commits desta fase (a seguir a `ca6bf44`) ainda não têm CI: será consultada após o push. Localmente, o conjunto relevante (pool, metamórficos, subconjunto OWL, harness, fontes/inventário, auditoria, rastreio, contrato, agnosticismo ×3, manifesto/C4.5) passa **158/158 em Python 3.11 e em Python 3.12** (venv 3.12 com `requirements-test.txt`).
`main` = `f1b2955`; o ramo tem à frente `557c381`, `d1121a4`, `06cd6e0`, `81adbc3`, `ad1a08d`, `ca6bf44` e esta fase. Nenhuma alteração a `core/` desde `f1b2955`.

## 2. Integridade dos hashes (bruto ≠ canónico; nunca trocados em silêncio)
Política (código `h2_hash_integrity.py`): **`sha256_raw`** = bytes tal como estão na cópia de trabalho (proveniência; no Windows, `autocrlf` converte LF→CRLF); **`sha256_canonical`** = após a única normalização documentada `\r\n→\n`, só para texto (binários: `canonicalization = none`, canónico = bruto). Os dois registam-se sempre lado a lado; o canónico **não substitui** o bruto de proveniência (nas fontes obtidas, `h2_sources.file_record` regista ambos).
Verificação independente da causa da falha Windows: o hash do `german.data` com `\n→\r\n` calculado aqui (`af2607959c93…`) é **igual** ao hash reportado pela CI Windows — a diferença é exclusivamente de fins de linha; o blob git (LF) coincide com o hash bruto/canónico desta cópia.

| Ficheiro | SHA-256 bruto (esta cópia, LF) | SHA-256 canónico (CRLF→LF) | bruto = canónico | blob git |
|---|---|---|---|---|
| `german.data` | `b21f3d81db807125…` | `b21f3d81db807125…` | sim | `b21f3d81db807125…` |
| `german_credit.owl` | `7fdc41808d4169b2…` | `7fdc41808d4169b2…` | sim | `7fdc41808d4169b2…` |
| `german_credit_O1.owl` | `b4cfb9aa49f17bdc…` | `b4cfb9aa49f17bdc…` | sim | `b4cfb9aa49f17bdc…` |
| `german_credit_O2.owl` | `b0752dbad9b4b384…` | `b0752dbad9b4b384…` | sim | `b0752dbad9b4b384…` |
| `german_credit_O3.owl` | `ca793f3f4a824497…` | `ca793f3f4a824497…` | sim | `ca793f3f4a824497…` |
| `german_credit_H.owl` | `d98aa79eeaaac58c…` | `d98aa79eeaaac58c…` | sim | `d98aa79eeaaac58c…` |
| `german_credit_I.owl` | `2300afa58e9f8b57…` | `2300afa58e9f8b57…` | sim | `2300afa58e9f8b57…` |

**Protegidos inalterados desde `f1b2955`:** `core/`, dados originais (`validation/counterfactual_research`), evidência B31/B32/B33, manifestos v1/v2/v3, ontologia congelada do German: `git diff --name-status` vazio (`hash_integrity.json`, `protected_unchanged.unchanged = True`). A correção do teste não tocou nos dados, baselines nem evidência histórica.

## 3. Estado das fontes German Credit
| Fonte | URL declarada pelo responsável | Estado |
|---|---|---|
| S1 `german.doc` | https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc | **bloqueada** (`403 CONNECT`); não consultada |
| S2 South German Credit | https://archive.ics.uci.edu/dataset/573/south+german+credit+update (522 mantido só como alternativo registado) | **bloqueada**; não consultada |
| S3 Grömping (2019) | https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf | **bloqueada**; não consultada |
Nova sonda em `network_probe.json` (2026-10-18): 3/3 recusados; sem contornar. Nenhuma fonte é declarada verificada; origem e versão ficam por registar até haver ficheiros (`h2_sources register`, que calcula hash bruto e canónico e só marca «consultado» se extrair texto). Quando existirem, executar pela ordem: registar origem/versão/hashes → consultar o conteúdo → comparar códigos clássicos com as correções → rever as 14 fichas → classificar relações verdadeiras/ambíguas/inválidas → confirmar matching às 61 features → preservar a semântica das categorias desconhecidas → validar OWL e inferências → preparar os pacotes de revisão → **não promover axiomas** sem a evidência exigida (e a documentação corrigida **não** justifica automaticamente os axiomas propostos).

## 4. Estado das 14 fichas
Todas `AI_DRAFT` / `HYPOTHESIS`, fonte citada de memória e marcada `verified=false`, três revisões `*_REVIEW_PENDING`. Decisões da auditoria: 6 `HOLD_PENDING_SOURCE`, 2 `HOLD_WEAK_SEMANTICS`, 2 `EXCLUDE_NO_DOCUMENTARY_BASIS`, 4 `TRIVIAL_SINGLETON`; conjunto avaliável **vazio**; nenhuma promovida; nenhuma alterada nesta fase.

## 5. Diagnóstico dos dez literais (estados formalmente separados)
Definição (`h2_pool_study.STATE_MEANING`): **`LITERAL_VALID`** = axioma disponível, ontologia consistente, facto compilado e atributos mapeados (validade **técnica**, não verdade de domínio, que fica em `domain_validation = PENDING`); **`LITERAL_ELIGIBLE`** = entrou no pool de features e no pool de literais do m-of-n numa pesquisa; **`LITERAL_EVALUATED`** = avaliado como candidato (simples ou em m-of-n); **`LITERAL_SELECTED`** = parte do teste escolhido num nó; **`SPLIT_CHANGED`** = a ablação controlada altera uma decisão de split; **`BENEFIT_OBSERVED`** = ganho conjunto accuracy+fidelidade com controlos (**`NOT_ASSESSED`**). Os estados são registados **independentemente** (nenhum implica o seguinte): um literal pode ser avaliado como candidato simples sem ser elegível para m-of-n.

| Braço | holdout | LITERAL_VALID | LITERAL_ELIGIBLE | LITERAL_EVALUATED | LITERAL_SELECTED | SPLIT_CHANGED | BENEFIT_OBSERVED |
|---|---|---|---|---|---|---|---|
| A_PLUS_ENRICHED | 42 | 10/10 | 1/10 | 2/10 | 1/10 | 1/10 | NOT_ASSESSED |
| A_PLUS_ENRICHED | 43 | 10/10 | 2/10 | 4/10 | 0/10 | 0/10 | NOT_ASSESSED |
| A_PLUS_ENRICHED | 44 | 10/10 | 2/10 | 3/10 | 0/10 | 0/10 | NOT_ASSESSED |
| C_OWL_PRIORITY | 42 | 10/10 | 8/10 | 10/10 | 1/10 | 1/10 | NOT_ASSESSED |
| C_OWL_PRIORITY | 43 | 10/10 | 4/10 | 10/10 | 1/10 | 1/10 | NOT_ASSESSED |
| C_OWL_PRIORITY | 44 | 10/10 | 6/10 | 10/10 | 1/10 | 1/10 | NOT_ASSESSED |

Leitura: com o pool do Original (A+), 8/10 não chegam à elegibilidade (exclusão pré-estatística por variância) e 0–1 é selecionado; com vagas reservadas a relações verdadeiras (C) todos são avaliados mas só 4–8 são elegíveis para m-of-n, **1** é selecionado e **1** altera um split (`AtLeast_savings_account_A63`, 3/3 holdouts, diferença de ganho de informação 0,0006–0,005 sobre etiquetas do oráculo no treino). Isto é **influência operacional**; o benefício preditivo **não foi medido** e este caso **não** é evidência de ganho ontológico nem motiva regra especial no algoritmo. Rastreio completo por literal (10 itens pedidos: axioma de origem, estado de validação, inferência do reasoner, features, entrada no pool, candidato gerado, avaliação, decisão, ablação, operacional vs preditivo) em `pool_study_german.json`, campo `literals`.

## 6. Comparabilidade dos pools
Braços (mesmas instâncias, mesmo MLP/oráculo, mesmos parâmetros exceto o pool): **A** limitado do Original (12 vagas); **B** alargado genericamente (+r=10 vagas por variância); **C** priorização por relações ontológicas (+r vagas reservadas às derivadas); **D** priorização permutada (ontologia H); **E** aleatória (E_BASE colunas base aleatórias; E_RELATIONS relações aleatórias, ontologia I); **F** pool genérico cujos candidatos m-of-n **avaliados** igualam os de C; A+ isola a representação.
**Tetos autorizados (proposta, antes de resultados; não congelados)** — `eligible_literals ≤ 40`, `evaluated_candidates ≤ 1200`, `oracle_queries ≤ 3000`, `nodes ≤ 7`; amostras consultadas e CPU como modalidades à parte (CPU só decide se tiver teto). Os **consumos efetivos** reportam-se separadamente; nenhum braço excedeu os tetos (3/3 holdouts, 8 braços). Vagas iguais **não** implicam candidatos ou consultas iguais (as binárias geram 2 candidatos simples; as numéricas até 48).
Consumos:

| Braço | s42: avaliados m-of-n; consultas; linhas; nós; CPU s | s43: avaliados m-of-n; consultas; linhas; nós; CPU s | s44: avaliados m-of-n; consultas; linhas; nós; CPU s |
|---|---|---|---|
| A_LIMITED | 392; 0; 750; 3; 0.58 | 411; 27; 777; 3; 0.557 | 329; 0; 750; 2; 0.56 |
| A_PLUS_ENRICHED | 371; 37; 787; 2; 0.575 | 418; 27; 777; 3; 0.576 | 171; 0; 750; 1; 0.516 |
| B_WIDE_GENERIC | 821; 0; 750; 3; 0.726 | 995; 57; 807; 3; 0.75 | 624; 21; 771; 3; 0.593 |
| C_OWL_PRIORITY | 758; 0; 750; 3; 0.787 | 786; 127; 877; 3; 0.701 | 674; 0; 750; 3; 0.647 |
| D_PERMUTED_PRIORITY | 813; 0; 750; 3; 0.625 | 828; 16; 766; 3; 0.797 | 788; 0; 750; 3; 0.618 |
| E_RANDOM_BASE | 811; 0; 750; 3; 0.604 | 813; 16; 766; 3; 0.704 | 879; 0; 750; 2; 0.615 |
| E_RANDOM_RELATIONS | 936; 0; 750; 3; 0.627 | 788; 13; 763; 3; 0.631 | 714; 19; 769; 2; 0.595 |
| F_GENERIC_MATCHED_EVALUATED | 769; 0; 750; 3; 0.736 | 789; 16; 766; 3; 0.637 | 691; 21; 771; 3; 0.676 |

Contrastes (estrito = vagas, candidatos, consultas, nós, profundidade e feixe dentro da tolerância proposta de 10 %; «candidatos+tetos» ignora as consequências da forma da árvore — consultas, nós, profundidade, linhas — e exige candidatos igualados e tetos respeitados):

| Contraste | s42 estrito / candidatos+tetos | s43 estrito / candidatos+tetos | s44 estrito / candidatos+tetos |
|---|---|---|---|
| C_OWL_PRIORITY vs B_WIDE_GENERIC | COMPARABLE / COMPARABLE | NOT_COMPARABLE / NOT_COMPARABLE | NOT_COMPARABLE / COMPARABLE |
| C_OWL_PRIORITY vs F_GENERIC_MATCHED_EVALUATED | COMPARABLE / COMPARABLE | NOT_COMPARABLE / COMPARABLE | NOT_COMPARABLE / NOT_COMPARABLE |
| C_OWL_PRIORITY vs D_PERMUTED_PRIORITY | COMPARABLE / COMPARABLE | NOT_COMPARABLE / COMPARABLE | NOT_COMPARABLE / NOT_COMPARABLE |
| C_OWL_PRIORITY vs E_RANDOM_BASE | COMPARABLE / COMPARABLE | NOT_COMPARABLE / COMPARABLE | NOT_COMPARABLE / NOT_COMPARABLE |
| C_OWL_PRIORITY vs E_RANDOM_RELATIONS | NOT_COMPARABLE / NOT_COMPARABLE | NOT_COMPARABLE / COMPARABLE | NOT_COMPARABLE / NOT_COMPARABLE |
| C_OWL_PRIORITY vs A_LIMITED | NOT_COMPARABLE / NOT_COMPARABLE | NOT_COMPARABLE / NOT_COMPARABLE | NOT_COMPARABLE / NOT_COMPARABLE |

Matching do braço F (percorre **todo** o intervalo de `mf`, sem paragem antecipada; regra pré-declarada, só contagens):

| holdout | mf de F | |Δ| relativo (avaliados) | viável |
|---|---|---|---|
| 42 | 21 | 0.0145 | True |
| 43 | 19 | 0.0038 | True |
| 44 | 24 | 0.0252 | True |

**Sensibilidade ao feixe, paragens, candidatos inválidos e duplicação** (`pool_study_german.json`): `mofn_evaluated` cresce com o feixe (≈ ×2 de 1 a 4) em todos os braços e o rácio C/B muda com o feixe e o holdout; candidatos m-of-n inválidos (`mofn_rejected_*`: folhas mínimas e partições não distintas) são 0,2–12,5 % dos gerados (todas as execuções); candidatos simples duplicados removidos = 0; as folhas terminam por `STOP_MAX_NODES` (185) ou `STOP_PRUNED` (9, poda de subárvores idênticas) — nenhuma por pureza nem por esgotamento do orçamento de consultas; no holdout 44, A e A+ ficam com 2 e 1 nós internos (paragem antecipada por poda ⇒ diferença de nós/profundidade ⇒ `NOT_COMPARABLE` estrito). Amostra por feixe:

| holdout | feixe | A avaliados / inválidos | B avaliados / inválidos | C avaliados / inválidos | literais selecionados C |
|---|---|---|---|---|---|
| 42 | 1 | 288 / 8 | 619 / 17 | 597 / 11 | 2 |
| 42 | 2 | 392 / 16 | 821 / 43 | 758 / 42 | 1 |
| 42 | 4 | 614 / 30 | 1311 / 77 | 1220 / 64 | 1 |
| 43 | 1 | 311 / 1 | 632 / 12 | 496 / 52 | 1 |
| 43 | 2 | 411 / 1 | 995 / 13 | 786 / 54 | 1 |
| 43 | 4 | 607 / 5 | 1521 / 19 | 1181 / 75 | 1 |
| 44 | 1 | 257 / 7 | 620 / 20 | 472 / 12 | 1 |
| 44 | 2 | 329 / 7 | 624 / 64 | 674 / 14 | 1 |
| 44 | 4 | 449 / 23 | 914 / 70 | 1039 / 21 | 1 |

**Conclusões do contrato:** (i) C vs A é sempre `NOT_COMPARABLE` (mais oportunidade); (ii) a comparabilidade depende do holdout — o contraste C vs B/F/D/E é `COMPARABLE` estrito em 1/3 holdouts e «candidatos+tetos» em 1–2/3 conforme o contraste; (iii) as **consultas** variam com a forma da árvore (0–127) e dominam as violações estritas: propõe-se igualá-las por **teto** e reportar o consumo, e não exigir igualdade relativa de contagens pequenas; (iv) F é viável nos 3 holdouts (|Δ| ≤ 2,5 %). **Não se eliminam candidatos de um braço após observar ganhos** (nenhuma eliminação ocorreu); onde a equiparação não é possível reporta-se `NOT_COMPARABLE`. O pool produtivo (`max_features_per_node`) **não** foi alterado.

## 7. Testes de agnosticismo
* Guard estático do núcleo (`scripts/check_dataset_agnosticism.py`): OK. Suites `test_agnosticism_contract`, `test_dataset_agnosticism`, `test_v92_static_guards`: passam em 3.11 e 3.12.
* **Maquinaria H2** sem nomes de datasets, códigos tipo A71/A410 ou formas fixas (teste dedicado).
* **Metamórficos** (`tests/test_h2_metamorphic.py`, 13 testes, ontologias sintéticas **desconhecidas** geradas por configuração): renomeação consistente de atributos e conceitos ⇒ mesma extensão, mesmas razões de elegibilidade e mesmas decisões de split (mutation test: renomear só os dados sem o OWL é detetado); permutação das colunas e da ordem de declaração do OWL ⇒ mesmo resultado; 1 a 3 famílias, 0 a 6 ruído, 0 a 3 numéricas, 2 a 7 níveis: a extensão verdadeira deriva da configuração e é escolhida na raiz (`max_n=1`); ontologias PERM (OWL válido, semântica errada), NEUTRAL e FLAT nunca exprimem a verdade; **equivalência sem ontologia** (2, 3 e 4 classes; mesmo fluxo de consultas) e ontologia válida sem estrutura ativa (S2) ≡ Original.
* Todo o conhecimento específico do German Credit vive em artefactos externos (ontologias `v0_draft`, fichas JSON) e em condutores `h2_*_german*.py`, fora da maquinaria.

## 8. Matching, reasoner e perfil OWL
Matching 61/61 (O1, O2, O3, H, I; sem rejeições P3); HermiT: consistente e sem classes insatisfazíveis (O2: 29 inferências HermiT; 55 factos do registo, reconciliados no Doc. 59); verificação **parcial** do subconjunto de construções OWL (`h2_owl_profile.py`): 5/5 PASS (lista branca de construções, sem punning, uniões só de classes declaradas, `range` XSD, sem ciclos, domínios declarados) com mutation tests — **não** é validação formal OWL 2 DL (sem OWL API neste ambiente; pendente). Categorias desconhecidas: linha sem nível ativo ⇒ derivadas 0; nível não ordinal fora das extensões (testes `test_h2_trace`).

## 9. Limiares ainda pendentes (nenhum congelado)
Δ_o = 0,05, κ = 0,375, τ_s = 0,8, τ_u = 0,5 continuam **candidatos**. Verificação da regra `média + 1,645·EP ≤ τ` sobre os valores por réplica (`threshold_rule_check.json`; reproduz exatamente as médias anteriores):
* valores por réplica discretos (∈ {0, 1/6, …, 1}), n = 20, limitados a [0,1];
* nível real da regra (simulação binomial, `P(aprovar)` quando a verdade é τ): 

| τ | P(aprovar) com verdade = τ | τ+0,05 | τ+0,10 |
|---|---|---|---|
| 0.5 | 0.057 | 0.0043 | 0.0001 |
| 0.6 | 0.052 | 0.0027 | 0.0001 |
| 0.7 | 0.048 | 0.0021 | 0.0000 |
| 0.8 | 0.044 | 0.0008 | 0.0000 |
| 0.9 | 0.038 | 0.0000 | 0.0000 |

  ≈ 4–6 % (nominal 5 %; 5,7 % em τ=0,5 → ligeiramente anticonservador), e quase nula a +0,05;
* concordância entre regras (normal, t₁₉, Bonferroni sobre 36 comparações, bootstrap percentil): discordam em 3, 2, 3, 1, 3 de 36 testes (τ = 0,5…0,9); em τ = 0,8 só `s2.0_n120|random_forest`;
* estimadores partilham os dados de cada réplica (correlação média entre estimadores por célula 0.31–0.49) ⇒ não independentes; `best_of_all` é um máximo de estimadores dependentes;
* multiplicidade: 36 comparações célula×estimador; dependência entre réplicas (sementes independentes) vs incerteza do gerador (única família sintética) não coberta.
**Recomendação metodológica (a decidir pelo responsável):** estimador de referência = união pré-declarada de correlação, L1, L1-CV, floresta e informação mútua (máximo por célula), fixada **antes** de avaliar variantes; regra t₁₉ com correção de Bonferroni como análise de sensibilidade; τ por célula, nunca escolhido por desempenho de ontologia; o resultado de uma porta é **critério de elegibilidade**, não evidência confirmatória.
Sensibilidade ao estimador em τ_s = 0,8: a decisão muda entre estimadores em 3 de 6 células (`threshold_sensitivity.json`).

## 10. Bloqueios para autorizar a H2
1. Ficheiros S1–S3 (ou acesso de rede) e a sua consulta efetiva. 2. Designação dos três revisores e revisão independente das fichas (domínio, ontologias, metodologia). 3. Validação formal OWL 2 DL e das inferências sobre axiomas já revistos. 4. Decisão sobre estimador de referência e limiares. 5. Aprovação das tolerâncias e dos tetos do contrato de pools (incl. tratamento das consultas). 6. Pré-registo (hipóteses, braços, contrastes, critérios, sementes ≥ 8000) em commit separado. 7. Autorização de orçamento. 8. CI Windows 3.11/3.12 verde dos commits desta fase. Enquanto 1–2 não existirem, o German Credit **não** pode ser usado como avaliação causal do benefício ontológico (só verificações técnicas e ontologias sintéticas de verdade conhecida).

## 11. Estimativa computacional atualizada (CPU-h; nada autorizado)
| Item | Estimativa |
|---|---|
| Esta fase (análise de pool com 8 braços, ablações, F, feixes, 3 holdouts; verificações; 158 testes ×2) | medido: ≈ 43 s reais e ≈ 80 s CPU por holdout; < 0,1 CPU-h no total |
| Fase R base (A–J, ablação, memória) | ≈ 0,7 |
| Braços de pool para 20 holdouts, **sem** a sensibilidade ao feixe (≈ 70 % do custo medido) | ≈ 0,3 |
| Sensibilidade ao feixe só em 3 holdouts | ≈ 0,07 |
| **Fase R com pools** | **≈ 1,0 — no limite proposto de 1,0; recomenda-se limite 1,2** |
| Fase P (sintético) | ≈ 1,5 |
| H1-B3.4 | 7,2 |
| Revisão humana | horas por revisor (a confirmar pelos designados) |

## 12. Recomendação sobre a próxima intervenção
1. **Do utilizador (desbloqueia tudo):** fornecer `german.doc`, a documentação do South German Credit e o PDF de Grömping (ou permitir os dois domínios). 2. Designar os três revisores. 3. Decidir o estimador de referência e a política de limiares, e as tolerâncias/tetos do contrato de pools (propostas aqui). 4. **Minha próxima intervenção, com ficheiros:** registar origem e hashes, consultar o conteúdo, comparar códigos, reavaliar as 14 fichas, atualizar o matching às 61 features, preparar os pacotes de revisão — ainda sem campanhas. 5. Só com axiomas revistos: pré-registo e autorização explícita da fase R; antes disso, nenhum resultado de desempenho deve ser calculado. A prioridade científica mantém-se: confirmar que a ontologia contém conhecimento **verdadeiro** e que o mecanismo genérico o oferece à pesquisa m-of-n em condições justas; só depois medir se isso melhora **em simultâneo** accuracy e fidelidade (`Acc(Reloaded_OWL) > Acc(Original) > Acc(C4.5)`; `Fid(Reloaded_OWL) > Fid(Original)` e `> Fid(C4.5)`, mesmas instâncias, mesmo MLP, sem mais consultas nem baselines enfraquecidos) — um **objetivo** experimental, não um resultado.
