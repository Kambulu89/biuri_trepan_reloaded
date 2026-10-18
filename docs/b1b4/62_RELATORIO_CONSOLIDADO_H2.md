# Doc. 62 — Relatório consolidado H2 (atualizado após a decisão sobre `0bafe28`): CI, fontes, validação documental, fichas, contrato de comparabilidade, piloto técnico, agnosticismo, limiares e decisão sobre a H2-R

Estado: **preparação técnica; H2-R, H2-P completa e H1 suspensas (não executadas)**; `core/`, TREPAN Original, C4.5 e MLP intocados; sem merge; sem datasets confirmatórios. As análises são **estruturais**: nenhuma accuracy, fidelidade a rótulos reais ou teste foi calculada ou consultada. **Os axiomas German Credit são rascunhos de IA não validados e as fontes não foram consultadas: nada aqui é avaliação causal do benefício ontológico nem conhecimento de domínio confirmado.** Objetivo experimental (não demonstrado): `Acc(Reloaded_OWL) > Acc(Original) > Acc(C4.5)`; `Fid(Reloaded_OWL) > Fid(Original)` e `> Fid(C4.5)`, nas mesmas instâncias e contra o mesmo MLP, por conhecimento ontológico verdadeiro; `Reloaded(None) ≡ Original`.

## 1. Estado final da CI
| Commit | Agnosticism guard | Windows 3.11 | Windows 3.12 |
|---|---|---|---|
| `f1b2955` (main) | ✔ | ✔ | ✔ |
| `557c381` | ✔ | ✔ | ✔ |
| `d1121a4` | ✔ | ✘ | ✘ — causa única: teste que comparava o hash **bruto** de `german.data` (CRLF no Windows); corrigido |
| `ca6bf44` | ✔ | ✔ | ✔ |
| `0bafe28` (Doc. 62 anterior, testes de equivalência, metamórficos e do adaptador H2 incluídos na suíte completa) | ✔ | ✔ (≈ 59 min) | ✔ |
| commits desta atualização | pendente (por consultar após o push) | pendente (por consultar após o push) | pendente (por consultar após o push) |
Local (conjunto relevante: pool, metamórficos, piloto, subconjunto OWL, harness, fontes/inventário, auditoria, rastreio, contrato, agnosticismo ×3, manifesto/C4.5): **Python 3.11: 165 passed in 276.63s (0:04:36); Python 3.12: 165 passed, 14 warnings in 276.13s (0:04:36)**. Nenhuma falha por diagnosticar.
`main` = `f1b2955`; o ramo tem à frente apenas trabalho de `validation/`, `tests/`, `docs/`. Nenhuma alteração a `core/` desde `f1b2955`; sem novos merges.

## 2. Fontes German Credit: recebidas ou pendentes
**Pendentes.** Nenhum ficheiro foi fornecido (nem existe no ambiente) e a rede continua a recusar os três URLs (`403 CONNECT`; `network_probe.json`, 2026-10-18): S1 `german.doc` (https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc); S2 South German Credit (https://archive.ics.uci.edu/dataset/573/south+german+credit+update); S3 Grömping (2019) (https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf). Nenhuma fonte é declarada verificada; origem e versão por registar.

## 3. Resultados da validação documental
**Não executável sem os ficheiros** (passos 1–10 do protocolo prontos: `h2_sources register` regista origem, versão declarada e hashes bruto e canónico e só marca «consultado» se extrair texto; `compare_codes` compara dicionários; depois reavaliar as 14 fichas, classificar relações verdadeiras/ambíguas/inválidas, confirmar categorias, validar hierarquias, identificar o que é formalizável em OWL, excluir axiomas sem justificação, preservar dados e preparar o pacote de revisão; a documentação corrigida **não** justifica automaticamente os axiomas propostos).
**Verificações técnicas independentes das fontes (feitas):**
* **Integridade de hashes** — duas identificações com finalidades distintas, nunca trocadas: `sha256_raw` (proveniência, bytes desta cópia; no Windows `autocrlf` converte LF→CRLF) e `sha256_canonical` (só `CRLF→LF`; identidade do conteúdo entre plataformas). O hash do `german.data` com `\n→\r\n` calculado aqui (`af2607959c93…`) coincide com o reportado pela CI Windows: a falha de `d1121a4` foi exclusivamente de fins de linha. Protegidos inalterados desde `f1b2955` (`core/`, dados originais, evidência B31–B33, manifestos, ontologia congelada): `git diff` vazio.

| Ficheiro | SHA-256 bruto | SHA-256 canónico (CRLF→LF) | igual | blob git |
|---|---|---|---|---|
| `german.data` | `b21f3d81db807125…` | `b21f3d81db807125…` | sim | `b21f3d81db807125…` |
| `german_credit.owl` | `7fdc41808d4169b2…` | `7fdc41808d4169b2…` | sim | `7fdc41808d4169b2…` |
| `german_credit_O1.owl` | `b4cfb9aa49f17bdc…` | `b4cfb9aa49f17bdc…` | sim | `b4cfb9aa49f17bdc…` |
| `german_credit_O2.owl` | `b0752dbad9b4b384…` | `b0752dbad9b4b384…` | sim | `b0752dbad9b4b384…` |
| `german_credit_O3.owl` | `ca793f3f4a824497…` | `ca793f3f4a824497…` | sim | `ca793f3f4a824497…` |
| `german_credit_H.owl` | `d98aa79eeaaac58c…` | `d98aa79eeaaac58c…` | sim | `d98aa79eeaaac58c…` |
| `german_credit_I.owl` | `2300afa58e9f8b57…` | `2300afa58e9f8b57…` | sim | `2300afa58e9f8b57…` |

* **Matching 61/61** e **HermiT** (O1, O2, O3, H, I consistentes, sem classes insatisfazíveis); **subconjunto OWL** usado: 5/5 PASS na verificação parcial de construções (não é validação formal OWL 2 DL; pendente).
* **Defeito de solidez encontrado e corrigido (nesta atualização):** se um atributo da família de um conceito tem matching **ambíguo** (P3), o OR das colunas mapeadas **não** é a extensão do conceito (omitia silenciosamente o atributo e produzia um literal incorreto, p. ex. `{L2,L4}` em vez de `{L2,L3,L4}`). `derived_union_literals` recusa agora esse literal e regista `INCOMPLETE_EXTENSION_UNRESOLVED_MATCH`; um atributo **sem** entidade candidata (não modelado) não bloqueia (mundo aberto). Os 10 literais do German (61/61 mapeados) não mudam. Testes de regressão em 3 configurações.

## 4. Situação das 14 fichas
Todas `AI_DRAFT` / `HYPOTHESIS`, fonte citada de memória (`verified=false`), três revisões `*_REVIEW_PENDING`; decisões da auditoria: 6 `HOLD_PENDING_SOURCE`, 2 `HOLD_WEAK_SEMANTICS`, 2 `EXCLUDE_NO_DOCUMENTARY_BASIS`, 4 `TRIVIAL_SINGLETON`; conjunto avaliável **vazio**; nenhuma promovida nem alterada. Os 10 literais derivados são **tecnicamente válidos** (`LITERAL_VALID` = axioma consistente, facto compilado, atributos mapeados) mas a validade de **domínio** está `PENDING`.

## 5. Contrato revisto de comparabilidade
Dois conceitos **separados** (`h2_pool_study.comparability`):
* **`CONFIGURATION_COMPARABLE`** — os braços receberam condições e limites compatíveis com o contraste: mesmas instâncias, mesmo MLP/oráculo, mesmos parâmetros exceto o pool, mesmo feixe, **ambos dentro dos tetos autorizados** e as dimensões que o *desenho* do contraste declara igualar (`design_equalized`: vagas nos contrastes B, D, E; candidatos avaliados no F) dentro da tolerância proposta. Caso contrário `NOT_COMPARABLE`, com a lista de violações.
* **`REALIZED_RESOURCE_MATCH`** — grau **descritivo** de correspondência do consumo efetivo (`EXACT` | `WITHIN_TOLERANCE` | `PARTIAL`, com diferenças relativas por dimensão). **Nunca decide** a comparabilidade: dois braços com o mesmo teto de consultas podem consumir quantidades diferentes e o contraste continua interpretável (o consumo é reportado e analisado, não igualado a posteriori). O CPU é modalidade distinta (decide só se tiver teto).
**Relatório obrigatório por braço** (`resource_report`): teto de candidatos e candidatos gerados/avaliados/inválidos; teto de consultas, consultas executadas, chamadas e linhas ao MLP; vagas e nº de literais (derivados, colunas forçadas); CPU; memória (pico `tracemalloc`); árvores construídas (inclui varrimento do F e ablações); complexidade (nós, profundidade, nº de testes, literais por teste, literais derivados nos testes); diferenças de representação. **Tetos autorizados (proposta antes de resultados; não congelados):** `eligible_literals ≤ 40`, `evaluated_candidates ≤ 1200`, `oracle_queries ≤ 3000`, `nodes ≤ 7`. Nenhum braço excedeu. Nenhum recurso adicional é dado ao braço OWL sem braço de controlo equivalente (B e E recebem as mesmas `r` vagas extra; F iguala os candidatos avaliados).
**Resultados estruturais no German Credit (holdouts 42, 43, 44; axiomas por validar):**

| Contraste (igualado pelo desenho) | s42: configuração / correspondência realizada | s43: configuração / correspondência realizada | s44: configuração / correspondência realizada |
|---|---|---|---|
| C_OWL_PRIORITY vs B_WIDE_GENERIC (feature_slots) | CONFIGURATION_COMPARABLE / WITHIN_TOLERANCE | CONFIGURATION_COMPARABLE / PARTIAL | CONFIGURATION_COMPARABLE / PARTIAL |
| C_OWL_PRIORITY vs F_GENERIC_MATCHED_EVALUATED (mofn_evaluated) | CONFIGURATION_COMPARABLE / WITHIN_TOLERANCE | CONFIGURATION_COMPARABLE / PARTIAL | CONFIGURATION_COMPARABLE / PARTIAL |
| C_OWL_PRIORITY vs D_PERMUTED_PRIORITY (feature_slots) | CONFIGURATION_COMPARABLE / WITHIN_TOLERANCE | CONFIGURATION_COMPARABLE / PARTIAL | CONFIGURATION_COMPARABLE / PARTIAL |
| C_OWL_PRIORITY vs E_RANDOM_BASE (feature_slots) | CONFIGURATION_COMPARABLE / WITHIN_TOLERANCE | CONFIGURATION_COMPARABLE / PARTIAL | CONFIGURATION_COMPARABLE / PARTIAL |
| C_OWL_PRIORITY vs E_RANDOM_RELATIONS (feature_slots) | CONFIGURATION_COMPARABLE / PARTIAL | CONFIGURATION_COMPARABLE / PARTIAL | CONFIGURATION_COMPARABLE / PARTIAL |
| C_OWL_PRIORITY vs A_LIMITED (feature_slots) | NOT_COMPARABLE / PARTIAL | NOT_COMPARABLE / PARTIAL | NOT_COMPARABLE / PARTIAL |

O braço F percorre **todo** o intervalo de `mf` (regra pré-declarada, só contagens, sem paragem antecipada): s42: mf 21, |Δ| 0.0145, viável=True, s43: mf 19, |Δ| 0.0038, viável=True, s44: mf 24, |Δ| 0.0252, viável=True. C vs A é `NOT_COMPARABLE` **por desenho** (C tem +r vagas): não atribuível à ontologia. Contrastes com consumo realizado `PARTIAL` (sobretudo consultas, que dependem da forma da árvore: 0–127) **permanecem configuração comparável**; a interpretação tem de os reportar.

Consumos por braço:

| Braço | holdout | vagas | tetos cand./cons. | cand. avaliados | consultas executadas (linhas) | literais derivados | CPU s | memória MB | árvores | nós/prof. | literais em testes (derivados) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_LIMITED | 42 | 12 | 1200/3000 | 392 | 0 (750) | 0 | 0.995 | 1.9 | 1 | 3/3 | 9 (0) |
| A_LIMITED | 43 | 12 | 1200/3000 | 411 | 27 (777) | 0 | 1.07 | 2.11 | 1 | 3/3 | 9 (0) |
| A_LIMITED | 44 | 12 | 1200/3000 | 329 | 0 (750) | 0 | 0.931 | 2.07 | 1 | 2/2 | 5 (0) |
| A_PLUS_ENRICHED | 42 | 12 | 1200/3000 | 371 | 37 (787) | 10 | 1.086 | 2.05 | 11 | 2/2 | 5 (1) |
| A_PLUS_ENRICHED | 43 | 12 | 1200/3000 | 418 | 27 (777) | 10 | 1.123 | 2.38 | 11 | 3/3 | 9 (0) |
| A_PLUS_ENRICHED | 44 | 12 | 1200/3000 | 171 | 0 (750) | 10 | 1.005 | 2.5 | 11 | 1/1 | 2 (0) |
| B_WIDE_GENERIC | 42 | 22 | 1200/3000 | 821 | 0 (750) | 0 | 1.412 | 2.1 | 1 | 3/3 | 9 (0) |
| B_WIDE_GENERIC | 43 | 22 | 1200/3000 | 995 | 57 (807) | 0 | 1.463 | 2.15 | 1 | 3/3 | 9 (0) |
| B_WIDE_GENERIC | 44 | 22 | 1200/3000 | 624 | 21 (771) | 0 | 1.529 | 2.15 | 1 | 3/3 | 7 (0) |
| C_OWL_PRIORITY | 42 | 22 | 1200/3000 | 758 | 0 (750) | 10 | 1.546 | 2.09 | 11 | 3/3 | 9 (1) |
| C_OWL_PRIORITY | 43 | 22 | 1200/3000 | 786 | 127 (877) | 10 | 1.426 | 2.58 | 11 | 3/3 | 9 (1) |
| C_OWL_PRIORITY | 44 | 22 | 1200/3000 | 674 | 0 (750) | 10 | 1.267 | 2.31 | 11 | 3/3 | 9 (1) |
| D_PERMUTED_PRIORITY | 42 | 22 | 1200/3000 | 813 | 0 (750) | 10 | 1.518 | 2.39 | 1 | 3/3 | 9 (1) |
| D_PERMUTED_PRIORITY | 43 | 22 | 1200/3000 | 828 | 16 (766) | 10 | 1.505 | 2.42 | 1 | 3/3 | 9 (0) |
| D_PERMUTED_PRIORITY | 44 | 22 | 1200/3000 | 788 | 0 (750) | 10 | 1.409 | 2.16 | 1 | 3/3 | 9 (2) |
| E_RANDOM_BASE | 42 | 22 | 1200/3000 | 811 | 0 (750) | 0 | 1.57 | 2.08 | 1 | 3/3 | 9 (0) |
| E_RANDOM_BASE | 43 | 22 | 1200/3000 | 813 | 16 (766) | 0 | 1.312 | 2.14 | 1 | 3/3 | 9 (0) |
| E_RANDOM_BASE | 44 | 22 | 1200/3000 | 879 | 0 (750) | 0 | 1.542 | 2.24 | 1 | 2/2 | 6 (0) |
| E_RANDOM_RELATIONS | 42 | 22 | 1200/3000 | 936 | 0 (750) | 10 | 1.359 | 2.21 | 1 | 3/3 | 9 (3) |
| E_RANDOM_RELATIONS | 43 | 22 | 1200/3000 | 788 | 13 (763) | 10 | 1.449 | 2.03 | 1 | 3/3 | 8 (1) |
| E_RANDOM_RELATIONS | 44 | 22 | 1200/3000 | 714 | 19 (769) | 10 | 1.279 | 2.36 | 1 | 2/2 | 6 (1) |
| F_GENERIC_MATCHED_EVALUATED | 42 | 21 | 1200/3000 | 769 | 0 (750) | 0 | 1.422 | 2.08 | 50 | 3/3 | 9 (0) |
| F_GENERIC_MATCHED_EVALUATED | 43 | 19 | 1200/3000 | 789 | 16 (766) | 0 | 1.29 | 2.1 | 50 | 3/3 | 9 (0) |
| F_GENERIC_MATCHED_EVALUATED | 44 | 24 | 1200/3000 | 691 | 21 (771) | 0 | 1.314 | 2.15 | 50 | 3/3 | 7 (0) |

**Estados dos literais** (independentes; nenhum implica o seguinte): `LITERAL_VALID` (validade técnica; domínio `PENDING`), `LITERAL_ELIGIBLE` (entrou nos pools de features e de literais), `LITERAL_EVALUATED` (avaliado como candidato simples ou em m-of-n), `LITERAL_SELECTED` (num teste da árvore), `SPLIT_CHANGED` (a ablação altera um split), `BENEFIT_OBSERVED` (**sempre `NOT_ASSESSED`**: exige accuracy **e** fidelidade com controlos).

| Braço | holdout | LITERAL_VALID | LITERAL_ELIGIBLE | LITERAL_EVALUATED | LITERAL_SELECTED | SPLIT_CHANGED | BENEFIT_OBSERVED |
|---|---|---|---|---|---|---|---|
| A_PLUS_ENRICHED | 42 | 10/10 | 1/10 | 2/10 | 1/10 | 1/10 | NOT_ASSESSED |
| A_PLUS_ENRICHED | 43 | 10/10 | 2/10 | 4/10 | 0/10 | 0/10 | NOT_ASSESSED |
| A_PLUS_ENRICHED | 44 | 10/10 | 2/10 | 3/10 | 0/10 | 0/10 | NOT_ASSESSED |
| C_OWL_PRIORITY | 42 | 10/10 | 8/10 | 10/10 | 1/10 | 1/10 | NOT_ASSESSED |
| C_OWL_PRIORITY | 43 | 10/10 | 4/10 | 10/10 | 1/10 | 1/10 | NOT_ASSESSED |
| C_OWL_PRIORITY | 44 | 10/10 | 6/10 | 10/10 | 1/10 | 1/10 | NOT_ASSESSED |

`AtLeast_savings_account_A63` é o único literal selecionado e que altera um split em 3/3 holdouts (diferença de ganho de informação sobre etiquetas do oráculo 0,0006–0,005): **caso de rastreabilidade experimental**, não evidência de ganho ontológico; nenhuma regra específica deste atributo existe ou será introduzida no motor. Rastreio completo por literal (axioma de origem, estado de validação, inferência, features, entrada no pool, candidato, avaliação, decisão, ablação, operacional vs preditivo) em `pool_study_german.json`.

## 6. Resultado do piloto técnico sintético
`h2_pilot_synthetic.py` (só `validation/`; **não** é campanha de desempenho): relações sintéticas de validade **conhecida**, 3 datasets sintéticos **desconhecidos** (12/16/27 atributos; 2/3/2 classes; 1–3 famílias; 2–7 níveis), `max_n` ∈ {1, 3}, ontologias TRUE/PERM/NEUTRAL/FLAT e uma variante com matching ambíguo. **Limites declarados antes da execução:** ≤ 300 s, ≤ 300 ajustes, ≤ 1500 MB (interrompe com `ABORTED_LIMIT`; teste dedicado). **Consumo real:** 40.7 s, 136 ajustes, 75.3 MB → estado **COMPLETED**.
Verificações (todas satisfeitas; ver `pilot_synthetic.json` e `tests/test_h2_pilot_synthetic.py`): (1) axiomas válidos compilam com a extensão exata esperada (derivada da configuração), PERM não exprime a verdade, NEUTRAL/FLAT não geram literais, o matching ambíguo recusa literais incompletos; (2) o literal verdadeiro está fora do pool no braço A+ (`NOT_IN_FEATURE_POOL`) e entra com vagas reservadas (`IN_LITERAL_POOL`); (3) é avaliado como candidato; (4) tetos aplicados sem excedências e um teto apertado de propósito produz `NOT_COMPARABLE`; (5) comparabilidade de configuração e correspondência realizada reportadas; (6) estados separados e `BENEFIT_OBSERVED = NOT_ASSESSED`; (7) `A ≡ Original` e ontologia válida sem estrutura ativa ≡ Original (regras, consultas); (8) funciona nos três datasets desconhecidos.

| Dataset sintético | atributos | classes | literais verdadeiros | max_n | F viável (gap) | C vs B / D / E_base / E_rel / F | verdade: pool A+ → C | avaliações | estados (V,E,Ev,S,Ch) |
|---|---|---|---|---|---|---|---|---|---|
| U1 | 12 | 2 | 3 | 1 | True (0.0) | CC/PART / CC/PART / CC/PART / CC/PART / CC/PART | NOT_IN_FEATURE → IN_LITERAL_POO | 4 | 11111 |
| U1 | 12 | 2 | 3 | 3 | False (0.6222) | CC/PART / CC/PART / CC/PART / CC/PART / NC/PART | NOT_IN_FEATURE → IN_LITERAL_POO | 34 | 11111 |
| U2 | 16 | 3 | 1 | 1 | True (0.0) | CC/EXAC / CC/PART / CC/WITH / CC/PART / CC/WITH | NOT_IN_FEATURE → IN_LITERAL_POO | 2 | 11111 |
| U2 | 16 | 3 | 1 | 3 | True (0.0) | CC/PART / CC/PART / CC/PART / CC/PART / CC/WITH | NOT_IN_FEATURE → IN_LITERAL_POO | 20 | 11111 |
| U3 | 27 | 2 | 5 | 1 | True (0.0) | CC/PART / CC/PART / CC/PART / CC/PART / CC/PART | NOT_IN_FEATURE → IN_LITERAL_POO | 4 | 11111 |
| U3 | 27 | 2 | 5 | 3 | True (0.0541) | CC/PART / CC/PART / CC/PART / CC/PART / CC/PART | NOT_IN_FEATURE → IN_LITERAL_POO | 46 | 11111 |

(CC = `CONFIGURATION_COMPARABLE`, NC = `NOT_COMPARABLE`; EXAC/WITH/PART = grau de correspondência realizada; estados na ordem VALID, ELIGIBLE, EVALUATED, SELECTED, SPLIT_CHANGED.) Com `max_n=1` o literal verdadeiro é selecionado na raiz e a ablação altera o split (mecanismo funcional, **verdade conhecida**); o contraste F é `NOT_COMPARABLE` quando não é viável igualar os candidatos avaliados (p. ex. U1, `max_n=3`). Isto verifica o **mecanismo**, não autoriza conclusões de superioridade.

## 7. Testes de agnosticismo
Guard estático do núcleo OK; `test_agnosticism_contract`, `test_dataset_agnosticism`, `test_v92_static_guards` passam em 3.11 e 3.12; maquinaria H2 sem nomes de datasets, códigos tipo A71/A410 nem formas fixas. **Metamórficos** (`test_h2_metamorphic.py`, 16 testes; ontologias sintéticas desconhecidas geradas por configuração): renomeação consistente de atributos/conceitos, permutação de colunas e da ordem de declaração OWL, nº variável de atributos/famílias/níveis, **atributos numéricos e categóricos na mesma matriz**, **categorias desconhecidas** (derivadas 0), 2/3/4 classes, ontologias planas, expressivas, PERM e neutras, **ausência de ontologia** (`Reloaded(None) ≡ Original`, mesmo fluxo de consultas), ontologia válida sem estrutura ativa, **matching ambíguo** e atributo não modelado. Todo o conhecimento específico do German fica em artefactos externos e condutores `h2_*_german*.py`.

## 8. Bloqueios para validar a ontologia
1. Ficheiros S1–S3 (ou acesso de rede) e consulta efetiva. 2. Designação dos três revisores (domínio, ontologias, metodologia) e revisão independente das fichas — sem nomes, assinaturas nem aprovações inventadas. 3. Comparação dicionário clássico × corrigido e reavaliação das 14 fichas. 4. Validação formal OWL 2 DL e das inferências sobre axiomas já revistos. 5. Conjunto avaliável de axiomas **não vazio** (hoje vazio). Enquanto 1–2 faltam, o German Credit **não** pode ser avaliação causal do benefício ontológico: só verificações técnicas e ontologias sintéticas de verdade conhecida.

## 9. Limiares metodológicos pendentes (nenhum congelado)
Δ_o = 0,05, κ = 0,375, τ_s = 0,8, τ_u = 0,5 continuam **candidatos**; o responsável pela metodologia aprova os critérios finais antes de qualquer avaliação confirmatória. Revisão da regra `média + 1,645·EP ≤ τ` (`threshold_rule_check.json`, valores por réplica; reproduz as médias anteriores):
* **Incerteza e discretização:** valores ∈ {0, 1/6, …, 1}, n = 20; nível real (simulação binomial, P(aprovar) com verdade = τ):

| τ | P(aprovar) verdade = τ | τ+0,05 | τ+0,10 |
|---|---|---|---|
| 0.5 | 0.057 | 0.0043 | 0.0001 |
| 0.6 | 0.052 | 0.0027 | 0.0001 |
| 0.7 | 0.048 | 0.0021 | 0.0000 |
| 0.8 | 0.044 | 0.0008 | 0.0000 |
| 0.9 | 0.038 | 0.0000 | 0.0000 |

  4–6 % (nominal 5 %; 5,7 % em τ=0,5: ligeiramente anticonservador) e quase nulo a +0,05.
* **Concordância entre regras** (normal, t₁₉, Bonferroni sobre 36 comparações, bootstrap): discordam em 3, 2, 3, 1, 3 de 36 testes (τ = 0,5…0,9).
* **Dependência entre réplicas / entre estimadores:** réplicas independentes por semente; os estimadores partilham os dados de cada réplica (correlação média por célula 0.31–0.49), pelo que `best_of_all` (máximo de estimadores dependentes) é optimista para o método genérico — **conservador** para a porta; exigir que **todos** os estimadores passem é uma conjunção, logo não aumenta a falsa aprovação.
* **Número de estimadores / múltiplas comparações:** 6 células × 6 estimadores = 36 comparações; Bonferroni como sensibilidade.
* **Estimadores competitivos:** correlação, L1, L1-CV, floresta e informação mútua (esta bem mais fraca: acerto ≤ 0,37, tornaria a porta vazia); em τ_s = 0,8 a decisão muda com o estimador em 3 de 6 células.
* **Risco de seleção favorável à ontologia:** a regra, o estimador de referência (união pré-declarada) e τ são fixados **antes** de avaliar variantes OWL, sobre dados sintéticos de semente independente e sem qualquer resultado de ontologia; cada limiar tem sensibilidade tabulada; uma porta é critério de **elegibilidade**, nunca evidência confirmatória; limites: 20 réplicas, uma única família sintética (R8).

## 10. Decisão proposta para autorizar a H2-R
**Recomendação: NÃO autorizar ainda.** Autorizar a H2-R apenas quando **todas** estas condições estiverem satisfeitas e registadas: (a) S1–S3 consultadas, hashes e versões registados, divergências de códigos documentadas; (b) 14 fichas reavaliadas e **pelo menos um conjunto não vazio de axiomas** revistos por três pessoas designadas (ou execução rotulada «rascunho de IA não validado», que **nunca** sustenta alegações de ganho); (c) validação formal OWL 2 DL e das inferências; (d) limiares, estimador de referência, tolerâncias e tetos aprovados pelo responsável pela metodologia; (e) pré-registo (hipóteses, braços A–F e controlos, contrastes, critérios, sementes ≥ 8000, métricas accuracy+fidelidade em conjunto, estados `JOINT_*`) em commit separado; (f) CI verde nas duas versões de Python; (g) aprovação explícita do orçamento. **Orçamento proposto (CPU-h; nada autorizado):** fase R base ≈ 0,7 + braços de pool para 20 holdouts ≈ 0,3 (medição: ≈ 43 s reais e ≈ 80 s CPU por holdout, sem feixe) ⇒ **≈ 1,0 (limite recomendado 1,2)**; piloto sintético executado: 40.7 s (< 0,02 CPU-h); fase P ≈ 1,5; H1 7,2 (ambas suspensas). **Próxima intervenção recomendada:** (1) o utilizador fornece S1–S3 e designa os revisores; (2) o responsável decide estimador/limiares e tolerâncias/tetos; (3) eu executo a validação documental (§3) e preparo o pacote de revisão; (4) só então proponho o pré-registo. Sucesso científico exige `Acc` e `Fid` melhorarem **em simultâneo**, nas mesmas instâncias e contra o mesmo MLP, sem mais consultas nem baselines enfraquecidos e atribuível a conhecimento ontológico verdadeiro (controlos permutado, aleatório e convencional) — um objetivo, não um resultado.
