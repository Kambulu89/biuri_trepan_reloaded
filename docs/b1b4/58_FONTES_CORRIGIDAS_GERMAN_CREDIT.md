# Doc. 58 — Fontes documentais corrigidas do German Credit: estado do desbloqueio

Estado: **BLOQUEADO por política de rede; nenhuma fonte foi obtida nem consultada.** Só `validation/` e documentação; `core/`, `german.data`, ontologia congelada e benchmarks históricos intocados. Nenhum conteúdo de fonte é afirmado neste documento: tudo o que depende de `german.doc`, da tabela corrigida do South German Credit ou do relatório de Grömping (2019) está `PENDING_SOURCE`.

## 0. Atualização (decisão do responsável após os Docs. 58–61)
URLs corrigidas pelo responsável: **S1** `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc`; **S2** `https://archive.ics.uci.edu/dataset/573/south+german+credit+update` (decisão mais recente do responsável; o URL `…/522/south+german+credit` indicado numa decisão intermédia fica registado como alternativo); **S3** inalterado. Nova tentativa (`network_probe.json`, 2026-10-18): os três pedidos continuam com `Tunnel connection failed: 403 Forbidden` (política de rede do ambiente). **Não se contornou, nenhuma fonte foi consultada, nenhuma é declarada verificada.** A instrução é aguardar os ficheiros fornecidos. Procedimento pronto para quando existirem (sem alterações de código): (1) `h2_sources register …` regista origem declarada e SHA-256; (2) comparar o dicionário original com a documentação corrigida (`compare_codes`); (3) listar divergências de códigos/categorias; (4) reavaliar as 14 fichas (Doc. 59); (5) verificar a correspondência 20 atributos ↔ 61 features (§3–4); (6) classificar categorias em ordinais/nominais/desconhecidas; (7) preservar dados e baselines; (8) manter os axiomas em revisão pendente até validação independente.
**Nota de integridade do hash (política completa e verificação no Doc. 62 §2):** o SHA-256 *bruto* de `german.data` difere entre Linux (LF) e Windows (CRLF por `autocrlf`); um teste que comparava o hash bruto falhou na CI Windows de `d1121a4`. Passou a registar-se também o hash **normalizado CRLF→LF** (`file_sha256_lf_normalized`), que é o estável entre plataformas; o ficheiro de dados não foi alterado.

## 1. Fontes pedidas e o que foi tentado
| Id | Fonte (URL declarada pelo responsável) | Ficheiro esperado | Resultado |
|---|---|---|---|
| S1 | https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc | `german.doc` | **não obtido** — proxy de saída: `403 CONNECT` |
| S2 | https://archive.ics.uci.edu/dataset/573/south+german+credit+update | tabela de códigos / ficheiros South German | **não obtido** — idem |
| S3 | https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf | `Report-2019-004.pdf` | **não obtido** — idem |
Evidência: `docs/evidence/b1b4/h2/network_probe.json` (tentativas com a mensagem de erro) e `sources_manifest.json` (`status = BLOCKED_NETWORK_POLICY_403`, `files = []`, `content_consulted = false`, `origin_verified = false`). A política **não foi contornada** (sem espelhos, sem cópias de terceiros, sem citar de memória).

## 2. Estado dos 10 passos pedidos
| # | Passo | Estado |
|---|---|---|
| 1 | obter os ficheiros | **bloqueado** (rede) |
| 2 | confirmar origem e guardar hashes | **não feito**: sem ficheiro não há hash; a ferramenta `h2_sources.py register` calcula SHA-256 e marca `content_consulted` só se extrair texto; `origin_verified` é sempre `false` (os repositórios não publicam hashes: a origem é a **declarada**, não verificada por máquina) |
| 3 | não declarar fonte verificada sem consulta | cumprido: todas as fontes `UNVERIFIED` |
| 4 | comparar códigos de `german.doc` com as correções do South German | **não executável** |
| 5 | tabela de correspondência dos 20 atributos | **feita só no lado verificável** (§3): colunas do ficheiro local, códigos e contagens **observados**; campos documentais `PENDING_SOURCE` |
| 6 | efeitos do one-hot nas 61 features | feito (§4) |
| 7 | distinguir ordinal / nominal / desconhecido / não aplicável | **só como questão aberta**: o que o projeto assume está registado, o que a fonte diz está `PENDING_SOURCE` (§5) |
| 8 | não assumir ordem a partir dos códigos | cumprido: nenhum código (A62 < A63…) foi usado para inferir ordem; as 14 fichas continuam hipóteses |
| 9 | não substituir `german.data` nem alterar benchmarks | cumprido (hash do ficheiro abaixo) |
| 10 | marcar interpretações ambíguas como pendentes | feito (§5) |

## 3. Tabela de correspondência dos 20 atributos (lado verificável: ficheiro local)
`german.data` local: SHA-256 `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871`, 1000 linhas, 20 atributos + coluna de classe. **A numeração dos atributos = ordem das colunas é uma suposição a confirmar em `german.doc`.** Contagens: todas as 1000 linhas (descrição do ficheiro, sem rótulos usados para seleção).
| Col. | Nome no projeto | Tratamento no carregador | Colunas codificadas | Códigos/valores observados | Definição documentada | Estatuto ordinal documentado | Correção South German |
|---|---|---|---|---|---|---|---|
| 1 | `status_checking_account` | nominal (one-hot) | 4 | A11:274, A12:269, A13:63, A14:394 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 2 | `duration_months` | numérico | 1 | num. [4, 72], 33 valores distintos | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 3 | `credit_history` | nominal (one-hot) | 5 | A30:40, A31:49, A32:530, A33:88, A34:293 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 4 | `purpose` | nominal (one-hot) | 10 | A40:234, A41:103, A410:12, A42:181, A43:280, A44:12, A45:22, A46:50, A48:9, A49:97 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 5 | `credit_amount` | numérico | 1 | num. [250, 18424], 921 valores distintos | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 6 | `savings_account` | nominal (one-hot) | 5 | A61:603, A62:103, A63:63, A64:48, A65:183 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 7 | `employment_since` | nominal (one-hot) | 5 | A71:62, A72:172, A73:339, A74:174, A75:253 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 8 | `installment_rate_pct_income` | numérico; marcado «ordinal-codificado» no projeto (direção NÃO verificada) | 1 | 1:136, 2:231, 3:157, 4:476 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 9 | `personal_status_sex` | nominal (one-hot) | 4 | A91:50, A92:310, A93:548, A94:92 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 10 | `other_debtors` | nominal (one-hot) | 3 | A101:907, A102:41, A103:52 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 11 | `residence_since` | numérico; marcado «ordinal-codificado» no projeto (direção NÃO verificada) | 1 | 1:130, 2:308, 3:149, 4:413 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 12 | `property` | nominal (one-hot) | 4 | A121:282, A122:232, A123:332, A124:154 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 13 | `age_years` | numérico | 1 | num. [19, 75], 53 valores distintos | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 14 | `other_installment_plans` | nominal (one-hot) | 3 | A141:139, A142:47, A143:814 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 15 | `housing` | nominal (one-hot) | 3 | A151:179, A152:713, A153:108 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 16 | `existing_credits_count` | numérico; marcado «ordinal-codificado» no projeto (direção NÃO verificada) | 1 | 1:633, 2:333, 3:28, 4:6 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 17 | `job` | nominal (one-hot) | 4 | A171:22, A172:200, A173:630, A174:148 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 18 | `people_liable_count` | numérico; marcado «ordinal-codificado» no projeto (direção NÃO verificada) | 1 | 1:845, 2:155 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 19 | `telephone` | nominal (one-hot) | 2 | A191:596, A192:404 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |
| 20 | `foreign_worker` | nominal (one-hot) | 2 | A201:963, A202:37 | PENDING_SOURCE | PENDING_SOURCE | PENDING_SOURCE |

## 4. Efeitos do one-hot sobre as 61 features
13 atributos nominais → 54 colunas binárias (4+5+10+5+5+4+3+4+3+3+4+2+2 = 54; níveis = os **observados**, por ordem lexicográfica) e 7 atributos numéricos inalterados ⇒ **61** features. Consequências (propriedades do carregador, verificáveis):
* A ordem lexicográfica dos níveis **não é** uma ordem de domínio; um nível não observado em treino não tem coluna (categoria desconhecida em dados novos ⇒ linha de zeros no grupo, nunca atribuída a um nível).
* Colunas de um mesmo atributo somam 1 por linha (verificado em treino: 0 linhas sem nível, 0 com vários) ⇒ são dependentes entre si; um teste «1-de-n» sobre um subconjunto de níveis é a união desses níveis.
* O Original ordena candidatos por variância (`_feature_priority_scores`): as 7 numéricas em escala bruta (montante, duração, idade) dominam a variância; colunas binárias raras ou muito prevalentes são preteridas (Doc. 59 §3).
* Quatro numéricas (`installment_rate`, `residence_since`, `existing_credits`, `people_liable`) são marcadas «ordinais-codificadas» nos metadados do projeto: o **sentido** do código (se 1 é menor ou maior) **não está verificado**.

## 5. Questões abertas (todas `PENDING_SOURCE`; nada resolvido por memória)
1. Definição e unidades de cada atributo e de cada código (incluindo se `A13`, `A65`, `A71`, `A14` são níveis ordinais, nominais, «desconhecido» ou «não aplicável»).
2. Se a tabela de códigos clássica tem erros (o responsável indica que Grömping, 2019, os identifica) e **quais** atributos/códigos afeta — não se pode afirmar quais sem ler S2/S3. Risco: atribuições de significado erradas propagam-se às fichas de axioma e aos agrupamentos de `purpose`.
3. Se o `german.data` local corresponde ao ficheiro clássico (códigos `A..`) e não à versão South German (códigos numéricos reordenados): **a comparação códigos-do-ficheiro vs tabelas é a verificação central e depende de S1/S2/S3**.
4. Direção dos códigos das quatro numéricas «ordinais».
5. Atributos potencialmente sensíveis (`personal_status_sex`, `age_years`, `foreign_worker`): fora do âmbito das fichas; tratamento a decidir à parte.

## 6. O que é necessário (um dos dois)
(a) **Permitir os anfitriões**: acrescentar `archive.ics.uci.edu` e `www1.beuth-hochschule.de` a *Allowed domains* da rede do ambiente (https://code.claude.com/docs/en/cloud-environments#network-access); ou (b) **fornecer os ficheiros** (anexo/repositório). Depois, sem outra alteração:
`python -m validation.benchmark.h2_sources register --german-doc PATH --south-file PATH [--south-file PATH …] --report-pdf PATH --out docs/evidence/b1b4/h2/sources_manifest.json`
e comparar códigos com `h2_sources.compare_codes`. Só depois de consultado o conteúdo se preencherão os campos documentais e se reavaliam as 14 fichas (Doc. 59).

## 7. Consequência
O conjunto de axiomas **avaliável** continua **vazio** (Doc. 54, `H2_EXPERIMENT_MANIFEST.DRAFT.json`). Os axiomas e as citações «de memória» que existam nas fichas **não** são fontes verificadas.
