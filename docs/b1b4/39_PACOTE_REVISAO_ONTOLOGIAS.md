# Doc. 39 — Pacote de revisão das ontologias e axiomas candidatos

Estado: **PACOTE PREPARADO; NENHUMA REVISÃO REALIZADA.** Não existem revisores humanos designados. **Nenhum nome, assinatura ou aprovação foi inventado.** Estados: `DOMAIN_REVIEW_PENDING`, `ONTOLOGY_REVIEW_PENDING`, `METHODOLOGY_REVIEW_PENDING`. Nenhuma ontologia enriquecida pode ser declarada independentemente validada (o verificador `b31_protocol.check_package` devolve `ontology_declared_independently_validated=False` para todos os pacotes). Gerador: `validation/benchmark/b32_review_package.py`; conteúdo: `docs/evidence/b1b4/b32/review_package/`; testes: `tests/test_b32_review_package.py`. Protocolo-mãe: Doc. 37.

## 1. Conteúdo do pacote
| pacote (`proposals/<id>/proposal.json`) | axiomas | estatuto atual | SHA-256 (LF) |
|---|---|---|---|
| `digits_spatial_topology_v1` | 1 | HYPOTHESIS_UNTIL_DOMAIN_REVIEW | `03ecfc8eda8744a4…` |
| `wdbc_compactness_identity_rejected_v1` | 1 | REJECTED_AS_IDENTITY_FOR_AGGREGATED_ATTRIBUTES | `49dca6818d39781b…` |
| `wdbc_geometric_approximations_v1` | 1 | HYPOTHESIS | `22f0ab7d733aa70c…` |
| `wdbc_order_relations_v1` | 10 | HYPOTHESIS_UNTIL_DOMAIN_REVIEW | `39834daaccbf7872…` |
| `wine_phenol_subset_rejected_v1` | 1 | REJECTED_AS_UNIVERSAL_CONSTRAINT | `67419fa4dc334cd3…` |

Formulários **em branco** (`review_forms/`): `DOMAIN_REVIEW.md`, `ONTOLOGY_REVIEW.md`, `METHODOLOGY_REVIEW.md`. Índice de hashes: `review_package/hashes.json` (inclui as ontologias DEV congeladas e os ficheiros de pré-registo B3.1).

## 2. Axiomas candidatos, proveniência e classificação
### 2.1 `wdbc_order_relations` (autorizada a preparação documental) — «worst ≥ mean» em 10 famílias
- **Fonte verificável (local):** texto incluído no scikit-learn 1.6.1 (`load_breast_cancer().DESCR`, cópia do UCI «Breast Cancer Wisconsin (Diagnostic)»): *«The mean, standard error, and "worst" or largest (mean of the three worst/largest values) of these features were computed for each image»*. Hash do DESCR em cada axioma. O revisor deve conferir contra o UCI original (`verification_required`).
- **Natureza proposta:** `MATHEMATICAL_IDENTITY_CONDITIONAL` — decorre das **definições** (a média dos k maiores valores dos mesmos núcleos é ≥ à média de todos), condicional a: «largest» = maior valor numérico; comportamento com < 3 núcleos por imagem; ausência de arredondamentos que invertam a ordem. **Não** se promove a axioma universal; as frequências observadas são **verificação de consistência**, não prova.
- **Cinco condições (Doc. 33 §10):** mesma grandeza *sim*; mesmo conjunto de observações *sim, segundo a descrição*; definições estatísticas compatíveis *sim, se «worst» = média dos 3 maiores (a confirmar)*; **pré-processamento: só na escala bruta** (padronizar colunas com médias/DP diferentes não preserva a ordem); valores em falta: nenhum, aplica-se a linhas completas.
- **Diagnóstico só com atributos** (569 linhas): violações = 0 nas 10 famílias; folga mínima `worst − mean` ≥ 0 por família (no JSON).
- **Estatuto atual:** `HYPOTHESIS_UNTIL_DOMAIN_REVIEW` → uso máximo `PREFERENCE`; **não** é restrição obrigatória até revisão de domínio e metodológica aprovadas.
- **Perguntas ao revisor de domínio:** nº mínimo de núcleos por imagem; «largest» também para simetria/dimensão fractal; arredondamento da cópia UCI.
### 2.2 `wdbc_compactness_identity_rejected` — «compactness_mean = perimeter_mean²/area_mean − 1»
Vale **por núcleo** (DESCR: «perimeter^2 / area − 1.0»); a média de um quociente não é o quociente das médias (Jensen) ⇒ **não é identidade para atributos agregados**; regularidade empírica, `REJECTED_AS_IDENTITY_FOR_AGGREGATED_ATTRIBUTES`. Mostra a distinção identidade matemática ↔ regularidade.
### 2.3 `wdbc_geometric_approximations` — perímetro ≈ 2π·raio, área ≈ π·raio²
Exato só para contornos circulares; razões medianas observadas ≈ 1,03 e 0,98 (atributos apenas). **Regularidade empírica/hipótese**; nunca restrição.
### 2.4 `wine_phenol_subset_rejected` (mantido REJEITADO como restrição universal)
«flavanoids ≤ total_phenols» falha em **36,0 %** das linhas; a inclusão conceptual não implica ordem entre medidas. **Teste negativo de validação semântica; não se força na ontologia para produzir ganho.**
### 2.5 `digits_spatial_topology` — adjacência 4-vizinhança da grelha 8×8
Fonte (a confirmar): DESCR de `load_digits` + descrição do pré-processamento (bitmaps 32×32 → blocos 4×4 → contagens). Relação **estrutural** entre atributos (preferência/relatedness), não restrição de valores; exige um tipo de facto novo (`SPATIAL_ADJACENCY`), ainda **inexistente** no registo. Diagnóstico: |correlação| média de pares adjacentes vs. não adjacentes (no JSON).

## 3. Revisões exigidas (todas pendentes)
| revisão | responsável | confirma | estado |
|---|---|---|---|
| **Domínio** | especialista do domínio (por designar) | verdade das relações; documentação; definições; âmbito de validade; pressupostos | `DOMAIN_REVIEW_PENDING` |
| **Ontológica** | engenheiro de ontologias (por designar) | OWL 2 DL (perfil formal), consistência, inferências, interpretação dos axiomas, mundo aberto, matching sem empates (Doc. 40) | `ONTOLOGY_REVIEW_PENDING` |
| **Metodológica** | responsável pela metodologia (por designar) | independência dos testes, ausência de *leakage*, pré-registo, versões/hashes, validade dos controlos | `METHODOLOGY_REVIEW_PENDING` |

## 4. O que a máquina já verificou e o que não pode
Automático (por pacote, `check_package`): fontes e justificações presentes; proveniência com citação + localizador; braços de comparação (`original, enriched, permuted, random, empty`) nos pacotes que vão a avaliação; revisores não humanos rejeitados. **Não verificável por máquina:** verdade no domínio, perfil OWL 2 DL formal (existe só um proxy: sem *punning* + classificação HermiT), pré-registo humano. **Não existe** ainda ficheiro OWL enriquecido (passos 5–7 e 9 do protocolo ficam `PENDING`).

## 5. Versões e hashes
Ambiente: scikit-learn 1.6.1, NumPy (no JSON). Hashes relevantes:
| ficheiro | SHA-256 (início) |
|---|---|
| `validation/benchmark/ontologies/iris.owl` | `fe73a34613393191…` |
| `validation/benchmark/ontologies/wine.owl` | `23381ea2d15f8027…` |
| `validation/benchmark/ontologies/breast_cancer.owl` | `cd79f1f009b4fd15…` |
| `validation/benchmark/ontologies/digits.owl` | `6142046aff2e52ea…` |
| `validation/benchmark/ontologies/german_credit.owl` | `7fdc41808d4169b2…` |
| `validation/benchmark/ontologies/synthetic_binary.owl` | `43e2650bc6701e97…` |
| `validation/benchmark/ontologies/synthetic_3class.owl` | `ca94db2aa39361ea…` |
| `b31/problems_frozen.json` | `a8ac2753b43df1cf…` |
| `b31/problems_frozen_v2.json` | `d90133e631ac55e8…` |
| `b31/freeze_B30.json` | `a1b9186123585e9c…` |

(Lista completa em `review_package/hashes.json`; ontologias históricas congeladas **inalteradas**.)

## 6. Pendências
1. Designar os três revisores (nomes, datas, referências de assinatura serão **escritos por eles**).
2. Confirmar a citação UCI e as perguntas abertas de §2.1.
3. Decidir quais pacotes seguem para ontologia OWL real (e escrever o OWL enriquecido **só depois** da revisão de domínio).
4. Congelar (hash + commit) a ontologia enriquecida e o pré-registo antes de qualquer avaliação; comparar com ontologia original, permutada ao nível da relação, aleatória, vazia, convencional de dimensão igualada e Original WIDE (Doc. 38).
