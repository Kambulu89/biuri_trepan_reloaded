# Doc. 54 — Auditoria dos axiomas, inferências e literais da proposta German Credit (`v0_draft`)

Estado: **auditoria de rascunho de IA; nenhum axioma está validado.** Só `validation/` e documentação; `core/`, TREPAN Original, C4.5, MLP e a ontologia congelada `german_credit.owl` **não** foram tocados (a proposta vive em `validation/benchmark/ontologies/proposals/german_credit/v0_draft/`, com versão e SHA-256 próprios em `docs/evidence/b1b4/b3q/h2_german_proposal_check.json`). Evidência desta auditoria: `h2_inference_audit.json`, `h2_german_reasoner_check.json`, `german_levels_x_only.json` (só X de treino do holdout seed 42; nenhum rótulo; nenhum teste).
Scripts: `validation/benchmark/h2_inference_audit.py`, `h2_german_proposal.py`; testes: `tests/test_h2_contract_and_mechanisms.py`, `tests/test_h2_trace.py`, `tests/test_h2_audit.py`.

## 1. Estado documental (H2-P0) — decisão
* **Não existe documentação local** do conjunto: o repositório só contém `german.data` (códigos `A11…`), sem o dicionário de dados; a rede (`archive.ics.uci.edu`) não resolve nesta sessão (erro de DNS). **Nenhuma fonte foi verificada.** As citações das fichas (`UCI Statlog German Credit Data, german.doc`) estão marcadas `verified = false` e foram escritas de memória: **não se citam como verificadas**.
* Consequência (regra do responsável: «excluir axiomas sem fundamentação documental»): **o conjunto avaliável hoje é vazio** (`evaluable_now = false` em todas as 14 fichas). As fichas ficam como *hipóteses candidatas* até que alguém forneça o dicionário de dados (ficheiro local ou acesso) e os revisores confirmem texto, numeração, ordem e âmbito.
* Único facto verificável por máquina: a **estrutura da codificação** (X de treino). Os códigos de nível existem e são mutuamente exclusivos (0 linhas sem nível, 0 com vários) nas cinco variáveis; contagens e níveis raros:

| Variável | Níveis presentes (contagem em treino) |
|---|---|
| status_checking_account | A11: 205, A12: 202, A13: 47, A14: 296 |
| savings_account | A61: 446, A62: 86, A63: 43, A64: 39, A65: 136 |
| employment_since | A71: 38, A72: 133, A73: 258, A74: 125, A75: 196 |
| job | A171: 13, A172: 149, A173: 480, A174: 108 |
| purpose | A40: 179, A41: 81, A410: 6, A42: 138, A43: 201, A44: 10, A45: 17, A46: 37, A48: 5, A49: 76 |

Observações verificáveis: `purpose` não tem coluna `A47` (códigos A40–A46, A48–A410); níveis muito raros (A410: 6, A48: 5, A44: 10, A171: 13) tornam literais que os incluam estatisticamente frágeis; `savings_account` e `status_checking_account` têm níveis «não ordinais» (A65, A14) presentes em muitas linhas (136/750 e 296/750), que **têm de ficar fora** das uniões ordinais.

## 2. Validade lógica e matching (H2-P1)
| Ontologia | HermiT consistente | Classes insatisfazíveis | Axiomas inferidos (HermiT) | Mapeamento |
|---|---|---|---|---|
| O1 | True | [] | 0 | 61/61 (nome exato, 0 ambiguidades, 0 colisões) |
| O2 | True | [] | 29 | 61/61 (nome exato, 0 ambiguidades, 0 colisões) |
| O3 | True | [] | 0 | 61/61 (nome exato, 0 ambiguidades, 0 colisões) |
| H | True | [] | 29 | 61/61 (nome exato, 0 ambiguidades, 0 colisões) |
| I | True | [] | 32 | 61/61 (nome exato, 0 ambiguidades, 0 colisões) |
Perfil OWL 2 DL: só o *proxy* existente (sem *punning* + classificação HermiT); o verificador formal continua por implementar. As classes definidas usam `ObjectUnionOf` e `EquivalentClasses` (OWL 2 DL).

## 3. As 14 fichas de axioma
Estrutura: 10 fichas **substantivas** (8 ordinais com ≥ 2 níveis na extensão + 2 taxonómicas) e **4 triviais** (a «união» de um único nível, equivalente ao próprio nível, sem literal). Todas: `authored_by = AI_DRAFT`, `epistemic_status = HYPOTHESIS`, fonte `NOT_VERIFIED`, três revisões `*_REVIEW_PENDING`.

| Ficha | Afirmação | Tipo reivindicado | Depende de hipóteses | Decisão de auditoria |
|---|---|---|---|---|
| `AX-GC-ORD-status_checking_account-A12` | Em «status_checking_account», os níveis A12, A13 são todos «pelo menos A12» na ordem documentada (A11 < A12 < A13). | ORDINAL parcial (A11<A12<A13 por montante); A14 «sem conta» é NOMINAL e fica de fora | A13 agrega «≥ 200 DM» e «cessão de ordenado ≥ 1 ano»: heterogéneo, ordem total assumida; A14 não ordenável | **HOLD_PENDING_SOURCE** |
| `AX-GC-ORD-status_checking_account-A13` | Em «status_checking_account», os níveis A13 são todos «pelo menos A13» na ordem documentada (A11 < A12 < A13). | ORDINAL parcial (A11<A12<A13 por montante); A14 «sem conta» é NOMINAL e fica de fora | A13 agrega «≥ 200 DM» e «cessão de ordenado ≥ 1 ano»: heterogéneo, ordem total assumida; A14 não ordenável | **TRIVIAL_SINGLETON_NO_CONTENT (união de um único nível ≡ o próprio nível; sem literal)** |
| `AX-GC-ORD-savings_account-A62` | Em «savings_account», os níveis A62, A63, A64 são todos «pelo menos A62» na ordem documentada (A61 < A62 < A63 < A64). | ORDINAL (intervalos de montante A61<A62<A63<A64); A65 «desconhecido/sem poupança» NOMINAL e fora | intervalos tal como no dicionário (não verificados); A65 fora da ordem | **HOLD_PENDING_SOURCE** |
| `AX-GC-ORD-savings_account-A63` | Em «savings_account», os níveis A63, A64 são todos «pelo menos A63» na ordem documentada (A61 < A62 < A63 < A64). | ORDINAL (intervalos de montante A61<A62<A63<A64); A65 «desconhecido/sem poupança» NOMINAL e fora | intervalos tal como no dicionário (não verificados); A65 fora da ordem | **HOLD_PENDING_SOURCE** |
| `AX-GC-ORD-savings_account-A64` | Em «savings_account», os níveis A64 são todos «pelo menos A64» na ordem documentada (A61 < A62 < A63 < A64). | ORDINAL (intervalos de montante A61<A62<A63<A64); A65 «desconhecido/sem poupança» NOMINAL e fora | intervalos tal como no dicionário (não verificados); A65 fora da ordem | **TRIVIAL_SINGLETON_NO_CONTENT (união de um único nível ≡ o próprio nível; sem literal)** |
| `AX-GC-ORD-employment_since-A72` | Em «employment_since», os níveis A72, A73, A74, A75 são todos «pelo menos A72» na ordem documentada (A71 < A72 < A73 < A74 < A75). | ORDINAL (duração: A71 desempregado < A72 <1 a < A73 1–4 < A74 4–7 < A75 ≥7) | A71 «desempregado» é um estado, não uma duração: colocá-lo abaixo de «<1 ano» é convenção a confirmar | **HOLD_PENDING_SOURCE** |
| `AX-GC-ORD-employment_since-A73` | Em «employment_since», os níveis A73, A74, A75 são todos «pelo menos A73» na ordem documentada (A71 < A72 < A73 < A74 < A75). | ORDINAL (duração: A71 desempregado < A72 <1 a < A73 1–4 < A74 4–7 < A75 ≥7) | A71 «desempregado» é um estado, não uma duração: colocá-lo abaixo de «<1 ano» é convenção a confirmar | **HOLD_PENDING_SOURCE** |
| `AX-GC-ORD-employment_since-A74` | Em «employment_since», os níveis A74, A75 são todos «pelo menos A74» na ordem documentada (A71 < A72 < A73 < A74 < A75). | ORDINAL (duração: A71 desempregado < A72 <1 a < A73 1–4 < A74 4–7 < A75 ≥7) | A71 «desempregado» é um estado, não uma duração: colocá-lo abaixo de «<1 ano» é convenção a confirmar | **HOLD_PENDING_SOURCE** |
| `AX-GC-ORD-employment_since-A75` | Em «employment_since», os níveis A75 são todos «pelo menos A75» na ordem documentada (A71 < A72 < A73 < A74 < A75). | ORDINAL (duração: A71 desempregado < A72 <1 a < A73 1–4 < A74 4–7 < A75 ≥7) | A71 «desempregado» é um estado, não uma duração: colocá-lo abaixo de «<1 ano» é convenção a confirmar | **TRIVIAL_SINGLETON_NO_CONTENT (união de um único nível ≡ o próprio nível; sem literal)** |
| `AX-GC-ORD-job-A172` | Em «job», os níveis A172, A173, A174 são todos «pelo menos A172» na ordem documentada (A171 < A172 < A173 < A174). | ORDEM FRACA (qualificação) com dimensões misturadas (residência, qualificação, trabalho por conta própria) | a ordem A171<A172<A173<A174 é inferida da ordem da lista, não de uma definição; A171 mistura «desempregado» e «não qualificado não residente» | **HOLD_WEAK_SEMANTICS (tratar como NOMINAL salvo fonte)** |
| `AX-GC-ORD-job-A173` | Em «job», os níveis A173, A174 são todos «pelo menos A173» na ordem documentada (A171 < A172 < A173 < A174). | ORDEM FRACA (qualificação) com dimensões misturadas (residência, qualificação, trabalho por conta própria) | a ordem A171<A172<A173<A174 é inferida da ordem da lista, não de uma definição; A171 mistura «desempregado» e «não qualificado não residente» | **HOLD_WEAK_SEMANTICS (tratar como NOMINAL salvo fonte)** |
| `AX-GC-ORD-job-A174` | Em «job», os níveis A174 são todos «pelo menos A174» na ordem documentada (A171 < A172 < A173 < A174). | ORDEM FRACA (qualificação) com dimensões misturadas (residência, qualificação, trabalho por conta própria) | a ordem A171<A172<A173<A174 é inferida da ordem da lista, não de uma definição; A171 mistura «desempregado» e «não qualificado não residente» | **TRIVIAL_SINGLETON_NO_CONTENT (união de um único nível ≡ o próprio nível; sem literal)** |
| `AX-GC-TAX-purpose-Vehicles` | Em «purpose», A40, A41 formam a categoria «Vehicles» (agrupamento taxonómico proposto). | NOMINAL; agrupamentos «Veículos» e «Bens domésticos» são proposta do rascunho | nenhuma base documental conhecida para os agrupamentos; critério de pertença por definir; categorias raras (A44: 10, A410: 6, A48: 5 em treino) | **EXCLUDE_NO_DOCUMENTARY_BASIS** |
| `AX-GC-TAX-purpose-HouseholdGoods` | Em «purpose», A42, A43, A44 formam a categoria «HouseholdGoods» (agrupamento taxonómico proposto). | NOMINAL; agrupamentos «Veículos» e «Bens domésticos» são proposta do rascunho | nenhuma base documental conhecida para os agrupamentos; critério de pertença por definir; categorias raras (A44: 10, A410: 6, A48: 5 em treino) | **EXCLUDE_NO_DOCUMENTARY_BASIS** |

Resumo: 6 `HOLD_PENDING_SOURCE`, 2 `HOLD_WEAK_SEMANTICS` (`job`: tratar como nominal salvo fonte), 2 `EXCLUDE_NO_DOCUMENTARY_BASIS` (taxonomia de finalidades: sem base documental conhecida), 4 `TRIVIAL_SINGLETON_NO_CONTENT`. Nenhuma está `evaluable_now`.

## 4. Auditoria das 55 inferências (O2)
Fonte do número: 55 factos compilados com derivação `INFERRED_BY_REASONER` no registo B3.1. Decomposição verificada:
| Tipo | N | Natureza |
|---|---|---|

| Subsunção nível ⊑ classe definida | 17 | consequência **imediata** da definição por união (12 ordinais + 5 taxonómicas); não acrescenta conhecimento além do axioma |
| Subsunção classe definida ⊑ classe definida (cadeia ordinal) | 8 | **não imediata**: segue da inclusão dos conjuntos de níveis; é a única inferência «genuína» (aninhamento das ordens) |
| Equivalência `AtLeast_último ≡ Lvl_último` | 4 | **trivial** (união singleton) |
| `ATTRIBUTE_APPLIES_TO_CONCEPT` inferido | 26 | **consequente** das subsunções + domínio asserido; derivável, não é inferência independente |
Total: 17 + 8 + 4 + 26 = 55. O HermiT reporta **29** axiomas inferidos (25 subsunções + 4 equivalências) — os restantes 26 factos são consequentes que o registo acrescenta. Logo: **55 não é uma medida de enriquecimento útil**; o conteúdo novo é de ordem 8 relações de aninhamento (que repetem a ordem declarada) e o resto restate as definições.

## 5. Auditoria dos 10 literais derivados (os 12 pontos)
Os 12 pontos pedidos: (1) axioma de origem; (2) entidades afetadas; (3) fonte documental; (4) estatuto epistémico; (5) resultado do reasoner; (6) atributos ARFF; (7) significado tabular; (8) condição de aplicabilidade; (9) candidato m-of-n; (10) mecanismo; (11) o Original pode gerá-lo?; (12) controlo que o neutraliza. Pontos comuns a todos: (3) fonte `NOT_VERIFIED`; (4) `HYPOTHESIS`, `AI_DRAFT`; (8) colunas one-hot mutuamente exclusivas da mesma variável, categoria desconhecida ⇒ literal 0, níveis não ordinais fora da extensão; (10) M-R (coluna binária derivada) + M-P (poda de candidatos redundantes); (12) O3 (classe primitiva), ablação do axioma, H (ordem permutada), I (união aleatória).

| Literal | (1) origem | (2)(6) atributos ARFF | (5) reasoner | (7) significado tabular | (9)(11) candidato e espaço do Original (`max_n=3`) | prevalência treino |
|---|---|---|---|---|---|---|
| `onto_any[AtLeast_employment_since_A72]` | `AX-GC-ORD-employment_since-A72` | A72, A73, A74, A75 | inferido (6 factos de suporte) | a linha pertence a um dos níveis A72, A73, A74, A75 da variável «employment_since» (equivalente a: nível ∉ {A71}) | TRIVIAL: negação de uma única coluna (A71 ≤ 0,5) — já é um literal simples do Original; **no espaço do Original: True** | 0.949 |
| `onto_any[AtLeast_employment_since_A73]` | `AX-GC-ORD-employment_since-A73` | A73, A74, A75 | inferido (5 factos de suporte) | a linha pertence a um dos níveis A73, A74, A75 da variável «employment_since» (equivalente a: nível ∉ {A71, A72}) | 1-de-3 (OR de literais positivos); **no espaço do Original: True** | 0.772 |
| `onto_any[AtLeast_employment_since_A74]` | `AX-GC-ORD-employment_since-A74` | A74, A75 | inferido (5 factos de suporte) | a linha pertence a um dos níveis A74, A75 da variável «employment_since» (equivalente a: nível ∉ {A71, A72, A73}) | 1-de-2 (OR de literais positivos); **no espaço do Original: True** | 0.428 |
| `onto_any[AtLeast_job_A172]` | `AX-GC-ORD-job-A172` | A172, A173, A174 | inferido (5 factos de suporte) | a linha pertence a um dos níveis A172, A173, A174 da variável «job» (equivalente a: nível ∉ {A171}) | TRIVIAL: negação de uma única coluna (A171 ≤ 0,5) — já é um literal simples do Original; **no espaço do Original: True** | 0.983 |
| `onto_any[AtLeast_job_A173]` | `AX-GC-ORD-job-A173` | A173, A174 | inferido (5 factos de suporte) | a linha pertence a um dos níveis A173, A174 da variável «job» (equivalente a: nível ∉ {A171, A172}) | 1-de-2 (OR de literais positivos); **no espaço do Original: True** | 0.784 |
| `onto_any[AtLeast_savings_account_A62]` | `AX-GC-ORD-savings_account-A62` | A62, A63, A64 | inferido (5 factos de suporte) | a linha pertence a um dos níveis A62, A63, A64 da variável «savings_account» (equivalente a: nível ∉ {A61, A65}) | 1-de-3 (OR de literais positivos); **no espaço do Original: True** | 0.224 |
| `onto_any[AtLeast_savings_account_A63]` | `AX-GC-ORD-savings_account-A63` | A63, A64 | inferido (5 factos de suporte) | a linha pertence a um dos níveis A63, A64 da variável «savings_account» (equivalente a: nível ∉ {A61, A62, A65}) | 1-de-2 (OR de literais positivos); **no espaço do Original: True** | 0.109 |
| `onto_any[AtLeast_status_checking_account_A12]` | `AX-GC-ORD-status_checking_account-A12` | A12, A13 | inferido (5 factos de suporte) | a linha pertence a um dos níveis A12, A13 da variável «status_checking_account» (equivalente a: nível ∉ {A11, A14}) | 1-de-2 (OR de literais positivos); **no espaço do Original: True** | 0.332 |
| `onto_any[Purpose_HouseholdGoods]` | `AX-GC-TAX-purpose-HouseholdGoods` | A42, A43, A44 | inferido (6 factos de suporte) | a linha pertence a um dos níveis A42, A43, A44 da variável «purpose» (equivalente a: nível ∉ {A40, A41, A410, A45, A46, A48, A49}) | 1-de-3 (OR de literais positivos); **no espaço do Original: True** | 0.465 |
| `onto_any[Purpose_Vehicles]` | `AX-GC-TAX-purpose-Vehicles` | A40, A41 | inferido (4 factos de suporte) | a linha pertence a um dos níveis A40, A41 da variável «purpose» (equivalente a: nível ∉ {A410, A42, A43, A44, A45, A46, A48, A49}) | 1-de-2 (OR de literais positivos); **no espaço do Original: True** | 0.347 |

**Conclusões da auditoria dos literais**
1. **Todos os 10 literais pertencem ao espaço de hipóteses do Original** com `max_n = 3`: ou como `1-de-k` (k ≤ 3 literais positivos) ou como o AND dos literais negados do complemento. O benefício potencial da ontologia está, por isso, na **orientação e eficiência da pesquisa** (candidatos pré-montados que o Original só encontra se o *pool* de literais por ganho individual e o *beam* lá chegarem), **não** na criação de hipóteses inéditas.
2. **2 literais são triviais:** `AtLeast_employment_since_A72` (≡ coluna `A71 ≤ 0,5`) e `AtLeast_job_A172` (≡ `A171 ≤ 0,5`) são a negação de uma única coluna — já são candidatos simples do Original. `AtLeast_job_A172` tem ainda prevalência 0,983 (quase constante).
3. **5 pares aninhados** na mesma variável (5); combinados num mesmo m-of-n, colapsam (M-P os poda).
4. As quatro variáveis ordinais contribuem 8 literais; a taxonomia, 2 (excluída por falta de base documental).
5. **Nenhuma quantidade (55, 10, 61/61) é prova de enriquecimento útil.**

## 6. Estado de validação e pendências humanas
Fichas: três pacotes de revisão separados (domínio, engenharia de ontologias, metodologia) em `…/v0_draft/review_forms/` (ver §7); nenhum preenchido; sem nomes, assinaturas ou aprovações. Condições para um axioma entrar no conjunto avaliável: fonte verificada com localizador; semântica das categorias confirmada; classificação nominal/ordinal confirmada; âmbito e exceções registados; revisão de domínio **e** de ontologia aprovadas por pessoas designadas; hash congelado antes de qualquer avaliação.

## 7. Fichas de revisão separadas
Geradas por `validation/benchmark/h2_review_forms.py`: cada papel recebe só o que é da sua competência — **domínio** (verdade e âmbito de cada afirmação, significado das categorias, nominal/ordinal, fonte), **ontologia** (OWL 2 DL, consistência, satisfatibilidade, inferências esperadas vs obtidas, matching e fecho de mundo aberto), **metodologia** (independência dos rótulos de teste, ausência de fuga, pré-registo, controlos, hashes). Campos de revisor, data e referência de assinatura estão **vazios** e só podem ser preenchidos por pessoas.
