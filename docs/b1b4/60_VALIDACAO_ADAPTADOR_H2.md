# Doc. 60 — Validação técnica do adaptador H2 e do arnês de ponta a ponta

Estado: **verificações de correção e equivalência apenas** (sem campanha, sem métricas de desempenho como evidência). Só `validation/`; `core/` intocado; `M0_M6_CORE_INTEGRATION = NOT_AUTHORIZED`. Código: `validation/benchmark/h2_harness.py`, `h2_synthetic.py`, `h2_trace.py`, `h2_axiom_mechanisms.py`, `h2_german_smoke.py`; testes: `tests/test_h2_harness.py`, `tests/test_h2_trace.py`, `tests/test_h2_contract_and_mechanisms.py`.

## 1. Cadeia ligada ao arnês experimental
`Axioma (OWL) → Reasoner (HermiT) → Fact Registry (B3.1) → Literal derivado → Candidato m-of-n → Decisão de split → Árvore`, com os braços do Doc. 56 (`ARMS = A_ORIG, B_WIDE, C_ENRICHED_NO_GUIDANCE, D_GROUPS_NO_OWL, E_FLAT_OWL, F_EXPRESSIVE, G_NEUTRALIZED, H_PERMUTED, I_CONVENTIONAL`) em `run_arm(arm, ctx)`; `H2Steered` combina `DerivedLiteralTrepan` (literais derivados + poda lógica) com `SemanticSteeredTrepan` (unidades B3.1). Cada execução devolve `instance_hash`, `oracle_id`, regras (`rules_sha`), consultas, chamadas ao oráculo e o rastreio por literal.

## 2. Propriedades verificadas (testes)
| Propriedade | Como se verifica |
|---|---|
| Equivalência sem semântica | braço sem literais/poda ≡ Original: mesmas regras, mesmo n.º e fluxo de consultas |
| Identidade do oráculo e das instâncias | `oracle_id` e `instance_hash` iguais entre braços; o oráculo só vê colunas base (`AugmentedOracle`) |
| Colunas derivadas coerentes em treino e consultas | recalculadas das colunas base também nas consultas sintéticas |
| Categorias desconhecidas | linha sem nível ativo ⇒ derivadas 0; nível não ordinal nunca entra na extensão |
| Rastreabilidade | por literal: axioma → `fact_ids` → avaliações como candidato → nós escolhidos → podas |
| Limites de candidatos/consultas | `max_nodes`, `max_queries`, `max_n`, `beam_width` iguais; n.º de colunas = n.º de literais |
| Integridade treino/teste e ausência de fuga | construção sem rótulos nem teste; `fit` não recebe teste |
| Independência de dataset | ontologias sintéticas de validade conhecida (`h2_synthetic.py`); sem nomes de datasets no núcleo; guard de agnosticismo |
| Atribuição contrafactual | ablação de um axioma remove exatamente um literal e regista mudança de decisão |
Resultados: `tests/test_h2_harness.py` 6 passed; `tests/test_h2_sources_inventory.py` 5 passed; `tests/test_h2_audit.py`, `test_h2_trace.py`, `test_h2_contract_and_mechanisms.py` a passar (execução final no §5).

## 3. Defeito encontrado e corrigido
`derived_union_literals` transformava classes organizacionais (p. ex. um grupo `Noise` com colunas binárias da mesma família) em literais. Correção: `defined_union_concepts(owl_path)` (apenas classes com `owl:equivalentClass` + `owl:unionOf`) e parâmetro `only_concepts`; teste de regressão; a evidência do German não muda (10 literais).

## 4. Verificação de funcionamento no German Credit (**não é avaliação**)
`h2_german_smoke.py` → `docs/evidence/b1b4/h2/german_functional_check.json`: **«verificação de funcionamento; rascunho de IA não validado»**, treino 750 linhas (mesmo `instance_hash`), parâmetros do Original, **sem accuracy, fidelidade ou qualquer métrica de desempenho**. Observado (estrutura, não mérito): o braço A não tem literais derivados; C e F têm 10; em C e F só 2 literais foram avaliados como candidatos (45 e 6 avaliações), 1 foi escolhido no nó 0, **sem podas**, e C e F produzem **o mesmo traço** — pelo critério do Doc. 56 §6 isso é um **sinal `UNTESTABLE`** para H2 nas condições do defeito (a poda M-P não teve efeito; a representação enriquecida mal entra no pool). Não é um resultado sobre a ontologia: o conjunto de axiomas é de rascunho e as fontes não foram verificadas.

## 5. Limitações
O adaptador é uma subclasse do Original em `validation/`, **não** o Reloaded do núcleo; M-G e os tipos T3–T9 permanecem desenho; o perfil OWL 2 DL formal não foi verificado; as ontologias sintéticas provam correção do mecanismo, não benefício; sem fontes verificadas, o German serve apenas de teste técnico.
