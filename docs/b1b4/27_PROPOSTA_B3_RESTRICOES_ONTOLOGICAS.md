# Doc. 27 — Proposta B3-a: restrições ontológicas na escolha dos candidatos m-of-n (DESENHO; nada implementado)

Estado: `B3A = DESIGN_AUTHORIZED` · `B3B = DESIGN_ONLY` · `B3C, B3D = DEFERRED` · `B4 = NOT_IMPLEMENTED`. Este documento é desenho técnico + auditoria do espaço de candidatos (`validation/experimental/b3a_contract.py`, 6 testes de propriedades em `tests/test_b3a_contract.py`). **Nenhuma linha de `core/` foi alterada para B3-a.** A implementação no núcleo exige autorização posterior.

## 1. O que já existe (leitura do código) e o que a auditoria mostrou
- `TrepanOriginalClassifier._best_mofn`: beam sobre um *pool* de literais (top `max(8, 3·max_features_per_node)` por score) e extensões `m∈{m, m+1}`; qualquer par de features pode combinar-se.
- Reloaded: (i) peso semântico multiplicativo do score (`_semantic_weight_for_test`, bónus de coesão de grupo/relatedness); (ii) `_ontology_guided_candidates`, que gera candidatos dentro de «componentes semânticos» (mesmo grupo **ou** relatedness ≥ `semantic_relation_threshold`=0,35) apenas em regiões de erro elegíveis (EFSR); (iii) gate EFSR de fidelidade local.
- **Achado da auditoria (evidência `docs/evidence/b1b4/b3a_design/`)**: nas TBoxes DEV (iris, wine, breast_cancer, digits, german_credit) a relatedness entre **todos** os pares de features originais é **exatamente 0,712** (= 0,88·0,88·0,92: dois saltos de subsunção via a classe-raiz comum `DomainMeasurement`), logo **≥ 0,35 em 100 % dos pares**. A relatedness é constante, não discrimina, e é idêntica para a ontologia verdadeira e a permutada. Consequência: o critério «grupo OU relatedness» do mecanismo existente colapsa num único componente e o espaço autorizado é todo o espaço (verdadeira = permutada, por construção) — compatível com o resultado do Doc. 23 (verdadeira ≈ permutada). O único canal que distingue verdadeira de permutada nestas TBoxes é a **partição em grupos**.

| Dataset (DEV) | pares permitidos, política «relatedness» (original) | pares permitidos, «só grupos» (original) | subconjuntos de 3 permitidos, «só grupos» |
|---|---|---|---|
| iris | 1,00 | 0,33 | 0,00 |
| wine | 1,00 | 0,30 | 0,07 |
| breast_cancer | 1,00 | 0,31 | 0,09 |
| digits | 1,00 | 0,33 | 0,11 |
| german_credit | 1,00 | 0,20 | 0,04 |
Permutar a atribuição feature→grupo (controlo `structure`) preserva **exatamente** estes números (mesmos tamanhos de grupo ⇒ mesmas oportunidades de pesquisa; testado). Aviso: com 2 grupos de 2 features (iris) não há m-of-n de ordem 3 dentro de um grupo.

## 2. Hipóteses científicas (a testar; não assumidas)
- **H_reg** (regularização genérica): restringir o espaço de hipóteses m-of-n reduz o sobreajuste da seleção (número efetivo de comparações) sobre amostras sintéticas finitas, **qualquer que seja** a restrição.
- **H_know** (conhecimento): restringir segundo relações **verdadeiras** da ontologia ajuda mais do que restringir segundo estrutura permutada/aleatória de igual tamanho.
- Só H_know sustenta uma afirmação ontológica. Se só H_reg se verificar, o ganho **não** é atribuído à ontologia (e seria uma melhoria independente da ontologia, proibida só no Reloaded pelo contrato).
Ponto lógico central: com critério estatístico inalterado, restringir o conjunto de candidatos nunca aumenta o score *amostral* máximo; um efeito só pode vir de generalização (menos sobreajuste) ou de diferenças de busca do beam. Por isso B3-a **não pode** ser avaliado pelo score interno, apenas por fidelidade/accuracy fora da amostra.

## 3. Arquitetura proposta
Ponto de integração único (futuro): `TrepanReloadedClassifier._best_mofn`/hook de extensão do beam, com flag `semantic_candidate_restriction ∈ {"off","prioritize","restrict"}` (default `off`). Sem ontologia efetiva (S0/S1/S2) ⇒ ramo `super()` ⇒ **idêntico ao Original** (EQ v2.0). B3-a **não** gera novos literais nem limiares (reutiliza `_candidate_literals`), **não** muda consultas ao oráculo, **não** altera o amostrador, **não** introduz m-of-n novos (isso é B4).
- **B3-a-1 `prioritize` (recomendado implementar primeiro)**: o pool de literais do beam é ordenado por (score, compatibilidade semântica) e as extensões compatíveis são avaliadas primeiro; o vencedor continua a ser escolhido pelo **mesmo score estatístico**; o desempate entre candidatos estatisticamente equivalentes (empate dentro de uma tolerância declarada a priori) favorece o compatível. Nunca aceita candidato de score inferior ⇒ cumpre «não forçar aceitação de candidatos inferiores». Efeito esperado: pequeno (só altera decisões em empates/limites do beam).
- **B3-a-2 `restrict` (segundo braço, só com autorização)**: no passo de extensão, só entram literais compatíveis com todos os literais já presentes; o literal-semente não é restringido. Pode escolher um candidato de score amostral inferior ao do Original ⇒ **só é aceite se passar um gate de não-inferioridade** (reutilização do gate EFSR de fidelidade local sobre amostra independente da usada na escolha). Sem o gate, viola o contrato — por isso é um braço separado e opcional.
- Mapeamento feature↔compatibilidade: função pura `build_compatibility(groups, relatedness, threshold, use_relatedness)` (protótipo e testes já existentes). A política **«só grupos/relações diretas»** (sem o atalho de raiz comum) é a única não-vazia nas TBoxes DEV; usar relatedness ≥ 0,35 tornaria B3-a vácuo (§1).

## 4. Contrato de mapeamento OWL → compatibilidade (mundo aberto)
| Axioma / fonte | Suportado hoje no grafo | Tradução proposta | Observação |
|---|---|---|---|
| `SubClassOf`/domínio comum (mesma classe-grupo) | sim (`parents`, `domains`) | par **permitido** (`same_group`) | prior de coerência, não consequência lógica |
| `EquivalentTo` | sim | permitido | |
| `SubPropertyOf` / propriedades relacionadas | parcial (`is_a` em propriedades) | permitido | |
| `domain`/`range` partilhado | sim (peso 0,68) | permitido se relação **direta** | o caminho via raiz comum não conta |
| `inverseOf` | sim | permitido | raro em tabelas |
| subsunção inferida pelo reasoner | sim (`inferred_subsumption`) | permitido, registado como inferido | exige reasoner consistente |
| `AllDisjoint`/`DisjointWith` | **não extraído** (existe nas TBoxes DEV) | par **desencorajado** (não proibido) | exige extensão do grafo; semântica «não misturar famílias disjuntas», decisão de desenho |
| `measurementFamily`/`statisticRole` (anotações) | só em breast_cancer, não entram na relatedness | permitido entre papéis da mesma família | anotação, não axioma |
| Cardinalidades, cadeias, SWRL, restrições existenciais sobre object properties | não | **não suportado** (sem tradução para colunas tabulares) | não inventar relações |
| Limites numéricos / `xsd:integer` | parcial (`feature_bounds`) | **fora de B3-a** (pertence à validade das consultas) | |
Regras de rigor: (a) feature sem mapeamento aceite, ou com mapeamento **ambíguo** (>1 entidade), é tratada como sem conhecimento ⇒ **sem restrição** e registada; (b) ontologia inconsistente/rejeitada pelo quality gate ⇒ S1 ⇒ Original; (c) a ausência de axioma nunca é lida como incompatibilidade (mundo aberto) — só pares **ambos mapeados**, de grupos distintos e sem relação direta ficam desencorajados; (d) sem fecho transitivo; (e) cada restrição regista a sua origem (`same_group`, `direct_relation:<tipo>`, `inferred`, `open_world_unmapped`).

## 5. Exemplos genéricos (sem datasets)
1. Medidas com papéis estatísticos de uma mesma grandeza (valor central/erro/extremo): permitem-se regras que combinem as três facetas da mesma grandeza e desencoraja-se misturá-las com grandezas de famílias disjuntas sem relação direta.
2. Atributos de um mesmo domínio temático (p.ex. «estado da conta» agrupando dois atributos): m-of-n dentro do tema.
3. Região espacial de um sinal (grupos de sensores/pixels): m-of-n dentro da região.
4. Nota categórica: níveis one-hot do **mesmo atributo** são mutuamente exclusivos; combiná-los num m-of-n é redundante. Evitá-lo é uma propriedade dos **dados** (independente da ontologia) e, aplicada só ao Reloaded, seria uma melhoria não ontológica: se se adotar, tem de valer igualmente para o Original (decisão do utilizador, §9).

## 6. Geração e filtragem de candidatos (pseudo-código do desenho)
```
pool ← _candidate_literals(X, y)                      # inalterado (mesmos literais, limiares e scores)
C    ← compatibility(ctx)                              # None se ontologia não efetiva  → ramo Original
beam ← seed(best literal)                              # semente sem restrição
para cada teste t no beam, para cada literal l no pool (ordem: prioritize→compatíveis primeiro; restrict→só compatíveis):
    se l.feature ∉ t.features  e  [modo restrict ⇒ ∀ f∈t.features: C[f, l.feature]]:
        avaliar m∈{m, m+1} com o MESMO score e o MESMO teste de partições distintas
escolha final = melhor score (prioritize: desempate semântico só dentro da tolerância; restrict: gate de não-inferioridade)
registar: candidatos gerados/avaliados/filtrados por origem; axiomas usados; decisão alterada vs Original
```
Orçamento de pesquisa: nº de avaliações m-of-n ≤ o do Original no mesmo nó (restrict) ou = (prioritize); zero consultas adicionais ao oráculo; custo de CPU medido (`mofn_evaluated`, tempo de busca).

## 7. Controlos ontológicos (mesmas oportunidades de pesquisa)
| Controlo | Construção | Propriedade garantida |
|---|---|---|
| Sem ontologia / OWL rejeitada | ramo Original | EQ exata |
| OWL válida sem influência efetiva (compatibilidade total) | `C ≡ tudo permitido` | ≡ Original (S2) |
| Estrutura **permutada** (`structure`) | permuta feature→grupo | mesmos tamanhos de grupo ⇒ **mesmo** nº de pares/subconjuntos autorizados (testado) |
| Grupos **aleatórios** de tamanhos iguais | partição aleatória com o mesmo multiconjunto de tamanhos | idem |
| Estrutura **neutralizada** | todas as features num grupo «general» | sem restrição (≡ Original) |
| Engenharia **convencional** | grupos por correlação (`ConventionalFeatureProvider`) com tamanhos igualados por reamostragem | separa conhecimento de proximidade estatística |
Contraste principal: **Reloaded + ontologia real** vs **Reloaded + ontologia permutada** (mesmo oráculo, orçamento, amostrador, sementes e réplicas); contrastes secundários vs Reloaded sem B3-a e vs Original.

## 8. Instrumentação e testes previstos
Métricas a registar (nomes pedidos): `semantic_candidates_generated`, `semantic_candidates_accepted`, `ontology_axioms_used`, `semantic_rules_used`, `semantic_split_gain`, `ontology_vs_permuted_fidelity`, `ontology_vs_permuted_accuracy`, `oracle_query_cost`, `tree_complexity` (o código atual já expõe parcialmente `_semantic_candidates_last_`, `semantic_split_audit_` e `oracle_accounting_`; os restantes são novos). Um candidato com referência OWL **não** conta como ganho; só diferenças favoráveis e mensuráveis face aos controlos.
Testes de propriedades (alguns já implementados no contrato):
- implementados: mundo aberto (não mapeada nunca restringe); simetria/diagonal; sem fecho transitivo; rótulo «general» = não mapeado; permutação preserva o tamanho do espaço; determinismo.
- a implementar com B3-a: (E1) sem ontologia ⇒ idêntico ao Original (assinatura E1–E9 e fluxo de consultas); (E2) flag `off` ⇒ idêntico ao Reloaded atual; (E3) conjunto de candidatos de `restrict` ⊆ conjunto do Original; (E4) `prioritize` nunca escolhe score inferior; (E5) zero consultas adicionais ao oráculo; (E6) nº de avaliações ≤ Original; (E7) determinismo com mesma semente; (E8) ambiguidade/mapeamento ausente ⇒ sem restrição e registada; (E9) nenhum nome de dataset (contrato de agnosticismo); (E10) ontologia permutada com o mesmo espaço autorizado.

## 9. Riscos e decisões pendentes
Riscos: relatedness vacuosa (§1); sobre-restrição (iris sem 3-subsets); H_reg confundido com H_know; interação com o gate EFSR já existente; poder estatístico limitado (EP ≈ 0,004–0,007; efeitos plausíveis ≲ 0,01); ontologias DEV manuais com apenas partição em grupos; one-hot (§5.4); `AllDisjoint` ainda não extraído.
Decisões para autorização: (1) implementar `prioritize` (B3-a-1) e/ou `restrict` com gate (B3-a-2); (2) política «só grupos/relações diretas» em vez de relatedness ≥ 0,35; (3) extrair `AllDisjoint` no grafo (extensão ontológica, auditável); (4) tratar níveis one-hot do mesmo atributo como incompatíveis em Original **e** Reloaded (melhoria não ontológica); (5) ontologias DEV mais ricas (hierarquias além da partição) antes de testar H_know.

## 10. Preparação de B3-b (desempate semântico) — SEM implementação
Ideia: quando dois candidatos têm qualidade estatística **equivalente**, desempatar por hierarquia/relatedness direta OWL. Propriedades a exigir (testes futuros, não escritos): (T1) só atua se |Δscore| ≤ tolerância declarada a priori (equivalência estatística), nunca altera um vencedor claro; (T2) sem ontologia ⇒ o desempate canónico do Original (ruído m-of-n do Doc. 19, limitação de reprodutibilidade registada) permanece inalterado; (T3) com ontologia, o desempate semântico é determinístico e documentado — mas **não** pode ser confundido com a política determinística de desempate como variante experimental isolada do Original (efeito não ontológico); (T4) controlo permutado com o mesmo número de empates resolvidos; (T5) taxa de empates efetivamente resolvidos registada (se for ≈0, o mecanismo é vácuo); (T6) zero consultas extra. Pré-requisito: medir primeiro a frequência de empates estatísticos nas árvores atuais (instrumentação apenas).
