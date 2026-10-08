# Auditoria do isolamento dos mundos OWL e da influência ontológica no TREPAN Reloaded

Relatório técnico. Estado: ramo `claude/repository-architecture-analysis-s1vl5a`, partindo da `main` em `c0b2405`.
Âmbito desta fase: corrigir defeitos de isolamento/estado; **não** foram alterados algoritmos, hiperparâmetros, tuning científico, contrato do
oráculo nem o benchmark oficial. Nenhuma relação OWL ou conhecimento de domínio foi acrescentado. Nenhum resultado abaixo é declarado como ganho
da ontologia: não existe evidência estatística reprodutível de ganho (ver §6 e §7).

Evidência bruta em `docs/evidence/ontology_isolation/`. As ontologias usadas nas medições (ficheiros do utilizador, não versionados) são
identificadas por SHA-256: **A = `ab99ff57…`** (a usada no treino da GUI) e **B = `551f538a…`** (`breast_cancer_domain.owl`).

## 1. Resumo

| Tema | Resultado |
|---|---|
| Defeito original | Com duas ontologias no mesmo processo, a primeira carregada **perdia as 50 features relacionais** (91→41 colunas). O resultado dependia da ordem de carga (A→B→A ≠ A). |
| Causa | (a) leitura de anotações por **nome curto** (`statisticRole`), que colide entre ontologias do mesmo mundo owlready2; (b) todas as ontologias no `default_world` global: cache por caminho, estado do raciocinador partilhado, propriedades partilhadas. |
| Correção | Um **mundo OWL novo por carga** (`core/owl_runtime.load_ontology_isolated`) + leitura de anotações **por propriedade/IRI** (`core/owl_annotations.annotation_values`), independente da ordem. |
| Verificação | 12 testes novos (12/12 passam; revertendo a correção 10/12 falham); 765 testes existentes ligados a ontologias passam; comparação antes/depois do benchmark: 0 diferenças nas primeiras cargas, 68 diferenças corrigidas na carga contaminada. |
| 60 features derivadas | 20 são combinações lineares das colunas originais; 40 trazem informação não linear; **todas** (60/60) seriam obtidas sem ontologia só a partir dos nomes das colunas. |
| Fidelidade do Reloaded | A diferença média (−0,42 pp com a ontologia B; +0,14 pp com A) **não é distinguível de zero** (desvio entre seeds ≈ 2,6–2,9 pp; 5 seeds). O mecanismo de aceitação das divisões tem fragilidades concretas (§5). |

## 2. Auditoria de estado e isolamento

| Componente | Estado mantido | Antes | Depois |
|---|---|---|---|
| owlready2 `default_world` | quadstore, IRIs, propriedades de anotação, indivíduos, resultados do raciocinador | partilhado por todo o processo; `get_ontology(path).load()` devolvia a ontologia em cache | **eliminado do caminho do produto**: 7 pontos de carga usam `load_ontology_isolated` (mundo próprio) |
| `run_owl_reasoner` / HermiT | `sync_reasoner([onto])` raciocina sobre o mundo da ontologia | contaminável por outras ontologias do mundo | âmbito = mundo próprio |
| Leitura de anotações | `getattr(entity, "statisticRole")` | resolvido por nome curto → colisão | por objeto/IRI (`prop[entity]`), união de todas as propriedades homónimas |
| Rótulos (`label/prefLabel/altLabel`) | idem | por atributo | idem (`annotation_values`) |
| `OwlSemanticProvider.build` (benchmark) | carregava no mundo global em cada split | estado acumulado entre splits/ontologias | mundo novo por chamada |
| `OntologyQualityGate`, `OntologyProcessor`, `OntologySemanticGraph` | só estado de instância | sem estado global | sem alteração (verificado por testes) |
| GUI `_ontology_match_cache` | chave = rótulo | atravessava ontologias se não fosse limpa | chave inclui `_ontology_state_id` |
| GUI resultados contrafactuais, estado «stale» | não eram invalidados ao trocar de ontologia | — | `_ontology_changed()`: novo id de estado, invalida `cf_*`, reavalia stale da auditoria |
| `TUNING_EXECUTIONS` (módulo de tuning) | contador de engenharia | global | sem efeito científico; mantido |
| `owlready2.JAVA_EXE` | configuração do processo (Java) | global | idempotente, não é estado de ontologia; mantido |

Outros locais com `get_ontology(...)` já usavam `World()` próprio (`core/benchmark/domain_tbox.py`, scripts de validação).

## 3. Causa-raiz

1. **Colisão por nome curto.** O owlready2 resolve `entity.statisticRole` pelo nome da propriedade dentro do mundo. Duas ontologias que declaram
   `statisticRole` e `measurementFamily` em namespaces distintos ficam com uma única resolução (a última definida); as entidades da outra
   parecem não ter valor. O ramo de recurso (`get_properties()`) não devolve anotações, pelo que nunca compensava. Efeito: sem família/papel,
   `OntologyProcessor._fit_relational_specs` não gera nenhuma feature relacional.
2. **Mundo partilhado.** Todas as cargas iam para o `default_world`: a mesma ontologia carregada de novo devolvia o objeto em cache (com o
   estado do raciocinador de cargas anteriores) e as propriedades de ontologias distintas partilhavam namespace de nomes.

Reprodução mínima (`scripts/owl_isolation_probe.py`, dados genéricos): `docs/evidence/ontology_isolation/probe_before.json` /
`probe_after.json`.

| Sequência | Antes (`main`) | Depois |
|---|---|---|
| A→B→A | A: 91 col. · B: 90 · **A: 41** (sem relacionais) | A: 91 · B: 90 · A: 91 (digest idêntico ao 1.º) |
| B→A→B | B: 90 · **A: 41** · B: 90 | B: 90 · A: 91 · B: 90 |

## 4. Alterações efetuadas

* `core/owl_runtime.py`: `new_owl_world`, `load_ontology_isolated` (RDF/XML e Turtle), `release_ontology`.
* `core/owl_annotations.py` (novo): `annotation_values` por propriedade/IRI; usado por `core/ontology_quality.py`
  (`entity_labels`, `_has_annotation`), `core/ontology_processor.py` e `core/ontology_semantic_graph.py`.
* Pontos de carga trocados para o carregamento isolado: `core/production_training.py`, `core/benchmark/semantic.py`, `core/trepan.py`,
  `core/trepan_reloaded_extractor.py` (RDF/XML, Turtle e TBox alternativo), `validation/ablation_study.py`.
* GUI (`gui/biuri_app_complete.py`): `_ontology_state_id`, `_ontology_changed`, chave da cache de matching.
* `validation/ontology_feature_audit.py` e `scripts/owl_isolation_probe.py`: ferramentas genéricas de auditoria (sem lógica de dataset).
* `tests/test_owl_world_isolation.py`: 12 testes.

Não alterado: algoritmos TREPAN (Original/Reloaded), `OntologyProcessor` (fórmulas), quality gate (limiares), tuning, oráculo, protocolo.

## 5. Evidências

### 5.1 Testes automatizados (determinísticos, metamórficos, regressão)

`tests/test_owl_world_isolation.py` — ontologias sintéticas e genéricas que declaram as mesmas anotações curtas:

* cada carga, mesmo do mesmo ficheiro, obtém um mundo distinto e nunca o `default_world`;
* **A→B→A, B→A→B, AABB, ABBA**: digest dos valores e colunas idênticos ao carregamento isolado; número de features relacionais = 5 × famílias;
* **colisão no mesmo mundo** (duas ontologias de propósito no mesmo `World`): resultado idêntico ao isolado e `annotation_values` correto;
* **correção numérica**: cada feature relacional comparada com a fórmula calculada em NumPy (`difference`, `relative_delta`, `contrast`,
  `normalized_error`, `ratio`) com tolerância 1e-12, e agregados = média padronizada com estatísticas só do treino; nenhuma feature constante
  (≥ 10 valores distintos, `std > 0`, tudo finito); valores constantes ou preenchidos artificialmente não são aceites;
* **metamórficos**: permutação de linhas, permutação de colunas e cópia do ficheiro com outro nome não alteram as features;
* raciocinador por mundo: uma ontologia inconsistente carregada entre duas cargas de A não contamina A;
* `OwlSemanticProvider.build`: contexto semântico idêntico antes e depois de carregar outra ontologia;
* GUI (`BiuriApp.load_ontology_from_file`, A→B→A): mundos e `_ontology_state_id` distintos por carga, cache de matching limpa, resultados CF
  antigos invalidados, features de A idênticas na 3.ª carga (testes do raciocinador/GUI só correm com Java presente; é uma dependência de ambiente).

Mutação (cópia do repositório com o comportamento anterior reposto): ambas as reversões → **10 falhas em 12**; só anotações → 1 falha; só
carregamento → 4 falhas.

Testes existentes ligados a ontologias (72 ficheiros): **765 passed, 0 failed**.

### 5.2 Benchmark antes/depois (protocolo preservado)

`benchmark_before_after_equivalence.json`: sequência ab99 → 551f → ab99 (seed 7, mesmo processo, `BenchmarkRunner`, 5 braços).

| Comparação | Diferenças |
|---|---|
| passo 1 (A) antes vs depois | **0** |
| passo 2 (B) antes vs depois | **0** |
| passo 3 (A repetida) antes vs depois | 68 |
| A (passo 1) vs A repetida, **antes** | 68 (`semantic_features_generated` 61→11 …) |
| A (passo 1) vs A repetida, **depois** | **0** |

Conclusão: onde a carga não estava contaminada os resultados científicos são bit-a-bit iguais (protocolo preservado); onde estava contaminada
agora são idênticos aos da carga isolada. Limitação: não verifiquei se resultados brutos históricos foram produzidos em processos com várias
ontologias; qualquer unidade calculada num processo que carregou mais de uma ontologia deve ser revista.

## 6. As 60 features derivadas (ontologia B; A: 61)

Ferramenta: `validation/ontology_feature_audit.py`. Geradas 90 (B): 20 indicadores de limite constantes removidos (`x >= 0` verdadeiro em todas
as linhas), 10 agregados duplicados removidos (a mesma família sob dois conceitos-pai), 60 retidas.

| Tipo | Operação | N | R² linear vs. TODAS as colunas originais | Informação não linear |
|---|---|---|---|---|
| agregado | média padronizada | 10 | 1,000 | não |
| relacional | `difference` (pior − média) | 10 | 1,000 | não |
| relacional | `relative_delta` | 10 | 0,64–0,99 | sim |
| relacional | `contrast` | 10 | 0,72–0,99 | sim |
| relacional | `normalized_error` | 10 | 0,65–0,98 | sim |
| relacional | `ratio` (erro/média) | 10 | 0,60–0,99 | sim |

* **Dependem de metadado da ontologia** (qual coluna é média/pior/erro de que família): as 50 relacionais; os agregados dependem do agrupamento por pai.
* **Aritmética**: 100% convencional (diferenças, rácios, médias padronizadas).
* **20 são combinações lineares das originais** (10 agregados + 10 diferenças): para um MLP, que já aprende combinações lineares, não acrescentam informação.
* **40 trazem não linearidade** (rácios/contrastes), mas **60/60 são reproduzíveis sem ontologia**: uma heurística só por nomes (sufixos
  `mean/worst/se`, prefixo de família) escolhe as mesmas fontes. Neste dataset a ontologia **formaliza o que os nomes das colunas já dizem**; o
  ganho potencial é de engenharia convencional de features, não de conhecimento exclusivamente ontológico.
* A ontologia só acrescentaria conhecimento não reproduzível por nomes quando o agrupamento/papel **não** estiver recuperável dos nomes
  (colunas opacas, hierarquias semânticas) — cenário a testar na ablação (§8).

## 7. Como a ontologia influencia as divisões do Reloaded e porque a fidelidade desce nalguns casos

Auditoria instrumentada (envolvendo `TrepanReloadedClassifier.fit` externamente, sem alterar o algoritmo): `reloaded_split_audit.json`;
5 seeds × 2 ontologias, configuração leve (não é o protocolo congelado), 143 amostras de teste por seed.

**Mecanismo (lido do código e das linhas de auditoria):**

1. Quando existe um candidato semântico, a pontuação de seleção é `ganho de informação + semantic_bonus`. O bónus observado (0,64–1,12) é
   **5–9× maior** que o ganho de informação dos nós internos não-raiz (0,07–0,24) e ≈1,4× na raiz (ganho 0,63–0,64). Um candidato semântico com **menor** ganho de
   informação ganha a seleção (p. ex. 0,0951 vs 0,1414 → pontuação 0,914 vs 0,141).
2. A única salvaguarda é o gate EFSR, que aceita a mudança por dois critérios: «melhora a fidelidade local e de treino» (ganho mínimo local
   0,002) ou «ganho de coerência semântica dentro da tolerância de fidelidade».
3. Das 10 decisões alteradas: 5 aceites por melhoria de fidelidade local+treino (ganhos locais de 0,0038 a 0,02); **5 aceites por coerência dentro
   da tolerância, das quais 3 com ganho de fidelidade NEGATIVO** (−0,0070, −0,0094, −0,0070), 1 com ganho nulo e 1 com ganho local
   positivo mas ganho nulo no treino real.
4. O limiar de ganho local (0,002) é **inferior à resolução dos dados em que é medido**: um nó com 265 amostras muda a fidelidade em passos de
   0,0038. Uma melhoria de ~1 amostra passou o gate (seed 42, `ganho 0,0038`) e o teste mudou −4,2 pp (8 previsões estragadas, 2 corrigidas).

**Efeito agregado (core vs. Reloaded com ontologia, mesmos dados e oráculo):**

| Ontologia | Previsões distintas | Corrigidas | Estragadas | Δ fidelidade média | Desvio entre seeds |
|---|---|---|---|---|---|
| A (`ab99`) | 39 | 20 | 19 | +0,14 pp | 2,90 pp |
| B (`551f`) | 37 | 17 | 20 | −0,42 pp | 2,55 pp |

Com 5 seeds o erro-padrão é ≈ 1,2 pp: **a descida média não é estatisticamente suportada**; o que está demonstrado é instabilidade estrutural
(poucas amostras trocam de lado em ambos os sentidos) e os três pontos frágeis do mecanismo acima (bónus dominante, aceitação por coerência
com ganho negativo, limiar abaixo da resolução). São observações para uma fase futura; **nada foi alterado** (protocolo congelado).

Predição (MLP, 25 dobras pareadas, IC corrigido de Nadeau–Bengio, `paired_cv_predictive.json`): + todas as features OWL: **+0,63 pp** (A,
IC −0,79 a +2,06) e **+0,56 pp** (B, IC −0,73 a +1,85) de accuracy; controlo embaralhado −1,6 / −2,0 pp. Em ambas o intervalo inclui zero.

## 8. Proposta experimental de ablação (para uma fase futura, com manifest novo congelado antes de correr)

Objetivo: separar (i) conhecimento OWL, (ii) engenharia convencional de features, (iii) alterações de algoritmo. Mesmos splits pareados, mesmo
oráculo congelado, mesmo tuning estrutural cego à ontologia, mesmas seeds mestras.

| Braço | Algoritmo | Features | Mede |
|---|---|---|---|
| E0 | TREPAN Original | originais | referência |
| E1 | Reloaded sem semântica (core) | originais | efeito do **algoritmo** (E1−E0) |
| E2 | Original | originais + **engenharia por nomes** (sem OWL) | efeito da **engenharia convencional** (E2−E0) |
| E3 | Original | originais + features OWL | OWL vs. convencional com o mesmo algoritmo (E3−E2; = 0 se as matrizes forem iguais, o que se verifica à partida) |
| E4 | Reloaded completo | originais + OWL + estrutura semântica | efeito conjunto |
| E5 | Reloaded | idem com **ontologia embaralhada** (mesma contagem de grupos/papéis) | controlo negativo (E4−E5) |
| E6 | Reloaded | originais + engenharia por nomes (sem OWL, sem estrutura semântica) | algoritmo + convencional (E6−E1) |
| E7 | Reloaded com bónus semântico desligado | originais + features OWL | separa **features** de **bónus/candidatos semânticos** (E4−E7) |
| M0–M4 (MLP) | MLP | originais / + convencional / + OWL / + embaralhado / + ruído de igual dimensão | ganho preditivo por tipo de feature |

* **Cenário decisivo**: repetir E2/E3/E4 com **colunas opacas** (`f1…fn`) mantendo o mapeamento por rótulos alternativos na OWL. Aí a heurística
  por nomes falha e a ontologia é a única fonte do agrupamento: é o único caso em que se pode atribuir ganho **ao conhecimento OWL**.
* **Datasets**: conjunto fixado antes de ver resultados, com estruturas distintas (famílias de estatísticas, hierarquias, categóricas,
  sem estrutura relacional). Resultados nulos reportados.
* **Estatística**: unidade = (dataset × seed × partição); endpoint primário = fidelidade ao oráculo em dados retidos; secundários = accuracy,
  balanced accuracy, complexidade. t reamostrado corrigido (Nadeau–Bengio) ou Wilcoxon por dataset (Demšar) com correção de Holm; IC por
  bootstrap; margem de equivalência ±1 pp (TOST) para afirmar «sem efeito». Potência: com desvio das diferenças ≈ 2,7 pp, detetar 1 pp (80%,
  α = 0,05) exige ≈ ((1,96+0,84)·2,7/1)² ≈ **57 unidades independentes** — bem acima das 5 seeds deste relatório.
* **Garantias**: manifest novo e hashes de código congelados antes da execução; resultados brutos imutáveis; equivalência exata das
  otimizações; nenhum ajuste de limiares com base nos resultados; sem ontologias criadas depois de ver os resultados.

## 9. Problemas ainda existentes

1. Resultados históricos: marcados `NOT_VERIFIED_OWL_ISOLATION` (§11); a repetição com o código corrigido continua por fazer.
2. GUI: ao trocar de ontologia depois de treinar, os modelos **não são descartados** mas ficam **bloqueados** (`_stale_models_reason`: explicação, árvore, métricas, exportações e contrafactuais exigem novo treino); contrafactuais são invalidados e a auditoria marca «stale». Os mundos OWL antigos não são fechados (memória) porque
   árvores treinadas podem referenciá-los.
3. Dependência de Java: testes de raciocinador/GUI/fornecedor semântico só correm com Java (sem Java são ignorados com justificação de ambiente);
   a confirmação em CI Linux/Windows 3.11/3.12 só existe depois do push.
4. Mecanismo semântico: pontos frágeis do §7 (bónus dominante, aceitação por coerência com ganho negativo, limiar abaixo da resolução) —
   observados, **não corrigidos** (algoritmo congelado).
5. As medições de §7 usam configuração leve (5 seeds, 143 amostras de teste, árvore até 7 nós), não o protocolo congelado; são exploratórias.
6. Duas ontologias, um só dataset; a heurística «por nomes» é simples (um token de papel por coluna) e subestima o que a engenharia convencional
   consegue.
7. Custo: cada chamada de `OwlSemanticProvider.build` volta a analisar o ficheiro OWL (mundo novo); não medido.

## 10. Reprodução

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_owl_world_isolation.py -q
python scripts/owl_isolation_probe.py --csv dados.csv --target classe --owl A=a.owl B=b.owl --order ABA BAB
python scripts/check_dataset_agnosticism.py
```

## 11. Resultados históricos potencialmente contaminados (registo não destrutivo)

A colisão exigia **duas ou mais ontologias carregadas no mesmo processo** (mundo owlready2 global). Resultados gerados antes desta correcção por
execuções que carregaram várias ontologias em sequência no mesmo processo podem ter features derivadas/métricas semânticas alteradas
(ver §3–§5: 91→41 colunas em A→B→A). Como o processo gerador de cada resultado **não foi registado**, o isolamento não é demonstrável a posteriori.

* Registo: `docs/evidence/ontology_isolation/historical_results_registry.json` (gerado por `python -m validation.historical_results_registry --out …`).
  Para cada ficheiro de dados (`results/**`, JSON/CSV na raiz) indica `sha256`, classificação e estado.
* Estado `NOT_VERIFIED_OWL_ISOLATION` = **não verificado**, não «errado». Classes: `potentially_contaminated_multi_ontology` (≥2 datasets/ontologias
  no mesmo artefacto), `unverified_single_ontology` (uma ontologia, processo desconhecido) e `NOT_AFFECTED_NO_ONTOLOGY` (braços sem ontologia: C4.5,
  TREPAN Original, MLP base, desempenho).
* Nenhum resultado bruto foi apagado ou alterado (teste `tests/test_historical_results_registry.py`). Resultados semânticos destes ficheiros não devem
  ser citados como evidência até serem repetidos com o código corrigido (uma unidade = um `World` isolado).
* Limitação: classificação por conteúdo; resultados em ficheiros sem identificadores de dataset podem escapar à classe «multi-ontologia».
