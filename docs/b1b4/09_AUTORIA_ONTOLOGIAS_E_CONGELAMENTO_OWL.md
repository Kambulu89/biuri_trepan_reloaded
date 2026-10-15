# Documento 9 — Protocolo de autoria independente de ontologias e contrato de congelamento OWL (Fase 0.5)

Âmbito: ontologias OWL 2 DL para os datasets CONFIRM (Doc. 8). **Nenhuma foi escrita nesta fase.** Este documento fixa *como* serão produzidas, validadas e congeladas.
Uma ontologia logicamente válida **não é por isso útil**: validade e utilidade são medidas em separado (§7).

## 1. Papéis e separação de funções
| Papel | Quem | Pode | Não pode |
|---|---|---|---|
| **Autor (A)** | pessoa designada por ti (ideal: não-programador do algoritmo) ou eu a pedido teu, **só** com as fontes de §2 | escrever a TBox a partir de conhecimento de domínio documentado | ver erros/accuracy/fidelidade/regras/ tuning/ seeds de **qualquer** modelo nesse dataset |
| **Validador (V)** | pessoa ≠ A (ou lista de verificação executada por ti) | rever axiomas vs fontes; executar validação técnica | alterar axiomas |
| **Desenvolvedor do algoritmo (D)** | eu / equipa BIURI | fornecer o validador técnico e o gate | editar ontologias de CONFIRM; ajustar o algoritmo depois de ver desempenho nelas |
Se eu (assistente) escrever alguma ontologia, fico no papel A e **não** executo nenhum modelo nesses datasets até ao congelamento; essa autoria fica registada no `authoring_log`. Recomendo que A seja humano.

## 2. Entradas permitidas e proibidas (a tua lista, integrada)
Permitidas: nomes e tipos dos atributos · dicionários de dados · documentação oficial · normas técnicas · literatura de domínio · metadados independentes dos resultados preditivos.
Proibidas: erros do MLP · accuracy/fidelidade do TREPAN · resultados nos conjuntos de teste · resultados de tuning · seeds vencedoras · regras geradas pelo classificador para ajustar retroativamente a ontologia ·
**valores do alvo ou das linhas** (nenhum indivíduo/ABox com linhas do dataset; ver anti-contaminação em §4).
O autor assina a declaração (§6) de que respeitou as proibições.

## 3. Procedimento
1. **Folha de conhecimento** (`knowledge_sheet.csv`): para cada atributo → nome publicado, tipo, unidade, definição oficial, **fonte (URL/DOI/página/citação)**, conceito/grupo proposto, relações propostas, texto-evidência. Axioma sem fonte = não entra.
2. **TBox em OWL 2 DL:** classes (conceitos de domínio), propriedades (`rdfs:subPropertyOf`, `domain`, `range`), hierarquias `subClassOf`, restrições documentadas (limites físicos com fonte), `rdfs:label` = **nome publicado do atributo** + `skos:altLabel` (sinónimos oficiais),
   anotações aceites pelo pipeline (`measurementFamily`, `statisticRole` quando a documentação distingue medidas/papéis). Proveniência por axioma (`dcterms:source`).
3. **Mapeamento atributo↔entidade** (`attribute_mapping.csv`): derivado do dicionário de dados (nomes), **não** dos dados. Para atributos nominais o mapeamento é ao atributo e aos seus valores via dicionário (EC11). Para o cenário de nomes opacos, `column_alias_map.csv` (índice/alias → IRI) gerado do dicionário (D-6).
4. **Validação técnica** (`validation/confirm/validate_ontology.py`, a criar na Fase 1b; só nomes de atributos, sem dados): carregamento num `World` isolado (PR #9); perfil OWL 2 DL; **reasoner (HermiT)** sem inconsistências nem classes insatisfazíveis; **quality gate do BIURI**
   (`core/ontology_quality.py`) sobre os nomes do dicionário: pontuação de matching, margem, cobertura, ≥ 2 entidades únicas, rácio genérico, TBox-only; registo do relatório JSON.
5. **Rondas de correção:** se a validação falhar por **defeito de autoria** (sintaxe, perfil DL, rótulos que não coincidem com o dicionário, falta de fonte), o autor corrige — **máximo 2 rondas**, cada uma com diff e justificação no `authoring_log`. Uma ontologia que passa a validação mas **não gera estrutura útil** é um *resultado* (reportado), nunca motivo de reedição.
6. **Revisão independente** (V): checklist de §4 + amostragem de ≥ 20 % dos axiomas contra as fontes.
7. **Congelamento** (§5). Depois do congelamento, qualquer alteração é proibida.

## 4. Verificações automáticas anti-contaminação (executadas antes do congelamento)
* Sem indivíduos/ABox com linhas do dataset (`rowIndex`, valores de linhas, classes «rótulo-alvo-como-indivíduo»): precedente `adult_census_full_ontology.owl` (`CONTAMINATED_ONTOLOGY`).
* Nenhuma classe/indivíduo/literal igual aos **valores** do alvo (excepto se o dicionário os documentar como conceito de domínio e V o aprovar por escrito).
* Nenhum limiar numérico, exceto limites físicos/normativos com fonte citada.
* Todo axioma tem `dcterms:source`; todo rótulo de atributo existe no dicionário; cobertura do dicionário reportada.
* Hash do ficheiro ≠ ontologia de qualquer dataset DEV.
* `independent` = verdadeiro apenas se A ≠ D e A não conheceu resultados (declaração).
Falha numa verificação ⇒ ronda de correção (defeito de autoria) ou ontologia declarada inválida (dataset sem ontologia efetiva → cenário R≡O, também informativo).

## 5. Contrato de congelamento OWL
Por dataset, antes de qualquer avaliação confirmatória, regista-se em `BENCHMARK_MANIFEST_v2.json` (secção `ontology_freeze`) e em `docs/b1b4/ontologies/<dataset_id>/OWL_FREEZE.json`:
`dataset_id`, `ontology_id`, `owl_sha256` (bytes do ficheiro), `owl_size_bytes`, `knowledge_sheet_sha256`, `attribute_mapping_sha256`, `authoring_log_sha256`, `sources_list_sha256`, `author_declaration_sha256`,
`validation_report_sha256` (reasoner + gate + anti-contaminação), versões (owlready2, Java, HermiT), `frozen_at_utc`, `git_commit` e **tag** `ontologies-confirm-v1`.
Regras: (1) o `sha256` do ficheiro usado em cada unidade é recalculado e **tem de coincidir** com o do manifest, senão a execução aborta; (2) alterações pós-congelamento só via nova versão do manifest e **novos** datasets CONFIRM; (3) o mesmo ficheiro é usado por todos os braços; a ontologia permutada/aleatória é derivada determinística do congelado (seed do manifest);
(4) os ficheiros OWL, as folhas e as declarações ficam versionados no repositório (não em `results/`).

## 6. Registo de proveniência e declaração
`docs/b1b4/ontologies/<dataset_id>/`: `<id>.owl`, `knowledge_sheet.csv`, `attribute_mapping.csv`, `column_alias_map.csv`, `authoring_log.md` (cronologia, rondas, diffs), `declaration.md`, `validation_report.json`, `OWL_FREEZE.json`.
Modelo de declaração: «Eu, <nome>, declaro que não vi nem usei erros, accuracy, fidelidade, regras, resultados de teste/tuning ou seeds de qualquer modelo neste dataset; usei apenas as fontes listadas; não inseri linhas ou valores do alvo; não editei a ontologia após o congelamento. Conflitos de interesse: <…>. Data/assinatura.»

## 7. Validade ≠ utilidade
Relatório separado por dataset: **(a)** validade (OWL 2 DL, reasoner, gate, anti-contaminação); **(b)** estrutura que o pipeline consegue explorar (nº de features derivadas aceites, grupos, relações, `OntoDepth`, parte replicável pelos nomes
— `validation/ontology_feature_audit.py`); **(c)** efeito (só depois do CONFIRM, via Δ_total, Δ_repr, Δ_guia, Δ_conhecimento, Δ_convencional). Nunca se afirma utilidade a partir de (a).

## 8. Agnosticismo
O gate e o processamento OWL continuam a funcionar com ontologias e datasets desconhecidos; nenhuma regra do pipeline depende do `dataset_id`/`ontology_id`. As ontologias vivem em `docs/` (e `validation/`), nunca em `core/`.
