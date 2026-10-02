# BIURI / TREPAN Reloaded V9.2 — Separação dos Gates Ontológicos

## Objetivo

Corrigir a interpretação errada em que uma ontologia estruturalmente válida era apresentada como "rejeitada" sempre que as features derivadas não melhoravam suficientemente o professor MLP. A implementação permanece agnóstica ao dataset: não contém aliases, nomes de datasets, thresholds de domínios específicos nem regras dedicadas aos ficheiros usados na validação manual.

## Alterações implementadas

### 1. Quality gate ontológico separado do gate preditivo

`core/ontology_quality.py` passa a produzir estados estruturais explícitos:

- `VALID_DOMAIN_ONTOLOGY`
- `CONTAMINATED_ONTOLOGY`
- `LOGICALLY_INCONSISTENT`
- `REASONER_NOT_VALIDATED`
- `INSUFFICIENT_SCHEMA_COVERAGE`
- `GENERIC_ONTOLOGY`
- `INVALID_ONTOLOGY`

O relatório continua a guardar `accepted`, `issues`, `warnings`, métricas e o matching feature-a-feature.

### 2. Matching agnóstico a convenções de nomes

O matching continua a usar nomes e labels da OWL, mas passa a reconhecer genericamente:

- snake_case;
- kebab-case;
- espaços;
- CamelCase;
- mesma composição de tokens em ordem diferente.

Exemplo genérico: `feature_total` e `TotalFeature` podem ser reconhecidos como os mesmos termos sem existir alias específico no código.

### 3. Gate de utilidade OOF com estados honestos

`core/semantic_utility_gate.py` deixou de transformar falhas de qualidade/matching em `REJECT_NO_INFORMATIONAL_GAIN`.

Estados adicionais incluem:

- `REJECT_ONTOLOGY_MATCHING`
- `REJECT_ONTOLOGY_QUALITY`
- `REJECT_ONTOLOGY_REASONER`
- `REJECT_NO_DERIVED_SEMANTIC_FEATURES`
- `REJECT_NO_INFORMATIONAL_NOVELTY`
- `REJECT_GAIN_BELOW_COMPLEXITY_COST`
- `REJECT_INSUFFICIENT_NET_UTILITY`

`REJECT_NO_INFORMATIONAL_GAIN` fica reservado para ausência real de ganho OOF.

### 4. Penalização de complexidade corrigida

Antes, adicionar 1 feature semântica num pool de 3 tinha custo `1/3`, independentemente de o modelo base ter 5, 30 ou 500 features.

Agora a razão é calculada relativamente ao modelo selecionado:

`semantic_count / (base_feature_count + semantic_count)`

Isto evita penalização desproporcional e continua totalmente agnóstico ao dataset.

### 5. OWL estrutural pode continuar disponível ao TREPAN Reloaded

Foi criado `core/ontology_stage_status.py` para separar:

- `ontology_quality_accepted`;
- `ontology_structural_available`;
- `ontology_feature_engineering_accepted`;
- `ontological_teacher_accepted`;
- `trepan_semantic_use_allowed`.

Uma OWL válida pode, portanto, orientar o TREPAN Reloaded mesmo quando o professor MLP ontológico é rejeitado pelo gate OOF.

### 6. Matching validado sincronizado com o núcleo Reloaded

`TrepanReloadedExtractor.register_quality_matches()` sincroniza os matches aceites pelo quality gate com `feature_semantics`, sem consultar rótulos e sem criar conhecimento específico de datasets.

### 7. Sem features derivadas deixou de ser erro fatal

A GUI já não aborta o treino apenas porque `OntologyProcessor` não gerou uma coluna derivada independente. Nesse caso:

- a OWL pode continuar estruturalmente válida;
- o feature gate do professor informa que não existem features derivadas independentes;
- o MLP Original permanece como oráculo;
- o TREPAN Reloaded pode continuar a usar o matching/estrutura OWL nos seus splits.

### 8. GUI não mostra métricas ausentes como 0,0%

Foi criado `gui/ontology_status_presenter.py`.

Métricas não calculadas são apresentadas como `N/A`. A interface mostra separadamente:

- estado estrutural da OWL;
- cobertura de matching;
- gate das features do MLP;
- gate do professor;
- disponibilidade semântica para TREPAN Reloaded.

As estatísticas de mapping agora preferem o `OntologyQualityReport`, em vez do estado legado `feature_semantics` vazio.

### 9. Comparação Original vs Reloaded com o mesmo orçamento

No caminho histórico principal, TREPAN Original e TREPAN Reloaded passam a receber os mesmos:

- `sample_size`;
- `max_queries`;
- `max_nodes`;
- `max_depth`;
- `min_samples_leaf`;
- seed.

A diferença experimental fica reservada à extensão semântica/OWL.

## Testes adicionados

`tests/test_ontology_gate_separation_v92.py` cobre:

1. matching independente da ordem/convenção dos tokens;
2. falha de matching não rotulada como ausência de ganho informacional;
3. ontologia válida sem features derivadas;
4. penalização de complexidade relativa ao modelo total;
5. distinção entre ganho bruto positivo e custo de complexidade;
6. estatísticas da GUI provenientes do quality gate;
7. separação dos estados estrutural/MLP/TREPAN;
8. `N/A` para métricas não calculadas;
9. contexto Reloaded mantém OWL estrutural com MLP Original;
10. núcleo Reloaded histórico é usado no espaço original quando a OWL estrutural é válida;
11. budgets históricos Original/Reloaded iguais.

## Validação manual com os ficheiros fornecidos pelo utilizador

Os ficheiros enviados foram utilizados apenas como fixture manual de diagnóstico e **não foram incorporados no código nem nos testes automáticos**.

Sem executar HermiT/owlready2 neste ambiente, foi feita uma auditoria de schema/matching por leitura do ARFF e RDF/XML. O matcher genérico corrigido encontrou:

- features do ARFF: 30;
- matches aceites: 30;
- cobertura de schema: 100%;
- entidades distintas: 30;
- confiança média do matching: 1.0.

Isto valida apenas o matching. A consistência lógica OWL/HermiT não foi executada neste ambiente e não é declarada como validada.

## Componentes não validados neste ambiente

- PyQt6 runtime: não disponível;
- owlready2/HermiT: não disponível;
- TensorFlow/CLEAR: não disponível;
- Windows: não executado;
- Python 3.11/3.12: runtime atual é Python 3.13.5.

## Confirmatório congelado

`results/confirmatory_v7/` permaneceu byte-a-byte inalterado. `run_confirmatory_locked.py --execute` não foi executado.
