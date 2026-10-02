# BIURI / TREPAN Reloaded V9.2 — Upgrade do Enriquecimento Semântico do MLP

## Objetivo

Implementar o caminho necessário para que uma OWL estruturalmente válida e corretamente mapeada possa produzir **features semânticas independentes, auditáveis e potencialmente úteis ao MLP**, sem reduzir artificialmente os gates científicos.

> Importante: a implementação **não força `ACCEPT_ONTOLOGY`**. A aceitação continua dependente de ganho OOF real no conjunto de desenvolvimento. O que foi corrigido é a capacidade do pipeline de gerar e avaliar conhecimento semântico mais informativo.

## Alterações implementadas

### 1. Agregações hierárquicas normalizadas no treino

Arquivo: `core/ontology_processor.py`

Antes, `onto_<Parent>_aggregate` era a média direta de features com escalas diferentes. Agora cada fonte é normalizada com média/desvio aprendidos apenas no treino e o agregado é calculado sobre z-scores.

- `method = standardized_mean`
- média/desvio guardados no spec de transformação;
- `scale=1.0` para features constantes, sem divisão por zero;
- nenhum `fill_value=0` introduzido.

### 2. Features relacionais dirigidas pela OWL

Arquivo: `core/ontology_processor.py`

O motor passou a interpretar metadados genéricos da ontologia:

- `measurementFamily`
- `statisticRole`

Quando uma família declara os papéis `mean`, `error` e/ou `worst`, o processador pode gerar:

- `onto_<family>_worst_minus_mean`
- `onto_<family>_relative_worst_delta`
- `onto_<family>_error_ratio`

A lógica do motor não contém nomes hardcoded do dataset. O conhecimento específico fica na OWL.

### 3. TBox Breast Cancer enriquecida semanticamente

Arquivos:

- `data/benchmark_ontologies/breast_cancer.owl`
- `data/benchmark_ontologies_v7/breast_cancer.owl`
- `core/benchmark_ontologies.py`

As 30 propriedades do benchmark Breast Cancer passam a declarar:

- família de medição: `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave_points`, `symmetry`, `fractal_dimension`;
- papel estatístico: `mean`, `error`, `worst`.

A TBox continua sem instâncias do dataset e sem consultar rótulos ou partições.

### 4. Gate de utilidade agora é realmente um gate de MLP

Arquivo: `core/semantic_utility_gate.py`

O proxy `LogisticRegression` foi removido da decisão de aceitação. O gate agora usa `MLPClassifier`.

Na aplicação principal, os hiperparâmetros/família do MLP Original são clonados para a comparação OOF. Wrappers de calibração são desembrulhados apenas para recuperar o estimador MLP de base.

Assim, a pergunta científica passa a ser:

`MLP Original` **vs** `mesmo MLP + features OWL`

nas mesmas folds de desenvolvimento.

### 5. Refit semântico dentro de cada fold OOF

Arquivos:

- `core/ontology_processor.py`
- `core/semantic_utility_gate.py`
- `core/trepan.py`
- `gui/biuri_app_complete.py`

Foi adicionado `OntologyProcessor.transform_fold_pair()`.

Durante o gate OOF:

1. o schema/proveniência OWL fica congelado pelo treino externo;
2. cada fold recalcula apenas parâmetros numéricos aprendidos, como média/desvio das agregações;
3. o validation fold é transformado com parâmetros do seu training fold;
4. o teste externo continua bloqueado.

A GUI passa o `X_train_raw` correspondente ao treino externo para permitir esta validação sem leakage.

Para pipelines CLI totalmente numéricos, existe fallback automático que reconstrói o frame original a partir das colunas base da matriz enriquecida quando não há semântica categórica.

### 6. Diagnóstico do gate na interface

Arquivo: `gui/ontology_status_presenter.py`

Além do código de decisão, a interface pode agora mostrar:

- família do estimador do gate (`MLPClassifier`);
- origem do modelo do gate;
- se houve refit fold-local das estatísticas OWL;
- quantidade de features OWL candidatas;
- melhor subconjunto OOF;
- Accuracy / Balanced Accuracy / Macro-F1 do MLP base;
- as mesmas métricas para MLP + OWL;
- utilidade base, utilidade OWL e ganho líquido.

Isto torna `REJECT_NO_INFORMATIONAL_GAIN`, `REJECT_GAIN_BELOW_COMPLEXITY_COST` ou `ACCEPT_*` auditáveis na própria aplicação.

## Regras científicas preservadas

Não foram relaxados artificialmente:

- `min_predictive_gain = 0.002`
- gate de matching/quality;
- verificação de consistência/reasoner;
- proibição de zero-fill;
- bloqueio do teste externo para seleção;
- separação entre TBox e ABox;
- fallback para MLP Original quando o ganho semântico não é comprovado.

## Novos testes

Arquivo: `tests/test_semantic_enrichment_v92_upgrade.py`

Cobertura adicionada para:

1. agregação hierárquica z-normalizada;
2. geração de features relacionais por família/papel OWL;
3. refit fold-local sem mutar o processor externo;
4. gate baseado em MLP com refit fold-local;
5. presença dos 30 pares `measurementFamily/statisticRole` na TBox Breast Cancer.

## Validação executada neste ambiente

- `tests/test_semantic_enrichment_v92_upgrade.py`: **4 passed**;
- conjunto dirigido de regressão semântica/produção: **39 passed**;
- execução ampliada `pytest -m 'not slow'`: avançou sem falhas observadas até ao limite de tempo do ambiente; a execução foi interrompida por timeout, não por erro de teste;
- `py_compile` concluído para todos os arquivos Python alterados;
- RDF/XML das duas TBoxes Breast Cancer validado com `xml.etree.ElementTree`.

### Limitação do ambiente de validação

`owlready2==0.47` não estava instalado no ambiente e a rede estava indisponível para o instalar. Por isso, os testes que dependem diretamente de Owlready2 continuam cobertos pelo requirements do projeto, mas não puderam ser executados aqui. Os testes independentes de Owlready2 e a validação XML foram executados normalmente.

## Resultado esperado ao executar novamente o Breast Cancer

O novo pipeline deve produzir aproximadamente:

- 3 agregações hierárquicas normalizadas (`MeanMorphology`, `ErrorMorphology`, `WorstMorphology`);
- até 30 features relacionais derivadas das 10 famílias semânticas;
- filtragem de constantes/duplicadas;
- seleção por estabilidade/MI;
- comparação OOF com MLP, não regressão logística;
- aceitação apenas se o melhor subconjunto ultrapassar o gate de utilidade real.

Se o resultado continuar como `REJECT_NO_INFORMATIONAL_GAIN`, isso passa a significar uma evidência mais forte: mesmo após relações OWL úteis, normalização correta, MLP pareado e refit fold-local, não houve ganho OOF suficiente nesse dataset/configuração.
