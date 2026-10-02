# Enriquecimento com Ontologia, Fidelidade e Precisão

Este documento descreve, com base no código atual do projeto **Trepa Reloaded / Biuri**, como a ontologia OWL é usada para enriquecer dados e modelos, e como são calculadas as métricas de **fidelidade** e **precisão**.

---

## Visão geral do pipeline

A ontologia entra no sistema em **dois níveis complementares**:

| Nível | Onde acontece | O que faz |
|-------|---------------|-----------|
| **1. Enriquecimento de dados (GUI)** | `gui/biuri_app_complete.py` | Adiciona colunas derivadas ao DataFrame ARFF antes do treino |
| **2. Extração Trepan-Reloaded (core)** | `core/trepan_reloaded_extractor.py` | Usa a ontologia para mapear features, gerar dados sintéticos, validar coerência semântica e calcular fidelidade durante a extração da árvore |

As métricas de **precisão** e **fidelidade** na comparação entre modelos são calculadas centralmente em `core/metrics_comparator.py`, acionadas pela GUI em **Comparar Modelos**.

```
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│ Carregar OWL    │────▶│ Enriquecer DataFrame     │────▶│ Treinar MLP + Árvores│
│ (owlready2)     │     │ (_augment_dataframe_...)  │     │ (Trepan / Reloaded) │
└─────────────────┘     └──────────────────────────┘     └──────────┬──────────┘
                                                                        │
                                                                        ▼
                                                             ┌─────────────────────┐
                                                             │ compare_all_models  │
                                                             │ (precisão + fidel.) │
                                                             └─────────────────────┘
```

---

## Mapa técnico — geração das três árvores

| Árvore | Fonte de dados | Oráculo / alvo | Lógica principal |
|--------|----------------|----------------|------------------|
| **C4.5-Nativo** | `original_data` (só colunas ARFF) | Rótulos reais `y` | Gain Ratio + missing fracionário + poda pessimista, sem MLP |
| **Trepan-Original** | `original_data` | **MLP original** | Dados sintéticos + destilação: a árvore imita o MLP, não treina directamente só em `y` |
| **Trepan-Reloaded** | `augmented_data` (original + colunas `onto_*`) | **MLP-Reloaded** (treino) | Sintéticos ontologia-aware; **priorização de splits** em `onto_*` (`ONTO_FEATURE_WEIGHT`, pesos de amostra, `apply_ontology_split_bias`) |

### Fidelidade na comparação de métricas

- **C4.5**, **Trepan-Original**, **Trepan-Reloaded** e **MLP** (baseline): fidelidade medida sempre face ao **MLP original** no `X_test` original (mesmas instâncias).
- **Predição** da árvore Reloaded: `X_test` enriquecido + `apply_ontology_split_bias` (mesma escala do treino).
- **Accuracy** do Reloaded: calculada no teste enriquecido vs rótulos reais.

Implementação: `gui/biuri_app_complete.py` (`train_model`, `compare_metrics`), `core/trepan_reloaded_extractor.py`, `core/metrics_comparator.py`.

---

## Parte 1 — Enriquecimento com ontologia

### 1.1 Carregamento da ontologia

**Ficheiros:** `gui/biuri_app_complete.py`, `core/trepan.py`

1. O utilizador carrega um ficheiro `.owl` (opcionalmente junto com o ARFF).
2. A ontologia é parseada com **owlready2** (`get_ontology(path).load()`).
3. O objeto fica em `self.loaded_ontology` (GUI) e em `self.trepan.ontology` (pipeline de treino).
4. Se existir ontologia, o extrator passa a ser `TrepanReloadedExtractor(ontology)` em vez de `TREPANExtractor`.

Ontologia padrão (se nenhuma for fornecida): `data/sample_ontology.owl`.

---

### 1.2 Enriquecimento do DataFrame (camada GUI)

**Método principal:** `_augment_dataframe_with_ontology(df)`  
**Ficheiro:** `gui/biuri_app_complete.py`  
**Momento:** ao carregar ficheiro ARFF, se `self.loaded_ontology is not None`.

#### Passo 1 — Obter entidades candidatas da ontologia (agnóstico ao dataset)

```python
matching_pairs, stats = matcher._get_ontology_matching_entities(ontology)
# matching_pairs: [(entity, 'class'|'datatype_property'|'object_property'), ...]
ontology_entities = TrepanReloadedExtractor._unwrap_matching_entities(matching_pairs)
```

O pool inclui **owl:Class**, **owl:DatatypeProperty** e **owl:ObjectProperty**, com exclusão de classes de alvo/diagnóstico genéricas (`Benign`, `Malignant`, `DiagnosisClass`, `TargetClass`, etc.).

Se não houver entidades elegíveis, o enriquecimento é ignorado.

Carregamento de ficheiros: `TrepanReloadedExtractor.load_ontology_file()` (RDF/XML, Turtle, fallback `*_TBox.owl`).

#### Passo 2 — Para cada feature (coluna de entrada, exceto o alvo)

O alvo é sempre a **última coluna** do DataFrame. O matching é **por nome/label**, nunca por índice ou ordem OWL.

#### Passo 3 — Matching de nomes (GUI)

**Comparação** (`_find_best_ontology_match` → `TrepanReloadedExtractor._find_matching_concept`):

- Nome interno OWL, `rdfs:label`, comentários RDFS
- Normalização: CamelCase→snake, remoção de pontuação, underscores, affixes (`feat_`, `_score`, …)
- `difflib.SequenceMatcher` e overlap de tokens
- **Threshold mínimo:** `self.ontology_match_threshold = 0.4` (GUI)
- Metadados: `matched_entity`, `entity_type` (`class` | `datatype_property` | `object_property`)
- Valores categóricos de features: matching só em **classes** (`for_class_values=True`)

#### Passo 4 — Colunas derivadas por feature

Para cada feature `X`, são criadas até **8 colunas**:

| Coluna | Conteúdo |
|--------|----------|
| `onto_{X}_concept` | Nome do conceito OWL mapeado, ou `"UNMAPPED"` |
| `onto_{X}_concept_score` | Score de matching (0.0–1.0) |
| `onto_{X}_depth` | Profundidade na hierarquia (`len(ancestors())`) ou `-1` |
| `onto_{X}_parents` | Nomes dos pais (`is_a`), separados por vírgula |
| `onto_{X}_num_children` | Número de subclasses directas |
| `onto_{X}_num_properties` | Número de propriedades da classe |

**Colunas extra para features categóricas** (`dtype == object`):

| Coluna | Conteúdo |
|--------|----------|
| `onto_{X}_value_concept` | Cada valor da coluna mapeado para um conceito OWL |
| `onto_{X}_value_score` | Score de matching por valor |

O mapeamento de valores usa `_map_series_values_to_ontology`, que aplica o mesmo algoritmo de matching a cada valor distinto da série.

#### Passo 5 — Metadados e treino

- O DataFrame enriquecido alimenta `self.current_data` (features + alvo).
- Metadados ARFF são actualizados com `derived_attributes` e `augmentation`.
- O MLP trata as novas colunas como features normais:
  - Numéricas → float directo
  - Categóricas → `LabelEncoder` por coluna (`core/mlp_trainer.py`)

**Importante:** o enriquecimento na GUI **expande o conjunto de features** que o MLP e as árvores veem. Não substitui os valores originais.

---

### 1.3 Enriquecimento semântico na extração Trepan-Reloaded

**Método principal:** `extract_tree_with_ontology(...)`  
**Ficheiro:** `core/trepan_reloaded_extractor.py`

Este fluxo usa a ontologia **durante a extração da árvore explicativa**, independentemente das colunas `onto_*` da GUI.

#### Fase 1 — Extração de conhecimento de domínio

**Método:** `_extract_domain_knowledge(feature_names, class_names)`

1. **Mapeamento de features** (`_map_features_to_concepts`):
   - Para cada feature, `_find_matching_concept` procura em Class, DatatypeProperty e ObjectProperty.
   - Algoritmo de scoring (0.0–1.0), por ordem de prioridade:
     1. Match exacto normalizado → score **1.0**
     2. Match em `rdfs:label` / `skos:prefLabel` → **0.95**
     3. Substring (contém/contido) → **0.6 + overlap_ratio × 0.2**
     4. Similaridade `difflib` > 0.7 → **similarity × 0.8**
     5. Similaridade em labels → **label_similarity × 0.85**
     6. Sobreposição de palavras → **word_overlap × 0.7**
   - Threshold mínimo para candidatos: **0.5**
   - Normalização remove prefixos como `has_`, `is_`, `contains_`.

2. **Mapeamento de classes alvo** (`_map_classes_to_concepts`):
   - Mesmo algoritmo para os nomes das classes de classificação.

3. **Relacionamentos semânticos** (`_extract_semantic_relationships`):
   - `subClassOf` (hierarquia)
   - `equivalentClass`
   - `disjointWith`
   - Propriedades OWL: `domain`, `range`, propriedades inversas
   - Restrições de datatype (`xsd:integer`, `xsd:float`, etc.)

4. **Fallback sem ontologia:** `_create_basic_domain_knowledge` infere tipos por palavras-chave no nome (temporal, dimensional, categorical, etc.).

Resultado guardado em:
- `self.feature_semantics` — por índice de feature
- `self.class_semantics` — por índice de classe
- `self.domain_knowledge` — agregado com estatísticas de mapeamento

#### Fase 2 — Dados sintéticos com consciência ontológica

**Método:** `_generate_ontology_aware_synthetic_data`

Combina 4 estratégias de amostragem (tamanho default: 2000):

| Estratégia | Proporção | Descrição |
|------------|-----------|-----------|
| Densidade | 40% | Amostras próximas de regiões densas dos dados reais |
| Incerteza | 25% | Regiões onde o MLP tem baixa confiança |
| Domínio | 25% | Valores gerados com base em tipos de conceito, datatypes e features relacionadas via ontologia |
| Aleatório | 10% | Uniforme entre min/max de cada feature |

A parte **domain-aware** (`_sample_by_domain_knowledge`):
- Agrupa features relacionadas por `domain` de propriedades OWL ou hierarquia
- Gera valores coerentes para grupos (`_generate_related_feature_values`)
- Respeita tipos inferidos (temporal, dimensional, boolean, etc.)

#### Fase 3 — Predição do MLP sobre sintéticos

```python
y_synthetic = mlp_model.predict(X_synthetic)
```

#### Fase 4 — Treino da árvore

**Método:** `_train_ontology_constrained_tree`

- `DecisionTreeClassifier` com parâmetros adaptados ao tamanho dos sintéticos:
  - `max_depth = min(7, log2(n) + 2)`
  - `min_samples_split = max(8, n // 80)`
  - `min_samples_leaf = max(4, n // 160)`
  - `criterion='gini'`
- Treino sobre `(X_synthetic, y_synthetic)` — **não** sobre dados reais directamente.
- Após treino: `_validate_semantic_coherence` verifica thresholds da árvore contra tipos semânticos e restrições de datatype.

#### Fase 5 — Fidelidade durante extração

Ver **Parte 2 — Fidelidade na extração Trepan-Reloaded**.

#### Fases 6–7 — Regras e relatório

- Regras exportadas com `sklearn.tree.export_text`
- Relatório textual via `_generate_ontology_aware_report`

---

## Parte 2 — Cálculo de fidelidade

### 2.1 O que é fidelidade neste projeto

**Fidelidade** mede o grau em que a **árvore explicativa reproduz as predições do MLP** (modelo opaco), **não** a exactidão face aos rótulos verdadeiros.

```
Fidelidade = concordância( predição_árvore , predição_MLP )
```

Isto segue a definição clássica do algoritmo TREPAN: a árvore deve ser um substituto interpretável **fiel** ao comportamento do MLP.

---

### 2.2 Fidelidade na extração (Trepan-Reloaded)

**Método:** `_calculate_ontology_aware_fidelity`  
**Ficheiro:** `core/trepan_reloaded_extractor.py`

```python
fidelity = accuracy_score(y_mlp_pred, y_tree_pred)
```

Onde:
- `y_tree_pred = tree.predict(X_data)`
- `y_mlp_pred = mlp_model.predict(X_data)`

#### Métricas produzidas

| Métrica | Conjunto de dados | Fórmula |
|---------|-------------------|---------|
| `overall_real` | Dados reais (`X_encoded`) | `accuracy(y_mlp, y_tree)` |
| `overall_synthetic` | Dados sintéticos | `accuracy(y_mlp, y_tree)` |
| `semantic_groups` | Dados reais, por grupo | Ver abaixo |

#### Fidelidade por grupo semântico

**Método:** `_calculate_semantic_group_fidelity`

1. Agrupa índices de features por `semantic_group` (medical, financial, technical, general, …)
2. Para cada grupo:
   - Extrai sub-matriz `X_group = X_real[:, feature_indices]`
   - Treina árvore temporária (`max_depth=3`) só com essas features
   - Compara predições dessa árvore parcial com predições do MLP (que usa **todas** as features)

**Nota:** esta fidelidade por grupo é **aproximada** — a árvore parcial não tem acesso às restantes features, mas o MLP sim.

#### Interpretação no relatório

Média `(overall_synthetic + overall_real) / 2`:

| Média | Classificação |
|-------|---------------|
| > 0.9 | Excelente |
| > 0.8 | Muito boa |
| > 0.7 | Boa |
| ≤ 0.7 | Moderada |

---

### 2.3 Fidelidade na comparação de modelos

**Método:** `_calculate_fidelity_metrics`  
**Ficheiro:** `core/metrics_comparator.py`  
**Acionado por:** `compare_all_models()` na GUI

Compara **MLP vs cada árvore** no conjunto de **teste** (30% hold-out, `random_state=42`, estratificado quando possível).

#### Modelos comparados

- Trepan-Original
- Trepan-Reloaded (se ontologia activa)
- C4.5-Nativo

#### Métricas de fidelidade por modelo

| Métrica | Cálculo | Interpretação |
|---------|---------|---------------|
| `overall_fidelity` | `accuracy_score(y_mlp_pred, y_tree_pred)` | % de amostras em que árvore e MLP concordam |
| `precision_fidelity` | `precision_score(y_mlp_pred, y_tree_pred, average='weighted')` | Precisão **weighted** tratando predições do MLP como "verdade" |
| `recall_fidelity` | `recall_score(..., average='weighted')` | Recall weighted MLP vs árvore |
| `f1_fidelity` | `f1_score(..., average='weighted')` | F1 weighted |
| `agreement_rate` | `mean(y_mlp_pred == y_tree_pred)` | Equivalente à fidelidade global |

**Distinção importante:**

- Em `overall_fidelity`, ambos os vectores são **rótulos de classe** (predições).
- Em `precision_fidelity`, `y_mlp_pred` actua como **rótulo de referência** (`y_true`) e `y_tree_pred` como **predição** — mede a qualidade da concordância por classe, ponderada pelo suporte.

#### Intervalos de confiança (bootstrap)

**Método:** `_bootstrap_confidence_interval`

- **1000 reamostragens** com reposição (`n_bootstrap=1000`)
- `random_state=42` para reprodutibilidade
- IC de **95%** via percentis 2.5 e 97.5
- Aplicado a `overall_fidelity` e `precision_fidelity`

#### Análises granulares adicionais

| Análise | Método | Descrição |
|---------|--------|-----------|
| Por região de confiança | `_analyze_fidelity_by_confidence_region` | Fidelidade quando MLP tem alta (>0.7) vs baixa confiança |
| Por tipo de erro | `_analyze_fidelity_by_error_type` | Discordâncias MLP-árvore cruzadas com erro face ao ground truth |
| Por região de features | `_analyze_fidelity_by_feature_region` | Fidelidade em subespaços de features |
| Discrepâncias | `_analyze_discrepancies` | Casos onde MLP e árvore divergem, com importância de features |

---

### 2.4 Fidelidade no Trepan-Original (sem ontologia)

**Ficheiro:** `core/trepan_extractor.py` — `_calculate_fidelity_metrics`

Calcula também:
- Fidelidade **por classe**
- Fidelidade em regiões de **alta confiança do MLP** (`max_proba > 0.8`)

---

## Parte 3 — Cálculo de precisão

### 3.1 O que é precisão neste projeto

**Precisão** (no sentido de métricas de classificação) mede a **exactidão das predições face aos rótulos verdadeiros** (`y_test`), não face ao MLP.

```
Precisão (weighted) = precision_score(y_test, y_pred, average='weighted')
```

Isto é distinto de `precision_fidelity`, que compara árvore **com o MLP**.

---

### 3.2 Precisão na comparação de modelos

**Método:** `_calculate_precision_metrics`  
**Ficheiro:** `core/metrics_comparator.py`

Para cada modelo (oráculo MLP, Trepan-Original, Trepan-Reloaded, C4.5-Nativo):

1. Obtém predições no conjunto de teste (com cache em `_cached_predict`)
2. Calcula métricas via `_calculate_metrics_with_ci(y_test, y_pred)`

#### Métricas calculadas

| Métrica | Função sklearn | Média bootstrap |
|---------|----------------|-----------------|
| `accuracy` | `accuracy_score` | Sim (IC 95%) |
| `precision` | `precision_score(average='weighted')` | Sim |
| `recall` | `recall_score(average='weighted')` | Sim |
| `f1` | `f1_score(average='weighted')` | Sim |

Também disponível internamente: `precision_macro`, `recall_macro`, `f1_macro`.

#### Bootstrap para precisão

Mesmo procedimento da fidelidade:
- 1000 amostras bootstrap
- Percentis 2.5 / 97.5 para IC 95%
- `zero_division=0` para evitar erros em classes sem predições

#### Métricas adicionais

**Método:** `_calculate_additional_metrics`

Inclui (quando aplicável):
- Matriz de confusão
- Cohen's Kappa
- AUC-ROC (multi-classe)
- Log loss, Brier score
- Detecção de desbalanceamento de classes

---

### 3.3 Precisão do MLP no treino (GUI)

**Ficheiro:** `gui/biuri_app_complete.py` — `train_model()`

Após treino, calcula acurácia simples no conjunto completo:

```python
accuracy = accuracy_score(y_true_labels, y_pred_labels)
```

Sem bootstrap neste passo — é uma métrica informativa imediata, não a comparação formal.

---

## Parte 4 — Tabela comparativa: Precisão vs Fidelidade

| Aspecto | Precisão (`precision_metrics`) | Fidelidade (`fidelity_metrics`) |
|---------|-------------------------------|--------------------------------|
| **Referência** | Rótulos verdadeiros (`y_test`) | Predições do MLP (`y_mlp_pred`) |
| **Pergunta respondida** | "O modelo acerta a classe real?" | "A árvore imita o MLP?" |
| **Aplica-se ao MLP?** | Sim | Não (MLP é a referência) |
| **Aplica-se às árvores?** | Sim | Sim |
| **Média usada** | `weighted` | `weighted` (em precision_fidelity) |
| **IC bootstrap** | Sim | Sim |
| **Usado na extração** | Não directamente | Sim (Trepan-Reloaded) |

---

## Parte 5 — Fluxo completo na aplicação

```
1. Carregar OWL (+ ARFF)
      ↓
2. _augment_dataframe_with_ontology  →  colunas onto_* adicionadas
      ↓
3. train_model()
      ├── train_mlp(X, y)  →  MLP treinado
      ├── TREPANExtractor.extract_tree  →  trepan_original_tree
      ├── TrepanReloadedExtractor.extract_tree_with_ontology  →  trepan_reloaded_tree
      │     (usa ontologia para sintéticos, validação semântica, fidelidade interna)
      └── C45Tree.train_c45_tree  →  c45_tree
      ↓
4. compare_models()  [requer MLP + árvores treinadas]
      ├── train_test_split(70/30)
      ├── _calculate_precision_metrics  →  exactidão vs y_test
      └── _calculate_fidelity_metrics   →  concordância vs y_mlp
      ↓
5. generate_comparison_report()  →  relatório na tab Métricas
```

---

## Parte 6 — Parâmetros e thresholds relevantes

| Parâmetro | Valor | Localização |
|-----------|-------|-------------|
| Threshold matching GUI | 0.65 | `biuri_app_complete.py` → `ontology_match_threshold` |
| Threshold matching extractor | 0.5 (candidatos) | `trepan_reloaded_extractor.py` → `_find_matching_concept` |
| Amostras sintéticas | 2000 | `extract_tree_with_ontology(sample_size=2000)` |
| Bootstrap iterations | 1000 | `MetricsComparator(n_bootstrap=1000)` |
| Test split | 30% | `train_test_split(test_size=0.3)` |
| Confiança alta MLP (fidelidade granular) | > 0.7 | `_analyze_fidelity_by_confidence_region` |
| Confiança alta MLP (Trepan original) | > 0.8 | `trepan_extractor.py` |

---

## Parte 7 — Ficheiros de referência

| Ficheiro | Responsabilidade |
|----------|------------------|
| `gui/biuri_app_complete.py` | Carregamento OWL, enriquecimento DataFrame, orquestração treino/comparação |
| `core/trepan.py` | Selecção do extrator, ligação ontologia ↔ pipeline |
| `core/trepan_reloaded_extractor.py` | Conhecimento de domínio, sintéticos, árvore, fidelidade na extração |
| `core/trepan_extractor.py` | Trepan original, fidelidade básica |
| `core/metrics_comparator.py` | Precisão e fidelidade formais, bootstrap, relatórios |
| `core/mlp_trainer.py` | Encoding de features (incluindo colunas `onto_*`) |
| `core/c45_j48_tree.py` | Baseline C4.5-Nativo e comparação local |

---

## Resumo executivo

1. **Enriquecimento na GUI** adiciona colunas `onto_*` ao dataset via matching fuzzy de nomes de features/valores com classes OWL (threshold 0.65).

2. **Enriquecimento no Trepan-Reloaded** usa a ontologia para mapear semanticamente features e classes, gerar dados sintéticos informados pelo domínio, validar coerência dos splits da árvore e reportar fidelidade real vs sintética.

3. **Fidelidade** = concordância entre predições da árvore e do MLP (`accuracy_score`); na comparação formal inclui também precision/recall/F1 weighted com o MLP como referência, mais intervalos de confiança bootstrap.

4. **Precisão** = qualidade de classificação face aos rótulos verdadeiros do test set (`precision_score` weighted, com bootstrap); mede desempenho predictivo, não imitação do MLP.
