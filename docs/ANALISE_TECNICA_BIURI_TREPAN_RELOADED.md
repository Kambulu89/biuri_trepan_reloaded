# Análise técnica — BIURI / Trepan Reloaded

Documento para revisão externa da implementação actual (Junho 2026). Foco: oráculo MLP, indução da árvore, gestão de features `onto_*`, métricas de Fidelidade e Precisão, e **pontos de extensão** para `MLP_Onto` e `ONTO_FEATURE_BIAS_WEIGHT`.

---

## 1. Arquitectura em alto nível

```mermaid
flowchart TB
    subgraph GUI["gui/biuri_app_complete.py"]
        ARFF[ARFF original]
        OWL[OWL opcional]
        ARFF --> X_orig
        OWL --> augment["_rebuild_augmented_dataset()"]
        X_orig --> MLP_train["_get_mlp_training_xy() → X_mlp"]
        augment --> X_aug[augmented_data]
        MLP_train --> MLP["MLPClassifier (n_features_in_ = |ARFF|)"]
        X_aug --> encode_aug["_encode_augmented_for_tree()"]
        encode_aug --> X_enc_aug
    end

    subgraph Extractor["core/trepan_reloaded_extractor.py"]
        X_enc_aug --> tree_train["Árvore: X_augmented + y_synthetic"]
        MLP --> oracle["_mlp_predict → só colunas ARFF"]
        oracle --> y_syn[y_synthetic = MLP(X_base)]
        y_syn --> tree_train
        tree_train --> DT["DecisionTreeClassifier + apply_ontology_split_bias"]
    end

    subgraph Metrics["core/metrics_comparator.py"]
        DT --> compare["compare_all_models()"]
        MLP --> compare
    end
```

| Componente | Ficheiro | Papel |
|----------|----------|--------|
| Orquestrador | `core/trepan.py` | `train_mlp`, delega extração ao extrator |
| Extrator Reloaded | `core/trepan_reloaded_extractor.py` | Pipeline ontologia + árvore + oráculo |
| Engenharia OWL | `core/ontology_processor.py` | Colunas `onto_*` (hierárquicas, relacionais, restrições) |
| Treino MLP | `core/mlp_trainer.py` | `MLPClassifier`, encoders por coluna |
| Baseline C4.5 | `core/c45_j48_tree.py` | C4.5 nativo: Gain Ratio + poda pessimista |
| GUI | `gui/biuri_app_complete.py` | Pipeline duplo ARFF / enriquecido |
| Métricas | `core/metrics_comparator.py` | Precisão vs rótulos; fidelidade vs MLP |

---

## 2. Fluxo do oráculo (MLP)

### 2.1 Resposta directa: `X_original` ou `X_enriched`?

**O MLP é treinado e consultado apenas no espaço ARFF original (`X_original`).**  
A árvore Trepan Reloaded treina-se na matriz enriquecida (`X_enriched` / `X_augmented`), mas os rótulos sintéticos vêm do MLP aplicado às **mesmas features base** que usou no treino.

Não existe hoje um segundo modelo `MLP_Onto` treinado em `X_enriched`.

### 2.2 Treino do MLP (GUI)

Com ontologia activa, a GUI activa um **pipeline duplo**:

1. `original_data` — cópia imutável do ARFF.
2. `augmented_data` — cópia com colunas `onto_*` (GUI +/ou `OntologyProcessor`).

Treino explícito só no original:

```1401:1410:gui/biuri_app_complete.py
    def _get_mlp_training_xy(self):
        """Oráculo MLP: apenas features ARFF originais."""
        X, y = self._get_original_xy()
        orig_names = self._get_original_feature_names()
        X_mlp = self._slice_matrix_to_original_features(X, orig_names)
        print(
            f"[DEBUG] Treinando MLP com {X_mlp.shape[1]} features "
            f"(originais ARFF: {len(orig_names)})."
        )
        return X_mlp, y
```

Fluxo de treino (`train_model`):

- `self.trepan.train_mlp(X_mlp, y)` → `MLPTrainer.train()` → `MLPClassifier.fit(X_encoded, y_encoded)`.
- `self.mlp_model_reloaded = self.mlp_model` (mesma instância, não um modelo separado enriquecido).
- Árvore Reloaded: `extract_tree_with_ontology(self.mlp_model, X_encoded_aug, ...)` com `original_feature_names` das features ARFF.

Codificação da matriz enriquecida **sem** retreinar o MLP:

```1417:1434:gui/biuri_app_complete.py
    def _encode_augmented_for_tree(self):
        """Codifica cópia enriquecida para a árvore Trepan (sem treinar MLP nela)."""
        ...
        X_enc = self.trepan.mlp_trainer.encode_columns_by_name(
            rows, feature_names_aug, orig_names
        )
```

Em `mlp_trainer.py`, colunas em `base_column_names` usam `feature_encoders` do treino; colunas `onto_*` recebem `LabelEncoder` auxiliar por nome (`_extra_column_encoders`).

### 2.3 Consulta do MLP durante a extração

Fases em `extract_tree_with_ontology`:

| Fase | Dados | Oráculo |
|------|--------|---------|
| 1 | `X_base` (ARFF) | Mapeamento OWL |
| 1b | `X_augmented` | Engenharia `onto_*` (se ainda não injectada pela GUI) |
| 2 | Sintéticos em `X_base` → expand → `X_synthetic` | Geração no espaço ARFF |
| 3 | `X_synthetic` (matriz completa, nomes `tree_feature_names`) | `_mlp_predict` |
| 4 | `X_synthetic`, `y_synthetic`, `X_augmented` | Treino da árvore |

Comentário no gerador de sintéticos:

```1593:1595:core/trepan_reloaded_extractor.py
    def _generate_ontology_aware_synthetic_data(self, mlp_model, X_encoded, sample_size, feature_names):
        # Sintéticos e oráculo MLP operam sempre no espaço ARFF original (n_features_in_)
        X_encoded, feature_names, _ = self._base_matrix_and_names(X_encoded, feature_names)
```

Predição do oráculo:

```918:921:core/trepan_reloaded_extractor.py
    def _mlp_predict(self, mlp_model, X, matrix_feature_names=None):
        return mlp_model.predict(
            self._features_for_mlp_oracle(X, matrix_feature_names, mlp_model)
        )
```

`_matrix_for_oracle` selecciona colunas por `_oracle_feature_names` (alinhado a `mlp_model.n_features_in_`), com fallback para índices não-`onto_*` e `_enforce_oracle_width` (corte posicional se necessário).

### 2.4 Resolução de esquema de features

```778:826:core/trepan_reloaded_extractor.py
    def _resolve_feature_schemas(
        self, X, feature_names, mlp_model, original_feature_names=None
    ):
        """
        Separa nomes da matriz, features ARFF originais e features do oráculo MLP.
        O oráculo usa exactamente n_features_in_ do MLP, resolvidos por nome.
        """
        ...
        elif n_mlp == len(base_names):
            # MLP treinado só em ARFF; matriz pode ter colunas extra (ruído/onto_*)
            oracle_names = list(base_names)
```

**Conclusão para `MLP_Onto`:** ponto natural de integração:

1. **Treino:** novo ramo em `biuri_app_complete.train_model` — `MLPTrainer` dedicado (ou segundo `fit`) sobre `X_encoded_aug` com `n_features_in_ == len(feature_names_aug)`.
2. **Consulta:** em `_resolve_feature_schemas`, se `n_mlp == len(matrix_names)`, usar `oracle_names = matrix_names` e `_mlp_predict` sem stripping (ou flag `use_ontology_oracle=True`).
3. **Fidelidade:** `MetricsComparator` hoje compara Reloaded ao **MLP original** (`fidelity_reference: 'mlp_original'`); com `MLP_Onto`, definir referência explícita (original vs enriquecido) para evitar métricas inconsistentes.

---

## 3. Lógica de split (ganho de informação / critério)

### 3.1 Onde está o critério de split?

**Não há implementação própria de ganho de informação (ID3/C4.5) no Trepan Reloaded.**  
O split é delegado ao **scikit-learn** `DecisionTreeClassifier`, que internamente optimiza impureza (**Gini** ou **entropia**) por feature e threshold.

Trecho exacto do treino principal (grid de profundidade orientado à fidelidade):

```2830:2838:core/trepan_reloaded_extractor.py
        for max_depth in depth_grid:
            candidate = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                criterion='gini',
            )
            candidate.fit(X_aug, y_aug, sample_weight=w_aug)
```

Entrada já com viés ontológico aplicado:

```2804:2813:core/trepan_reloaded_extractor.py
    def _fit_tree_with_semantic_tuning(self, mlp_model, X_syn, y_syn, sample_weights,
                                       X_real, y_real, feature_names, class_names,
                                       competitor_trees=None):
        ...
        X_aug = self.apply_ontology_split_bias(X_aug)
```

Fallback sem tuning semântico completo:

```2898:2906:core/trepan_reloaded_extractor.py
            X_aug = self.apply_ontology_split_bias(X_aug)
            tree = DecisionTreeClassifier(
                max_depth=min(7, int(np.log2(len(X_aug)) + 2)),
                ...
                criterion='gini',
            )
            tree.fit(X_aug, y_aug, sample_weight=w_aug)
```

Em buscas expandidas (`ensure_ontology_dominance` / `_fit_tree_dominance_grid`), o código testa também `criterion='entropy'` — ainda via sklearn, não IG explícito.

### 3.2 Viés de split ontológico (equivalente funcional a `ONTO_FEATURE_BIAS_WEIGHT`)

Em vez de alterar a fórmula de IG, o sistema **escala colunas** antes de `fit`/`predict`, o que altera efectivamente a impureza reduzida por feature:

```2052:2089:core/trepan_reloaded_extractor.py
    def _build_feature_gain_weights(self, feature_names):
        """Peso elevado no ganho de informação para colunas onto_* e features mapeadas na OWL."""
        weights = np.ones(len(feature_names), dtype=float)
        for idx, name in enumerate(feature_names):
            info = self.feature_semantics.get(idx, {})
            if self._is_ontology_derived_feature(name) or info.get('ontology_derived'):
                weights[idx] = self.ONTO_FEATURE_WEIGHT
            elif info.get('ontology_mapped') and info.get('mapping_score', 0) >= self.PARTIAL_MATCH_THRESHOLD:
                weights[idx] = self.MAPPED_FEATURE_GAIN_BOOST
        return weights

    def _build_ontology_split_multipliers(self, feature_names):
        """
        Escala colunas no treino/inferência da árvore Reloaded para priorizar splits
        semânticos (onto_*) quando o ganho de informação é semelhante ao das originais.
        """
        multipliers = self._build_feature_gain_weights(feature_names)
        ...

    def apply_ontology_split_bias(self, X):
        """Aplica a mesma escala usada no treino da árvore Reloaded (obrigatório em predict)."""
        ...
        return X_arr * m
```

Constantes actuais (classe `TrepanReloadedExtractor`):

| Constante | Valor | Efeito |
|-----------|-------|--------|
| `ONTO_FEATURE_WEIGHT` | `2.5` | Multiplicador em colunas `onto_*` / `ontology_derived` |
| `MAPPED_FEATURE_GAIN_BOOST` | `1.2` | Features ARFF mapeadas na OWL com score ≥ 0.4 |
| `split_priority` | dinâmico | Reforço extra em `onto_*` via `feature_semantics` |

Reforço adicional por **pesos de amostra** (`_boost_sample_weights_for_mapped_features`), não por feature no critério sklearn.

### 3.3 C4.5-Nativo (baseline)

O baseline usa entropia para calcular ganho, selecciona atributos por **Gain
Ratio**, trata categorias por ramos multivalor, distribui missing values por
pesos fracionários e aplica poda pessimista. Não usa CART, Weka ou Java.

Treina só em `X_train` original (GUI passa `ontology_mode=False`) e recebe os
tipos declarados no cabeçalho ARFF.

### 3.4 Onde ligar `ONTO_FEATURE_BIAS_WEIGHT`

| Local | Acção sugerida |
|-------|----------------|
| Linha 31 | Renomear/parametrizar `ONTO_FEATURE_WEIGHT` → configurável (env, GUI, construtor) |
| `_build_feature_gain_weights` | Usar `ONTO_FEATURE_BIAS_WEIGHT` como factor único exportável |
| `apply_ontology_split_bias` | Garantir mesma constante em treino, predict, `reload_test_transform` e `ensure_ontology_dominance` |
| Alternativa avançada | `DecisionTreeClassifier` custom + `criterion` com pesos por feature (não existe hoje) |

**Importante:** qualquer mudança exige manter `apply_ontology_split_bias` simétrico em treino e inferência; a GUI já passa este método a `MetricsComparator` como `reload_test_transform`.

---

## 4. Gestão de features em tempo de execução

### 4.1 Regra de identificação `onto_*`

```719:749:core/trepan_reloaded_extractor.py
    def _is_ontology_derived_feature(self, feature_name):
        return str(feature_name).lower().startswith(self.ONTO_FEATURE_PREFIX)

    def _identify_base_feature_names(self, feature_names, original_feature_names=None):
        """Features ARFF originais — agnóstico ao número de colunas."""
        if original_feature_names:
            return list(original_feature_names)
        return [
            n for n in feature_names
            if not self._is_ontology_derived_feature(n)
        ]
```

Metadados GUI (`onto_{feat}_concept`, `_depth`, …) são `onto_*` mas tratados à parte em `_is_ontology_metadata_column` para priorização de splits.

### 4.2 Evitar erro de dimensão no MLP

Cadeia de defesa:

1. **`n_features_in_`** do `MLPClassifier` após treino em ARFF.
2. **`_resolve_feature_schemas`** — `oracle_names` ⊆ `base_names` quando `n_mlp == len(base_names)`.
3. **`_matrix_for_oracle`** — indexação por nome; fallback `non_onto_idx[:n_exp]`.
4. **`_enforce_oracle_width`** — corta colunas extra: `X[:, :n_exp]`.
5. **GUI `_slice_matrix_to_original_features`** — antes do `train_mlp`.
6. **`MLPTrainer.reset()`** — evita encoders de dimensão anterior entre sessões.

Warnings típicos no log: `[WARN] Oráculo MLP: a cortar N→M colunas.`

### 4.3 Codificação dual (original vs enriquecido)

```70:110:core/mlp_trainer.py
    def encode_columns_by_name(
        self, X, column_names, base_column_names, extra_encoders=None
    ):
        """
        Colunas em base_column_names usam feature_encoders do treino ARFF;
        demais colunas (onto_*) usam encoders auxiliares por nome de coluna.
        """
```

A árvore vê `X_encoded_aug` com dimensão `|ARFF| + |onto_*|`; o MLP continua a ver só `|ARFF|`.

### 4.4 `OntologyProcessor`

Prefixo partilhado `ONTO_PREFIX = "onto_"`. Gera colunas hierárquicas, relacionais, de contexto e `_High`/`_Low`. Integrado no extrator via `_apply_semantic_feature_engineering` quando a GUI ainda não injectou inferidas.

---

## 5. Cálculo de Fidelidade e Precisão

### 5.1 Durante a extração (relatório inline)

```3081:3088:core/trepan_reloaded_extractor.py
    def _calculate_fidelity(self, tree, mlp_model, X_data, y_data):
        X_tree = self.apply_ontology_split_bias(X_data)
        y_tree_pred = tree.predict(X_tree)
        y_mlp_pred = self._mlp_predict(mlp_model, X_data)

        # Fidelidade = concordância entre árvore e MLP
        fidelity = accuracy_score(y_mlp_pred, y_tree_pred)

        return fidelity
```

**Fórmula implementada (fidelidade):**

\[
\text{Fidelidade} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_{\text{árvore}}(x_i) = \hat{y}_{\text{MLP}}(x_i)]
\]

com \(\hat{y}_{\text{árvore}}\) sobre `apply_ontology_split_bias(X)` e \(\hat{y}_{\text{MLP}}\) sobre submatriz ARFF via `_mlp_predict`.

`y_data` (rótulos reais) **não entram** nesta fidelidade inline (só concordância árvore–MLP).

Refino interno / dominância:

```2130:2138:core/trepan_reloaded_extractor.py
    def _measure_tree_performance(self, tree, mlp_model, X, y_true):
        ...
        return {
            'fidelity': float(accuracy_score(y_mlp, y_tree)),
            'precision': float(precision_score(y_true, y_tree, average='weighted', zero_division=0)),
        }
```

**Precisão (extrator):** `sklearn.metrics.precision_score(y_true, y_tree, average='weighted')`.

### 5.2 Comparação formal (GUI → `MetricsComparator`)

Após treino, botão «Comparar Métricas» chama `compare_all_models`. Com pipeline duplo:

```2414:2421:gui/biuri_app_complete.py
            if dual_pipeline:
                compare_kwargs.update(
                    X_test_reloaded=X_test_encoded_aug,
                    ...
                    reload_test_transform=self.trepan.extractor.apply_ontology_split_bias,
                )
```

#### Precisão (vs rótulos reais `y_test`)

```1394:1410:core/metrics_comparator.py
    def _calculate_metrics_with_ci(self, y_true, y_pred, include_macro=True):
        ...
        metrics_with_ci['precision'] = self._bootstrap_confidence_interval(
            y_true, y_pred, 
            lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0)
        )
```

\[
\text{Precisão weighted} = \text{precision\_score}(y_{\text{test}}, \hat{y}_{\text{modelo}}, \text{average='weighted'})
\]

Árvore Reloaded: `X_rel = apply_ontology_split_bias(X_test_reloaded)` antes de `predict`.

#### Fidelidade (vs MLP)

```1068:1098:core/metrics_comparator.py
            overall_fidelity_ci = self._bootstrap_confidence_interval(
                y_mlp_ref, y_trepan_reloaded_pred, accuracy_score
            )
            ...
            fidelity['trepan_reloaded'] = {
                'overall_fidelity': overall_fidelity_ci['mean'],
                ...
                'agreement_rate': np.mean(y_mlp_ref == y_trepan_reloaded_pred),
                'fidelity_reference': 'mlp_original',
```

\[
\text{Fidelidade geral} = \text{accuracy\_score}(\hat{y}_{\text{MLP original}}, \hat{y}_{\text{Trepan Reloaded}})
\]

- **Trepan-Original / C4.5:** mesma fórmula com `X_test` original (sem `apply_ontology_split_bias`).
- **MLP:** fidelidade não é auto-referência; precisão MLP = concordância com `y_test`.
- Intervalos de confiança: bootstrap em `_bootstrap_confidence_interval`.

### 5.3 Tabela resumo de métricas

| Métrica | Onde | Referência | Espaço de features |
|---------|------|------------|-------------------|
| Fidelidade inline | `_calculate_fidelity` | MLP | Real/sintético augmentado + bias árvore |
| Fidelidade comparador | `_calculate_fidelity_metrics` | MLP original | Reloaded: teste augmentado + bias |
| Precisão extrator | `_measure_tree_performance` | `y_true` | Augmentado + bias |
| Precisão comparador | `_calculate_precision_metrics` | `y_test` | Reloaded: augmentado + bias |
| Dominância | `_tree_dominates` | Margens `DOMINANCE_MARGIN`, `PRECISION_MARGIN` | Conjunto eval augmentado |

---

## 6. Pipeline GUI de treino (sequência)

1. Carregar ARFF → `original_data`.
2. Se OWL: `_rebuild_augmented_dataset()` → `augmented_data`, `feature_names_augmented`.
3. `train_model`:
   - `X_mlp, y = _get_mlp_training_xy()` → `trepan.train_mlp`.
   - `X_encoded_aug, y_encoded_aug = _encode_augmented_for_tree()` (se dual).
   - C4.5 em `X_train_dom` (original).
   - Trepan-Original: `TREPANExtractor.extract_tree(mlp_model, X_encoded, ...)`.
   - Trepan-Reloaded: `extract_tree_with_ontology(mlp_model, X_encoded_aug, ..., original_feature_names=feature_names)`.
   - Opcional: `ensure_ontology_dominance(...)` no conjunto eval augmentado.

Mensagem explícita na UI: *«MLP oráculo só no ARFF original; Trepan-Reloaded na cópia enriquecida»*.

---

## 7. Pontos de extensão recomendados

### 7.1 `MLP_Onto` (oráculo enriquecido)

| # | Ficheiro | Alteração |
|---|----------|-----------|
| 1 | `gui/biuri_app_complete.py` | Treinar `mlp_model_onto` com `X_encoded_aug`; não reutilizar `mlp_model_reloaded = self.mlp_model` |
| 2 | `core/mlp_trainer.py` | Instância separada ou `reset` + treino dedicado; metadata `oracle_space='augmented'` |
| 3 | `core/trepan_reloaded_extractor.py` | Parâmetro `mlp_model_onto`; ramo em `_resolve_feature_schemas` quando `n_features_in_ == len(matrix_names)` |
| 4 | `core/trepan_reloaded_extractor.py` | Fase 3: opcionalmente etiquetar sintéticos com oráculo enriquecido (ou ensemble original+onto) |
| 5 | `core/metrics_comparator.py` | `fidelity_reference: 'mlp_onto'` e `y_mlp_pred_reloaded` já previsto no dual pipeline |
| 6 | Testes | `tests/test_ontology_feature_matching.py` — regressão de dimensões |

Risco: se a árvore imita `MLP_Onto` mas a fidelidade reportada for vs `MLP_original`, métricas parecem piores sem serem comparáveis.

### 7.2 `ONTO_FEATURE_BIAS_WEIGHT`

| # | Ficheiro | Alteração |
|---|----------|-----------|
| 1 | `TrepanReloadedExtractor.__init__` ou config GUI | Expor peso (substitui `ONTO_FEATURE_WEIGHT = 2.5`) |
| 2 | `_build_feature_gain_weights` / `_build_ontology_split_multipliers` | Factor único documentado |
| 3 | `apply_ontology_split_bias` | Usado em treino, predict, comparador, dominância — **um único sítio de verdade** |
| 4 | Avaliação | Grid em [1.0, 1.5, 2.5, 3.5] com `_tree_dominates` no hold-out |

Opcional: alinhar critério da árvore Reloaded a `entropy` no baseline C4.5 para comparação mais justa (hoje Reloaded prefere `gini`).

### 7.3 Garantir dominância sobre C4.5 e Trepan Original

Mecanismos **já existentes**:

- `ensure_ontology_dominance` — refino de precisão preservando âncora de fidelidade.
- `_fit_tree_dominance_grid` — selecção por `_score_tree_dominance` / `_tree_dominates`.
- Amostragem sintética 5000 com mix centrality + fronteira MLP.
- `ONTO_FEATURE_WEIGHT` + augmentação de amostras `onto_*`.

Lacunas para revisão externa:

- Oráculo não vê features semânticas → tecto de fidelidade quando `onto_*` são preditivas e MLP não.
- Viés de split é heurístico (escala), não IG ponderado formal.
- C4.5 usa `entropy` em original; Reloaded usa `gini` em augmentado — assimetria metodológica.

---

## 8. Ficheiros de referência cruzada

| Tema | Documentação existente |
|------|------------------------|
| Fidelidade / precisão | `docs/CALCULO_PRECISAO_FIDELIDADE.md`, `docs/ENRIQUECIMENTO_ONTOLOGIA_FIDELIDADE_PRECISAO.md` |
| Testes features | `tests/test_ontology_feature_matching.py` |
| Trepan clássico | `core/trepan_extractor.py` |

---

## 9. Conclusão executiva

1. **Oráculo actual:** um único MLP em **X_original**; consulta via `_mlp_predict` com stripping/corte para `n_features_in_`. **Não** recebe `X_enriched`.
2. **Split:** **sklearn** Gini/entropia; viés ontológico = **`X * multipliers`** (`ONTO_FEATURE_WEIGHT=2.5`) + pesos de amostra — **não** IG custom.
3. **Features:** prefixo `onto_`, `original_feature_names`, `n_features_in_`, encoders separados em `encode_columns_by_name`.
4. **Métricas:** Fidelidade = `accuracy(MLP, árvore)`; Precisão = `precision_score(y_true, árvore, weighted)`; comparador com bootstrap e `apply_ontology_split_bias` no Reloaded.

Implementar `MLP_Onto` e `ONTO_FEATURE_BIAS_WEIGHT` nos pontos da Secção 7 mantém a arquitectura dual e torna configurável o que hoje está hardcoded nas constantes da classe extratora.

---

*Gerado para análise externa do repositório BIURI/Trepan Reloaded.*
