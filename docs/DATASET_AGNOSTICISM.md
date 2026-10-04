# Agnosticismo ao dataset

O núcleo científico (`core/`, `gui/`, `counterfactuals/` genérico) não contém nomes de datasets, números conhecidos de
features/classes, hiperparâmetros dedicados nem caminhos de execução por dataset. Toda a decisão emerge de propriedades
observadas no treino e da configuração científica genérica (grelha de tuning configurável, CV 5×3, política lexicográfica).

```
dataset arbitrário -> schema -> alvo -> tipos -> pré-processamento genérico -> split estratificado
  -> treino do oráculo -> FrozenOracle (oracle_id) -> tuning só no treino -> seleção por evidência estatística
  -> TREPAN Original / Reloaded (mesmo oracle_id) -> avaliação final (teste usado uma só vez)
```

## Onde vivem os datasets conhecidos
`validation/` (fora do núcleo; o núcleo nunca o importa): catálogos de ontologias de benchmark, estudo de ablação e o
registo/pipelines em lote de contrafactuais sobre datasets nomeados (`validation/counterfactual_research/`).
O CLEAR vendorizado já não traz defaults de nenhum dataset. A exclusão de classes-alvo no matching ontológico deriva dos
nomes das classes observados (`set_target_metadata`), com vocabulário genérico de papéis (class/target/label/...).

## Garantias automáticas (`tests/test_dataset_agnosticism.py`, `tests/test_v92_static_guards.py`)
1. sem nomes de datasets conhecidos nos módulos científicos; sem ramos por identidade do dataset nem por nº fixo de features/classes;
2. nº arbitrário de features (2, 7, 25); 3. binário e multiclasse (2, 3, 5 classes); 4. numéricas e categóricas, com valores em falta;
5. nomes de alvo arbitrários (incl. unicode e alvo na 1.ª coluna); 6. nomes e números de classes arbitrários;
7. ontologia presente ou ausente; 8. um novo ARFF compatível funciona sem alterar o código-fonte (hash dos fontes inalterado);
9. renomear colunas/alvo/classes dá a mesma seleção e as mesmas estatísticas; a grelha é só configuração.
