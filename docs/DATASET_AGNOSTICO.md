# BIURI / TREPAN Reloaded V9.2 — Uso com datasets arbitrários

## Objectivo

A V9.2 introduz um caminho de treino agnóstico ao dataset. O caminho novo parte de um `pandas.DataFrame`, exige a confirmação explícita da coluna-alvo e executa contrato de dados, divisão treino/teste, pré-processamento ajustado apenas no treino, MLP adaptativo, árvores substitutas/baselines e, opcionalmente, serialização num artefacto versionado.

A ontologia OWL é opcional. O fluxo sem OWL deve ser considerado o caso normal.

## Contrato de dados

`core.data_contract.build_data_contract(df, target, user_overrides=None)` inspecciona as colunas e produz um objecto serializável. O alvo pode ser indicado pelo nome ou índice. Sem confirmação explícita, uma sugestão automática é apenas uma sugestão e o contrato fica marcado como `alvo_nao_confirmado`.

O contrato regista, por coluna, tipo inferido, cardinalidade, taxa de falta e decisão de tratamento. Os marcadores de falta reconhecidos incluem `NaN`, `None`, string vazia, `?`, `NA`, `N/A` e `null`.

São sinalizados, entre outros: colunas constantes, ID-like, elevada taxa de falta, forte desbalanceamento, poucas linhas, mais colunas do que linhas, duplicados e possível fuga de alvo.

## Pré-processamento

`core.preprocessing.DataPreprocessor` é ajustado **apenas nas linhas de treino**. O split é feito antes do `fit`.

- Numéricas: imputação pela mediana e indicadores de falta quando aplicável.
- Categóricas de baixa/média cardinalidade: one-hot com categorias desconhecidas ignoradas.
- Alta cardinalidade: frequency encoding ou descarte, de acordo com a decisão do contrato.
- ID-like, constantes e texto livre: descartados do caminho de modelo quando o contrato assim determinar.

A transformação mantém `feature_names_out` e `feature_origins` para ligar cada feature derivada à coluna original.

## Treino genérico

O caminho de referência é `core.pipeline_v92.train_dataset`.

Exemplo conceptual:

```python
from core.pipeline_v92 import train_dataset

result = train_dataset(
    df,
    target="classe",
    seed=42,
    artifact_dir="results/modelo_v92",
)
```

O MLP é criado por `core.mlp_factory` com perfil adaptado ao número de linhas, número de features e número de classes, sem nomes ou presets específicos por dataset.

## Inferência

Os artefactos V9.2 podem ser carregados por `core.inference.load_artifact`.

```python
from core.inference import load_artifact

predictor = load_artifact("results/modelo_v92")
pred = predictor.predict(novos_dados)
proba = predictor.predict_proba(novos_dados)
report = predictor.validate(novos_dados)
```

DataFrames podem ter as colunas em ordem diferente; o schema é realinhado pelos nomes. Categorias novas e valores em falta são tratados pelo preprocessor.

## Formatos de entrada

`core.data_loading` suporta CSV/TSV, ARFF, Parquet e Excel. Dependências opcionais ausentes são comunicadas com erro accionável.

A CLI experimental encontra-se em `scripts/biuri_cli.py` e possui comandos de treino, avaliação, predição e explicação.

## OWL

OWL é opcional. O caminho genérico não selecciona ontologias por nome de dataset. Enriquecimento só deve ser activado quando o utilizador fornece uma ontologia e esta passa os quality gates existentes.

## Avisos importantes desta implementação candidata

A integração completa do novo fluxo com todos os caminhos históricos da GUI e dos contrafactuais ainda não está concluída. Em particular, existem caminhos legados que continuam a usar os antigos treinadores/configurações. Consulte `IMPLEMENTATION_REPORT_V9_2.md` para o estado validado e as pendências.

## Estados da ontologia na V9.2

A presença de um ficheiro OWL não implica automaticamente que um professor MLP ontológico será aceite. O sistema mantém três decisões independentes:

1. **Qualidade estrutural e matching** — verifica TBox/ABox, reasoner e correspondência com o schema do dataset.
2. **Feature engineering OOF** — verifica se features derivadas da OWL acrescentam utilidade preditiva no desenvolvimento.
3. **Uso semântico no TREPAN Reloaded** — uma OWL estruturalmente válida pode orientar prioridades/splits do Reloaded mesmo quando o passo 2 não justifica alterar o professor MLP.

Os estados são inferidos a partir do schema e da ontologia carregados; não existem tabelas de aliases ou regras específicas para datasets embutidas no caminho principal.
