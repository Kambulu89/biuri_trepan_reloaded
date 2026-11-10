# Configuração e validação de datasets (CSV / ARFF)

Carregue ARFF, CSV ou TSV em **📁 Carregar dados**. A configuração pode ser revista em **⚙ Configurar dataset**.
Lógica em `core/dataset_schema.py` (agnóstica a datasets); integração em `gui/dataset_config_support.py`.

## Coluna-alvo
Precedência: escolha manual › configuração guardada do dataset › diretiva ARFF explícita (`% biuri:target = nome` ou `@relation ... -C n`) › **última coluna**. A coluna tem de existir; senão o treino é bloqueado.

## Ordem das classes
Precedência: escolha manual (se divergir da de um modelo já treinado, é **bloqueada** — nunca alterada em silêncio) › ordem do modelo/encoder treinado › ordem nominal declarada no ARFF › ordenação determinística (numérica se todos os rótulos forem numéricos; senão lexicográfica). O mapeamento rótulo→índice fica guardado e é o único usado por MLP, TREPAN, métricas, matrizes de confusão e explicações (todos derivam do `OrderedLabelEncoder` construído a partir desta ordem).

## Tipos
`numeric`, `categorical`, `ordinal`, `boolean`, `datetime`. Prioridade: declarado no ARFF › manual/guardado › inferência pelos valores. Inteiros com ≤ 10 valores distintos são tratados como **categóricos (a confirmar)** — nunca numéricos só por serem números. O MLP codifica essas colunas como categóricas (hook `forced_categorical_columns`, inativo sem esquema). Conversões (CSV): boolean→0/1, datetime→segundos UTC, ordinal→posto na ordem definida.

## Validação (bloqueia o treino se houver incompatibilidade crítica)
Alvo inexistente/ausentes/uma só classe · ordem inválida · colunas duplicadas · tipos inválidos · valores ausentes (CSV; em ARFF por omissão é aviso, como antes) · categorias desconhecidas · incompatibilidade com ARFF de referência, com o modelo/encoder treinado (alvo, features, ordem) · códigos numéricos categóricos mapeados na ontologia (o enriquecimento OWL tratá-los-ia como numéricos).

## Persistência e reprodutibilidade
`~/.biuri/dataset_configs/<hash do ficheiro>.json` (ou `BIURI_DATASET_CONFIG_DIR`); também em `arff_meta['dataset_schema']`. A assinatura do esquema entra na chave da cache de modelos **apenas** quando o esquema não é o comportamento anterior (chaves existentes ficam iguais). ARFF por omissão: dataframe intacto, nada gravado.

## Limitações
O modo SCIENTIFIC / BENCHMARK usa o seu próprio serviço de dados; a validação pré-treino aplica-se-lhe, mas a conversão de tipos do esquema não.
