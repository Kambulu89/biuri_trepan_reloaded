# Esquema do dataset resolvido automaticamente (ARFF/CSV + ontologia)

Não há configuração manual. O BIURI/TREPAN Reloaded **lê a própria ontologia ativa** e resolve coluna-alvo, ordem das classes e
tipos das colunas, validando sempre contra o ficheiro de dados, o ARFF e o modelo treinado. Código genérico, sem nomes de datasets:
`core/dataset_schema.py` (resolução/validação), `core/ontology_schema.py` (leitura na ontologia),
`core/ontology_schema_vocabulary.json` (vocabulário multilingue editável), `gui/dataset_schema_support.py` (integração).

## Precedências
| Elemento | Ordem de decisão |
|---|---|
| **Coluna-alvo** | diretiva ARFF explícita (`% biuri:target = nome` ou `@relation … -C n`) › alvo declarado na ontologia (entidade marcada, ou a **única** coluna cujos valores coincidem com as etiquetas de classe da ontologia) › última coluna |
| **Ordem das classes** | ordem do modelo/encoder já treinado (mesmo dataset) › ordem nominal do ARFF › ordem da ontologia › ordenação determinística (numérica se todos os rótulos forem numéricos; senão lexicográfica) |
| **Tipos** | declarado no ARFF › declarado na ontologia › inferência pelos valores |

Inteiros com ≤ 10 valores distintos ficam **categóricos «a confirmar»** (nunca numéricos só por serem códigos), salvo se a ontologia os declarar numéricos.
O mapeamento rótulo→índice fica em `arff_meta['class_mapping']` e o `OrderedLabelEncoder` do MLP/TREPAN usa exatamente essa ordem.

## O que a ontologia pode declarar
Em qualquer vocabulário do ficheiro `ontology_schema_vocabulary.json` (por omissão sinónimos EN/ES/PT; acrescente os seus):
- **Atributos** (indivíduos de uma classe tipo `Attribute`/`Atributo`): nome da coluna (`columnName`/`nombreAtributo`…), tipo (`dataType`/`tipoDato`: numeric, categorical, ordinal, boolean, datetime; `target`/`class` marca o alvo), `ordinalOrder`.
- **Etiquetas de classe** (indivíduos de `TargetLabel`/`EtiquetaObjetivo`): valor (`classValue`/`valorEtiqueta`) e, opcionalmente, `classIndex`; sem índice usa-se a ordem natural dos nomes dos indivíduos (`etiqueta_1`, `etiqueta_2`…).
- **Anotações canónicas** `biuriColumnName`, `biuriDataType`, `biuriIsTarget`, `biuriClassOrder` ("a,b,c"), `biuriClassIndex`, `biuriClassValue`, `biuriOrdinalOrder` em qualquer entidade.
- **Propriedades por coluna** (rótulo/nome == coluna): o tipo vem do `rdfs:range` (xsd:double→numeric, xsd:string→categorical, xsd:boolean, xsd:dateTime).

## Validação (bloqueia o treino se crítica)
Do ficheiro: alvo inexistente/ausentes/uma só classe, ordem inválida, colunas duplicadas, tipos inválidos, valores ausentes (CSV), colunas-identificador (quase uma categoria por linha), ARFF de referência.
Da ontologia: **alvo** declarado inexistente ou em conflito com o ficheiro; **classes** da ontologia diferentes das do alvo; **ontologia de outro domínio** (nenhum atributo coincide); tipo declarado incompatível com o ARFF; códigos numéricos categóricos mapeados na OWL (o enriquecimento OWL tratá-los-ia como numéricos).
Do modelo treinado (mesmo dataset): alvo, features e **ordem das classes** — nunca alterada em silêncio. Um modelo treinado noutro ficheiro não bloqueia dados novos.
Avisos: ordem da ontologia diferente da usada (prevalece ficheiro/modelo), atributos da ontologia ausentes no dataset, categorias de ocorrência única, códigos numéricos ambíguos.

## Re-resolução e reprodutibilidade
Ao carregar/remover uma ontologia depois dos dados, o esquema é recalculado e, se mudar, o ficheiro é recarregado (modelos treinados são descartados, como já acontecia). O esquema resolvido e a evidência (`arff_meta['dataset_schema']`, `['ontology_schema']`) são registados em `~/.biuri/dataset_configs/<hash>.json` (ou `BIURI_DATASET_CONFIG_DIR`) só para auditoria. A assinatura do esquema entra na chave da cache de modelos apenas quando difere do comportamento anterior (ARFF por omissão: chave igual, dataframe intacto, nada gravado).

## Notas científicas
O esquema é partilhado por TREPAN Original, C4.5, MLP e Reloaded na mesma execução (mesmos dados, mesma ordem de classes). Com ontologia ativa a ordem das classes pode diferir da de uma execução sem ontologia em dados sem declaração própria; fica registada em `class_order_source`.

## Limitações
O modo SCIENTIFIC / BENCHMARK usa o seu próprio serviço de dados: a validação pré-treino aplica-se-lhe, a conversão de tipos do esquema não. Ontologias inconsistentes (reasoner falha) não ficam ativas e, portanto, não são lidas.
