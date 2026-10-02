# C4.5 nativo no BIURI / TREPAN Reloaded

## Identidade do algoritmo

O comparador contém exatamente três árvores:

1. TREPAN Original — árvore substituta do MLP Original;
2. C4.5-Nativo — árvore supervisionada pelos rótulos reais;
3. TREPAN Reloaded — árvore substituta com enriquecimento semântico quando uma
   ontologia válida é carregada.

A MLP continua necessária como caixa-preta/oráculo, mas não participa do ranking
visual das árvores. CART-Entropy foi removido. Weka e Java não são utilizados.

## Implementação

`core/c45_j48_tree.py` preserva o nome físico histórico por compatibilidade com
artefactos antigos, mas o símbolo novo é `C45Tree` e o estimador é
`C45Classifier`.

O algoritmo implementa:

- entropia de Shannon;
- ganho de informação corrigido pela proporção de valores conhecidos;
- Gain Ratio e filtro de ganho médio antes da maximização;
- limiares numéricos entre valores adjacentes de classes diferentes;
- split nominal com um ramo por categoria;
- distribuição fracionária de missing values segundo os pesos dos ramos;
- remoção do atributo nominal após o split e reutilização de atributos contínuos;
- poda pessimista baseada no limite superior do erro e no `confidence_factor`;
- pesos por instância, classificação multiclasse e probabilidades de folha.

## Protocolo de comparação

C4.5 é treinado somente no treino externo, recebe apenas atributos ARFF originais
e nunca recebe colunas `onto_*`. A seleção de parâmetros usa exclusivamente folds
internos. O teste bloqueado é usado apenas para a avaliação final, nas mesmas
instâncias usadas por TREPAN Original e TREPAN Reloaded.

Na interface, C4.5 deve aparecer sempre como `C4.5-Nativo`. O identificador
interno legado `c45_j48` é conservado para que resultados e modelos antigos
continuem legíveis; ele não significa integração com J48.
