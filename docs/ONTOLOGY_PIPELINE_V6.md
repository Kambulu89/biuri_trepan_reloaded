# Pipeline ontológico v6

## Invariantes científicos

1. O enriquecimento é um transformador com estado. `fit` recebe somente o
   treino externo; validação e teste usam apenas `transform`.
2. A TBox fornece classes, propriedades, hierarquias e axiomas. Indivíduos de
   vocabulário controlado podem representar categorias; indivíduos semelhantes
   a registos/pacientes ou identificadores do teste bloqueiam a ontologia.
3. O schema de entrada e saída é nominal e imutável. Colunas ausentes,
   inesperadas ou fora de ordem provocam erro; não existe truncamento nem
   preenchimento com zero.
4. Features derivadas constantes, duplicadas ou quase duplicadas de features
   originais são removidas durante o `fit`.
5. Matching abaixo de `0.72`, ambíguo por margem inferior a `0.08`, cobertura
   insuficiente ou ontologia genérica rejeitam o enriquecimento.
6. A ontologia só é aceite após execução real do reasoner HermiT/Pellet. Falha
   de execução, inconsistência ou classes insatisfazíveis não contam como
   validação bem-sucedida.
7. A ordem nominal da classe vem da declaração `@ATTRIBUTE` do ARFF e é
   preservada pelo encoder, pelas probabilidades OOF e pelos relatórios.

## Fluxo de treino

```text
ARFF -> split externo -> fit do matching/gate/TBox no treino
     -> transform do treino -> seleção/OOF/treino residual
     -> transform do teste bloqueado -> avaliação final única
```

O carregamento da ontologia executa primeiro o reasoner e uma auditoria ABox
preliminar. Quando os dados e o split estão disponíveis, o quality gate é
reexecutado com os nomes das features e os identificadores de teste. Uma
ontologia rejeitada interrompe o treino; o sistema não continua silenciosamente
como se OWL tivesse sido aplicada.

## Matching de categorias por indivíduos

Indivíduos OWL de vocabulário controlado são indexados por `name`, `label` e
sinónimos. Os seus tipos (`is_a`) definem o grupo semântico da categoria. Uma
feature categórica só é criada quando o agrupamento reduz efetivamente o espaço
de categorias; um mapeamento 1:1 redundante é eliminado.

## Benchmark OWL

O benchmark usa três TBoxes RDF/XML específicas de domínio, sem indivíduos de
dataset: Iris, Wine e Breast Cancer Wisconsin. Cada execução usa o mesmo split
e seed para a ablação pareada `sem OWL` versus `com OWL`. São 3 datasets × 3
repetições = 9 pares, com teste externo bloqueado.

| Modelo | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| C4.5-Nativo | 0.951 | 0.952 | 0.951 |
| TREPAN Original | 0.748 | 0.743 | 0.718 |
| TREPAN Reloaded — sem OWL | 0.748 | 0.743 | 0.718 |
| TREPAN Reloaded — com OWL | 0.922 | 0.915 | 0.917 |

O efeito pareado de OWL no Reloaded foi `+0.174` em accuracy (IC bootstrap 95%
`[0.053, 0.315]`, Wilcoxon `p=0.0391`), `+0.172` em balanced accuracy (IC
`[0.049, 0.303]`, `p=0.0391`) e `+0.199` em macro-F1 (IC `[0.047, 0.364]`,
`p=0.0391`). OWL melhorou significativamente esta amostra de engenharia, mas o
Reloaded não superou o C4.5-Nativo em média. Isto não é validação clínica
externa nem autoriza uma alegação universal de superioridade.

Resultados reproduzíveis: `results/ablation/ablation_results.json`,
`ablation_rows.csv` e `ablation_report.md`.

