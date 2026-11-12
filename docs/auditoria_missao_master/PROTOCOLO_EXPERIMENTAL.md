# Protocolo experimental

Este documento fixa o protocolo usado nesta auditoria (secção 1) e o protocolo proposto para a comparação confirmatória (secção 2). O segundo foi escrito antes de existirem resultados obtidos com ele.

## 1. Protocolo usado na reprodução do baseline

- **Harness:** `python -m validation.benchmark.run --main` (o do próprio projeto), manifest `BENCHMARK_MANIFEST_v1.json`.
- **Datasets:** iris, wine, breast_cancer, digits (os quatro de ranking do manifest).
- **Sementes:** 42, 7, 123, 2024, 11 (as cinco do manifest). Nenhuma foi escolhida, repetida ou excluída.
- **Partição:** holdout estratificado, 25 % de teste, uma por semente.
- **Braços:** A MLP; B C4.5 (rótulos reais); C TREPAN Original; D Reloaded sem ontologia; E Reloaded + OWL; F Reloaded + OWL permutada.
- **Oráculo:** um MLP por unidade, congelado (`oracle_id`), consultado por C, D, E e F.
- **Estrutura das árvores:** escolhida só no treino pelo tuning do projeto (CV 3×5), sem informação ontológica, igual em C–F.
- **Código:** commit `df045ee` com 5 ficheiros «congelados» diferentes do manifest (P05). As correções `af02d5d` e `16e6be8` não tocam em nenhum módulo importado pelo benchmark.
- **Ambiente:** Python 3.12, numpy 1.26.4, scikit-learn 1.6.1, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, 4 CPU.
- **Natureza:** DEV. Estes datasets já foram observados pelo projeto; nada aqui é confirmatório.

Verificação independente: `evidencia/verify_metrics.py` recalcula accuracy, F1, precisão, recall e fidelidade com scikit-learn a partir das previsões guardadas e confirma que todos os braços de cada unidade usam as mesmas instâncias de teste e as mesmas previsões do oráculo.

## 2. Protocolo proposto para a hipótese H1

### 2.1 Definição de «>»

`X > Y` num dataset significa: a diferença média emparelhada (mesmas partições e sementes) é positiva **e** o intervalo de confiança a 95 % dessa diferença exclui zero. Diferença positiva com intervalo a incluir zero é «observada, não sustentada».

H1 exige, por dataset, cinco condições em conjunto:

| Métrica | Condições |
|---|---|
| Accuracy (rótulos reais) | Reloaded+OWL > Original; Reloaded+OWL > C4.5; Original > C4.5 |
| Fidelidade ao MLP | Reloaded+OWL > Original; Reloaded+OWL > C4.5 |

Condições adicionais para atribuir o ganho à ontologia: Reloaded+OWL > Reloaded+OWL permutada, > grupos aleatórios e > engenharia convencional, nas duas métricas.

Dimensões reportadas sem entrarem na decisão, mas que impedem declarar sucesso se piorarem: número de nós, número de regras, profundidade, literais por regra, tempo, amostras do oráculo.

### 2.2 Comparação justa

- Mesmas instâncias de teste, mesmo oráculo, mesmo pré-processamento em todos os braços.
- Mesmo orçamento de amostras do oráculo no Original e no Reloaded (ação A3).
- Comparação a complexidade igual (ação A4).
- C4.5 com rótulos reais é um classificador supervisionado; a sua fidelidade ao MLP mede concordância, não extração. Uma variante **C4.5 treinado com rótulos do oráculo** é reportada à parte, identificada como tal.

### 2.3 Unidade de inferência e repetições

- A unidade de inferência é o dataset. Sementes e réplicas dentro de um dataset não são independentes entre datasets.
- Por dataset: 5 sementes × ≥ 5 réplicas do extrator.
- Estatística por dataset: diferença emparelhada, IC a 95 % por bootstrap sobre unidades, tamanho de efeito (diferença média / desvio das diferenças).
- Entre datasets: teste dos sinais ou de Wilcoxon sobre as diferenças médias por dataset. Com 4 datasets nenhum teste atinge p < 0,05 bilateral (mínimo 0,125): são precisos pelo menos 6 datasets com o mesmo sinal.

### 2.4 Regras fixas

- Datasets e sementes escolhidos antes de correr; nenhum é excluído depois.
- Critério de elegibilidade de datasets, fixado a priori: o MLP tem de superar o C4.5 em validação cruzada interna do treino. Onde não supera, a unidade é reportada como `HIERARCHY_INFEASIBLE`.
- O teste nunca é usado para escolher hiperparâmetros, estruturas ou configurações semânticas.
- Resultados desfavoráveis reportam-se com o mesmo detalhe dos favoráveis.

### 2.5 Evidência por execução

Identificador, commit, hash do dataset, hash da ontologia, configuração, semente, comando, ambiente numérico, registo, métricas e estado final. O harness do projeto já grava quase tudo em `RUN_META.txt`, `UNIT_SHA256.json` e nos JSON por braço.
