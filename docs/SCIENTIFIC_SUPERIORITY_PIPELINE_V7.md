# Pipeline científico V7 — TREPAN Reloaded

## Objetivo

Esta versão melhora o professor e o substituto sem usar o teste externo para
seleção. Superioridade não é uma configuração: é uma conclusão condicionada ao
benchmark confirmatório pareado.

## Alterações implementadas

1. MLP Original e MLP Ontológico/Residual: busca interna por balanced accuracy,
   `StandardScaler` dentro do pipeline e calibração sigmoide por CV no treino.
2. Gate OOF estrito entre o MLP Original e o professor ontológico no mesmo
   holdout interno: accuracy, balanced accuracy e macro-F1 têm de ser
   simultaneamente não-inferiores dentro da margem pré-registada. O C4.5-Nativo
   permanece baseline supervisionado por rótulos reais; a sua concordância com
   outro modelo é apenas análise auxiliar e não define o oráculo TREPAN.
3. Splits multiobjetivo combinando ganho ao oráculo, rótulo real, semântica,
   estabilidade e complexidade.
4. TREPAN canónico com fila best-first, orçamento de nós e testes `m`-of-`n`
   procurados por beam search.
5. Destilação híbrida: rótulos reais, probabilidades calibradas do professor e
   confiança semântica.
6. Consultas ativas que priorizam discordâncias TREPAN–oráculo e
   C4.5–oráculo, entropia, diversidade e proximidade ao manifold.
7. Contrafactuais de fronteira ajustados apenas no treino, ponderados por
   plausibilidade, proximidade e validade.
8. Soft-tree e seleção global, seguidas por redestilação crisp. Uma barreira de
   não-degradação na validação interna impede perda de fidelidade ou balanced
   accuracy após redestilação/poda.
9. Cinco TBoxes de domínio independentes de rótulos e do teste: iris, wine,
   breast cancer, diabetes progression e digits. Todas passam pelo quality gate
   e pelo reasoner HermiT antes do benchmark.
10. Protocolo confirmatório congelado e executável uma única vez por pacote,
    com `claim_guard`: só `superiority_supported` autoriza linguagem de
    superioridade; estimativas pontuais isoladas nunca bastam.

## Árvores contrafactuais

Cada árvore contrafactual usa o próprio modelo-alvo como oráculo. O serviço
regista fingerprints separados do C4.5, TREPAN Original e TREPAN Reloaded; a
MLP não substitui silenciosamente nenhum desses modelos.

## Regra de superioridade

A versão só declara superioridade forte se o TREPAN Reloaded superar os dois
concorrentes em balanced accuracy e macro-F1, com limite inferior do IC bootstrap
pareado de 95% acima de zero e Wilcoxon unilateral com `p < 0,05` em todas as
quatro comparações.

As TBoxes incluídas são ontologias de benchmark curadas a partir das definições
públicas das features; são independentes dos rótulos e partições, mas não devem
ser apresentadas como ontologias clínicas validadas por especialistas.
