# Semantic Real-Gain Contract — V9.2

O TREPAN Reloaded V9.2 usa a ontologia apenas quando ela consegue produzir efeito
mensurável no treino interno.

## Contrato

1. A OWL passa primeiro por quality gate e matching.
2. O matching aprovado é convertido em `OntologySemanticGraph`.
3. Regiões de discordância entre TREPAN e MLP são priorizadas por erro, alcance e
   incerteza.
4. A semântica propõe candidatos `m-of-n` apenas nessas regiões prioritárias.
5. Uma alteração só é aceite se melhorar fidelidade no decision sample e também nas
   linhas reais de treino do nó.
6. Se nenhuma alteração/reforço semântico for aceite, o Reloaded final espelha
   exactamente o TREPAN Original.
7. O teste externo nunca participa destas decisões.

## Diagnóstico esperado

Uma execução com OWL deve conseguir distinguir:

- `OWL válida/mapeada`;
- `grafo semântico disponível`;
- `regiões EFSR avaliadas/elegíveis`;
- `intervenções tentadas/aceites`;
- `ganho de fidelidade real interno`;
- `semantic_effect_mirror_applied` quando não houver contribuição.

Uma OWL com 100% de matching mas sem relações úteis não é tratada como ganho
semântico. Nessa situação o modelo final deve ser igual ao Original.
