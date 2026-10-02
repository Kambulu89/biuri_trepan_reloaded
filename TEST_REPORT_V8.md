# Relatório de testes V8

Data: 2026-08-31

## Resultado

- Suíte não gráfica: **170 aprovados, 1 ignorado**.
- Testes novos V8: **7 aprovados**.
- Três módulos que importam Qt não foram recolhidos porque o contentor não possui
  `libEGL.so.1`.
- A interface e todos os ficheiros Python são validados adicionalmente por
  compilação estática antes do empacotamento.

## Cobertura nova

- aceitação de feature OWL útil;
- eliminação de duplicada e constante;
- rejeição de ontologia contaminada;
- pesos híbridos e cobertura OOF exata;
- transporte das probabilidades OOF para o extrator;
- consultas por erro real, minoria e gap semântico;
- folds idênticos nas 23 variantes de ablação;
- bloqueio de métricas do teste dentro da ablação;
- bloqueio confirmatório sem ontologia externa independente.

## Avisos não bloqueantes

- Alguns MLPs pequenos alcançam `max_iter` durante testes sintéticos.
- Testes legados que retornam booleano geram `PytestReturnNotNoneWarning`.
- Estes avisos não produziram falhas, mas devem ser limpos numa manutenção futura.

## Benchmark

O benchmark confirmatório V8 não foi executado. O resultado negativo e já
observado da V7 permanece em `results/confirmatory_v7/` e não foi reutilizado
para tuning.
