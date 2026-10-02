# Benchmark confirmatório V7

Data: 2026-08-30  
Configuração SHA-256: `4d38fe2b09ac1d6b1dd9f372249d5b26dc02e36a84830d640012404183b906ef`

## Conclusão pré-especificada

**A superioridade forte do TREPAN Reloaded não foi demonstrada.** A execução
foi realizada uma única vez e bloqueada por
`CONFIRMATORY_RUN_COMPLETED.json`; não houve repetição nem ajuste posterior aos
resultados do teste.

## Médias nos 15 testes externos bloqueados

| Modelo | Accuracy | Balanced accuracy | Macro-F1 | Fidelidade ao MLP Original |
|---|---:|---:|---:|---:|
| C4.5-Nativo | 0,8712 | 0,8720 | 0,8699 | 0,8622 |
| TREPAN Original | 0,8528 | 0,8496 | 0,8510 | 0,8764 |
| TREPAN Reloaded | 0,8462 | 0,8438 | 0,8447 | 0,8785 |

O Reloaded obteve a maior fidelidade média ao MLP Original, mas isso não se
converteu em melhor desempenho contra os rótulos reais.

## Diferenças pareadas: Reloaded menos concorrente

| Concorrente | Métrica | Diferença média | IC bootstrap 95% | Wilcoxon unilateral p | V/E/D |
|---|---|---:|---:|---:|---:|
| TREPAN Original | Accuracy | -0,0066 | [-0,0308; 0,0165] | 0,6235 | 6/2/7 |
| TREPAN Original | Balanced accuracy | -0,0059 | [-0,0306; 0,0183] | 0,6401 | 7/0/8 |
| TREPAN Original | Macro-F1 | -0,0063 | [-0,0309; 0,0172] | 0,6192 | 7/0/8 |
| C4.5-Nativo | Accuracy | -0,0250 | [-0,0456; -0,0029] | 0,9749 | 4/2/9 |
| C4.5-Nativo | Balanced accuracy | -0,0282 | [-0,0496; -0,0056] | 0,9836 | 4/1/10 |
| C4.5-Nativo | Macro-F1 | -0,0252 | [-0,0467; -0,0019] | 0,9579 | 4/1/10 |

`V/E/D` significa vitórias/empates/derrotas do Reloaded.

## Balanced accuracy por dataset

| Dataset | C4.5-Nativo | TREPAN Original | TREPAN Reloaded |
|---|---:|---:|---:|
| Breast cancer | 0,9508 | 0,8918 | 0,8993 |
| Diabetes progression | 0,6736 | 0,7449 | 0,6968 |
| Digits | 0,8267 | 0,7489 | 0,7622 |
| Iris | 0,9651 | 0,9466 | 0,9309 |
| Wine | 0,9438 | 0,9160 | 0,9296 |

## Diagnóstico interno, sem alterar a conclusão

- O gate OOF do oráculo ontológico indicou superioridade sobre C4.5 em 6/15
  partições, não-inferioridade em 3/15 e rejeição em 6/15.
- As rejeições concentraram-se em Iris e Wine; o professor ontológico ainda não
  é consistentemente melhor nesses domínios.
- Os testes `m`-of-`n` foram efetivamente utilizados em 15/15 árvores canónicas.
- A seleção final interna escolheu 12 árvores redestiladas, 2 podadas por Pareto
  e 1 árvore ativa.
- O Reloaded melhorou o TREPAN Original em Breast Cancer, Digits e Wine, mas a
  perda em Diabetes e Iris anulou o ganho médio.

## Interpretação

As alterações solicitadas foram implementadas e auditadas, mas não fabricaram
uma vitória. O gargalo confirmado é o desempenho do professor ontológico em
alguns datasets e a conversão de fidelidade em generalização contra o rótulo
real. Estes resultados não autorizam afirmar que o TREPAN Reloaded é superior ao
C4.5 ou ao TREPAN Original.

As ontologias do benchmark são TBoxes de domínio independentes das classes e
dos testes, validadas pelo reasoner. Não são ontologias clínicas externamente
validadas por especialistas e o benchmark não constitui validação clínica.
