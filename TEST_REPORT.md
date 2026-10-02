# Relatório de testes desta entrega

Data: 2026-08-30

## Estado funcional

- Suíte não gráfica: **154 testes aprovados** após a inclusão do teste de
  isolamento das árvores contrafactuais.
- Teste de integração OWL/reasoner e fidelidade Reloaded: aprovado.
- Testes novos: otimização/calibração dos oráculos, gate OOF contra C4.5,
  destilação híbrida, `m`-of-`n`, best-first, consultas por discordância,
  contrafactuais plausíveis, soft-tree, seleção global e fingerprints por modelo.
- Cinco TBoxes RDF/XML passam por quality gate e HermiT antes do benchmark.
- O código Python é verificado por `compileall` antes do empacotamento.

## Benchmark confirmatório único

Foram executados cinco datasets × três repetições, com teste externo bloqueado.
O ficheiro `results/confirmatory_v7/CONFIRMATORY_RUN_COMPLETED.json` impede uma
segunda execução neste pacote.

| Modelo | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| C4.5-Nativo | 0,8712 | 0,8720 | 0,8699 |
| TREPAN Original | 0,8528 | 0,8496 | 0,8510 |
| TREPAN Reloaded | 0,8462 | 0,8438 | 0,8447 |

Conclusão: **a superioridade forte não foi suportada**. O Reloaded foi 0,59 p.p.
inferior ao Original e 2,82 p.p. inferior ao C4.5 em balanced accuracy média.
O relatório estatístico completo está em
`results/confirmatory_v7/CONFIRMATORY_REPORT.md`.

## Árvores contrafactuais

TREPAN Original, C4.5 e TREPAN Reloaded utilizam agora o respetivo modelo-alvo
como oráculo contrafactual. Os testes verificam referências e fingerprints
distintos, evitando as árvores iguais causadas pelo MLP partilhado.

## Limitação gráfica

Os cinco testes que importam PyQt6 não foram recolhidos neste contentor porque a
biblioteca gráfica não está instalada. Os módulos Python são compilados
estaticamente; a interface completa deve ainda ser aberta no ambiente Windows
de destino.
