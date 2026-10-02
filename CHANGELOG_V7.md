# Changelog V7

- Otimização por CV e calibração sigmoide dos MLPs Original, Ontológico e
  Residual, sem consulta ao teste externo.
- Gate OOF pareado do oráculo ontológico contra C4.5-Nativo.
- TREPAN canónico com expansão best-first e testes `m`-of-`n` por beam search.
- Merit function multiobjetivo e poda Pareto.
- Destilação híbrida de rótulos reais, probabilidades do professor e confiança
  semântica.
- Consultas ativas por discordância com o oráculo e com C4.5.
- Contrafactuais de fronteira ponderados por plausibilidade.
- Soft-tree, seleção global e redestilação crisp com guard de não-degradação.
- Catálogo ampliado para cinco TBoxes independentes de rótulos/testes.
- Árvores contrafactuais isoladas por modelo-alvo.
- Benchmark confirmatório executado e bloqueado; a conclusão negativa é
  preservada integralmente no pacote.
