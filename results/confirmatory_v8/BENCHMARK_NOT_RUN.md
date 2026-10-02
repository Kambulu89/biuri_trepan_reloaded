# Benchmark confirmatório V8 — não executado

Data: 2026-08-31

O teste confirmatório V7 já foi observado e permanece congelado em
`results/confirmatory_v7/`. Ele **não foi reutilizado** para selecionar features,
pesos, árvores ou hiperparâmetros da V8.

O benchmark V8 está bloqueado porque o catálogo ainda não contém uma ontologia
de domínio externa, versionada, licenciada, independente do projeto e aprovada
pelo quality gate. As TBoxes incluídas são adequadas para testes de engenharia,
mas não satisfazem esse requisito confirmatório.

Estado: `BLOCKED — NO_INDEPENDENT_VERSIONED_DOMAIN_ONTOLOGY`

Consequência: esta entrega não afirma que o TREPAN Reloaded V8 supera C4.5 ou
TREPAN Original. Uma única execução futura deverá usar protocolo congelado e um
teste externo novo e independente.
