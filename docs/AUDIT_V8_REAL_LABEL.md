# Auditoria V8 — ontologia convertida em desempenho contra rótulos reais

## Conclusão

A V8 corrige o objetivo de otimização: fidelidade ao MLP deixa de ser o objetivo
dominante. A aceitação semântica, a escolha do professor, os splits, as consultas,
a seleção global e a poda passam a ser comandados principalmente por balanced
accuracy, macro-F1 e recall das classes minoritárias, sempre no desenvolvimento.

Isto **aumenta a possibilidade**, mas não garante, que a ontologia melhore o
desempenho externo. Se o OWL não demonstrar ganho OOF estável, é rejeitado ou
apenas um subconjunto de features é mantido.

## Auditoria inicial

| Problema observado | Risco | Correção V8 |
|---|---|---|
| Features OWL válidas eram ativadas sem prova preditiva | Ruído semântico | `SemanticUtilityGate` OOF obrigatório |
| Pesos do professor favoreciam fidelidade | Árvore fiel, mas menos correta | Rótulo real 60%; professor 30%; semântica 10% |
| Probabilidades das linhas reais podiam ser in-sample | Sobreajuste da destilação | Probabilidades OOF obrigatórias no híbrido |
| Seleção da árvore dava 48% à fidelidade e 10% à accuracy | Objetivo científico invertido | BA, macro-F1 e recall mínimo tornados primários |
| Consultas privilegiavam apenas árvore↔oráculo | Erros reais não priorizados | Proxy local do rótulo real, minoria e gap semântico |
| TBoxes curadas eram descritas como independentes | Alegação metodológica excessiva | Catálogo distingue `project-curated` de externa |
| Ontologia Adult contém linhas e target | Leakage direto | Marcada `CONTAMINATED_ONTOLOGY` e proibida |

## Fluxo V8

1. O enriquecedor ontológico é ajustado apenas no treino e aplicado por
   `transform` às restantes partições.
2. Quality gate valida TBox/ABox, matching, reasoner, cobertura e genericidade.
3. `SemanticUtilityGate` remove features inválidas e seleciona subconjuntos por
   OOF no subtreino.
4. MLP Original, MLP Ontológico direto e MLP Residual são ajustados no
   desenvolvimento.
5. Pesos do professor híbrido são escolhidos numa validação interna separada.
6. O professor ontológico é comparado ao C4.5 nos mesmos folds OOF.
7. A destilação das linhas reais usa probabilidades OOF; sintéticos podem ser
   consultados no professor final porque não foram linhas de treino observadas.
8. TREPAN canónico usa best-first e testes `m`-of-`n`, com ganho real e balanceado
   maior do que ganho do professor.
9. Consultas, soft-tree, redestilação e poda só são aceites se não degradarem BA,
   macro-F1 e recall mínimo na validação interna.
10. O teste externo permanece fechado até uma única confirmação futura.

## Quality gate semântico

Saídas implementadas:

- `ACCEPT_ONTOLOGY`;
- `ACCEPT_PARTIAL_FEATURES`;
- `REJECT_NO_INFORMATIONAL_GAIN`;
- `REJECT_UNSTABLE`;
- `REJECT_COMPLEXITY_COST`;
- `REJECT_INCONSISTENT`;
- `REJECT_LEAKAGE`.

O relatório inclui features candidatas, constantes, duplicadas, selecionadas,
confiança/cobertura do matching, reasoner, MI média, estabilidade por fold,
novidade, custo de complexidade e deltas OOF.

## Oráculos e calibração

| Componente | Entrada | Treino | Papel |
|---|---|---|---|
| MLP Original | Features ARFF | CV + calibração no desenvolvimento | Baseline/professor |
| MLP Ontológico direto | ARFF + OWL selecionadas | CV multiobjetivo + calibração | Sinal semântico direto |
| MLP Residual | Probabilidades OOF do Original + OWL | Desenvolvimento | Correção residual |
| Professor híbrido | Probabilidades dos três | Pesos na validação interna | Professor do Reloaded |

Se o híbrido não melhorar a utilidade predefinida sobre o Original, os pesos
voltam ao Original. O relatório mostra também se o professor selecionado supera
o C4.5 na validação interna; isso não é tratado como superioridade externa.

## Ablação

`core/component_ablation.py` define e valida 23 variantes:

- oito modelos/controlos principais;
- reasoner, matching estrutural, categorias, discordância, contrafactuais e poda
  com/sem cada componente;
- destilação hard, soft e híbrida.

Todas recebem exatamente os mesmos folds repetidos. O avaliador é bloqueado se
tentar devolver qualquer campo de teste externo. A execução confirmatória não
foi feita nesta versão.

## Leakage e schemas

- O teste externo não participa do gate semântico, pesos híbridos, otimização,
  consultas, seleção global ou poda.
- O professor híbrido falha se as probabilidades OOF estiverem ausentes,
  desalinhadas ou não finitas.
- Nomes duplicados e colunas originais ausentes são erros bloqueantes.
- Não há truncagem nem preenchimento silencioso com zero.
- Classes inesperadas geram `ClassOrderMismatchError`; espaços incompatíveis
  geram `FeatureSpaceMismatchError`.
- A ordem nominal do ARFF continua preservada pelo carregador existente e pelos
  testes de regressão.

## Ontologias disponíveis

As cinco TBoxes do benchmark são curadas pelo projeto a partir das descrições das
features, sem indivíduos do dataset. Servem para testes de engenharia, não como
ontologias externas independentes. A ontologia Adult foi rejeitada por conter
indivíduos de linhas, índices e classe de rendimento.

O manifesto `data/ontology_catalog_v8.json` mantém origem, hash, licença,
independência e estado de qualidade. Como nenhuma ontologia externa licenciada
passou ainda pelo gate, o confirmatório V8 está bloqueado.

## Resultado anterior preservado

O benchmark V7 já observado concluiu que a superioridade não foi demonstrada:
balanced accuracy média de 0,8438 para Reloaded, 0,8496 para TREPAN Original e
0,8720 para C4.5. A V8 não foi afinada nem reavaliada nesse teste.

## Limitações restantes

- Falta obter e auditar ontologias externas adequadas a cada dataset.
- A melhoria interna do professor pode não generalizar para um novo teste.
- O proxy de rótulo real nas consultas ativas usa vizinhança do treino; não é um
  rótulo real de uma amostra sintética.
- A validação clínica exige dados externos por centro/tempo e revisão de regras
  por especialistas.
- Três módulos Qt não carregam neste contentor por falta de `libEGL.so.1`.

## Critério futuro de superioridade

Só declarar `SUPERIOR` se, no teste confirmatório novo e único, o Reloaded tiver
ganho pareado contra ambos os baselines em métricas reais primárias, intervalo de
confiança compatível, tamanho de efeito relevante, consistência entre datasets,
ausência de degradação clínica por classe e complexidade aceitável. Caso contrário
o resultado deve ser `EQUIVALENT`, `INCONCLUSIVE` ou `INFERIOR`.
