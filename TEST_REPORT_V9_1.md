# Relatório de testes V9.1

Data: 2026-09-02

## Erros reproduzidos e corrigidos

| Fluxo | Causa | Correção | Resultado |
|---|---|---|---|
| ARFF + OWL | Owlready2 iniciava HermiT com `java` não resolvido e a GUI convertia o relatório em erro fatal | descoberta/configuração explícita do Java; aviso acionável e continuação segura sem OWL quando indisponível | aprovado |
| ARFF sem OWL | `trepan_reloaded_audit` só era criado no ramo com ontologia | inicialização obrigatória e auditoria espelhada no ramo sem OWL | aprovado |

## Validação focada

- 5 regressões V9.1 aprovadas: inicialização GUI, ordem criação/leitura da
  auditoria, deteção por `JAVA_HOME`, ausência simulada de Java e configuração
  da binding interna do Owlready2.
- 5 testes do pipeline ontológico aprovados, incluindo HermiT real com Java 17,
  fit/transform no treino, ABox/quality gate e ordem ARFF.
- 2 testes de espelhamento sem OWL aprovados, com igualdade de predições,
  métricas e auditoria.

## Regressão consolidada

- **170 testes aprovados e 1 ignorado** na suíte não gráfica `tests/`.
- **4 regressões legadas não gráficas** da raiz aprovadas.
- **32 testes `unittest` críticos** aprovados.
- **145 ficheiros Python** compilados sem erros.
- `TrepanApp.spec` compilado sintaticamente sem erros.

Os avisos observados foram `ConvergenceWarning` de testes deliberadamente
curtos e `PytestReturnNotNoneWarning` em quatro scripts legados. Não houve
falhas funcionais.

## Reasoner e validade científica

- HermiT foi executado de verdade numa TBox de teste e devolveu
  `executed=True`, `consistent=True`.
- A simulação de Java ausente devolveu `executed=False`, `consistent=None` e
  `error_type=java_not_found`; não houve falso positivo de consistência.
- A GUI rejeita a ativação da OWL sem reasoning, informa como configurar Java
  e permite continuar com o ARFF sem enriquecimento.
- Ontologias inconsistentes ou com ABox de registos continuam bloqueadas.

## Limitação gráfica do contentor

Os dois módulos de teste que importam widgets Qt não foram executados porque
PyQt6/runtime EGL não estão disponíveis neste contentor. O módulo GUI completo
foi compilado e as alterações de estado/ordem foram verificadas pela AST.
O `requirements-win.txt` mantém PyQt6 fixado para o ambiente-alvo Windows.

## Empacotamento

- `unzip -t`: nenhum erro no conteúdo comprimido.
- Cópia extraída: 145 ficheiros Python recompilados.
- Cópia extraída: 12 testes focados de reasoner, ontologia e modo sem OWL
  aprovados.
- O SHA-256 é calculado sobre o pacote final e comunicado na entrega.
