# Changelog V9.1 — correções de execução Windows

- Corrigida a carga ARFF + OWL quando o Java está instalado fora do `PATH`.
- O reasoner procura explicitamente `OWLREADY2_JAVA_EXE`, `JAVA_EXE`,
  `JAVA_HOME`, `JRE_HOME`, o `PATH` e diretórios usuais de JDK/JRE no Windows.
- O caminho resolvido é aplicado ao namespace público e ao módulo interno de
  reasoning do Owlready2 antes de executar HermiT/Pellet.
- O ficheiro PyInstaller passou a incluir os JARs e restantes dados do
  Owlready2 e deixou de depender de um caminho absoluto de outro computador.
- Ausência de Java produz diagnóstico acionável; deixa de expor apenas
  `FileNotFoundError [WinError 2]`.
- Uma OWL sem reasoner não é aceite como logicamente consistente: o ARFF
  continua em modo sem ontologia, sem criar features sem validação.
- Inicializadas sempre as auditorias de TREPAN Original e TREPAN Reloaded.
- No treino sem OWL, a auditoria Reloaded espelha explicitamente a auditoria
  Original e identifica o modo `mirrored_no_ontology`.
- O valor da auditoria é criado antes do registo de desempenho, eliminando o
  erro `BiuriApp object has no attribute trepan_reloaded_audit`.
- Ao carregar outro ARFF, árvores e auditorias antigas são invalidadas para
  impedir resultados residuais entre datasets.

## 2026-09-27 — Correção do baseline C4.5 e Precision Macro

- A métrica primária usada no refinamento/dominância do TREPAN Reloaded passou de `Precision Weighted` para `Precision Macro`.
- `precision_weighted` continua disponível apenas para auditoria; o alias legado `precision` no extrator aponta agora para `precision_macro`.
- C4.5 deixou de participar em qualquer condição de fidelidade/dominância como se fosse oráculo. Continua exclusivamente baseline supervisionado contra os rótulos reais.
- Adicionado `core/c45_baseline_gate.py` com gate separado para TREPAN Original e TREPAN Reloaded contra C4.5.
- No teste bloqueado, o gate C4.5 é apenas descritivo e nunca altera o modelo selecionado.
- O `surrogate_quality_audit` passou a exigir explicitamente as métricas macro e, quando C4.5 está disponível, audita a não-inferioridade em relação ao baseline.
- A interface de comparação mostra uma linha de baseline C4.5 e marca cada TREPAN como `OK C4.5` ou `ABAIXO C4.5`, sem modificar percentagens.
- O relatório textual inclui uma secção `GATE C4.5 — BASELINE SUPERVISIONADO (NÃO É ORÁCULO)`.
- Adicionados testes de regressão em `tests/test_c45_baseline_policy_v9_1.py`.
