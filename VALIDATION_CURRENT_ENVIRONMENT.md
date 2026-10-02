# Validação desta execução — ambiente disponível

Os ficheiros de evidência estão em `results/audit_remediation_v9_1/`.

- Runtime: Linux, Python 3.13.5.
- Java: OpenJDK 21 disponível.
- Graphviz `dot`: disponível.
- Ausentes neste runtime: `PyQt6`, `owlready2`, `dtreeviz`.
- A tentativa de instalar dependências não pôde usar a rede/DNS; por isso não foi possível transformar este runtime no ambiente Windows/Python 3.11 declarado pelo projeto.
- `compileall`: OK.
- `tests/test_audit_remediation_v9_1.py`: 13 passed.
- Subconjunto regressivo selecionado: 65 passed, 46 `ConvergenceWarning` de fixtures com limites de iteração deliberadamente curtos.
- Testes GUI: collection bloqueada por ausência de PyQt6.
- Teste OWL: collection bloqueada por ausência de owlready2.
- Benchmark externo confirmatório: propositalmente não consumido; o launcher bloqueou a execução no preflight.

Esta distinção é importante: a implementação e os testes executáveis foram corrigidos/validados, mas a reprodutibilidade Windows/GUI/HermiT e a evidência confirmatória final só podem ser declaradas depois da execução no ambiente alvo.
