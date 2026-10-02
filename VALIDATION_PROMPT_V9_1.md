# Validação - Prompt BIURI/TREPAN Reloaded V9.1

- compileall: PASS
- regressão crítica: 77 passed, 0 failed, 20 warnings
- testes novos do prompt: 14 incluídos nos 77, todos PASS
- smoke test do comparador: PASS
- Precision Macro e Accuracy: calculadas separadamente
- comparação visual primária: Precision Macro
- leakage guard: ativo
- gate MLP Ontológico: Precision Macro + Recall Macro + Macro-F1 + Balanced Accuracy + Accuracy
- gate TREPAN Reloaded: mesmas métricas + fidelidade + complexidade
- GUI: não executada neste runtime (PyQt6 ausente)
- OWL/HermiT via owlready2: não executado neste runtime (owlready2 ausente; rede indisponível)
- Java: disponível (OpenJDK 21)
