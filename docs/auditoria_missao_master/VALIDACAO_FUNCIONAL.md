# Validação funcional

Ambiente: Linux, 4 CPU, Python 3.12, numpy 1.26.4, scikit-learn 1.6.1, pandas 2.2.3, owlready2 0.47, PyQt6 (offscreen), Java com HermiT. Os registos estão em `evidencia/`.

| Componente | IMPLEMENTADO | EXECUTADO | VALIDADO | Evidência e âmbito |
|---|---|---|---|---|
| Suite de testes (antes das correções) | sim | sim | sim | 1732 aprovados, 40 ignorados, 70 min; `pytest_full_baseline.log` |
| Suite de testes (depois das correções) | sim | sim | sim | 1742 aprovados, 40 ignorados; `pytest_full_after_fixes.log` |
| Cobertura | — | sim | — | 72,1 % de `core/` + `counterfactuals/` (27 284 linhas) |
| Leitura ARFF | sim | sim | sim | 3 ficheiros reais (vehicle 200×4, adult 48 842×15, breast cancer 569×31) e casos sintéticos com ausentes |
| Leitura CSV | sim | sim | sim, após correção | separador `;`, ausentes `?`; defeito P09 corrigido com teste |
| Contrato de dados e tipos | sim | sim | parcial | identificadores, binárias, categóricas e numéricas verificadas; datas e texto livre NÃO TESTADOS |
| Divisão treino/teste estratificada | sim | sim | sim | mesmas instâncias de teste em todos os braços de cada unidade (verificado nas 20) |
| Treino do MLP | sim | sim | sim | 20 treinos no benchmark, 3 na produção (mais 6 no recálculo de 2026-11-20: 3 de controlo com o código antigo e 3 com o código corrigido), 5 no benchmark com ontologia enriquecida; accuracy de teste 0,953–0,969 em média por dataset |
| Reprodutibilidade do MLP | sim | sim | **falhou em parte** | 18 de 20 reproduzidos fora do harness (P13) |
| Convergência e tempo de treino do MLP | sim | sim | NÃO VERIFICADO | registados pelo harness; não os auditei individualmente |
| C4.5 | sim | sim | sim | 20 unidades; métricas recalculadas |
| TREPAN Original | sim | sim | sim (execução) | 20 unidades; equivalência à referência: ver desvios em `AUDITORIA_ALGORITMOS.md` |
| TREPAN Reloaded sem ontologia | sim | sim | sim | idêntico ao Original nas 20 unidades |
| TREPAN Reloaded com ontologia | sim | sim | sim (execução) | 20 unidades com ontologia estrutural, 5 com a enriquecida |
| Controlo com ontologia permutada | sim | sim | sim | 20 + 5 unidades |
| Controlos de grupos aleatórios e engenharia convencional | sim | **não** | NÃO TESTADO | existem no código; não os corri |
| Carregamento OWL | sim | sim | sim | 41 ficheiros |
| Consistência lógica (HermiT) | sim | sim | sim | 41 consistentes, 0 classes insatisfazíveis |
| Perfil OWL 2 DL estrito | — | não | NÃO TESTADO | sem validador de perfil |
| Pipeline de produção (CLI) com ARFF + OWL | sim | sim | sim (execução) | Breast Cancer, 3 sementes, 11–14 min cada; `producao_bc/`. Recalculado em 2026-11-20 com o código corrigido (`010619f`): as 3 sementes terminam com código 0; o código antigo reproduz exatamente as execuções originais; com a ordem declarada das classes mudam a partição, o MLP e as árvores. Accuracy do MLP 0,986 / 0,958 / 0,958 (antes 0,979 / 0,951 / 0,965), sementes 7 / 42 / 123 (`PRODUCAO_BC_RECALCULO.md`, `evidencia/producao_bc_recalculo/`) |
| Cálculo das métricas | sim | sim | sim | 700 valores recalculados com scikit-learn, diferença máxima 0 |
| Persistência de resultados | sim | sim | sim | JSON por braço, `RUN_META.txt`, `UNIT_SHA256.json` por unidade |
| Persistência e recarga de modelos | sim | pela suite | NÃO TESTADO por mim | `production_bundle.joblib` gravado; não o recarreguei |
| GUI | sim | pela suite (offscreen) | NÃO TESTADO manualmente | os testes de GUI passam; não operei a interface |
| Windows | — | não | NÃO TESTADO | só Linux |
| Contrafactuais (`counterfactuals/`) | sim | pela suite | NÃO AUDITADO | fora do âmbito desta passagem |

## Incidentes de execução

- O supervisor do baseline foi terminado pelo limite de 2 horas da tarefa em segundo plano, com 18 de 20 unidades concluídas. As duas unidades em falta (digits, sementes 2024 e 11) foram executadas depois individualmente com o mesmo comando por unidade; retomaram dos pontos de controlo do tuning e terminaram com código de saída 0.
- Os testes de GUI falharam na coleção até instalar `libEGL` (P18).
