# B3.0 — evidência da auditoria (Docs. 31–33)
Reprodução (só leitura sobre os brutos M29 / só treino nas ontologias DEV; nenhum dataset confirmatório):
- `python -m validation.benchmark.b30_controls_audit --sc docs/evidence/b1b4/sampler_c/dev_numeric --m29 docs/evidence/b1b4/m29_campaign/main docs/evidence/b1b4/m29_campaign/g29_pilot_patch --out g29_audit.json` → `g29_audit.json`, `b30_g29_table.md`
- `reconcile()` de `b30_controls_audit.py` sobre `m29_campaign/report/m29_report.json` → `m29_attribution_reconciliation.json`
- `python -m validation.benchmark.b30_ontology_audit --out onto_dev.json` → `onto_dev.json` (censo de axiomas, reasoner, matching, grafo, grupos, relatedness, ativação estrutural)
- `validation/benchmark/b30_axiom_probe.py` (`axiom_cases`, `probe`) → `axiom_probe.json`
- `python -m validation.benchmark.b30_query_constraint_violation query_constraint_violation_bc.json` → violações de ordem definicional pelos amostradores de consultas (breast_cancer, só treino)
- `python -m validation.benchmark.b30_identifiability --seeds 0..11 --out ident_main.json` → `ident_main.json`
- testes: `tests/test_b30_semantic_audit.py`
