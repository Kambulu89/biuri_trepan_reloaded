# CHANGELOG — BIURI / TREPAN Reloaded V9.2

## Adicionado

- Contrato de dados genérico e serializável (`core/data_contract.py`).
- Pré-processador sklearn-compatible ajustado apenas no treino (`core/preprocessing.py`).
- Fábrica adaptativa única de MLP e perfis baseados na geometria dos dados.
- Auditoria de capacidade preditiva contra `DummyClassifier`.
- Convergência com captura de `ConvergenceWarning` e suporte a `CalibratedClassifierCV`.
- Estratégias/tolerâncias de avaliação dependentes da dimensão efectiva da validação.
- Deadline/cancelamento cooperativo para pesquisa de hiperparâmetros.
- Excepções científicas/operacionais tipadas adicionais.
- Logging central com suporte opcional a JSON e ficheiro rotativo.
- `TrepanOriginalClassifier` canónico; imports históricos de `TREPANExtractor` são redireccionados para o núcleo histórico, sem árvore destilada.
- Artefactos V9.2 (`manifest.json` + `model.joblib`) e API headless de inferência.
- Carregamento genérico CSV/TSV/ARFF/Parquet/Excel.
- CLI headless experimental.
- Pipeline V9.2 agnóstico ao dataset.
- Matriz de testes sintéticos dataset-agnostic.
- Script de auditoria/migração de pickles antigos.
- Runner de ablação V9.2 separado dos resultados congelados.

## Corrigido

- Alvo do Hepatitis passa a ser a primeira coluna.
- Removida imputação antes do split no preparador de Hepatitis.
- Removido `os.chdir()` do bootstrap de contrafactuais.
- Selecção baseline/Grid/Optuna deixa de consultar o teste externo durante a escolha do candidato.
- Decisão de calibração usa apenas dados internos de treino; métricas finais são calculadas sobre o modelo efectivamente entregue.
- Hash de cache usa todo o conteúdo do dataset.
- Construção do MLP da ablação passa pela fábrica adaptativa, evitando o MLP fixo de 24 neurónios/220 iterações.
- Compatibilidade do preprocessing com colunas categóricas/binárias numéricas em scikit-learn 1.8.

## CI/packaging

- `pyproject.toml` actualizado para versão 9.2.0 e Python `>=3.11,<3.13`.
- Extras opcionais para GUI/OWL/CLEAR/Excel/Parquet/testes.
- CI Windows em Python 3.11 e 3.12.

## Ainda pendente

- Wiring integral do pipeline V9.2 em toda a GUI histórica.
- Refactor integral de `counterfactuals/cf_mlp_trainer.py` e remoção do requisito de configs por dataset.
- Migração total do armazenamento de cache para o novo formato de artefacto.
- Validar a release final no ambiente-alvo Windows/Python 3.11–3.12 com GUI e reasoner; a build de runtime já bloqueia reintrodução de árvores sklearn auxiliares.
- Validação cruzada da implementação histórica contra o código TREPAN original preservado pela Univ. Wisconsin, quando executável.
- Revisão completa de todos os `except Exception` e de todos os `print()` legados.
- Execução/validação em Python 3.11 e 3.12, Windows, PyQt6, OWL/HermiT e TensorFlow/CLEAR.

## Incremento: TREPAN Reloaded sobre o núcleo histórico (29/09/2026)

- Adicionado `core/trepan_reloaded_historical.py` com `TrepanReloadedClassifier`, extensão directa de `TrepanOriginalClassifier`.
- Com semântica neutra, Reloaded e Original são exactamente equivalentes para a mesma seed/configuração.
- O efeito OWL entra directamente no score/prioridade dos splits e na projecção das membership queries, sem mudar a família algorítmica.
- `extract_tree_with_ontology()` deixou de chamar CART/soft-tree/active-query/multiobjective no caminho principal.
- Na etapa seguinte de hardening de produção, `_legacy_cart_baseline()` foi removido integralmente e os caminhos legados incompatíveis passaram a ser bloqueados.
- A fidelidade por grupo semântico deixou de treinar CARTs temporários.
- Árvores TREPAN descartam a referência ao oráculo após o fit, tornando o artefacto de inferência autónomo.
- Nova bateria `tests/test_reloaded_historical_core_v92.py`.
- Validação crítica desta ronda: **30 passed, 8 warnings em 28.79 s**.
- `results/confirmatory_v7/`: hashes SHA-256 inalterados.

Relatório: `IMPLEMENTATION_REPORT_V9_2_HISTORICAL_RELOADED_CORE.md`.

## Auditoria experimental controlada Original vs Reloaded

- criado `core/controlled_trepan_experiment.py`;
- ablação OWL passa a usar exactamente o mesmo MLP Original nos dois braços;
- removido `ResidualOntologicalOracle` do protocolo de ablação controlada;
- `OriginalOracleProjection` adapta o espaço OWL sem alterar o professor;
- seed e orçamento TREPAN passam a ser verificados como idênticos;
- gate do oráculo usa apenas CV de treino contra Dummy e C4.5;
- datasets com oráculo inválido são excluídos do efeito OWL agregado;
- bootstrap/Wilcoxon passam a usar dataset como unidade de análise;
- `scripts/run_ablation_v9_2.py` regenerado com split estratificado e protocolo controlado;
- braço OWL não executado neste runtime por ausência de owlready2/HermiT.

## Ontology gate separation — 2026-09-29

- Separados quality/matching OWL, feature-engineering OOF e aceitação do professor ontológico.
- Matching genérico passou a reconhecer variações de tokenização/ordem sem aliases por dataset.
- Corrigida penalização de complexidade do gate semântico para considerar o modelo completo.
- Novos estados de rejeição distinguem matching, reasoner, novidade, custo e ganho preditivo.
- OWL estrutural válida pode orientar TREPAN Reloaded mesmo quando o MLP ontológico é rejeitado.
- GUI usa o quality report para estatísticas de mapping e mostra métricas ausentes como `N/A`.
- Original e Reloaded usam o mesmo orçamento do núcleo histórico na comparação principal.

## Semantic audit & exact fidelity fix — 2026-09-29

- Professor ontológico rejeitado/fallback deixa de aparecer como barra `MLP Ontológico`.
- `projected_original` é tratado explicitamente como MLP Original.
- Fidelidade de controlo Original/Reloaded usa a mesma estimativa pontual exacta; bootstrap fica apenas para IC.
- Adicionada auditoria por nó `data-only vs semantic` com `Ontology Usage Rate` e `Semantic Decision Impact`.
- Adicionada coerência de grupos OWL em regras m-of-n para evitar que matching uniforme 100% seja neutralizado pela normalização dos pesos.
- Extracção OWL passa a unir data/object/annotation properties e registar `subPropertyOf` de propriedades.
- Relatórios deixam de inferir contribuição semântica apenas a partir de fidelidade alta.
- Nova suite `tests/test_semantic_audit_metrics_v92.py`.

## Production candidate hardening
- Added generic `OntologySemanticGraph` and semantic relatedness matrices.
- Added `SemanticContributionGate` distinct from ontology quality and feature OOF utility.
- Added paired production training, versioned production bundle and headless inference.
- Added multi-seed controlled ablation runner and scientific claim guard in production reports.
- Split optional dependencies (OWL, CLEAR, Excel/Parquet) from core runtime.
- Added production readiness checker and frozen V7 hash reference outside the frozen directory.

## Scientific-mode production lock — 2026-09-30
- Corrigido `int(None)` no TREPAN Reloaded quando o preset Científico usa profundidade opcional.
- Original e Reloaded passam a resolver os mesmos limites estruturais.
- GUI/worker/orquestrador bloqueados no modo Científico para produção.
- Defaults de treino MLP passam a Científico.
- Corrigida recursão no helper CART legado.
- Adicionados testes de regressão `test_scientific_mode_lock_v92.py`.

## Production No-CART hardening

- removido `DecisionTreeClassifier` de todos os módulos Python de runtime (`core`, `gui`, `counterfactuals`, `scripts`);
- `TREPANExtractor` legado redireccionado para o TREPAN Original histórico;
- removidos fallbacks/refinos de árvore de outra família do TREPAN Reloaded;
- LORE/árvore contrafactual local migrados para TREPAN histórico;
- Data Contract deixou de usar árvore sklearn para leakage check;
- bundles de produção passam a usar `biuri-v9.2-production-2-no-cart`;
- loader recusa famílias de árvore incompatíveis;
- adicionado guard AST permanente `test_production_no_cart_v92.py`;
- confirmatório V7 mantido congelado e não reexecutável na build de produção.

## Adaptive Semantic TREPAN — 2026-09-30

- candidatos m-of-n orientados pela ontologia;
- active membership queries sem aumentar consultas ao oráculo;
- tuning científico train-only de capacidade TREPAN e força semântica;
- capacidade final idêntica entre Original e Reloaded;
- relatedness OWL ponderada por tipo de relação + ancestors inferidos;
- formato de bundle `biuri-v9.2-production-3-adaptive-semantic`;
- 100 testes críticos disjuntos aprovados nesta revisão.

## Error-Focused Semantic Refinement (EFSR)

- A semântica passa a actuar prioritariamente em regiões onde TREPAN e MLP discordam.
- Adicionado perfil local feature↔erro e oportunidade semântica.
- Membership queries são priorizadas por anchors de erro sem aumentar o orçamento.
- Propostas OWL só substituem o split data-only quando melhoram fidelidade local.
- O gerador ontológico avalia ambas as orientações dos literais em regras m-of-n.
- Auditoria por nó reporta fidelidade local antes/depois, decisão e motivo do gate.
- Novo formato de produção: `biuri-v9.2-production-4-error-focused-semantic`.

## Semantic Real-Gain Contract

- Quality-gate matches passam a alimentar um `OntologySemanticGraph` efectivo.
- Relações/grupos OWL são propagados ao motor histórico Reloaded.
- EFSR ganha fallback top-k para regiões com erro real abaixo do threshold fixo.
- Candidato semântico precisa melhorar também fidelidade nas linhas reais internas de treino.
- Sem contribuição semântica mensurável, Reloaded espelha exactamente o Original.
- GUI passa a mostrar ficheiro OWL, versionInfo e SHA-256 para rastreabilidade.
- Formato de artefacto actualizado para `biuri-v9.2-production-5-semantic-real-gain`.

## 2026-09-30 — Semantic enrichment / MLP utility gate upgrade

- Agregações OWL hierárquicas convertidas para `standardized_mean` train-only.
- Novo motor genérico de features relacionais por `measurementFamily` + `statisticRole`.
- TBox Breast Cancer enriquecida com família/papel para as 30 features.
- `SemanticUtilityGate` passou de proxy LogisticRegression para MLP pareado ao MLP Original.
- Refit de estatísticas semânticas dentro de cada fold OOF.
- GUI agora expõe métricas Base vs Base+OWL e ganho líquido do gate.
- Adicionado `tests/test_semantic_enrichment_v92_upgrade.py`.
