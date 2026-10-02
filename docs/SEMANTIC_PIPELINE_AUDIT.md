# Auditoria do pipeline ontológico (estado anterior às alterações)

Auditoria feita sobre o commit `b49e8ff`, **antes** de qualquer alteração desta fase.
Método: leitura do código, execução dos testes (344 passam, 3 falhas pré-existentes
não relacionadas), e medições com as ontologias do repositório. Nada abaixo é
inferido da documentação: cada afirmação foi confirmada no código ou numa execução.

## 1. Fluxo existente e ficheiros

```
ARFF/CSV ──► core/data_loading.py, core/data_contract.py, core/preprocessing.py
OWL      ──► owlready2 (get_ontology) em core/production_training.py, gui/biuri_app_complete.py
Quality  ──► core/ontology_quality.py        (OntologyQualityGate: matching, ABox, genéricas)
Reasoner ──► core/ontology_reasoner.py       (HermiT/Pellet via owlready2; só consistência)
Grafo    ──► core/ontology_semantic_graph.py (relatedness, grupos, limites, profundidade)
Enriq.   ──► core/ontology_processor.py      (fit/transform; agregados, relacionais, restrições, categóricos)
Seleção  ──► core/semantic_utility_gate.py   (novidade, refit por fold, estabilidade MI, OOF)
         ──► core/onto_feature_selector.py, core/semantic_contribution_gate.py
Estados  ──► core/ontology_stage_status.py, core/ontology_acceptance.py
MLP      ──► core/mlp_factory.py, mlp_trainer.py, mlp_optimizer.py, mlp_convergence.py
TREPAN   ──► core/trepan_original.py, trepan_reloaded_historical.py,
             trepan_reloaded_extractor.py, error_focused_semantic_refinement.py
Cache    ──► core/model_cache.py
GUI      ──► gui/biuri_app_complete.py, gui/ontology_status_presenter.py
```

Há **dois pipelines paralelos**:

| | Legado / GUI (`core/trepan.py`, `gui/biuri_app_complete.py`) | Headless V9.2 (`core/production_training.py`, `core/pipeline_v92.py`) |
|---|---|---|
| Usa `OntologyProcessor` / `onto_*` | sim | **não** |
| Usa `SemanticUtilityGate` | sim | **não** |
| "MLP Ontológico" | sim (residual/híbrido) | **não existe** |
| Ontologia usada para | features, gate, professor, TREPAN | só pesos/grupos/relatedness do TREPAN |
| Reasoner | sim (GUI) | sim (`require_reasoner`) |

## 2. O que existe e funciona (a preservar)

- **Quality gate** com matching por score, `runner_up_score`, margem de ambiguidade,
  ontologias genéricas, cobertura, ABox (registos de dataset e colisões de identificadores).
- **`OntologyProcessor`** com API `fit/transform`, fit só no treino, esquema de saída fixo,
  proveniência por feature (`feature_audit_`), deduplicação de constantes/duplicados,
  agregados padronizados (z-score aprendido no treino), relacionais declarados por
  `measurementFamily` + `statisticRole`, restrições com limites OWL explícitos,
  grupos categóricos por tipo de indivíduo.
- **`SemanticUtilityGate`**: novidade (constantes/duplicados), refit do processor **por fold**,
  estabilidade por frequência de seleção e informação mútua, comparação OOF base vs +OWL
  com utilidade ponderada e penalização de complexidade, aceitação parcial.
- **Separação de estados** em `ontology_stage_status.py` (estrutural vs engenharia vs professor).
- **TREPAN Reloaded**: matriz de relatedness, grupos, pesos semânticos, EFSR, auditoria
  por nó (`information_gain`, `semantic_bonus`, `selection_score`, `decision_changed`).
- **Reasoner** HermiT funcional neste ambiente (Java presente; ~0,5 s numa OWL pequena).

## 3. Lacunas confirmadas (o que está incompleto ou não influencia o modelo)

| # | Lacuna | Evidência |
|---|---|---|
| L1 | Ontologias do repositório são esqueletos (3–4 classes, 0 propriedades de objeto, 0 limites) e o quality gate dava `VALID` na mesma | medição; já endereçado em `4e9db3a` (aviso de riqueza) |
| L2 | Agregados são combinações lineares exatas das fontes (R² = 1) | já auditado em `3d2f0bc` (opt-in de descarte) |
| L3 | Matching: nada impede várias features → mesma entidade; o registo não guarda **qual** foi o segundo candidato nem um `status` explícito | `match_features`, `MatchDecision` |
| L4 | ABox devolve só `accepted: bool`; sem estado nomeado (`SAFE`/`REJECT_*`) nem contagens TBox/ABox | `audit_abox` |
| L5 | Reasoner só verifica consistência: sem duração, sem nº de axiomas/relações inferidas, sem hierarquia inferida | `run_owl_reasoner` |
| L6 | Métricas de ontologia genérica incompletas: faltam `distinct_entity_ratio`, `mapping_entropy`, profundidade, especificidade | `evaluate` |
| L7 | Restrições: só `min/maxInclusive`; limites aprendidos no treino ficam com `kind="constraint"` (não distinguidos de conhecimento OWL) | `_fit_constraint_specs` |
| L8 | Relacionais limitados a diferença, delta relativo e rácio de erro | `_fit_relational_specs` |
| L9 | Sem resumo "gerado / constante / duplicado / quase duplicado / retido" | `last_engineering_stats` |
| L10 | Estabilidade sem desvio-padrão do ganho preditivo por fold | `SemanticUtilityGate.evaluate` |
| L11 | Estados de decisão incompletos: faltam `ACCEPT_SIGNIFICANT_GAIN`, `ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN`, `REJECT_UNSTABLE_FEATURES`, `REJECT_DEGRADATION`, `REJECT_INVALID_ONTOLOGY`; sem intervalo de confiança | grep de estados |
| L12 | O MLP ontológico não é otimizado separadamente com o mesmo orçamento do original | `semantic_utility_gate._model` reutiliza o estimador de referência |
| L13 | Não existe `semantic_mlp_accepted` / `semantic_trepan_available` como campos independentes | grep |
| L14 | Relatedness do TREPAN só cobre features mapeadas diretamente; features derivadas `onto_*` não herdam relação com as suas fontes | `_semantic_inputs`, `feature_relatedness_matrix` |
| L15 | Auditoria por split sem `selected_feature`, `ontology_entity`, `semantic_reason` | `_pending_semantic_decision_audit` |
| L16 | Chave de cache sem versão do pipeline semântico nem hash da configuração do gate/processor | `build_cache_key` |
| L17 | Sem relatório por feature exportável; sem controlo negativo (OWL aleatória) nem ablação por componente | — |
| L18 | Guard anti-hardcode falha em `gui/ontology_status_presenter.py`: falso positivo (parâmetro chamado `digits`) | teste `test_no_dataset_names_...` |
| L19 | Pipeline headless sem etapa de enriquecimento/MLP ontológico | secção 1 |

## 4. Duplicado / aparentemente implementado mas sem efeito

- `core/improve_surrogate.py` ↔ `counterfactuals/improve_surrogate.py` (re-export circular entre pacotes).
- `core/soft_global_tree.py` sempre lança `RuntimeError` mas ainda é testado.
- No pipeline headless, a ontologia valida-se e produz um grafo, mas **nenhuma feature `onto_*`
  chega ao MLP** (L19): o "enriquecimento" aí é só metadados para o TREPAN.
- Confirmatório V7: o Reloaded não superou o Original; V8 bloqueado por falta de OWL de
  domínio independente. Nenhuma alteração de código substitui essa evidência.

## 5. Limitações que esta fase NÃO pode resolver

- Sem ontologias de domínio reais e independentes, o ganho semântico continua limitado
  pelo conteúdo da OWL (ver `4e9db3a`).
- `PyQt6` e `tensorflow` não estão instalados neste ambiente: a GUI e o CLEAR não podem ser
  executados aqui (`BLOCKED_BY_ENVIRONMENT`); alterações de GUI só são validadas
  estaticamente e pelo apresentador (código Python puro).
