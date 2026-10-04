"""Camada de strings da interface (i18n) e terminologia fixa.

Idioma principal do produto: **português (pt-PT)**. Existe um catálogo ``en`` para os textos
novos; quando falta uma chave no idioma pedido usa-se pt e, em último caso, a própria chave
(nunca uma exceção). Os termos técnicos abaixo são escritos **sempre** da mesma forma, em
qualquer idioma, e nunca devem ser hardcoded noutros módulos: usar ``TERMS``/``tr``.

Nenhum texto científico novo deve ser espalhado pela GUI: tudo passa por aqui.
"""
from __future__ import annotations

from typing import Dict

DEFAULT_LANGUAGE = "pt"
SUPPORTED_LANGUAGES = ("pt", "en")
_language = DEFAULT_LANGUAGE

# Terminologia canónica (Parte 33): não alternar entre variantes.
TERMS: Dict[str, str] = {
    "mlp_original": "MLP Original",
    "mlp_ontological": "MLP Ontológico",
    "c45": "C4.5",
    "trepan_original": "TREPAN Original",
    "trepan_reloaded": "TREPAN Reloaded",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "macro_f1": "Macro-F1",
    "precision_macro": "Precision Macro",
    "recall_macro": "Recall Macro",
    "fidelity": "Fidelity",
    "oracle": "Oracle",
    "ontology": "Ontologia",
}

# Variantes proibidas na interface (usadas por um teste de consistência terminológica).
FORBIDDEN_VARIANTS = (
    "Exactitud", "Exatidão", "Precisión", "Precisão Macro", "Fidelidad", "Fidelidade", "Oráculo", "Oraculo",
    "Trepan-Original", "Trepan-Reloaded", "Trepan Reloaded", "TREPAN-Reloaded", "C4.5-Nativo", "C4.5 Nativo",
    "MLP Ontologico", "MLP Onto ", "Ontología",
)

_CATALOG: Dict[str, Dict[str, str]] = {
    "pt": {
        # --- estados do experimento
        "state.NO_DATA": "Sem dados", "state.DATA_LOADED": "Dados carregados", "state.MODEL_TRAINED": "Modelo treinado",
        "state.SEMANTIC_VALIDATED": "Semântica validada", "state.TREES_BUILT": "Árvores construídas",
        "state.RESULTS_READY": "Resultados prontos", "state.ERROR": "Erro", "state.busy": "A executar…",
        # --- ações (botões)
        "action.load_data": "Carregar dados", "action.train": "Treinar modelo", "action.explain": "Gerar explicação",
        "action.visualize": "Visualizar árvore", "action.compare": "Comparar métricas",
        "action.natural": "Explicações em linguagem natural", "action.counterfactual": "Contrafactuais",
        "action.improve": "Melhorar árvore substituta",
        "action.export_results": "Exportar resultados (dados)", "action.export_tree": "Exportar árvore (imagem)",
        "action.disabled.NO_DATA": "Carregue um dataset primeiro.",
        "action.disabled.need_model": "Treine o modelo primeiro.",
        "action.disabled.need_trees": "Construa as árvores primeiro (treine o modelo).",
        "action.disabled.need_results": "Ainda não há resultados para exportar.",
        "action.disabled.busy": "Aguarde: há uma operação em curso.",
        # --- secções
        "section.experiment": "Experiência", "section.dataset": "Dataset", "section.ontology": "Ontologia",
        "section.enrichment": "Enriquecimento semântico", "section.models": "Modelos", "section.trees": "Árvores",
        "section.diagnostics": "Diagnóstico das árvores", "section.semantic_features": "Features semânticas",
        "section.semantic_splits": "Splits semânticos (TREPAN Reloaded)", "section.controls": "Controlo negativo",
        "section.ablation": "Ablação", "section.benchmark": "Benchmark", "section.log": "Registo científico",
        "section.metrics.predictive": "Desempenho preditivo (vs rótulos reais)",
        "section.metrics.fidelity": "Surrogate Fidelity (vs Oracle)", "section.metrics.complexity": "Complexidade",
        "tab.audit": "Auditoria", "tab.summary": "Resumo", "tab.ontology": "Ontologia", "tab.models": "Modelos",
        "tab.trees": "Árvores", "tab.semantics": "Semântica", "tab.experiment": "Experiência", "tab.log": "Registo",
        "mode.basic": "Modo básico", "mode.scientific": "Modo científico",
        # --- etiquetas de campos
        "field.build": "Build", "field.commit": "Commit", "field.semantic_pipeline": "Pipeline semântico",
        "field.dataset": "Dataset", "field.rows": "Linhas", "field.features": "Features", "field.classes": "Classes",
        "field.train": "Treino", "field.test": "Teste", "field.seed": "Seed", "field.file": "Ficheiro",
        "field.structural": "Estado estrutural", "field.reasoner": "Reasoner", "field.mapping": "Mapeamento",
        "field.coverage": "Cobertura", "field.ambiguous": "Ambíguos", "field.tbox": "TBox", "field.abox": "ABox",
        "field.richness": "Riqueza semântica", "field.ontology_validity": "Validade da ontologia",
        "field.mlp_enrichment": "Enriquecimento do MLP", "field.trepan_semantics": "Semântica no TREPAN",
        "field.reason": "Motivo", "field.base_utility": "Utilidade base", "field.owl_utility": "Utilidade com OWL",
        "field.delta": "Δ utilidade", "field.generated": "Geradas", "field.stable": "Estáveis",
        "field.selected": "Selecionadas", "field.oracle": "Oracle", "field.status": "Estado", "field.cached": "Em cache",
        "field.nodes": "Nós", "field.depth": "Profundidade", "field.leaves": "Folhas", "field.queries": "Queries",
        "field.experiment_id": "ID da experiência", "field.dataset_hash": "Hash do dataset", "field.owl_hash": "Hash da OWL",
        "field.timestamp": "Data/hora", "field.config_hash": "Hash da configuração", "field.cache": "Cache",
        "field.teacher": "Professor (Oracle) do TREPAN", "field.reloaded_mode": "Modo do Reloaded",
        "field.logical_nodes": "Nós lógicos", "field.rendered_nodes": "Nós desenhados",
        "field.nodes_before": "Nós antes da poda", "field.nodes_after": "Nós depois da poda",
        "field.query_budget": "Orçamento de queries", "field.budget_exhausted": "Orçamento esgotado",
        "field.node_budget": "Orçamento de nós", "field.loop_end": "Motivo de paragem global",
        "field.stop_reasons": "Motivos de paragem por nó", "field.m_of_n": "Splits m-of-n",
        "field.hyperparameters": "Hiperparâmetros", "field.eval_samples": "Amostras de avaliação",
        "field.agreement": "Concordância com o MLP (diagnóstico)",
        # --- estados/valores
        "status.VALID": "VÁLIDA", "status.CONSISTENT": "CONSISTENTE", "status.INCONSISTENT": "INCONSISTENTE",
        "status.NOT_EXECUTED": "NÃO EXECUTADO", "status.SAFE": "SEGURA", "status.INVALID": "INVÁLIDA",
        "status.NOT_EVALUATED": "NÃO AVALIADO", "status.ACCEPTED": "ACEITE", "status.REJECTED": "REJEITADO",
        "status.NOT_AVAILABLE": "INDISPONÍVEL", "status.AVAILABLE": "DISPONÍVEL",
        "status.yes": "Sim", "status.no": "Não", "status.stale": "DESATUALIZADO", "status.cached": "RESULTADO EM CACHE",
        "status.fresh": "Calculado neste treino",
        # --- NA com significado (Parte 9)
        "na.not_calculated": "Não calculado", "na.not_applicable": "Não aplicável", "na.not_available": "Indisponível",
        "na.not_executed": "Não executado",
        "reason.teacher_rejected": "professor semântico rejeitado",
        "reason.no_oracle": "este modelo não tem Oracle",
        "reason.no_ontology": "ontologia não carregada",
        "reason.tree_not_built": "árvore não construída",
        "reason.not_rendered": "árvore ainda não desenhada",
        "reason.not_trained": "modelo não treinado",
        "reason.comparison_not_run": "comparação de métricas ainda não executada",
        "reason.enrichment_not_evaluated": "enriquecimento não avaliado",
        "reason.not_reported_by_backend": "o backend não reportou este valor",
        "reason.no_semantic_features": "nenhuma feature semântica gerada",
        # --- explicações humanas das decisões do enriquecimento (Parte 8)
        "decision.REJECT_NO_INFORMATIONAL_GAIN":
            "O conjunto de features semânticas não melhorou a utilidade OOF em relação ao MLP base.",
        "decision.REJECT_DEGRADATION": "As features semânticas pioraram a utilidade OOF face ao MLP base.",
        "decision.REJECT_UNSTABLE_FEATURES": "Nenhuma feature semântica foi selecionada de forma estável entre folds.",
        "decision.REJECT_NO_NOVEL_FEATURES": "Todas as features semânticas eram constantes, duplicadas ou redundantes.",
        "decision.REJECT_INVALID_ONTOLOGY": "A ontologia não passou o quality gate estrutural.",
        "decision.REJECT_LEAKAGE_RISK": "A ABox contém instâncias do dataset ou do teste (risco de leakage).",
        "decision.ACCEPT_SIGNIFICANT_GAIN": "A utilidade OOF melhorou de forma significativa (IC acima de zero).",
        "decision.ACCEPT_PARTIAL_FEATURE_SET":
            "Um subconjunto estável de features semânticas melhorou a utilidade OOF.",
        "decision.ACCEPT_NON_INFERIOR_WITH_SECONDARY_GAIN":
            "Não-inferior ao MLP base, com ganho numa métrica secundária (evidência fraca).",
        "decision.ONTOLOGY_VALID_BUT_NO_PREDICTIVE_UTILITY":
            "A ontologia é válida, mas as features semânticas não mostraram utilidade preditiva validada em validação interna.",
        "decision.NOT_EVALUATED": "O enriquecimento não foi avaliado nesta experiência.",
        "decision.unknown": "Decisão técnica: {code}.",
        "enrichment.nothing_to_show": "Sem relatório de enriquecimento para esta experiência.",
        # --- trepan
        "stop.node_budget_exhausted": "orçamento de nós esgotado",
        "stop.no_expandable_nodes_left": "não restavam nós expansíveis",
        "stop.STOP_PURE_NODE": "nó puro (uma só classe do Oracle)", "stop.STOP_MAX_DEPTH": "profundidade máxima",
        "stop.STOP_MIN_SAMPLES": "amostra mínima não atingida", "stop.STOP_NO_VALID_SPLIT": "nenhum split válido",
        "stop.STOP_MIN_GAIN": "ganho de informação abaixo do mínimo",
        "stop.STOP_QUERY_BUDGET_EXHAUSTED": "orçamento de queries esgotado", "stop.STOP_MAX_NODES": "orçamento de nós esgotado",
        "stop.STOP_QUERY_GENERATION_FAILURE": "falha na geração de queries", "stop.STOP_NUMERICAL_FAILURE": "falha numérica",
        "stop.STOP_PRUNED": "removido pela poda", "stop.STOP_UNRECORDED": "motivo não registado",
        "stop.pure_node": "nó puro (uma só classe do Oracle)", "stop.no_valid_split": "nenhum split válido",
        "stop.min_samples_leaf": "split rejeitado (folha mínima)", "stop.max_depth": "profundidade máxima",
        "stop.query_budget_before_min_sample": "orçamento de queries insuficiente para o nó",
        "tree.small_diagnostic": "Árvore pequena: diagnóstico disponível (ver Diagnóstico das árvores).",
        "tree.oracle_line": "Oracle: {oracle}",
        # --- métricas: ajuda curta (Parte 31)
        "help.accuracy": "Accuracy: concordância com os rótulos reais.",
        "help.balanced_accuracy": "Balanced Accuracy: média do recall por classe; não favorece a classe maioritária.",
        "help.macro_f1": "Macro-F1: média do F1 de cada classe, com o mesmo peso para todas.",
        "help.precision_macro": "Precision Macro: média da precision por classe.",
        "help.recall_macro": "Recall Macro: média do recall por classe.",
        "help.fidelity": "Fidelity: concordância com o Oracle (não com os rótulos reais).",
        "help.agreement": "Concordância com o MLP: diagnóstico; não é a fidelity principal (o C4.5 não tem Oracle).",
        "help.nodes": "Nós lógicos da árvore depois da poda.", "help.depth": "Profundidade máxima da árvore.",
        "help.queries": "Consultas ao Oracle usadas na extração.",
        # --- métricas: cabeçalhos
        "metrics.model": "Modelo", "metrics.oracle_used": "Oracle usado", "metrics.metric": "Métrica",
        "metrics.value": "Valor", "metrics.fidelity_vs": "Fidelity vs",
        # --- proveniência
        "prov.title": "Proveniência", "prov.model": "Modelo", "prov.dataset": "Dataset", "prov.seed": "Seed",
        "prov.split": "Split", "prov.oracle": "Oracle", "prov.build": "Build", "prov.metric": "Métrica",
        "prov.evaluated_on": "Avaliado em",
        # --- stale / cache
        "stale.banner": "RESULTADOS DESATUALIZADOS — {reasons}. Treine de novo para os atualizar.",
        "stale.dataset_changed": "o dataset mudou", "stale.owl_changed": "a ontologia mudou",
        "stale.config_changed": "a configuração mudou", "stale.seed_changed": "a seed mudou",
        "cache.banner": "RESULTADO EM CACHE (chave {key})",
        "cache.fresh": "Calculado neste treino",
        # --- mensagens
        "level.INFO": "INFO", "level.WARNING": "AVISO", "level.ERROR": "ERRO", "level.SCIENTIFIC_WARNING": "AVISO CIENTÍFICO",
        "msg.dataset_loaded": "Dataset carregado: {name} ({rows} linhas, {features} features).",
        "msg.ontology_loaded": "Ontologia carregada: {name}.",
        "msg.ontology_validated": "Ontologia validada: {status}.",
        "msg.enrichment_rejected": "Enriquecimento semântico rejeitado: {decision}.",
        "msg.enrichment_accepted": "Enriquecimento semântico aceite: {decision}.",
        "msg.trees_built": "Árvores construídas (TREPAN Original e TREPAN Reloaded).",
        "msg.training_started": "Treino iniciado (experiência {id}).",
        "msg.training_finished": "Treino concluído (experiência {id}).",
        "msg.cache_hit": "Modelo recuperado da cache (chave {key}).",
        "msg.small_sample": "Amostra de avaliação pequena ({n} exemplos). Os resultados são indicativos.",
        "msg.single_seed": "Resultado de uma única seed ({seed}). Não mede variabilidade.",
        "msg.weak_evidence": "Enriquecimento aceite com evidência fraca (IC da utilidade inclui zero).",
        "msg.small_tree": "{tree}: árvore com {nodes} nós. Abra o diagnóstico para ver a razão.",
        "msg.budget_limited": "{tree}: a árvore parou por esgotamento do orçamento de queries ({used}/{budget}) com apenas {nodes} nós; "
                              "não reflete a capacidade do método. Aumente o orçamento de queries ou reduza a amostra mínima por nó.",
        "msg.surrogate_above_oracle": "{tree}: accuracy superior à do MLP Original ({tree_acc} vs {mlp_acc}, +{diff} pp, cerca de {n_samples} de {n_total} "
                                      "amostras de teste). O substituto é avaliado contra os rótulos reais; a diferença é pequena e não é evidência de que o "
                                      "substituto seja melhor que o MLP. Convém ver a fidelity ao Oracle.",
        "msg.tuning_unstable": "Tuning da estrutura do TREPAN instável: a configuração vencedora muda entre seeds da validação cruzada "
                               "(concordância {agreement}, limiar {threshold}). A escolha é pouco suportada pelos dados.",
        "msg.tuning_failed": "O tuning da estrutura do TREPAN falhou ({reason}); foi usada a configuração canónica.",
        "field.trepan_tuning": "Estrutura do TREPAN (tuning)",
        "msg.stale": "Resultados anteriores marcados como desatualizados: {reasons}.",
        "msg.cancelled": "Operação cancelada pelo utilizador.",
        # --- erros
        "error.title": "Ocorreu um erro", "error.what": "O que falhou", "error.where": "Onde",
        "error.details": "Detalhes técnicos", "error.action": "Ação sugerida", "error.id": "ID da experiência",
        "error.training": "O treino do modelo falhou.", "error.comparison": "A comparação de métricas falhou.",
        "error.counterfactual": "A geração de contrafactuais falhou.", "error.load_data": "Não foi possível carregar o dataset.",
        "error.load_ontology": "Não foi possível carregar a ontologia.", "error.export": "A exportação falhou.",
        "error.action.training": "Verifique o dataset, a coluna alvo e a configuração; consulte o registo.",
        "error.action.comparison": "Verifique que o modelo está treinado e que os dados são válidos.",
        "error.action.generic": "Consulte os detalhes técnicos e o registo (ID da experiência acima).",
        "error.log_hint": "O traceback completo foi guardado no registo.",
        # --- progresso (Parte 20)
        "progress.training_mlp": "A treinar o MLP…", "progress.building_original": "A construir o TREPAN Original…",
        "progress.building_reloaded": "A construir o TREPAN Reloaded…", "progress.metrics": "A calcular métricas…",
        "progress.reasoner": "A executar o reasoner…", "progress.counterfactual": "A gerar contrafactuais…",
        "progress.working": "A trabalhar…",
        # --- exportação
        "export.results_done": "Resultados exportados para {path}.", "export.tree_done": "Árvore exportada para {path}.",
        "export.stale_warning": "Atenção: estes resultados estão marcados como desatualizados.",
        # --- vários
        "misc.not_loaded": "não carregada", "misc.none": "—", "misc.selected_only": "Só selecionadas",
        # --- colunas de tabelas / linhas de resumo (apresentador)
        "col.name": "Nome", "col.type": "Tipo", "col.source": "Origem", "col.stability": "Estabilidade",
        "col.selected": "Selecionada", "col.reason": "Motivo", "col.node": "Nó", "col.feature": "Feature",
        "col.base_score": "Score base", "col.semantic_bonus": "Bónus semântico", "col.final_score": "Score final",
        "col.ontology_reason": "Razão ontológica", "col.configuration": "Configuração",
        "col.semantic_contribution": "Contribuição semântica", "col.arm": "Braço", "col.runs": "Execuções",
        "col.mean_std": "Média ± desvio", "col.metric": "Métrica", "col.value": "Valor",
        "arm.no_semantics": "Sem semântica", "arm.real_owl": "OWL real", "arm.shuffled_owl": "OWL baralhada",
        "arm.random_control": "Controlo aleatório (features derivadas aleatórias)",
        "summary.dataset": "Dataset ativo", "summary.owl": "OWL carregada", "summary.build": "Build",
        "summary.mlp": "MLP selecionado", "summary.ontology_valid": "Ontologia validada?",
        "summary.enrichment": "Enriquecimento aceite?", "summary.oracle_original": "Oracle do TREPAN Original",
        "summary.oracle_reloaded": "Oracle do TREPAN Reloaded", "summary.nodes": "Nós (Original / Reloaded)",
        "summary.stop": "Porque parou a árvore", "summary.queries": "Queries (Original / Reloaded)",
        "summary.accuracy": "Accuracy (Original / Reloaded)", "summary.fidelity": "Fidelity (Original / Reloaded)",
        "summary.owl_features": "Features OWL selecionadas", "summary.semantic_splits": "Splits semânticos",
        "summary.cache": "Cache", "summary.seed_config": "Seed / configuração",
        "summary.fidelity_against": "vs Oracle", "summary.no_result": "Sem resultados: treine o modelo.",
        "value.mlp_original_only": "MLP Original", "value.none_selected": "nenhuma",
        "value.not_loaded": "Não carregada", "value.stump": "árvore pequena",
        "field.semantic_splits_count": "Splits semânticos", "field.semantic_score_total": "Score semântico total (soma dos bónus reportados)",
        "field.tuning": "Tuning",
        "mode.hint": "O modo só muda o que é mostrado; nunca os cálculos.",
        "progress.training_onto": "A treinar o MLP Ontológico…", "progress.ontology": "A validar a ontologia…",
        "progress.preparing": "A preparar a experiência…", "progress.auditing": "A calcular métricas e auditoria…",
        "progress.finished": "Concluído.", "progress.cancelling": "A cancelar…",
        "progress.title.training": "Treino BIURI", "progress.title.metrics": "Comparação de métricas",
        "progress.cancel": "Cancelar",
        "export.choose_dir": "Escolha a pasta para guardar os resultados",
        "export.tree_title": "Exportar árvore (imagem)", "export.tree_filter": "Imagem PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        "export.no_tree": "Ainda não há árvore para exportar.",
        "msg.state_changed": "Estado: {state}.",
        "msg.data_loaded_state": "Dataset carregado; resultados anteriores ficam marcados como desatualizados.",
        "msg.metrics_finished": "Comparação de métricas concluída (experiência {id}).",
        "error.metrics_where": "Comparação de métricas", "error.training_where": "Treino do pipeline",
        "error.copy_hint": "Copie o ID da experiência ao reportar o problema.",
        "action.hint.state": "Estado atual: {state}.",
        "misc.show_log": "Mostrar registo", "misc.copy_id": "Copiar ID",
    },
    "en": {
        "state.NO_DATA": "No data", "state.DATA_LOADED": "Data loaded", "state.MODEL_TRAINED": "Model trained",
        "state.SEMANTIC_VALIDATED": "Semantics validated", "state.TREES_BUILT": "Trees built",
        "state.RESULTS_READY": "Results ready", "state.ERROR": "Error", "state.busy": "Running…",
        "action.load_data": "Load data", "action.train": "Train model", "action.explain": "Generate explanation",
        "action.visualize": "Visualize tree", "action.compare": "Compare metrics",
        "action.export_results": "Export results (data)", "action.export_tree": "Export tree (image)",
        "section.experiment": "Experiment", "section.dataset": "Dataset", "section.ontology": "Ontology",
        "stale.banner": "STALE RESULTS — {reasons}. Retrain to refresh them.",
        "na.not_calculated": "Not calculated", "na.not_applicable": "Not applicable", "na.not_available": "Not available",
        "na.not_executed": "Not executed",
        "reason.teacher_rejected": "semantic teacher rejected", "reason.no_oracle": "this model has no oracle",
        "reason.no_ontology": "ontology not loaded", "reason.tree_not_built": "tree not built",
        "decision.REJECT_NO_INFORMATIONAL_GAIN":
            "The semantic feature set did not improve the OOF utility over the base MLP.",
        "help.accuracy": "Accuracy: agreement with the real labels.",
        "help.fidelity": "Fidelity: agreement with the oracle (not with the real labels).",
        "msg.small_sample": "Small evaluation sample ({n} examples). Results are indicative.",
    },
}


def set_language(lang: str) -> None:
    global _language
    _language = lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def get_language() -> str:
    return _language


def tr(key: str, /, lang: str | None = None, **fmt) -> str:
    """Texto traduzido; cai para pt e depois para a chave. Nunca lança exceção."""
    language = lang or _language
    text = _CATALOG.get(language, {}).get(key)
    if text is None:
        text = _CATALOG[DEFAULT_LANGUAGE].get(key, key)
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def term(key: str) -> str:
    """Nome canónico de um modelo/métrica/conceito (terminologia fixa)."""
    return TERMS.get(key, key)


def catalog_keys(lang: str = DEFAULT_LANGUAGE):
    return set(_CATALOG[lang])
