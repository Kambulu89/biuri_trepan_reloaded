"""Textos centralizados da visualização de árvores (futura internacionalização).

O idioma principal da GUI é o espanhol; ``pt`` está disponível. Nenhum texto
novo deve ficar disperso pelo código do renderer.
"""
from __future__ import annotations

LANG = "es"

_TEXT = {
    "es": {
        "leaf": "Hoja", "internal": "Nodo interno", "node": "Nodo", "edge": "Rama",
        "no_tree": "Ningún árbol disponible.", "reason": "Motivo",
        "reason_untrained": "modelo no entrenado, construcción fallida o árbol no generado",
        "legend_internal": "Nodo interno", "legend_leaf": "Hoja", "legend_semantic": "S = split semántico",
        "legend_mofn": "m/n = regla m-of-n", "legend_onto": "onto: = feature ontológica",
        "node_details": "DETALLES DEL NODO", "edge_details": "DETALLES DE LA RAMA",
        "node_id": "ID del nodo", "depth": "Profundidad", "type": "Tipo", "samples": "Muestras",
        "split": "SPLIT", "feature_full": "Feature completa", "threshold": "Umbral",
        "split_type": "Tipo de split", "simple": "simple", "m_of_n": "m-of-n", "categorical": "categórico",
        "prediction": "PREDICCIÓN", "dominant_class": "Clase dominante", "class_distribution": "Distribución de clases",
        "uncertainty_def": "Incertidumbre = entropía normalizada de la distribución de clases del nodo (0 = puro, 1 = máxima mezcla)",
        "confidence": "Confianza (prob. de la clase dominante)", "entropy": "Entropía normalizada",
        "trepan_info": "INFORMACIÓN TREPAN", "real_samples": "Muestras reales", "effective_samples": "Muestras efectivas (reales + sintéticas)", "synthetic_samples": "Muestras sintéticas",
        "queries": "Queries al oráculo", "stop_reason": "Motivo de parada",
        "reloaded_info": "INFORMACIÓN RELOADED", "base_score": "Score base (estadístico)",
        "semantic_score": "Bonus semántico", "final_score": "Score final", "ontology_entity": "Entidad ontológica",
        "semantic_reason": "Razón semántica", "semantic_features": "Features semánticas",
        "rule_path": "CAMINO DE LA REGLA", "if": "SI", "and": "Y", "then": "ENTONCES clase",
        "true_when": "VERDADERO cuando al menos {m} de las {n} condiciones se cumplen",
        "mofn_details": "Regla m-of-n", "conditions": "Condiciones",
        "parent": "Padre", "child": "Hijo", "condition": "Condición", "branch": "Rama",
        "samples_child": "Muestras que llegan al hijo", "no": "no", "yes": "sí",
        "hidden_notice": "Mostrando {shown} / {total} nodos ({hidden} ocultos por filtro/colapso)",
        "collapsed_more": "+ {n} nodos",
        "viz_error": "TREE_VISUALIZATION_ERROR",
        "selected_tree": "Árbol seleccionado",
        "search": "Buscar feature/nodo", "no_results": "Sin resultados",
        "provenance": "Procedencia", "family": "Familia ontológica",
    },
    "pt": {
        "leaf": "Folha", "internal": "Nó interno", "node": "Nó", "edge": "Ramo",
        "no_tree": "Nenhuma árvore disponível.", "reason": "Motivo",
        "reason_untrained": "modelo não treinado, construção falhou ou árvore não gerada",
        "legend_internal": "Nó interno", "legend_leaf": "Folha", "legend_semantic": "S = split semântico",
        "legend_mofn": "m/n = regra m-of-n", "legend_onto": "onto: = feature ontológica",
        "node_details": "DETALHES DO NÓ", "edge_details": "DETALHES DO RAMO",
        "node_id": "ID do nó", "depth": "Profundidade", "type": "Tipo", "samples": "Amostras",
        "split": "SPLIT", "feature_full": "Feature completa", "threshold": "Limiar",
        "split_type": "Tipo de split", "simple": "simples", "m_of_n": "m-of-n", "categorical": "categórico",
        "prediction": "PREDIÇÃO", "dominant_class": "Classe dominante", "class_distribution": "Distribuição de classes",
        "uncertainty_def": "Incerteza = entropia normalizada da distribuição de classes do nó (0 = puro, 1 = máxima mistura)",
        "confidence": "Confiança (prob. da classe dominante)", "entropy": "Entropia normalizada",
        "trepan_info": "INFORMAÇÃO TREPAN", "real_samples": "Amostras reais", "effective_samples": "Amostras efectivas (reais + sintéticas)", "synthetic_samples": "Amostras sintéticas",
        "queries": "Queries ao oráculo", "stop_reason": "Motivo de paragem",
        "reloaded_info": "INFORMAÇÃO RELOADED", "base_score": "Score base (estatístico)",
        "semantic_score": "Bónus semântico", "final_score": "Score final", "ontology_entity": "Entidade ontológica",
        "semantic_reason": "Razão semântica", "semantic_features": "Features semânticas",
        "rule_path": "CAMINHO DA REGRA", "if": "SE", "and": "E", "then": "ENTÃO classe",
        "true_when": "VERDADEIRO quando pelo menos {m} das {n} condições forem satisfeitas",
        "mofn_details": "Regra m-of-n", "conditions": "Condições",
        "parent": "Pai", "child": "Filho", "condition": "Condição", "branch": "Ramo",
        "samples_child": "Amostras que chegam ao filho", "no": "não", "yes": "sim",
        "hidden_notice": "A mostrar {shown} / {total} nós ({hidden} ocultos por filtro/colapso)",
        "collapsed_more": "+ {n} nós",
        "viz_error": "TREE_VISUALIZATION_ERROR",
        "selected_tree": "Árvore seleccionada",
        "search": "Procurar feature/nó", "no_results": "Sem resultados",
        "provenance": "Proveniência", "family": "Família ontológica",
    },
}


def tr(key: str, **kw) -> str:
    text = _TEXT.get(LANG, _TEXT["es"]).get(key) or _TEXT["es"].get(key) or key
    return text.format(**kw) if kw else text


def set_language(lang: str) -> None:
    global LANG
    LANG = lang if lang in _TEXT else "es"
