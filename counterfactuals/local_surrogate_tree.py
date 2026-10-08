"""``CounterfactualLocalTreeExplainer``: explicador local único (Local Surrogate m-of-n) partilhado por todos os oráculos.

A árvore CF NÃO é a árvore interna do modelo seleccionado. O modelo seleccionado é apenas o *oráculo* que rotula a vizinhança local
da instância (e valida os contrafactuais); a árvore é sempre induzida pelo mesmo explicador (lógica m-of-n do TREPAN histórico) com
a mesma semente, para que as fronteiras locais dos oráculos sejam comparáveis. Este módulo acrescenta:

* proveniência por construção (``oracle_id``, ``instance_id``, ``local_dataset_hash``, ``prediction_hash``, ``tree_signature``,
  identidade do objecto da árvore) e um *ledger* que prova que nenhuma árvore/previsão foi reutilizada entre oráculos;
* concordância local entre oráculos (matriz par-a-par, concordância total, discordância local);
* o resumo mostrado na interface.

Nada aqui conhece datasets: só valores numéricos, hashes e nomes de modelos.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from counterfactuals.cf_tree import build_local_surrogate_tree

EXPLAINER_NAME = "CounterfactualLocalTreeExplainer"
BUILDER_LABEL = "Local Surrogate m-of-n"
TREE_TITLE_PREFIX = "Árvore CF local — Oráculo: "
# Oráculos comparados na matriz de concordância (nomes de modelos do painel; não são datasets).
CONCORDANCE_ORACLES = ("MLP Original", "C4.5-Nativo", "Trepan Original", "Trepan Reloaded")


class LocalTreeReuseError(RuntimeError):
    """Uma árvore, previsão ou oráculo foi reutilizado indevidamente entre construções."""


def tree_title(oracle_name: str) -> str:
    return f"{TREE_TITLE_PREFIX}{oracle_name}"


class LocalTreeLedger:
    """Regista a proveniência de cada árvore local e verifica que nada foi reutilizado entre oráculos.

    Mantém referências fortes às últimas ``capacity`` árvores para que ``id()`` nunca seja reciclado enquanto o registo existir.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = int(capacity)
        self._records: List[Dict[str, Any]] = []
        self._trees: List[Any] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._records]

    def record(self, provenance: Mapping[str, Any], tree: Any, oracle: Any) -> None:
        """Verifica e regista. Levanta ``LocalTreeReuseError`` se o objecto-árvore ou o seu token já tinham sido registados."""
        with self._lock:
            token, obj_id = provenance.get("tree_build_token"), provenance.get("tree_object_id")
            if getattr(tree, "_cf_build_token", None) != token:
                raise LocalTreeReuseError("O token de construção da árvore não corresponde à proveniência registada.")
            if id(tree) != obj_id:
                raise LocalTreeReuseError("A identidade do objecto-árvore não corresponde à proveniência registada.")
            for old, old_tree in zip(self._records, self._trees):
                if old_tree is tree or old["tree_build_token"] == token or old["tree_object_id"] == obj_id:
                    raise LocalTreeReuseError(
                        f"A árvore construída para '{old['oracle_name']}' foi reutilizada para "
                        f"'{provenance.get('oracle_name')}' (mesmo objecto/token).")
            entry = dict(provenance)
            entry.pop("neighbor_index", None)
            entry["oracle_object_id"] = id(oracle)
            self._records.append(entry)
            self._trees.append(tree)
            if len(self._records) > self.capacity:
                del self._records[0], self._trees[0]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._trees.clear()


DEFAULT_LEDGER = LocalTreeLedger()


def explain_equality(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Demonstra, a partir da proveniência, PORQUÊ duas árvores locais coincidem (ou não).

    Duas árvores iguais são legítimas quando o conjunto local, os rótulos e as regras coincidem e os objectos são distintos
    (nada de cache/reutilização). Aceita resultados completos ou apenas o dicionário ``local_provenance``.
    """
    pa = dict(a.get("local_provenance") or a)
    pb = dict(b.get("local_provenance") or b)
    same_rules = pa.get("tree_signature") == pb.get("tree_signature")
    same_labels = pa.get("prediction_hash") == pb.get("prediction_hash")
    same_local = pa.get("local_dataset_hash") == pb.get("local_dataset_hash")
    distinct_objects = (pa.get("tree_object_id") != pb.get("tree_object_id")
                        and pa.get("tree_build_token") != pb.get("tree_build_token"))
    if same_rules and same_labels and same_local and distinct_objects:
        verdict = "iguais_por_mesmos_rotulos_e_regras_locais"
    elif same_rules and not same_labels:
        verdict = "regras_iguais_com_rotulos_diferentes"
    elif same_rules and not distinct_objects:
        verdict = "SUSPEITO_mesmo_objecto_ou_token"
    else:
        verdict = "arvores_diferentes"
    return {"same_local_dataset": same_local, "same_prediction_hash": same_labels, "same_tree_signature": same_rules,
            "distinct_tree_objects": distinct_objects, "same_oracle_id": pa.get("oracle_id") == pb.get("oracle_id"),
            "verdict": verdict}


def local_oracle_concordance(predictions: Mapping[str, Any]) -> Dict[str, Any]:
    """Concordância entre oráculos sobre os MESMOS pontos locais.

    ``predictions``: ``{nome: previsões}`` (mesmo comprimento). Devolve a matriz par-a-par (fracção de pontos com a mesma previsão),
    a concordância total (todos os oráculos concordam) e a discordância local (complemento).
    """
    names = [str(n) for n in predictions]
    labels = {n: np.asarray(predictions[n]).astype(str).reshape(-1) for n in names}
    sizes = {len(v) for v in labels.values()}
    if len(sizes) > 1:
        raise ValueError("Os oráculos devem prever exactamente os mesmos pontos locais.")
    n_points = sizes.pop() if sizes else 0
    matrix = [[float(np.mean(labels[a] == labels[b])) if n_points else float("nan") for b in names] for a in names]
    if names and n_points:
        stacked = np.vstack([labels[n] for n in names])
        all_agree = float(np.mean(np.all(stacked == stacked[0], axis=0)))
    else:
        all_agree = float("nan")
    return {
        "oracles": names, "n_points": int(n_points), "pairwise_agreement": matrix,
        "total_agreement": all_agree,
        "local_disagreement": (1.0 - all_agree) if n_points and names else float("nan"),
        "prediction_hash": {n: hashlib.sha256(labels[n].astype(str).tobytes()).hexdigest() for n in names},
    }


def collect_oracle_predictions(session: Mapping[str, Any], options: Mapping[str, Any], neighbor_index: Sequence[int],
                               n_reference_rows: int, resolve_context) -> Dict[str, Any]:
    """Previsões de cada oráculo disponível nos pontos locais (linhas ``neighbor_index`` do espaço de referência).

    ``resolve_context(session, options)`` devolve o contexto do modelo-alvo. Oráculos indisponíveis, com número de linhas
    diferente do espaço de referência ou que falhem são excluídos com razão explícita (nunca se inventa alinhamento).
    """
    idx = np.asarray(list(neighbor_index), dtype=int)
    predictions: Dict[str, Any] = {}
    excluded: Dict[str, str] = {}
    for name in CONCORDANCE_ORACLES:
        try:
            ctx = resolve_context(dict(session), {**dict(options), "target_model": name})
            X = np.asarray(ctx["X"], dtype=float)
            if len(X) != int(n_reference_rows):
                excluded[name] = f"linhas diferentes do espaço de referência ({len(X)} != {n_reference_rows})"
                continue
            predictions[name] = np.asarray(ctx["oracle"].predict(X[idx])).reshape(-1)
        except Exception as exc:                      # um oráculo em falta não impede os restantes
            excluded[name] = f"{type(exc).__name__}: {exc}"
    return {"predictions": predictions, "excluded": excluded}


class CounterfactualLocalTreeExplainer:
    """Explicador local único: mesma lógica m-of-n, mesma semente, qualquer oráculo."""

    name = EXPLAINER_NAME
    builder = BUILDER_LABEL

    def __init__(self, *, max_depth: int = 5, neighborhood_size: int = 300, seed: int = 42,
                 ledger: Optional[LocalTreeLedger] = None):
        self.max_depth, self.neighborhood_size, self.seed = int(max_depth), int(neighborhood_size), int(seed)
        self.ledger = DEFAULT_LEDGER if ledger is None else ledger

    def explain(self, oracle: Any, X_reference: Any, generation_result: Mapping[str, Any], feature_names: Sequence[str], *,
                original_tree: Any = None, class_labels: Optional[Mapping[Any, Any]] = None, oracle_name: str = "modelo",
                dataset_name: str = "dataset_carregado", concordance_provider=None) -> Dict[str, Any]:
        """``concordance_provider(neighbor_index) -> {"predictions": {...}, "excluded": {...}}`` (opcional) fornece as previsões
        dos restantes oráculos nos MESMOS pontos locais."""
        result = build_local_surrogate_tree(
            oracle, X_reference, generation_result, feature_names, original_tree=original_tree, class_labels=class_labels,
            max_depth=self.max_depth, neighborhood_size=self.neighborhood_size, seed=self.seed, model_name=oracle_name,
            dataset_name=dataset_name)
        prov = result["local_provenance"]
        self.ledger.record(prov, result["_runtime_tree_model"], oracle)
        prov["ledger_checked"] = True
        if concordance_provider is not None:
            collected = concordance_provider(prov["neighbor_index"])
            preds = collected.get("predictions") or {}
            concordance = local_oracle_concordance(preds) if preds else {"oracles": [], "n_points": 0}
            if collected.get("excluded"):
                concordance["excluded"] = dict(collected["excluded"])
            result["local_concordance"] = concordance
        result["oracle_name"] = str(oracle_name)
        result["tree_title"] = tree_title(oracle_name)
        result["ui_summary"] = ui_summary(result)
        return result


def _pct(value: Any) -> str:
    try:
        return "n/d" if value is None or not np.isfinite(float(value)) else f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/d"


def ui_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Campos mostrados na interface: oráculo, construtor, instância, amostras locais, contrafactuais e fidelidade local."""
    prov = result.get("local_provenance") or {}
    metrics = result.get("aggregate_metrics") or {}
    inst = prov.get("instance_id") or {}
    return {
        "oracle": str(result.get("oracle_name") or result.get("target_model") or prov.get("oracle_name") or "?"),
        "builder": BUILDER_LABEL,
        "instance": inst.get("index"),
        "instance_hash": str(inst.get("hash", ""))[:12],
        "n_local_samples": prov.get("n_local_samples"),
        "n_counterfactuals": prov.get("n_counterfactuals"),
        "local_fidelity": metrics.get("fidelity_to_oracle"),
        "evaluation_mode": metrics.get("evaluation_mode"),
    }


def summary_lines(result: Mapping[str, Any]) -> List[str]:
    s = result.get("ui_summary") or ui_summary(result)
    instance = "?" if s["instance"] is None else str(s["instance"])
    return [
        f"Oráculo: {s['oracle']}",
        f"Construtor da árvore CF: {s['builder']}",
        f"Instância analisada: {instance} (hash {s['instance_hash']})",
        f"Amostras locais: {s['n_local_samples']}",
        f"Contrafactuais válidos: {s['n_counterfactuals']}",
        f"Fidelidade local ao oráculo: {_pct(s['local_fidelity'])} ({s['evaluation_mode']})",
    ]


def concordance_lines(concordance: Optional[Mapping[str, Any]]) -> List[str]:
    """Matriz de concordância em texto monoespaçado + concordância total e discordância local."""
    if not concordance or not concordance.get("oracles"):
        reason = (concordance or {}).get("excluded")
        return ["Concordância entre oráculos: indisponível" + (f" ({reason})" if reason else "")]
    names = concordance["oracles"]
    width = max(len(n) for n in names) + 2
    lines = [f"Concordância local entre oráculos ({concordance['n_points']} pontos)",
             " " * width + "".join(f"{n[:12]:>14}" for n in names)]
    for name, row in zip(names, concordance["pairwise_agreement"]):
        lines.append(f"{name:<{width}}" + "".join(f"{_pct(v):>14}" for v in row))
    lines.append(f"Concordância total (todos iguais): {_pct(concordance['total_agreement'])}")
    lines.append(f"Discordância local: {_pct(concordance['local_disagreement'])}")
    for name, why in (concordance.get("excluded") or {}).items():
        lines.append(f"Excluído da matriz: {name} — {why}")
    return lines


__all__ = [
    "BUILDER_LABEL", "CONCORDANCE_ORACLES", "CounterfactualLocalTreeExplainer", "DEFAULT_LEDGER", "EXPLAINER_NAME",
    "LocalTreeLedger", "LocalTreeReuseError", "collect_oracle_predictions", "concordance_lines", "explain_equality",
    "local_oracle_concordance", "summary_lines", "tree_title", "ui_summary",
]
