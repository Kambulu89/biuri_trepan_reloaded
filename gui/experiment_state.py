"""Modelo de estado explícito da experiência e deteção de resultados desatualizados (Partes 3, 17, 18). Qt-free.

A máquina de estados **só** governa a interface (que ações estão disponíveis e porquê); nunca
altera nem recalcula resultados científicos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.experiment_result import ExperimentState
from gui.strings import tr

S = ExperimentState

_ORDER = [S.NO_DATA, S.DATA_LOADED, S.MODEL_TRAINED, S.SEMANTIC_VALIDATED, S.TREES_BUILT, S.RESULTS_READY]

# Transições permitidas (além de ERROR, acessível de qualquer estado, e do reset para NO_DATA).
_ALLOWED: Dict[S, Tuple[S, ...]] = {
    S.NO_DATA: (S.DATA_LOADED,),
    S.DATA_LOADED: (S.DATA_LOADED, S.MODEL_TRAINED, S.SEMANTIC_VALIDATED, S.TREES_BUILT, S.RESULTS_READY),
    S.MODEL_TRAINED: (S.DATA_LOADED, S.MODEL_TRAINED, S.SEMANTIC_VALIDATED, S.TREES_BUILT, S.RESULTS_READY),
    S.SEMANTIC_VALIDATED: (S.DATA_LOADED, S.MODEL_TRAINED, S.TREES_BUILT, S.RESULTS_READY),
    S.TREES_BUILT: (S.DATA_LOADED, S.MODEL_TRAINED, S.SEMANTIC_VALIDATED, S.RESULTS_READY),
    S.RESULTS_READY: (S.DATA_LOADED, S.MODEL_TRAINED, S.SEMANTIC_VALIDATED, S.TREES_BUILT, S.RESULTS_READY),
    S.ERROR: (S.NO_DATA, S.DATA_LOADED, S.MODEL_TRAINED, S.SEMANTIC_VALIDATED, S.TREES_BUILT, S.RESULTS_READY),
}

ACTIONS = ("load_data", "train", "explain", "visualize", "compare", "natural", "counterfactual",
           "improve", "export_results", "export_tree")


class InvalidTransition(ValueError):
    pass


def _rank(state: S) -> int:
    return _ORDER.index(state) if state in _ORDER else -1


@dataclass
class ExperimentStateMachine:
    state: S = S.NO_DATA
    busy: bool = False
    last_good: S = S.NO_DATA
    history: List[S] = field(default_factory=list)
    error_message: Optional[str] = None

    def transition(self, target: S) -> S:
        target = S(target)
        if target == S.ERROR:
            return self.fail("")
        if target not in _ALLOWED[self.state]:
            raise InvalidTransition(f"{self.state.value} -> {target.value}")
        self.history.append(self.state)
        self.state = target
        self.last_good = target
        self.error_message = None
        return self.state

    def fail(self, message: str) -> S:
        """Entra em ERROR; ``recover()`` volta ao último estado válido."""
        if self.state != S.ERROR:
            self.history.append(self.state)
            self.last_good = self.state
        self.state = S.ERROR
        self.busy = False
        self.error_message = message
        return self.state

    def recover(self) -> S:
        if self.state == S.ERROR:
            self.state = self.last_good
            self.error_message = None
        return self.state

    def reset(self) -> S:
        self.state, self.last_good, self.busy, self.error_message = S.NO_DATA, S.NO_DATA, False, None
        return self.state

    def reached(self, state: S) -> bool:
        base = self.last_good if self.state == S.ERROR else self.state
        return _rank(base) >= _rank(S(state))

    def begin(self) -> None:
        self.busy = True

    def end(self) -> None:
        self.busy = False

    # --- ações disponíveis (os botões refletem o estado real)
    def action_enabled(self, action: str, *, has_trees: Optional[bool] = None) -> Tuple[bool, str]:
        """(ativo?, motivo se desativado). ``has_trees`` permite à GUI confirmar a existência real."""
        if action not in ACTIONS:
            raise KeyError(action)
        if self.busy:
            return False, tr("action.disabled.busy")
        if action == "load_data":
            return True, ""
        if not self.reached(S.DATA_LOADED):
            return False, tr("action.disabled.NO_DATA")
        if action == "train":
            return True, ""
        if action in ("explain", "natural", "counterfactual", "improve"):
            return (True, "") if self.reached(S.MODEL_TRAINED) else (False, tr("action.disabled.need_model"))
        trees_ok = self.reached(S.TREES_BUILT) if has_trees is None else (has_trees and self.reached(S.TREES_BUILT))
        if action in ("visualize", "compare", "export_tree"):
            return (True, "") if trees_ok else (False, tr("action.disabled.need_trees"))
        if action == "export_results":
            return (True, "") if self.reached(S.RESULTS_READY) or trees_ok else (False, tr("action.disabled.need_results"))
        return False, ""

    def available_actions(self, *, has_trees: Optional[bool] = None) -> Dict[str, Tuple[bool, str]]:
        return {a: self.action_enabled(a, has_trees=has_trees) for a in ACTIONS}

    def label(self) -> str:
        text = tr(f"state.{self.state.value}")
        return f"{text} — {tr('state.busy')}" if self.busy else text


@dataclass(frozen=True)
class Fingerprint:
    """Identidade da configuração com a qual um resultado foi calculado."""

    dataset_hash: Optional[str] = None
    owl_hash: Optional[str] = None
    config_hash: Optional[str] = None
    seed: Optional[int] = None


_STALE_KEYS = (("dataset_hash", "stale.dataset_changed"), ("owl_hash", "stale.owl_changed"),
               ("config_hash", "stale.config_changed"), ("seed", "stale.seed_changed"))


class StaleTracker:
    """Compara a configuração atual com a do último resultado e indica porque está desatualizado."""

    def __init__(self):
        self.result_fp: Optional[Fingerprint] = None

    def mark_result(self, fp: Fingerprint) -> None:
        self.result_fp = fp

    def clear(self) -> None:
        self.result_fp = None

    def reasons(self, current: Fingerprint) -> List[str]:
        if self.result_fp is None:
            return []
        out = []
        for attr, key in _STALE_KEYS:
            if getattr(self.result_fp, attr) != getattr(current, attr):
                out.append(key)
        return out

    def is_stale(self, current: Fingerprint) -> bool:
        return bool(self.reasons(current))

    def banner(self, current: Fingerprint) -> str:
        keys = self.reasons(current)
        if not keys:
            return ""
        return tr("stale.banner", reasons=", ".join(tr(k) for k in keys))
