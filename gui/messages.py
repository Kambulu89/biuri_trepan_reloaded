"""Mensagens de diagnóstico estruturadas (Partes 21, 22, 39). Qt-free.

Quatro níveis: INFO, WARNING, ERROR e SCIENTIFIC_WARNING. Um aviso científico (p.ex. amostra
pequena) **não** é uma falha. Todas as mensagens ligam-se a um ``experiment_id``; o traceback
completo vai para o logging e apenas um resumo + detalhes expansíveis chegam à interface.
"""
from __future__ import annotations

import logging
import traceback
from enum import Enum
from typing import Iterable, List, Optional

from core.experiment_result import ExperimentResult, Message
from gui.strings import term, tr

logger = logging.getLogger("biuri.gui")

SMALL_SAMPLE_THRESHOLD = 30
STUMP_MAX_NODES = 3


class Level(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SCIENTIFIC_WARNING = "SCIENTIFIC_WARNING"


_LOG_LEVEL = {
    Level.INFO: logging.INFO,
    Level.WARNING: logging.WARNING,
    Level.SCIENTIFIC_WARNING: logging.WARNING,
    Level.ERROR: logging.ERROR,
}


def make_message(level: Level, code: str, key: str, experiment_id: Optional[str] = None, *, where: Optional[str] = None,
                 details: Optional[str] = None, action: Optional[str] = None, **fmt) -> Message:
    return Message(level=Level(level).value, code=code, text=tr(key, **fmt), experiment_id=experiment_id,
                   where=where, details=details, action=action)


class MessageLog:
    """Registo de mensagens da sessão (vista opcional do registo científico, Parte 38)."""

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._items: List[Message] = []

    def add(self, message: Message) -> Message:
        self._items.append(message)
        if len(self._items) > self.capacity:
            del self._items[: len(self._items) - self.capacity]
        logger.log(_LOG_LEVEL.get(Level(message.level), logging.INFO), "[%s] %s: %s",
                   message.experiment_id or "-", message.code, message.text)
        return message

    def extend(self, messages: Iterable[Message]) -> None:
        for m in messages:
            self.add(m)

    def items(self, level: Optional[Level] = None, experiment_id: Optional[str] = None) -> List[Message]:
        out = self._items
        if level is not None:
            out = [m for m in out if m.level == Level(level).value]
        if experiment_id is not None:
            out = [m for m in out if m.experiment_id == experiment_id]
        return list(out)

    def clear(self) -> None:
        self._items.clear()

    def format_lines(self, experiment_id: Optional[str] = None) -> List[str]:
        import time as _t
        lines = []
        for m in self.items(experiment_id=experiment_id):
            stamp = _t.strftime("%H:%M:%S", _t.localtime(m.timestamp))
            lines.append(f"{stamp} [{tr('level.' + m.level)}] [{m.experiment_id or '-'}] {m.text}")
        return lines


def format_error(exc: BaseException | str, *, what_key: str = "error.training", where: Optional[str] = None,
                 action_key: str = "error.action.generic", experiment_id: Optional[str] = None,
                 traceback_text: Optional[str] = None) -> Message:
    """Mensagem de erro estruturada: o que falhou, onde, detalhes expansíveis, ação sugerida.

    O traceback completo é enviado para o logging (nunca perdido); a mensagem leva-o em ``details``.
    """
    if isinstance(exc, BaseException):
        tb = traceback_text or "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        summary = f"{type(exc).__name__}: {exc}"
    else:
        tb = traceback_text or str(exc)
        summary = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
    logger.error("[%s] %s\n%s", experiment_id or "-", tr(what_key), tb)
    details = f"{summary}\n\n{tb}" if tb and tb.strip() != summary else summary
    return Message(level=Level.ERROR.value, code=what_key, text=tr(what_key), experiment_id=experiment_id,
                   where=where, details=details, action=tr(action_key))


def derive_messages(result: ExperimentResult) -> List[Message]:
    """Avisos derivados **só** do que já está no ``ExperimentResult`` (nada é recalculado)."""
    eid = result.provenance.experiment_id
    out: List[Message] = []

    sizes = [c.evaluation_samples for c in result.models.values() if c.evaluation_samples]
    if sizes and min(sizes) < SMALL_SAMPLE_THRESHOLD:
        out.append(make_message(Level.SCIENTIFIC_WARNING, "small_sample", "msg.small_sample", eid, n=min(sizes)))

    if result.benchmark is None and result.provenance.seed is not None:
        out.append(make_message(Level.SCIENTIFIC_WARNING, "single_seed", "msg.single_seed", eid, seed=result.provenance.seed))

    enr = result.enrichment
    if enr.mlp_status == "ACCEPTED" and enr.evidence_strength and str(enr.evidence_strength).lower() in ("weak", "fraca", "none"):
        out.append(make_message(Level.SCIENTIFIC_WARNING, "weak_evidence", "msg.weak_evidence", eid))
    if enr.mlp_status == "REJECTED":
        out.append(make_message(Level.INFO, "enrichment_rejected", "msg.enrichment_rejected", eid,
                                decision=enr.decision or "-"))

    for key in ("trepan_original", "trepan_reloaded"):
        diag = result.trees.get(key)
        if diag is None or not diag.available:
            continue
        nodes = diag.logical_nodes.value
        if nodes is not None and nodes <= STUMP_MAX_NODES:
            # Uma árvore pequena é um diagnóstico, nunca um "erro de árvore" (Parte 24).
            out.append(make_message(Level.INFO, "small_tree", "msg.small_tree", eid, tree=term(key), nodes=int(nodes)))
    return out
