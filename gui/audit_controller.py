"""Controlador da camada de auditoria da GUI (Partes 3, 17-22, 37, 39).

Liga o ``BiuriApp`` ao modelo de estado, à deteção de resultados desatualizados, ao registo de
mensagens e ao painel de auditoria. **Nunca** calcula nem altera resultados científicos: apenas
lê os atributos já preenchidos pelo pipeline (via ``result_builder``) e reflete o estado real
nos botões. Pode ser usado sem Qt (``panel`` e botões opcionais), o que torna o fluxo testável.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional

from core.experiment_result import ExperimentResult, ExperimentState as S
from gui.experiment_state import ExperimentStateMachine, InvalidTransition, StaleTracker
from gui.messages import Level, MessageLog, derive_messages, format_error, make_message
from gui.result_builder import build_experiment_result, current_fingerprint
from gui.strings import tr

# ação lógica -> nome do atributo botão no BiuriApp
BUTTON_ATTRS = {
    "load_data": "btn_load_data", "train": "btn_train_model", "explain": "btn_generate_explanation",
    "visualize": "btn_visualize_tree", "compare": "btn_compare_metrics", "natural": "btn_natural_explanations",
    "counterfactual": "btn_generate_counterfactuals", "improve": "btn_improve_surrogate", "export_results": "btn_export_results",
    "export_tree": "btn_export_tree",
}

PROGRESS_KEYS = {
    "init": "progress.preparing", "mlp": "progress.training_mlp", "onto_gate": "progress.ontology",
    "onto": "progress.ontology", "mlp_residual": "progress.training_onto", "trepan_original": "progress.building_original",
    "trepan_reloaded": "progress.building_reloaded", "audit": "progress.auditing", "compare": "progress.metrics",
    "report": "progress.metrics", "load": "progress.preparing", "split": "progress.preparing", "encode": "progress.preparing",
    "done": "progress.finished",
}


def progress_text(stage: Optional[str]) -> str:
    """Texto de progresso a partir da *etapa* (nunca de uma percentagem inventada)."""
    return tr(PROGRESS_KEYS.get(stage or "", "progress.working"))


def dataset_fingerprint_of(X, y) -> str:
    import numpy as np
    import pandas as pd
    h = hashlib.sha256()
    for part in (np.asarray(X, dtype=object), np.asarray(y, dtype=object)):
        frame = pd.DataFrame(part.reshape(len(part), -1) if part.ndim == 1 else part).astype(str)
        h.update(str(frame.shape).encode())
        h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return h.hexdigest()[:16]


class AuditController:
    def __init__(self, app: Any, panel: Any = None):
        self.app = app
        self.panel = panel
        self.sm = ExperimentStateMachine()
        self.stale = StaleTracker()
        self.log = MessageLog()
        self.result: Optional[ExperimentResult] = None
        self.experiment_id: Optional[str] = None
        self._stale_reasons: tuple = ()

    # ------------------------------------------------------------ botões
    def _has_trees(self) -> bool:
        return (getattr(self.app, "trepan_original_tree", None) is not None
                or getattr(self.app, "trepan_reloaded_tree", None) is not None)

    def apply_buttons(self) -> Dict[str, bool]:
        """Ativa/desativa os botões conforme o estado real; o tooltip explica porque está desativado."""
        out = {}
        actions = self.sm.available_actions(has_trees=self._has_trees())
        for action, attr in BUTTON_ATTRS.items():
            ok, why = actions.get(action, (True, ""))
            out[action] = ok
            btn = getattr(self.app, attr, None)
            if btn is None:
                continue
            btn.setEnabled(ok)
            if why:
                btn.setToolTip(why)
            elif hasattr(btn, "setToolTip"):
                btn.setToolTip(tr("action.hint.state", state=tr(f"state.{self.sm.state.value}")))
        if self.panel is not None:
            self.panel.set_state(self.sm.label())
        return out

    def _transition(self, target: S) -> None:
        try:
            self.sm.transition(target)
        except InvalidTransition:
            pass  # a interface nunca falha por causa do modelo de estado

    # ------------------------------------------------------------ eventos do ciclo de vida
    def add_message(self, message) -> None:
        self.log.add(message)
        if self.result is not None:
            self.result.messages.append(message)
        if self.panel is not None:
            self.panel.set_log_lines(self.log.format_lines())

    def on_data_loaded(self) -> None:
        """Novo dataset/OWL: resultados anteriores passam a desatualizados até novo treino."""
        self.sm.reset()
        self._transition(S.DATA_LOADED)
        self.check_stale()
        if self.result is not None:
            self.add_message(make_message(Level.INFO, "data_loaded", "msg.data_loaded_state", self.result.provenance.experiment_id))
        self.apply_buttons()

    def on_training_started(self) -> str:
        res = build_experiment_result(self.app, state=S.DATA_LOADED.value)
        self.experiment_id = res.provenance.experiment_id
        self.sm.begin()
        self.add_message(make_message(Level.INFO, "training_started", "msg.training_started", self.experiment_id,
                                      id=self.experiment_id))
        if self.panel is not None:
            self.panel.set_error("")
        self.apply_buttons()
        return self.experiment_id

    def on_training_finished(self) -> ExperimentResult:
        self.sm.end()
        if self.sm.state == S.ERROR:
            self.sm.recover()
        if self.sm.state == S.NO_DATA:
            self._transition(S.DATA_LOADED)
        self._transition(S.MODEL_TRAINED)
        if getattr(self.app, "ontology_acceptance", None) is not None:
            self._transition(S.SEMANTIC_VALIDATED)
        if self._has_trees():
            self._transition(S.TREES_BUILT)
        self._publish(final=True)
        return self.result

    def on_benchmark_finished(self, result: ExperimentResult) -> ExperimentResult:
        """Publica um resultado SCIENTIFIC / BENCHMARK já montado pelo serviço científico (nada é recalculado na GUI)."""
        self.sm.end()
        if self.sm.state == S.ERROR:
            self.sm.recover()
        if self.sm.state == S.NO_DATA:
            self._transition(S.DATA_LOADED)
        for target in (S.MODEL_TRAINED, S.TREES_BUILT, S.RESULTS_READY):
            self._transition(target)
        self.experiment_id = result.provenance.experiment_id
        result.messages.extend(m for m in derive_messages(result) if m.code not in {x.code for x in result.messages})
        self.result = result
        self.stale.mark_result(current_fingerprint(self.app))
        self._stale_reasons = ()
        for m in result.messages:
            if m not in self.log.items():
                self.log.add(m)
        if self.panel is not None:
            self.panel.set_result(result)
            self.panel.set_log_lines(self.log.format_lines())
        self.apply_buttons()
        return result

    def on_metrics_finished(self) -> ExperimentResult:
        self.sm.end()
        if self._has_trees() and not self.sm.reached(S.TREES_BUILT):
            self._transition(S.TREES_BUILT)
        self._transition(S.RESULTS_READY)
        self._publish(final=True)
        if self.result is not None:
            self.add_message(make_message(Level.INFO, "metrics_finished", "msg.metrics_finished",
                                          self.result.provenance.experiment_id, id=self.result.provenance.experiment_id))
        return self.result

    def on_failure(self, error: Any, *, what_key: str, where_key: str, action_key: str, details: Optional[str] = None):
        """Erro estruturado: o que falhou, onde, detalhes expansíveis, ação sugerida, ID da experiência."""
        msg = format_error(error, what_key=what_key, where=tr(where_key), action_key=action_key,
                           experiment_id=self.experiment_id, traceback_text=details)
        self.sm.fail(msg.text)
        self.log.add(msg)
        if self.panel is not None:
            self.panel.set_error(f"{msg.text} [{tr('error.id')}: {msg.experiment_id or '-'}]")
            self.panel.set_log_lines(self.log.format_lines())
        self.sm.recover()  # volta ao último estado válido; os botões deixam de ficar bloqueados
        self.apply_buttons()
        return msg

    def _publish(self, final: bool) -> None:
        result = build_experiment_result(self.app, experiment_id=self.experiment_id, state=self.sm.state.value,
                                         previous=self.result if self.result is not None and
                                         self.result.provenance.experiment_id == self.experiment_id else None)
        result.messages.extend(m for m in derive_messages(result) if m.code not in {x.code for x in result.messages})
        self.result = result
        self.stale.mark_result(current_fingerprint(self.app))
        self._stale_reasons = ()
        for m in result.messages:
            if m not in self.log.items():
                self.log.add(m)
        if self.panel is not None:
            self.panel.set_result(result)
            self.panel.set_log_lines(self.log.format_lines())
        self.apply_buttons()

    # ------------------------------------------------------------ stale
    def check_stale(self) -> bool:
        """Compara a configuração atual com a do último resultado; marca/limpa o estado stale."""
        if self.result is None:
            return False
        reasons = tuple(self.stale.reasons(current_fingerprint(self.app)))
        self.result.stale = bool(reasons)
        self.result.stale_reasons = list(reasons)
        if reasons != self._stale_reasons:
            if reasons:
                self.add_message(make_message(Level.WARNING, "stale", "msg.stale", self.result.provenance.experiment_id,
                                              reasons=", ".join(tr(r) for r in reasons)))
            self._stale_reasons = reasons
            if self.panel is not None:
                self.panel.set_result(self.result)
        return bool(reasons)
