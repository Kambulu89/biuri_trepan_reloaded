"""Worker assíncrono para contrafactuais na GUI BIURI."""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal


class CounterfactualWorker(QThread):
    progress = pyqtSignal(str, int, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    STAGE_GENERATE = 'generate'
    STAGE_GLOBAL = 'global'
    STAGE_TREE = 'tree'
    STAGE_GENERATE_TREE = 'generate_tree'     # um clique: gera os CFs locais (se preciso) e constrói a árvore CF
    STAGE_TRANSFER = 'transfer'
    STAGE_IMPROVE = 'improve'
    STAGE_FULL = 'full'

    def __init__(
        self, session, mode=STAGE_GENERATE, cf_result=None, seed=42,
        options=None, parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.mode = mode
        self.cf_result = cf_result
        self.seed = seed
        self.options = dict(options or {})
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def _emit(self, stage: str, pct: int, message: str) -> None:
        self.progress.emit(stage, pct, message)

    def run(self) -> None:
        try:
            from counterfactuals.service import (
                evaluate_transfer_from_session,
                build_counterfactual_tree_from_session,
                generate_explanation_from_session,
                generate_global_counterfactuals_from_session,
                generate_counterfactuals_from_session,
                improve_surrogates_from_session,
            )

            cancel_fn = self.is_cancelled
            progress_fn = self._emit

            if self.mode == self.STAGE_GLOBAL:
                global_result = generate_global_counterfactuals_from_session(
                    self.session,
                    self.options,
                    progress_fn=progress_fn,
                    cancel_fn=cancel_fn,
                )
                if self._cancel_requested:
                    self.failed.emit('Operación cancelada por el usuario.')
                    return
                self.finished_ok.emit({'counterfactuals': global_result})
                return

            if self.mode == self.STAGE_TREE:
                tree_result = build_counterfactual_tree_from_session(
                    self.session,
                    self.cf_result,
                    self.options,
                    progress_fn=progress_fn,
                    cancel_fn=cancel_fn,
                )
                if self._cancel_requested:
                    self.failed.emit('Operación cancelada por el usuario.')
                    return
                self.finished_ok.emit({'counterfactuals': tree_result})
                return

            if self.mode == self.STAGE_GENERATE_TREE:
                cf_result = generate_explanation_from_session(
                    self.session, self.options, progress_fn=progress_fn, cancel_fn=cancel_fn,
                )
                if self._cancel_requested:
                    self.failed.emit('Operación cancelada por el usuario.')
                    return
                if not any(bool((item.get('metrics') or {}).get('validity')) for item in cf_result.get('candidates') or []):
                    raise ValueError(
                        "Não foi possível gerar contrafactuais VÁLIDOS para esta instância com o método escolhido, pelo que a árvore CF "
                        "não pode ser construída. Escolha outra instância, outra classe alvo ou outro método."
                    )
                tree_result = build_counterfactual_tree_from_session(
                    self.session, cf_result, self.options, progress_fn=progress_fn, cancel_fn=cancel_fn,
                )
                if self._cancel_requested:
                    self.failed.emit('Operación cancelada por el usuario.')
                    return
                self.finished_ok.emit({'counterfactuals': tree_result, 'local_counterfactuals': cf_result})
                return

            if self.mode == self.STAGE_TRANSFER:
                transfer_result = evaluate_transfer_from_session(
                    self.session,
                    self.options,
                    progress_fn=progress_fn,
                    cancel_fn=cancel_fn,
                )
                if self._cancel_requested:
                    self.failed.emit('Operación cancelada por el usuario.')
                    return
                self.finished_ok.emit({'transfer': transfer_result})
                return

            if self.mode in (self.STAGE_GENERATE, self.STAGE_FULL):
                if self.options:
                    cf_result = generate_explanation_from_session(
                        self.session,
                        self.options,
                        progress_fn=progress_fn,
                        cancel_fn=cancel_fn,
                    )
                else:
                    cf_result = generate_counterfactuals_from_session(
                        self.session, progress_fn=progress_fn, cancel_fn=cancel_fn,
                    )
                if self._cancel_requested:
                    self.failed.emit('Operación cancelada por el usuario.')
                    return
            else:
                cf_result = self.cf_result

            if self.mode in (self.STAGE_IMPROVE, self.STAGE_FULL):
                improve_result = improve_surrogates_from_session(
                    self.session, cf_result,
                    progress_fn=progress_fn,
                    cancel_fn=cancel_fn,
                    seed=self.seed,
                    options=self.options,
                )
                if self._cancel_requested:
                    self.failed.emit('Operación cancelada por el usuario.')
                    return
                payload = {'improve': improve_result}
                if cf_result is not None:
                    payload['counterfactuals'] = cf_result
                self.finished_ok.emit(payload)
            else:
                self.finished_ok.emit({'counterfactuals': cf_result})

        except InterruptedError:
            self.failed.emit('Operación cancelada por el usuario.')
        except Exception as exc:
            self.failed.emit(str(exc))
