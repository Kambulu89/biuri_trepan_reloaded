"""Worker em background para treino BIURI sem bloquear a UI."""
from __future__ import annotations

import logging
import traceback

from PyQt6.QtCore import QThread, pyqtSignal
from core.training_config import enforce_scientific_preset

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    progress = pyqtSignal(str, int, str)  # stage, percent, message
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    failed_detail = pyqtSignal(str, str)  # mensagem, traceback (emitido antes de ``failed``)

    def __init__(self, app, preset):
        super().__init__()
        self.app = app
        self.preset = enforce_scientific_preset(preset)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, stage: str, pct: int, message: str) -> None:
        self.progress.emit(stage, pct, message)

    def run(self) -> None:
        try:
            result = self.app._execute_training_pipeline(
                self.preset,
                progress_fn=self._emit_progress,
                cancel_fn=self.is_cancelled,
            )
            if self._cancel_requested:
                self.failed.emit("Treino cancelado pelo utilizador.")
                return
            self.finished_ok.emit(result)
        except InterruptedError:
            self.failed.emit("Treino cancelado pelo utilizador.")
        except Exception as exc:
            logger.exception("Falha no treino científico BIURI")
            self.failed_detail.emit(str(exc), traceback.format_exc())
            self.failed.emit(str(exc))
