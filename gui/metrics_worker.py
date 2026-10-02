"""Background worker for metrics comparison — keeps the UI responsive."""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal


class MetricsWorker(QThread):
    progress = pyqtSignal(str, int, str)  # stage, percent, message
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, app, X, y):
        super().__init__()
        self.app = app
        self.X = X
        self.y = y
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, stage: str, pct: int, message: str) -> None:
        self.progress.emit(stage, pct, message)

    def run(self) -> None:
        try:
            result = self.app._execute_metrics_comparison(
                self.X,
                self.y,
                progress_fn=self._emit_progress,
                cancel_fn=self.is_cancelled,
            )
            if self._cancel_requested:
                self.failed.emit("Comparación cancelada por el usuario.")
                return
            self.finished_ok.emit(result)
        except InterruptedError:
            self.failed.emit("Comparación cancelada por el usuario.")
        except Exception as exc:
            self.failed.emit(str(exc))
