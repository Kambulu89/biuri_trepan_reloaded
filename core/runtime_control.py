"""Controlo cooperativo de deadline/cancelamento para treino."""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Callable, Optional

@dataclass
class Deadline:
    seconds: Optional[float]=None
    started_at: float=0.0
    def __post_init__(self):
        if not self.started_at: self.started_at=time.perf_counter()
    def expired(self)->bool:
        return self.seconds is not None and time.perf_counter()-self.started_at >= float(self.seconds)
    def remaining(self):
        return None if self.seconds is None else max(0.0,float(self.seconds)-(time.perf_counter()-self.started_at))

def check_interruption(*,deadline:Optional[Deadline]=None,cancel_fn:Optional[Callable[[],bool]]=None):
    if cancel_fn and cancel_fn(): raise InterruptedError('Treino cancelado pelo utilizador.')
    if deadline and deadline.expired(): raise TimeoutError('Orçamento de tempo do treino esgotado.')
