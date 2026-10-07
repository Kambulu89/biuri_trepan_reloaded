"""Checkpoint em disco dos ajustes do tuning (engenharia: permite RETOMAR uma unidade interrompida sem alterar nenhum resultado).

Cada linha da CV (um ajuste determinístico de uma configuração numa dobra) é gravada ao terminar. Ao retomar, uma linha só é reutilizada se
a chave coincidir EXATAMENTE: hash do candidato (todos os campos) + repetição/seed/dobra + hash dos índices de treino + ``oracle_id`` + sal
do código (hash dos ficheiros que definem o algoritmo). Qualquer alteração científica muda a chave e força o recálculo.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

_CODE_FILES = ("trepan_original.py", "trepan_reloaded_historical.py", "trepan_scientific_tuning.py", "controlled_trepan_experiment.py",
               "scientific_experiment_contract.py")


def code_salt() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in _CODE_FILES:
        f = here / name
        h.update(name.encode()); h.update(f.read_bytes() if f.exists() else b"-")
    return h.hexdigest()[:16]


class FitCheckpoint:
    """Fluxo append-only de pares ``(chave, linha)`` em pickle; tolera uma cauda truncada (processo morto a meio da escrita)."""

    def __init__(self, path: Optional[str], salt: Optional[str] = None):
        self.path = Path(path) if path else None
        self.salt = salt or code_salt()
        self.rows: Dict[str, Any] = {}
        self.resumed = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        good = 0
        with open(self.path, "rb") as fh:
            while True:
                pos = fh.tell()
                try:
                    key, row = pickle.load(fh)
                except EOFError:
                    break
                except Exception:                      # cauda truncada/corrompida: descarta-a e continua a partir do último registo válido
                    with open(self.path, "r+b") as w:
                        w.truncate(pos)
                    break
                self.rows[key] = row
                good += 1

    def key(self, *parts: Any) -> str:
        return hashlib.sha256(("|".join(map(str, parts)) + "|" + self.salt).encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if self.path is None:              # sem ficheiro de checkpoint o mecanismo é inerte (não funciona como memo escondida)
            return None
        row = self.rows.get(key)
        if row is not None:
            self.resumed += 1
        return row

    def put(self, key: str, row: Any) -> None:
        if self.path is None:
            return
        self.rows[key] = row
        with open(self.path, "ab") as fh:
            pickle.dump((key, row), fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())


def from_environment() -> FitCheckpoint:
    return FitCheckpoint(os.environ.get("BIURI_TUNING_CHECKPOINT") or None)
