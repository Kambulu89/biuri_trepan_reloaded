"""Checkpoint em disco dos ajustes do tuning (engenharia: permite RETOMAR uma unidade interrompida sem alterar nenhum resultado).

Cada linha da CV (um ajuste determinístico de uma configuração numa dobra) é gravada ao terminar. Ao retomar, uma linha só é reutilizada se
a chave coincidir EXATAMENTE: hash do candidato (todos os campos) + repetição/seed/dobra + hash dos índices de treino + ``oracle_id`` + sal
do código (hash dos ficheiros que definem o algoritmo). Qualquer alteração científica muda a chave e força o recálculo.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------------------------------------------------------------
# FINGERPRINT versionado das DEPENDÊNCIAS EFETIVAS do tuning estrutural (substitui o sal por lista fixa de ficheiros).
#
# O que o checkpoint guarda: linhas da CV do tuning, produzidas EXCLUSIVAMENTE por ajustes do ``TrepanOriginalClassifier`` contra o oráculo
# congelado (``evaluate`` em ``trepan_scientific_tuning``), com chave = hash do candidato (todos os campos) + repetição/seed/dobra + hash dos
# índices de treino + ``oracle_id`` + fingerprint. Logo as dependências científicas efetivas são o código que EXECUTA nesse caminho:
# Original, o próprio tuning (CV, objetivo, métricas de estrutura), a configuração controlada/candidatos, o orçamento de queries e este mecanismo.
# O Reloaded NÃO executa nesse caminho (a etapa semântica não é checkpointada e o benchmark desliga-a: ``run_semantic_stage=False``) e por isso NÃO
# integra o fingerprint — por demonstração, não por exclusão arbitrária: ``tests/test_tuning_fingerprint.py`` traça a execução real do tuning e falha
# se algum ficheiro do repositório fora de ``EFFECTIVE_DEPENDENCIES`` for executado (qualquer nova dependência obriga a declará-la).
# Em dúvida invalida-se: o fingerprint inclui também versões de Python/NumPy/SciPy/scikit-learn, plataforma e o ambiente numérico fixado.
# Dependências que entram noutras chaves (não no fingerprint): oráculo (``oracle_id``), pré-processamento/treino (hash dos índices de treino e,
# via oráculo, dos dados), pesos do MLP (``oracle_id``), configuração do candidato (hash), RNG (seed por repetição na chave).
# ---------------------------------------------------------------------------------------------------------------------------------
FINGERPRINT_VERSION = 2
EFFECTIVE_DEPENDENCIES = (
    "trepan_original.py",                     # ajustes do TREPAN Original (algoritmo, RNG, amostragem, pontuação, m-of-n)
    "trepan_scientific_tuning.py",            # CV, objetivo, estatísticas, estrutura da árvore, protocolo de seleção
    "controlled_trepan_experiment.py",        # ControlledTrepanConfig / candidatos
    "scientific_experiment_contract.py",      # contrato científico (conservador: não executa no caminho traçado, mas define o protocolo)
    "training_config.py",                     # orçamento de queries
    "tuning_checkpoint.py",                   # o próprio mecanismo de chave/fingerprint
)
_NUMERIC_ENV = ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES")


def _read_normalised(path: Path) -> bytes:
    data = path.read_bytes() if path.exists() else b"-"
    return data.replace(b"\r\n", b"\n")          # CRLF/LF não alteram a identidade do código


def fingerprint_report(root: Optional[Path] = None) -> Dict[str, Any]:
    """Relatório auditável do fingerprint: ficheiros (SHA-256 com fins de linha normalizados), versões de bibliotecas, plataforma e ambiente numérico."""
    import platform
    import sys
    here = Path(root) if root is not None else Path(__file__).resolve().parent
    libs = {}
    for name in ("numpy", "scipy", "sklearn"):
        try:
            libs[name] = __import__(name).__version__
        except Exception:                                   # biblioteca ausente ⇒ valor distinto ⇒ invalida
            libs[name] = "unavailable"
    report = {"fingerprint_version": FINGERPRINT_VERSION,
              "files": {name: hashlib.sha256(_read_normalised(here / name)).hexdigest() for name in EFFECTIVE_DEPENDENCIES},
              "python": f"{sys.version_info.major}.{sys.version_info.minor}", "libs": libs,
              "platform": f"{platform.system()}-{platform.machine()}",
              "numeric_env": {k: os.environ.get(k) for k in _NUMERIC_ENV}}
    report["fingerprint"] = hashlib.sha256(json.dumps({k: v for k, v in report.items()}, sort_keys=True).encode()).hexdigest()[:16]
    return report


def code_salt() -> str:
    return "v%d:%s" % (FINGERPRINT_VERSION, fingerprint_report()["fingerprint"])


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
            self._write_sidecar()

    def _write_sidecar(self) -> None:
        """Regista o fingerprint (auditável) ao lado do checkpoint. Se já existe um diferente, as chaves antigas simplesmente não coincidem (sem reutilização)."""
        side = self.path.with_name(self.path.name + ".fingerprint.json")
        try:
            report = fingerprint_report()
            if side.exists():
                prev = json.loads(side.read_text(encoding="utf-8"))
                if prev.get("fingerprint") == report["fingerprint"]:
                    return
                history = prev.get("history", []) + [{k: prev.get(k) for k in ("fingerprint", "fingerprint_version")}]
                report["history"] = history[-10:]
            side.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
        except Exception:                                    # o registo nunca afeta o resultado científico
            pass

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
