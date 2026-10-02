"""Manifest experimental e persistência (sem sobrescrever resultados anteriores)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

BUILD_VERSION = "benchmark-v1.0"


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def hash_dataset(X, y, feature_names=None) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, dtype=float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y).astype(str)).tobytes())
    h.update(json.dumps(list(map(str, feature_names or []))).encode())
    return h.hexdigest()


def hash_file(path: Optional[str]) -> str:
    if not path or not Path(path).exists():
        return "none"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(root: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root or Path(__file__).resolve().parents[2]),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def library_versions() -> Dict[str, str]:
    versions = {}
    for name in ("numpy", "scipy", "pandas", "sklearn", "owlready2"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not-installed"
    return versions


def build_manifest(*, experiment_id: str, dataset_name: str, dataset_hash: str, ontology_hash: str,
                   config: Dict[str, Any], seeds, split_hash: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    m = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "build_version": BUILD_VERSION,
        "git_commit": git_commit(),
        "dataset": dataset_name,
        "dataset_hash": dataset_hash,
        "ontology_hash": ontology_hash,
        "config_hash": stable_hash(config),
        "seeds": list(map(int, seeds)),
        "split_hash": split_hash,
        "library_versions": library_versions(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    if extra:
        m.update(extra)
    return m


def unique_run_dir(root: Path, experiment_id: str) -> Path:
    """Cria results/<experiment_id>/ sem sobrescrever: sufixo _2, _3... se já existir."""
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / experiment_id
    n = 2
    while candidate.exists():
        candidate = root / f"{experiment_id}_{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate
