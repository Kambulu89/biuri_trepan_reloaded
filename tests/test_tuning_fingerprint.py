"""Fingerprint versionado das dependências efetivas do tuning (checkpoints). Em dúvida invalida; nunca reutiliza resultados incompatíveis."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from core import tuning_checkpoint as tc

CORE = Path(tc.__file__).resolve().parent
ROOT = CORE.parent


def _clone_core(tmp_path):
    d = tmp_path / "core"
    d.mkdir()
    for f in list(tc.EFFECTIVE_DEPENDENCIES) + ["trepan_reloaded_historical.py"]:
        shutil.copy(CORE / f, d / f)
    return d


def test_fingerprint_is_deterministic_and_versioned():
    a, b = tc.fingerprint_report(), tc.fingerprint_report()
    assert a == b and a["fingerprint_version"] == tc.FINGERPRINT_VERSION and tc.code_salt().startswith(f"v{tc.FINGERPRINT_VERSION}:")
    assert set(a["files"]) == set(tc.EFFECTIVE_DEPENDENCIES) and "trepan_reloaded_historical.py" not in a["files"]


def test_changing_an_effective_dependency_invalidates(tmp_path):
    d = _clone_core(tmp_path)
    base = tc.fingerprint_report(d)["fingerprint"]
    for name in tc.EFFECTIVE_DEPENDENCIES:
        f = d / name
        old = f.read_bytes()
        f.write_bytes(old + b"\n# alteracao\n")
        assert tc.fingerprint_report(d)["fingerprint"] != base, name
        f.write_bytes(old)
    assert tc.fingerprint_report(d)["fingerprint"] == base


def test_reloaded_changes_do_not_invalidate_but_are_justified_by_trace(tmp_path):
    d = _clone_core(tmp_path)
    base = tc.fingerprint_report(d)["fingerprint"]
    (d / "trepan_reloaded_historical.py").write_bytes((d / "trepan_reloaded_historical.py").read_bytes() + b"\n# alteracao\n")
    assert tc.fingerprint_report(d)["fingerprint"] == base


def test_line_endings_do_not_change_identity(tmp_path):
    d = _clone_core(tmp_path)
    f = d / "trepan_original.py"
    lf = f.read_bytes().replace(b"\r\n", b"\n")              # o checkout pode já ter CRLF (Windows): parte-se de LF puro
    f.write_bytes(lf)
    base = tc.fingerprint_report(d)["fingerprint"]
    f.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert tc.fingerprint_report(d)["fingerprint"] == base


def test_library_platform_and_numeric_env_invalidate(monkeypatch):
    base = tc.fingerprint_report()["fingerprint"]
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    assert tc.fingerprint_report()["fingerprint"] != base
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    import numpy
    monkeypatch.setattr(numpy, "__version__", "0.0.0-test")
    assert tc.fingerprint_report()["fingerprint"] != base


def test_executed_repo_files_during_tuning_are_declared():
    """Prova dinâmica: traça a execução REAL do tuning estrutural (sem etapa semântica) e exige que todo o código do repositório executado esteja
    declarado em EFFECTIVE_DEPENDENCIES (ou seja o oráculo de teste). Uma nova dependência no caminho do tuning faz este teste falhar."""
    from core.controlled_trepan_experiment import ControlledTrepanConfig
    from core.trepan_scientific_tuning import ScientificTrepanSearchConfig, tune_scientific_trepan
    from tests.eq_support import ClosedFormOracle
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4)); orc = ClosedFormOracle(4, 3, 1); y = orc.predict(X)
    base = ControlledTrepanConfig(max_nodes=7, max_depth=3, min_samples_leaf=2, min_sample=120, max_n=2, beam_width=2, max_features_per_node=4,
                                  max_queries=400, random_state=0)
    seen = set()

    def prof(frame, event, arg):
        if event == "call":
            fn = frame.f_code.co_filename
            if fn.startswith(str(ROOT)):
                seen.add(os.path.relpath(fn, ROOT).replace(os.sep, "/"))
    sys.setprofile(prof)
    try:
        tune_scientific_trepan(X, y, oracle=orc, feature_names=[f"a{i}" for i in range(4)], base_config=base,
                               search=ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1), run_semantic_stage=False)
    finally:
        sys.setprofile(None)
    declared = {f"core/{n}" for n in tc.EFFECTIVE_DEPENDENCIES}
    undeclared = {f for f in seen if f.startswith("core/") and f not in declared}
    assert not undeclared, f"dependências executadas e não declaradas: {sorted(undeclared)}"
    assert "core/trepan_reloaded_historical.py" not in seen


def test_resume_only_with_identical_fingerprint(tmp_path):
    path = tmp_path / "u.ckpt"
    c1 = tc.FitCheckpoint(str(path), salt="v2:AAAA")
    k1 = c1.key("cand", 0, 1, 0, "h", "oracle")
    c1.put(k1, {"x": 1})
    c2 = tc.FitCheckpoint(str(path), salt="v2:AAAA")
    assert c2.get(c2.key("cand", 0, 1, 0, "h", "oracle")) == {"x": 1}
    c3 = tc.FitCheckpoint(str(path), salt="v2:BBBB")                 # fingerprint diferente ⇒ chave diferente ⇒ nenhuma reutilização
    assert c3.get(c3.key("cand", 0, 1, 0, "h", "oracle")) is None


def test_sidecar_records_fingerprint_and_history(tmp_path, monkeypatch):
    path = tmp_path / "u.ckpt"
    tc.FitCheckpoint(str(path))
    side = Path(str(path) + ".fingerprint.json")
    rep = json.loads(side.read_text())
    assert rep["fingerprint"] == tc.fingerprint_report()["fingerprint"] and rep["fingerprint_version"] == tc.FINGERPRINT_VERSION
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "9")
    tc.FitCheckpoint(str(path))
    rep2 = json.loads(side.read_text())
    assert rep2["fingerprint"] != rep["fingerprint"] and rep2["history"][-1]["fingerprint"] == rep["fingerprint"]
