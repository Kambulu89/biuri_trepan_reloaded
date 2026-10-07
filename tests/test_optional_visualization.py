"""Graphviz (pacote Python e binário ``dot``) é dependência OPCIONAL de visualização: nunca bloqueia treino, benchmark, MLP, ontologia ou TREPAN."""
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_core_extractor_imports_and_trains_without_the_graphviz_package():
    code = (
        "import sys; sys.modules['graphviz'] = None; sys.modules['dtreeviz'] = None\n"
        "import core.trepan_reloaded_extractor as m\n"
        "import core.trepan_original as t\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-800:]


def test_preflight_is_ready_without_the_dot_binary_and_reports_visualization_as_optional(monkeypatch):
    import environment_preflight as pre
    importlib.reload(pre)
    real_which = pre.shutil.which
    monkeypatch.setattr(pre.shutil, "which", lambda name, *a, **k: None if name == "dot" else real_which(name, *a, **k))
    report = pre.build_preflight()
    assert report["graphviz_dot"] is None
    assert "graphviz_dot" not in report["checks"]
    assert report["optional_visualization"]["graphviz_dot_binary"] is False and report["optional_visualization"]["available"] is False
    required = {k: v for k, v in report["checks"].items()}
    assert report["ready"] == all(required.values())


def test_preflight_does_not_require_the_graphviz_python_packages(monkeypatch):
    import environment_preflight as pre
    assert "graphviz" not in pre.RUNTIME_IMPORTS and "dtreeviz" not in pre.RUNTIME_IMPORTS
    assert {"graphviz", "dtreeviz"} == set(pre.OPTIONAL_VISUALIZATION_IMPORTS)


def test_checkpoint_without_a_file_is_inert_and_not_a_hidden_memo():
    from core.tuning_checkpoint import FitCheckpoint
    cp = FitCheckpoint(None)
    cp.put(cp.key("a"), {"x": 1})
    assert cp.get(cp.key("a")) is None and cp.resumed == 0 and cp.rows == {}


def test_checkpoint_with_a_file_resumes_exactly(tmp_path):
    from core.tuning_checkpoint import FitCheckpoint
    path = tmp_path / "ck.pkl"
    cp = FitCheckpoint(str(path), salt="s")
    cp.put(cp.key("a"), {"x": 1})
    again = FitCheckpoint(str(path), salt="s")
    assert again.get(again.key("a")) == {"x": 1} and again.resumed == 1
    assert FitCheckpoint(str(path), salt="other").get(FitCheckpoint(str(path), salt="other").key("a")) is None
