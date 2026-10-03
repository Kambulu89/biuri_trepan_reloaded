"""Regressão: NameError em export_results (GUI) e BASE_SEED (pipeline_train)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "gui/biuri_app_complete.py",
    "counterfactuals/pipelines/pipeline_train.py",
]


@pytest.mark.parametrize("rel", FILES)
def test_no_undefined_names(rel):
    """Detecta nomes indefinidos sem precisar de importar PyQt6/TensorFlow."""
    pytest.importorskip("pyflakes")
    from pyflakes.api import check
    from pyflakes.reporter import Reporter
    import io

    out, err = io.StringIO(), io.StringIO()
    src = (ROOT / rel).read_text(encoding="utf-8")
    check(src, rel, Reporter(out, err))
    undefined = [l for l in out.getvalue().splitlines() if "undefined name" in l]
    assert not undefined, undefined


def test_pipeline_train_defines_base_seed():
    pytest.importorskip("sklearn")
    from counterfactuals.pipelines import pipeline_train

    assert pipeline_train.BASE_SEED == 42
