"""CONTRATO EXECUTÁVEL de agnosticismo ao dataset (guardrail arquitetural, obrigatório na CI antes de qualquer merge).

Critério final: «um dataset compatível e nunca visto durante o desenvolvimento entra em produção e atravessa o pipeline completo sem
modificação do código científico». Este ficheiro transforma essa regra (CLAUDE.md, secção 1) em testes:

  A. verificação estática (nomes, ramificações e políticas por dataset; núcleo ↛ validation) e mutation tests do próprio verificador;
  B. testes metamórficos (mesmo conteúdo, outro nome/ficheiro/ids → mesmo resultado científico);
  C. dataset totalmente desconhecido; dimensionalidade, classes, tipos de features e ontologia variáveis;
  D. política de recursos global (nunca por dataset); E. CI configurada para falhar automaticamente.

``validation → core`` é permitido; ``core → validation`` nunca.
"""
from __future__ import annotations

import ast
import hashlib
import secrets
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from scripts import check_dataset_agnosticism as chk  # noqa: E402
from test_dataset_agnosticism import _search, make_frame  # noqa: E402

import core.scientific_benchmark_service as svc  # noqa: E402
from core.benchmark.runner import BenchmarkConfig, BenchmarkRunner, Dataset, TreeBudget, default_arms  # noqa: E402
from core.benchmark.semantic import GroupSemanticProvider  # noqa: E402
from core.trepan_scientific_tuning import ScientificTrepanSearchConfig  # noqa: E402
from validation.benchmark import equivalence as eq  # noqa: E402

PRODUCTION = chk.PRODUCTION_DIRS


# ==================================================================================================== A. verificação estática
def test_production_code_passes_the_static_agnosticism_contract():
    violations = chk.scan(ROOT) + chk.scan_policies(ROOT)
    assert not violations, "\n".join(f"{f}:{l} [{r}] {m}" for f, l, r, m in violations)


def test_the_cli_checker_exits_non_zero_on_a_violation_and_zero_on_clean_code(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "ok.py").write_text("def f(x):\n    return x + 1\n")
    run = lambda: subprocess.run([sys.executable, str(ROOT / "scripts" / "check_dataset_agnosticism.py"), "--root", str(tmp_path)],
                                 capture_output=True, text=True)
    assert run().returncode == 0
    (tmp_path / "core" / "bad.py").write_text('def f(dataset_name):\n    if dataset_name == "digits":\n        return 1\n')
    out = run()
    assert out.returncode == 1 and "bad.py:2" in out.stdout and "R2" in out.stdout


def _violations(source: str, root: Path = ROOT):
    names = chk.known_dataset_regex(root)
    return {v[2] for v in chk.check_source("core/x.py", textwrap.dedent(source), names)
            + chk.check_imports("core/x.py", textwrap.dedent(source)) + chk.check_policies("core/x.py", textwrap.dedent(source))}


@pytest.mark.parametrize("source,rule", [
    ('x = "digits"', "R1"), ('wine_config = {}', "R1"), ('DATASETS = ["adult", "census"]', "R1"), ('label = "Iris"', "R1"),
    ('if dataset_name == "digits":\n    pass', "R2"), ('if dataset_id in ("a", "b"):\n    pass', "R2"),
    ('if meta["file_name"] == "x.arff":\n    pass', "R2"), ('if dataset_name.lower() != "z":\n    pass', "R2"),
    ('if file_name.startswith("ad"):\n    pass', "R2"),
    ('v = CONFIG[dataset_name]', "R3"), ('v = tables.get(dataset_id)', "R3"),
    ('limit = resource_policy[dataset_name]', "R6"), ('t = LIMITS.get(dataset_id)', "R6"), ('c = max_nodes_by[ds_name]', "R6"),
    ('if n_features == 30:\n    pass', "R4"), ('if len(class_names) == 10:\n    pass', "R4"),
    ('import validation.benchmark.datasets', "R5"), ('from validation import benchmark', "R5"), ('from scripts import check_dataset_agnosticism', "R5"),
    ('p = "validation/benchmark/ontologies/x.owl"', "R5"),
])
def test_the_checker_detects_every_forbidden_pattern(source, rule):
    assert rule in _violations(source)


@pytest.mark.parametrize("source", [
    'from owlready2 import IRIS as OWLREADY_IRIS', '# resolve os IRIs completos', 'iri_map = {}', 'x = SOMETHING_IRIS',
    'result["dataset_name"] = dataset_name', 'info = "arquivo: {}".format(meta["file_name"])', 'if mode == "fast":\n    pass',
    'if n_features == 3:\n    pass', 'spec = REGISTRY[key]', 'limit = max_wall_time_per_unit', 'if kind in ("a", "b"):\n    pass',
    'import numpy as np', 'from core.benchmark import runner',
])
def test_the_checker_has_no_false_positives_on_legitimate_code(source):
    assert not _violations(source)


def test_known_dataset_names_come_from_the_registry_so_new_datasets_are_banned_automatically(tmp_path):
    (tmp_path / "validation" / "benchmark").mkdir(parents=True)
    (tmp_path / "validation" / "benchmark" / "datasets.py").write_text('SPEC = DatasetSpec("zebra_crossing_set", "real_offline", "o", "v", None, None, None)\n')
    names = chk.known_dataset_regex(tmp_path)
    assert chk.check_source("core/y.py", 'name = "zebra_crossing_set"\n', names)
    assert not chk.check_source("core/y.py", 'name = "zebra"\n', names)           # só a palavra isolada registada
    real = chk.known_dataset_regex(ROOT)
    for dataset_id in ("wine", "breast_cancer", "digits", "synthetic_binary"):    # os ids do registo real estão cobertos
        assert real.search(f"x = '{dataset_id}'")


def test_core_gui_and_counterfactuals_never_import_validation_scripts_or_tests():
    """Dependência permitida: validation → core. Nunca: core → validation."""
    offenders = []
    for d in PRODUCTION:
        for path in (ROOT / d).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            for lineno, name in chk.imports_of(tree):
                if name.split(".")[0] in ("validation", "scripts", "tests"):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno} -> {name}")
    assert not offenders, offenders


def test_validation_is_allowed_to_know_the_core_and_hosts_the_known_datasets():
    imports_core = [p for p in (ROOT / "validation" / "benchmark").glob("*.py")
                    if any(n.split(".")[0] == "core" for _l, n in chk.imports_of(ast.parse(p.read_text(encoding="utf-8"))))]
    assert imports_core                                                            # validation → core existe (e é permitido)
    assert (ROOT / "validation" / "benchmark" / "datasets.py").exists() and not (ROOT / "core" / "datasets.py").exists()
    assert (ROOT / "validation" / "benchmark" / "ontologies").is_dir()


def test_policies_are_global_in_the_validation_benchmark_too():
    assert not chk.scan_policies(ROOT, dirs=("validation/benchmark",))


# ==================================================================================================== D. política de recursos global
def test_resource_policy_is_global_configurable_and_never_keyed_by_dataset(tmp_path, monkeypatch):
    import inspect
    from validation.benchmark import manifest as mf
    from validation.benchmark import run as runner_cli
    sig = inspect.signature(runner_cli._spawn)
    assert "max_wall_time_per_unit" in sig.parameters and sig.parameters["max_wall_time_per_unit"].default is None   # global e opcional
    policy = mf.build_manifest(["iris", "wine"])["resource_policy"]
    assert policy["scope"] == "global_operational_not_scientific" and policy["max_wall_time_per_unit_s"] is None
    assert not (set(policy) & {d.dataset_id for d in __import__("validation.benchmark.datasets", fromlist=["x"]).REGISTRY.values()})
    # o MESMO limite aplica-se a qualquer dataset: duas unidades com identidades diferentes são ambas marcadas, sem mudar a configuração científica
    class Hung:
        def __init__(self, *a, **k): self.terminated = False
        def poll(self): return 0 if self.terminated else None
        def terminate(self): self.terminated = True
        def wait(self, timeout=None): return 0
        def kill(self): self.terminated = True
    monkeypatch.setattr(runner_cli.subprocess, "Popen", Hung)
    failed = runner_cli._spawn([("dataset_alpha", 1), ("totally_other_name", 2)], tmp_path, workers=2, max_wall_time_per_unit=0.05)
    status = {f.stem: __import__("json").loads(f.read_text()) for f in (tmp_path / "status").glob("*.json")}
    assert failed == 2 and {s["status"] for s in status.values()} == {"resource_limit_exceeded"}
    assert {s["limit_s"] for s in status.values()} == {0.05}
    assert not any("budget" in s or "max_nodes" in s for s in status.values())             # nada científico é alterado para terminar


# ==================================================================================================== utilitários dinâmicos
CHEAP = ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1, purity_epsilon_grid=(0.05,), max_nodes_grid=(5,))
ARMS_AF = ("mlp_original", "c45", "trepan_original", "reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled")


def fast_cfg(*, tuning: bool = True, seed: int = 7) -> BenchmarkConfig:
    return BenchmarkConfig(seeds=(seed,), tree=TreeBudget(min_sample=120, max_queries=1200, max_nodes=5, max_depth=4), extra_ablations=False,
                           different_oracle_experiment=False, structure_tuning=tuning, structure_search=CHEAP if tuning else None,
                           n_boot=30, inner_cv_splits=2)


def arms_af(cfg):
    return [a for a in default_arms(cfg) if a.arm_id in ARMS_AF]


def arrays(n: int, p: int, k: int, seed: int):
    X, y = make_classification(n_samples=n, n_features=p, n_informative=min(p, max(2, p // 2)), n_redundant=0, n_classes=k,
                               n_clusters_per_class=1, flip_y=0.02, random_state=seed)
    return X, y


def positional_provider(feature_names):
    """Ontologia programática genérica: duas metades das colunas (por POSIÇÃO). Os braços semânticos (E/F) exigem uma ontologia por desenho;
    sem ela são saltados com razão explícita (ver ``test_without_an_ontology_...``). Não conhece nenhum dataset."""
    names = [str(n) for n in feature_names]
    half = max(1, len(names) // 2)
    return GroupSemanticProvider({"G1": names[:half], "G2": names[half:]} if len(names) > 1 else {"G1": names})


_AUTO = object()


def run_arrays(name: str, X, y, feature_names, provider=_AUTO, tuning: bool = True):
    cfg = fast_cfg(tuning=tuning)
    if provider is _AUTO:
        provider = positional_provider(feature_names)
    return BenchmarkRunner(cfg, arms_af(cfg)).run(Dataset(name, X, y, feature_names), provider)


# Identificadores de ARTEFACTO/cache: incluem legitimamente nomes (dataset, features, ficheiro da ontologia) para rastreabilidade.
# A identidade CIENTÍFICA equivalente (scientific_preprocessing_id, dataset_content_hash, ontology_structure_signature) NÃO está aqui:
# é comparada como qualquer outro resultado (ver ``test_scientific_identity_is_name_free_while_artifact_identity_keeps_names``).
NAME_BEARING = ["dataset", "dataset_hash", "features_used_names", "root_feature", "split_signatures", "experiment_id",
                "preprocessing_id", "ontology_hash", "original_ontology_hash", "shuffled_ontology_hash"]


def comparable(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in NAME_BEARING if c in df.columns])


def source_digest() -> str:
    h = hashlib.sha256()
    for d in ("core", "gui", "validation/benchmark"):
        for path in sorted((ROOT / d).rglob("*.py")):
            h.update(path.read_bytes())
    return h.hexdigest()


# ==================================================================================================== B. metamórficos
def test_dataset_identity_name_and_feature_names_do_not_influence_the_scientific_result():
    X, y = arrays(180, 6, 3, seed=5)
    a = run_arrays("identity_a", X, y, [f"m{i}" for i in range(6)])
    b = run_arrays("zz_completely_unrelated_name", X, y, [f"coluna_ñ_{i} (°C)" for i in range(6)])
    assert not a.skipped and not b.skipped
    diffs = eq.compare_frames(comparable(a.frame()), comparable(b.frame()))
    assert not diffs, diffs[:8]
    assert a.semantic_report[0]["structural_protocol"]["selected"] == b.semantic_report[0]["structural_protocol"]["selected"]
    assert np.array_equal(a.predictions.filter(like="pred__").to_numpy(), b.predictions.filter(like="pred__").to_numpy())


def test_file_name_and_dataset_id_do_not_influence_the_result(tmp_path):
    X, y = arrays(150, 4, 2, seed=8)
    frame = pd.DataFrame(X, columns=[f"c{i}" for i in range(4)]); frame["target"] = y
    p1, p2 = tmp_path / "primeiro_ficheiro.csv", tmp_path / "x9_outro_nome_qualquer.csv"
    frame.to_csv(p1, index=False); frame.to_csv(p2, index=False)
    d1, d2 = Dataset.from_file(str(p1), "target"), Dataset.from_file(str(p2), "target", name="id_diferente_zzz")
    assert d1.name != d2.name and np.array_equal(d1.X, d2.X) and np.array_equal(d1.y, d2.y)
    cfg = fast_cfg()
    r1 = BenchmarkRunner(cfg, arms_af(cfg)).run(d1, positional_provider(d1.feature_names))
    r2 = BenchmarkRunner(cfg, arms_af(cfg)).run(d2, positional_provider(d2.feature_names))
    assert not eq.compare_frames(comparable(r1.frame()), comparable(r2.frame()))


def test_semantic_arms_depend_on_the_ontology_content_not_on_the_dataset_name():
    X, y = arrays(180, 6, 2, seed=11)
    n1, n2 = [f"a{i}" for i in range(6)], [f"zz_{i}_nome" for i in range(6)]
    groups = lambda names: {"G1": names[:3], "G2": names[3:]}
    r1 = run_arrays("same_content_1", X, y, n1, GroupSemanticProvider(groups(n1)))
    r2 = run_arrays("same_content_2_renamed", X, y, n2, GroupSemanticProvider(groups(n2)))
    assert set(r1.frame()["arm"]) == set(ARMS_AF)
    assert not eq.compare_frames(comparable(r1.frame()), comparable(r2.frame()))
    e1 = r1.frame().set_index("arm").loc["reloaded_owl_full"]
    assert e1["mapped_feature_count"] == 6 and e1["mapping_rate"] == 1.0


def test_scientific_identity_is_name_free_while_artifact_identity_keeps_names():
    X, y = arrays(180, 6, 2, seed=11)
    n1, n2 = [f"a{i}" for i in range(6)], [f"zz_{i}_nome" for i in range(6)]
    f1 = run_arrays("same_content_1", X, y, n1).frame()
    f2 = run_arrays("zz_other_dataset_id", X, y, n2).frame()
    for col in ("scientific_preprocessing_id", "dataset_content_hash", "ontology_structure_signature", "split_hash", "oracle_id"):
        assert f1[col].tolist() == f2[col].tolist(), col                      # mesma identidade científica
    assert f1["preprocessing_id"].tolist() != f2["preprocessing_id"].tolist()   # o artefacto continua rastreável pelos nomes
    assert f1["dataset_hash"].iloc[0] != f2["dataset_hash"].iloc[0]


def test_without_an_ontology_semantic_arms_are_skipped_explicitly_and_the_rest_ignores_names():
    X, y = arrays(180, 6, 3, seed=5)
    a = run_arrays("identity_a", X, y, [f"m{i}" for i in range(6)], None)
    b = run_arrays("zz_unrelated", X, y, [f"coluna_ñ_{i} (°C)" for i in range(6)], None)
    for out in (a, b):
        assert {s["arm"] for s in out.skipped} == {"reloaded_owl_full", "reloaded_owl_shuffled"}
        assert all("semantic_available=false" in s["reason"] for s in out.skipped)
        assert set(out.frame()["arm"]) == set(ARMS_AF) - {"reloaded_owl_full", "reloaded_owl_shuffled"}
    assert not eq.compare_frames(comparable(a.frame()), comparable(b.frame()))
    assert np.array_equal(a.predictions.filter(like="pred__").to_numpy(), b.predictions.filter(like="pred__").to_numpy())


# ==================================================================================================== C. dataset desconhecido
def test_a_brand_new_unknown_dataset_crosses_the_whole_pipeline_without_any_code_change():
    before = source_digest()
    name = f"zq{secrets.token_hex(5)}_nunca_visto"
    rng = np.random.default_rng(secrets.randbelow(10**6))
    n_features = int(rng.integers(3, 11))
    X, y = arrays(170, n_features, int(rng.integers(2, 5)), seed=int(rng.integers(1, 10**5)))
    names = [f"{secrets.token_hex(2)}_sensor_{i}" for i in range(n_features)]
    out = run_arrays(name, X, y, names)
    assert set(out.frame()["arm"]) == set(ARMS_AF) and not out.skipped
    assert (out.frame()["tuning_execution_count"] == 1).all()
    # o mesmo dataset no pipeline de produção (DataFrame, alvo com nome arbitrário e classes em texto)
    labels = np.array([f"classe_{secrets.token_hex(1)}_{k}" for k in range(len(np.unique(y)))])
    frame = pd.DataFrame(X, columns=names); frame["desfecho_final_x"] = labels[y]
    prod = svc.run_scientific_benchmark(frame, target="desfecho_final_x", seed=4, search=_search())
    assert prod.usable_as_benchmark and prod.oracle_id
    assert set(map(str, prod.result.dataset.class_names)) == set(labels[np.unique(y)])
    assert source_digest() == before


@pytest.mark.parametrize("n_features", [2, 7, 25, 120])
def test_variable_dimensionality_never_triggers_a_special_path(n_features):
    X, y = arrays(170, n_features, 2, seed=n_features)
    out = run_arrays(f"dim_{n_features}", X, y, [f"v{i}" for i in range(n_features)])
    frame = out.frame()
    assert set(frame["arm"]) == set(ARMS_AF) and not out.skipped
    assert (frame["tuning_execution_count"] == 1).all() and frame["n_features_raw" if "n_features_raw" in frame else "split_id"].notna().all()
    assert out.semantic_report[0]["work"]["n_features_raw"] == n_features                  # o nº vem dos dados, não de uma tabela


@pytest.mark.parametrize("labels", [
    ["no", "yes"], ["alpha", "beta", "gamma"], [3, 7, 11], [0, 1, 2, 3], ["a", "b", "c", "d", "e", "f", "g"], [10, 20],
], ids=lambda v: f"{len(v)}classes_{type(v[0]).__name__}")
def test_binary_multiclass_string_and_numeric_class_labels(labels):
    k = len(labels)
    df = make_frame(n=170, n_features=5, n_classes=k, class_names=[str(v) for v in labels], target="resultado", seed=2 + k)
    if isinstance(labels[0], (int, np.integer)):
        df["resultado"] = df["resultado"].astype(int)
    out = svc.run_scientific_benchmark(df, target="resultado", seed=3, search=_search())
    assert out.result.dataset.classes == k and out.usable_as_benchmark
    assert set(map(str, out.result.dataset.class_names)) == {str(v) for v in labels}


@pytest.mark.parametrize("kw", [
    dict(n_features=4, categorical=0), dict(n_features=3, categorical=2), dict(n_features=1, categorical=3),
    dict(n_features=5, categorical=0, missing=True), dict(n_features=4, categorical=2, missing=True),
], ids=["numeric", "mixed", "mostly_categorical", "numeric_missing", "mixed_missing"])
def test_numeric_categorical_mixed_and_missing_features_follow_the_input_contract(kw):
    df = make_frame(n=170, n_classes=3, class_names=["p", "q", "r"], target="rotulo", seed=13, **kw)
    out = svc.run_scientific_benchmark(df, target="rotulo", seed=3, search=_search())
    assert out.usable_as_benchmark and out.result.dataset.features == df.shape[1] - 1


def test_the_validation_benchmark_input_contract_is_numeric_and_rejects_missing_values_explicitly():
    X, y = arrays(60, 3, 2, seed=1)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="imputa"):
        Dataset("novo", X, y, ["a", "b", "c"])


# ==================================================================================================== C7. ontologia
def _tbox(path: Path, name: str, groups):
    pytest.importorskip("owlready2")
    from core.benchmark.domain_tbox import create_domain_tbox
    return create_domain_tbox(path, dataset_name=name, groups=groups)


def test_ontology_behaviour_depends_on_ontology_properties_not_on_dataset_identity(tmp_path):
    pytest.importorskip("owlready2")
    X, y = arrays(180, 6, 2, seed=17)
    names = [f"m{i}" for i in range(6)]
    none = run_arrays("onto_none", X, y, names, None)
    assert set(none.frame()["arm"]) == {"mlp_original", "c45", "trepan_original", "reloaded_core"}              # E/F exigem ontologia
    assert {s["arm"] for s in none.skipped} == {"reloaded_owl_full", "reloaded_owl_shuffled"}
    assert all("semantic_available=false" in s["reason"] for s in none.skipped)

    valid = _tbox(tmp_path / "valid.owl", "qualquer_nome", {"GA": names[:3], "GB": names[3:]})
    r_valid = run_arrays("onto_valid", X, y, names, valid)
    e = r_valid.frame().set_index("arm")
    assert {"reloaded_owl_full", "reloaded_owl_shuffled"} <= set(e.index) and bool(e.loc["reloaded_owl_full", "ontology_valid"])
    assert e.loc["reloaded_owl_full", "mapping_rate"] > 0 and e.loc["reloaded_core", "semantic_features_available"] == 0

    partial = _tbox(tmp_path / "partial.owl", "outro_nome", {"GA": names[:2]})
    r_partial = run_arrays("onto_partial", X, y, names, partial)
    sem = r_partial.semantic_report[0]
    if sem["semantic_available"]:                                      # parcialmente mapeada: aceite, mas a taxa de mapeamento reflete-o
        pe = r_partial.frame().set_index("arm").loc["reloaded_owl_full"]
        assert 0 < pe["mapping_rate"] < 1 and pe["unmapped_feature_count"] > 0
    else:                                                              # ou rejeitada pelo quality gate por cobertura insuficiente
        assert sem["ontology_valid"] is not None and "quality gate" in sem["reason"].lower() or "rejeit" in sem["reason"].lower()

    broken = tmp_path / "invalid.owl"
    broken.write_text("isto <<não>> é uma ontologia OWL válida")
    r_broken = run_arrays("onto_invalid", X, y, names, str(broken))
    assert not r_broken.semantic_report[0]["semantic_available"] and r_broken.semantic_report[0]["ontology_valid"] in (False, None)
    assert set(r_broken.frame()["arm"]) == {"mlp_original", "c45", "trepan_original", "reloaded_core"}          # o pipeline não colapsa

    # a mesma ontologia sob outro nome de dataset produz o mesmo comportamento
    again = run_arrays("completely_other_dataset_id", X, y, names, valid)
    assert not eq.compare_frames(comparable(r_valid.frame()), comparable(again.frame()))
    assert r_valid.semantic_report[0]["semantic_available"] == again.semantic_report[0]["semantic_available"]


# ==================================================================================================== E. CI
def test_ci_runs_the_agnosticism_guard_on_every_pull_request_and_push():
    workflow = (ROOT / ".github" / "workflows" / "agnosticism.yml").read_text(encoding="utf-8")
    assert "pull_request" in workflow and "push" in workflow
    assert "scripts/check_dataset_agnosticism.py" in workflow
    for test_file in ("tests/test_agnosticism_contract.py", "tests/test_dataset_agnosticism.py", "tests/test_v92_static_guards.py"):
        assert test_file in workflow
    assert "agnosticism-guard" in workflow                              # nome estável do job: é o «required status check» do ramo
    full = (ROOT / ".github" / "workflows" / "windows-ci.yml").read_text(encoding="utf-8")
    assert "pytest" in full and "pull_request" in full                  # a suite completa também corre em cada PR


def test_claude_md_keeps_the_architectural_principle_and_points_to_the_executable_contract():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "AGNÓSTICO" in text and "check_dataset_agnosticism.py" in text and "test_agnosticism_contract.py" in text
