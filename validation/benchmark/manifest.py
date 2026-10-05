"""BENCHMARK MANIFEST imutável (pertence a ``validation/benchmark/``, nunca ao núcleo).

Regista, por dataset, a identidade dos dados e da ontologia, e, globalmente, as master seeds, os braços A–F, o commit do
código, os hashes do código congelado e as versões das bibliotecas. NÃO contém hiperparâmetros do TREPAN/Reloaded: o desenho
experimental é único e igual para todos os datasets. O ficheiro é escrito uma só vez (não sobrescreve) e protegido por hash.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.benchmark import manifest as core_mf
from core.benchmark.runner import BenchmarkConfig, default_arms
from validation.benchmark import datasets as ds_mod
from validation.benchmark.c45_audit import audit_native_c45

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "1"
MANIFEST_PATH = Path(__file__).resolve().parent / f"BENCHMARK_MANIFEST_v{MANIFEST_VERSION}.json"
MASTER_SEEDS = (42, 7, 123, 2024, 11)                        # fixadas ANTES de qualquer resultado
MAIN_ARM_GROUPS = ("A", "B", "C", "D", "E", "F")
EVALUATION = {"scheme": "holdout", "test_size": 0.25, "stratified": True, "one_split_per_master_seed": True}

# Ficheiros cujo conteúdo define o algoritmo avaliado (detecção de deriva do código congelado).
FROZEN_CODE_FILES = (
    "core/trepan_scientific_tuning.py", "core/scientific_experiment_contract.py", "core/trepan_original.py",
    "core/trepan_reloaded_historical.py", "core/c45_j48_tree.py", "core/benchmark/runner.py", "core/benchmark/semantic.py",
    "core/benchmark/ontology_gate.py", "core/benchmark/structure.py", "core/benchmark/metrics.py", "core/benchmark/analysis.py",
)
# Chaves proibidas por dataset: hiperparâmetros do TREPAN/Reloaded e da seleção estrutural.
FORBIDDEN_DATASET_KEYS = {"max_nodes", "purity_epsilon", "alpha", "beta", "lam", "lambda", "min_sample", "max_queries", "max_depth",
                          "max_n", "beam_width", "min_samples_leaf", "confidence_factor", "semantic_active_query_fraction",
                          "error_focused_refinement", "onto_weight", "trepan", "reloaded", "hyperparameters", "hyperparams"}


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _feature_types(X: np.ndarray) -> Dict[str, Any]:
    kinds = []
    for j in range(X.shape[1]):
        col = X[:, j]
        kinds.append("binary" if len(np.unique(col)) == 2 else "integer" if np.all(np.equal(np.mod(col, 1), 0)) else "continuous")
    counts = {k: kinds.count(k) for k in sorted(set(kinds))}
    return {"counts": counts, "per_feature": kinds, "all_numeric": True}


def dataset_entry(spec: ds_mod.DatasetSpec) -> Dict[str, Any]:
    ds, target, classes = spec.load()
    onto = spec.ontology_path
    return {
        "dataset_id": spec.dataset_id, "kind": spec.kind, "ranking_eligible": bool(spec.ranking_eligible),
        "origin": spec.origin, "version": spec.version,
        "data_hash": core_mf.hash_dataset(ds.X, ds.y, ds.feature_names),
        "source_file_sha256": ds_mod.source_file_sha256(spec.source_file),
        "n_samples": int(len(ds.y)), "n_features": int(ds.X.shape[1]),
        "feature_names": list(ds.feature_names), "feature_types": _feature_types(ds.X),
        "target_column": target, "n_classes": int(len(classes)), "classes": list(classes),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(ds.y, return_counts=True))},
        "has_missing_values": bool(np.isnan(ds.X).any()),
        "ontology_id": spec.ontology_id,
        "ontology_hash": core_mf.hash_file(str(onto)) if onto is not None else None,
        "ontology_file": None if onto is None else str(onto.relative_to(ROOT)),
        "expected_mapping_rate": ds_mod.expected_mapping_rate(spec),
        "expected_mapping_rate_note": "metadado informativo (cobertura dos grupos da TBox); não é threshold nem critério de seleção",
        "master_seeds": list(MASTER_SEEDS),
    }


def arms_entry() -> List[Dict[str, Any]]:
    cfg = BenchmarkConfig(extra_ablations=False)
    return [{"group": a.group, "arm_id": a.arm_id, "family": a.family, "description": a.description, "control": bool(a.control)}
            for a in default_arms(cfg) if a.group in MAIN_ARM_GROUPS]


def manifest_digest(manifest: Dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def assert_no_hyperparameters(manifest: Dict[str, Any]) -> None:
    """Nenhum dataset pode ter hiperparâmetros específicos do TREPAN/Reloaded."""
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() in FORBIDDEN_DATASET_KEYS:
                    raise ValueError(f"Hiperparâmetro/chave proibida no manifesto de dataset: {'.'.join(path + [str(k)])}")
                walk(v, path + [str(k)])
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + [str(i)])
    for entry in manifest.get("datasets", []):
        walk(entry, [str(entry.get("dataset_id"))])


def build_manifest(dataset_ids=None) -> Dict[str, Any]:
    ids = list(dataset_ids or (ds_mod.MAIN_DATASETS + ds_mod.CONTROLLED_DATASETS))
    entries = [dataset_entry(ds_mod.REGISTRY[i]) for i in ids]
    audit = audit_native_c45()
    manifest = {
        "manifest_version": MANIFEST_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_frozen": True,
        "design_statement": ("Desenho experimental CONGELADO: algoritmos e hiperparâmetros não são alterados em resposta aos resultados; "
                             "um resultado inesperado é evidência a investigar. A configuração estrutural é escolhida por split, só no treino, "
                             "pelo tuning científico congelado (ontology-blind) e é igual em C, D, E e F."),
        "datasets": entries,
        "main_ranking_datasets": [e["dataset_id"] for e in entries if e["ranking_eligible"]],
        "controlled_validation_datasets": [e["dataset_id"] for e in entries if not e["ranking_eligible"]],
        "master_seeds": list(MASTER_SEEDS), "evaluation": dict(EVALUATION), "arms": arms_entry(),
        "different_oracle_experiment_excluded_from_A_to_F": ["mlp_ontological", "reloaded_e2e"],
        "c45": {"label": audit["algorithm_name"], "audit_is_c45": audit["is_c45"], "audit_checks": audit["checks"],
                "known_deviations": audit["known_deviations"], "canonical_no_node_cap": True},
        "code_commit": _git("rev-parse", "HEAD"), "code_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_dirty_at_freeze": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "frozen_code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN_CODE_FILES},
        "library_versions": core_mf.library_versions(), "python_version": sys.version, "platform": platform.platform(),
    }
    assert_no_hyperparameters(manifest)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def write_manifest(path: Path = MANIFEST_PATH, dataset_ids=None) -> Path:
    """Escreve o manifesto UMA vez; recusa sobrescrever (um desenho diferente exige uma nova versão)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"O manifesto {path.name} é imutável e já existe; crie uma nova versão em vez de o sobrescrever.")
    manifest = build_manifest(dataset_ids)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    path.chmod(0o444)
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(path: Path = MANIFEST_PATH, *, check_code: bool = True) -> Dict[str, Any]:
    """Verifica integridade do manifesto, dados e ontologias atuais e deriva do código congelado. Devolve um relatório."""
    m = load_manifest(path)
    problems: List[str] = []
    if manifest_digest(m) != m.get("manifest_sha256"):
        problems.append("hash do manifesto não corresponde ao conteúdo (manifesto alterado)")
    try:
        assert_no_hyperparameters(m)
    except ValueError as exc:
        problems.append(str(exc))
    for e in m["datasets"]:
        spec = ds_mod.REGISTRY.get(e["dataset_id"])
        if spec is None:
            problems.append(f"dataset desconhecido: {e['dataset_id']}")
            continue
        cur = dataset_entry(spec)
        for key in ("data_hash", "source_file_sha256", "n_samples", "n_features", "ontology_hash", "classes", "has_missing_values"):
            if cur[key] != e[key]:
                problems.append(f"{e['dataset_id']}: {key} mudou desde o manifesto")
    drift = []
    if check_code:
        for f, h in m["frozen_code_sha256"].items():
            if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h:
                drift.append(f)
    return {"ok": not problems, "problems": problems, "frozen_code_drift": drift, "manifest_sha256": m.get("manifest_sha256"),
            "manifest_code_commit": m.get("code_commit"), "current_code_commit": _git("rev-parse", "HEAD")}
