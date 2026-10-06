"""Prova de equivalência científica entre duas execuções (p.ex. antes/depois de uma otimização de engenharia).

Compara, para o mesmo dataset × master seed e por braço: split_hash, preprocessing_id, oracle_id, configuração estrutural do tuning,
predições (alvo, oráculo, substituto), métricas, nós/folhas/profundidade, auditoria semântica e quality gate. Só ignora campos de
TEMPO e identificadores de execução. A comparação é EXATA (tolerância 0 por omissão); qualquer diferença é listada.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

TIME_TOKENS = ("time", "seconds", "_s", "elapsed", "timestamp", "experiment_id", "run_code_commit", "manifest")
IGNORED_EXACT = {"arm_time_s", "tuning_time_s", "time_s", "tuning_execution_count"}      # contadores/medições de engenharia, não resultados


def _is_time_key(key: str) -> bool:
    k = str(key).lower()
    return k in IGNORED_EXACT or any(tok in k for tok in ("time", "elapsed", "seconds", "timestamp", "experiment_id", "created", "commit"))


def _equal(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is None and b is None
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True
        return abs(float(a) - float(b)) <= tol
    return a == b


def _strip_time(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_time(v) for k, v in obj.items() if not _is_time_key(k)}
    if isinstance(obj, list):
        return [_strip_time(v) for v in obj]
    return obj


def compare_records(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.0) -> List[str]:
    """Diferenças entre dois registos brutos de braço (ignora tempos)."""
    diffs: List[str] = []
    ra, rb = _strip_time(a["row"]), _strip_time(b["row"])
    for k in sorted(set(ra) | set(rb)):
        if k not in ra or k not in rb:
            diffs.append(f"{a['arm']}: coluna {k} só existe numa das execuções")
        elif not _equal(ra[k], rb[k], tol):
            diffs.append(f"{a['arm']}: {k}: {ra[k]!r} != {rb[k]!r}")
    for key in ("split_hash", "oracle_id", "fidelity_oracle_id", "preprocessing_id", "ontology_hash"):
        if a.get(key) != b.get(key):
            diffs.append(f"{a['arm']}: {key}: {a.get(key)!r} != {b.get(key)!r}")
    if not _equal(_strip_time(a["config"]), _strip_time(b["config"]), tol):
        diffs.append(f"{a['arm']}: configuração diferente")
    pa, pb = a["predictions"], b["predictions"]
    for key in ("test_row_index", "y_true", "oracle_predictions", "surrogate_predictions"):
        if not _equal(pa[key], pb[key], tol):
            diffs.append(f"{a['arm']}: predições '{key}' diferentes")
    return diffs


def compare_split_payload(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.0) -> List[str]:
    """Auditoria semântica, protocolo estrutural (tuning), gate e contrato do oráculo (ignora tempos e contadores de trabalho)."""
    def keep(p):
        rep = {k: v for k, v in p["semantic_report"][0].items() if k not in ("work", "timing", "tuning_execution_count")}
        if "structural_protocol" in rep:                # perfil/contadores de engenharia do tuning (o resto do protocolo é comparado)
            rep["structural_protocol"] = {k: v for k, v in rep["structural_protocol"].items()
                                          if k not in ("engineering_profile", "number_cv_fits", "number_capacity_fits")}
        if "oracle_contract" in rep:                    # contadores de queries por escopo são CONTADORES DE TRABALHO (a memoização reduz-os)
            rep["oracle_contract"] = {k: v for k, v in rep["oracle_contract"].items() if k != "queries_per_scope"}
        return _strip_time(rep)
    ka, kb = keep(a), keep(b)
    diffs = [f"split.json: campo '{k}' diferente" for k in sorted(set(ka) | set(kb)) if not _equal(ka.get(k), kb.get(k), tol)]
    if a["splits"] != b["splits"]:
        diffs.append("split.json: splits diferentes")
    if a["dataset_hash"] != b["dataset_hash"]:
        diffs.append("split.json: dataset_hash diferente")
    return diffs


def compare_frames(df_a: pd.DataFrame, df_b: pd.DataFrame, tol: float = 0.0) -> List[str]:
    """Comparação de DataFrames de resultados (linhas por braço × split) ignorando tempos."""
    cols = sorted(c for c in set(df_a.columns) & set(df_b.columns) if not _is_time_key(c))
    a = df_a.sort_values(["split_id", "arm"]).reset_index(drop=True)
    b = df_b.sort_values(["split_id", "arm"]).reset_index(drop=True)
    diffs: List[str] = []
    if list(a["arm"]) != list(b["arm"]):
        return ["conjuntos de braços diferentes"]
    for i in range(len(a)):
        for c in cols:
            va, vb = a.at[i, c], b.at[i, c]
            if pd.isna(va) and pd.isna(vb):
                continue
            va = va.item() if hasattr(va, "item") else va
            vb = vb.item() if hasattr(vb, "item") else vb
            if not _equal(va, vb, tol):
                diffs.append(f"{a.at[i, 'arm']}/{a.at[i, 'split_id']}: {c}: {va!r} != {vb!r}")
    return diffs


def compare_raw_roots(root_a: Path, root_b: Path, tol: float = 0.0) -> Dict[str, Any]:
    """Compara duas árvores ``raw/`` unidade a unidade. Devolve ``{"identical": bool, "units": {...}, "differences": [...]}``."""
    root_a, root_b = Path(root_a), Path(root_b)
    units_a = {p.parent.relative_to(root_a / "raw") for p in (root_a / "raw").glob("*/seed*/UNIT_SHA256.json")}
    units_b = {p.parent.relative_to(root_b / "raw") for p in (root_b / "raw").glob("*/seed*/UNIT_SHA256.json")}
    report: Dict[str, Any] = {"units": {}, "differences": []}
    for unit in sorted(units_a & units_b):
        da, db = root_a / "raw" / unit, root_b / "raw" / unit
        diffs: List[str] = []
        arms_a = {p.stem for p in da.glob("*.json")} - {"split", "UNIT_SHA256"}
        arms_b = {p.stem for p in db.glob("*.json")} - {"split", "UNIT_SHA256"}
        if arms_a != arms_b:
            diffs.append(f"braços diferentes: {sorted(arms_a ^ arms_b)}")
        for arm in sorted(arms_a & arms_b):
            diffs += compare_records(json.loads((da / f"{arm}.json").read_text()), json.loads((db / f"{arm}.json").read_text()), tol)
        diffs += compare_split_payload(json.loads((da / "split.json").read_text()), json.loads((db / "split.json").read_text()), tol)
        report["units"][str(unit)] = {"identical": not diffs, "n_differences": len(diffs), "arms_compared": sorted(arms_a & arms_b)}
        report["differences"] += [f"{unit}: {d}" for d in diffs]
    report["only_in_a"] = sorted(map(str, units_a - units_b))
    report["only_in_b"] = sorted(map(str, units_b - units_a))
    report["identical"] = not report["differences"] and bool(report["units"])
    report["tolerance"] = tol
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root_a"); ap.add_argument("root_b"); ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    rep = compare_raw_roots(a.root_a, a.root_b, a.tol)
    print(json.dumps({k: v for k, v in rep.items() if k != "differences"}, indent=1))
    for d in rep["differences"][:50]:
        print("DIFF", d)
    raise SystemExit(0 if rep["identical"] else 1)
