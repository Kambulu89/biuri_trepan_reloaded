"""Resultados BRUTOS imutáveis por dataset × master_seed × braço, escritos antes de qualquer agregação.

Layout: ``<root>/raw/<dataset>/seed<seed>/<arm>.json`` (+ ``split.json`` e ``UNIT_SHA256.json``). Os ficheiros são só de leitura
e nunca são sobrescritos. Todas as tabelas agregadas devem poder ser reconstruídas EXCLUSIVAMENTE a partir destes ficheiros.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from core.benchmark.io import _clean
from core.benchmark.metrics import OracleInfo, classification_bundle, fidelity_to_oracle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_dir(root: Path, dataset: str, seed: int) -> Path:
    return Path(root) / "raw" / dataset / f"seed{seed}"


def _write_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Resultado bruto imutável já existe: {path}")
    path.write_text(json.dumps(_clean(payload), indent=1, ensure_ascii=False), encoding="utf-8")
    path.chmod(0o444)


def write_unit(result, root: Path, *, manifest_sha256: str, labels: List[int]) -> List[Path]:
    """Escreve os brutos de UM dataset × master seed (``result`` vem de um BenchmarkRunner com uma única seed)."""
    df = result.frame()
    seeds = sorted(int(s) for s in df["seed"].unique())
    if len(seeds) != 1:
        raise ValueError("write_unit exige exatamente uma master seed por unidade.")
    seed = seeds[0]
    udir = unit_dir(root, result.dataset, seed)
    udir.mkdir(parents=True, exist_ok=False)
    pred = result.predictions
    written: List[Path] = []
    for _, row in df.iterrows():
        sel = pred[pred["split_id"] == row["split_id"]]
        col = f"pred__{row['arm']}"
        oracle_cols = {c[len("oraclepred__"):]: sel[c].tolist() for c in sel.columns if c.startswith("oraclepred__")}
        rec = {"dataset": result.dataset, "master_seed": seed, "arm": row["arm"], "family": row["family"], "group": row.get("benchmark_group"),
               "split_id": row["split_id"], "split_hash": row["split_hash"], "oracle_id": row.get("oracle_id"),
               "fidelity_oracle_id": row.get("fidelity_oracle_id"), "preprocessing_id": row.get("preprocessing_id"),
               "ontology_hash": row.get("ontology_hash"), "manifest_sha256": manifest_sha256,
               "config": {"config_hash": row.get("config_hash"), "arm": next((a for a in result.arms if a["arm_id"] == row["arm"]), None),
                          "structural": {k: row.get(k) for k in row.index if str(k).startswith("structural_") or k in
                                         ("tree_max_nodes", "purity_epsilon", "query_budget", "tree_max_n")}},
               "labels": list(map(int, labels)), "minority_label": row.get("minority_label"),
               "row": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in row.to_dict().items()},
               "predictions": {"test_row_index": sel["test_row_index"].tolist(), "y_true": sel["y_real"].tolist(),
                               "oracle_predictions": oracle_cols,
                               "surrogate_predictions": sel[col].tolist() if col in sel else None}}
        path = udir / f"{row['arm']}.json"
        _write_once(path, rec)
        written.append(path)
    split_payload = {"dataset": result.dataset, "master_seed": seed, "splits": [s for s in result.splits], "semantic_report": result.semantic_report, "skipped_arms": result.skipped,
                     "dataset_hash": result.dataset_hash, "ontology_hash": result.ontology_hash, "manifest": result.manifest}
    _write_once(udir / "split.json", split_payload)
    written.append(udir / "split.json")
    index = {p.name: _sha(p) for p in sorted(written)}
    _write_once(udir / "UNIT_SHA256.json", index)
    return written


def unit_complete(root: Path, dataset: str, seed: int) -> bool:
    return (unit_dir(root, dataset, seed) / "UNIT_SHA256.json").exists()


def verify_raw_integrity(root: Path) -> Dict[str, Any]:
    problems: List[str] = []
    n = 0
    for idx in sorted((Path(root) / "raw").glob("*/seed*/UNIT_SHA256.json")):
        listed = json.loads(idx.read_text(encoding="utf-8"))
        for name, h in listed.items():
            f = idx.parent / name
            n += 1
            if not f.exists() or _sha(f) != h:
                problems.append(f"{f}: ausente ou alterado")
        extra = {p.name for p in idx.parent.glob("*.json")} - set(listed) - {"UNIT_SHA256.json"}
        problems += [f"{idx.parent / e}: ficheiro bruto não indexado" for e in sorted(extra)]
    return {"ok": not problems and n > 0, "files": n, "problems": problems}


def load_raw(root: Path, *, verify: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reconstrói (linhas por braço, registos brutos, splits) lendo SÓ os ficheiros brutos."""
    if verify:
        v = verify_raw_integrity(root)
        if not v["ok"]:
            raise RuntimeError(f"Resultados brutos corrompidos ou ausentes: {v['problems'][:5]}")
    records, splits = [], []
    for f in sorted((Path(root) / "raw").glob("*/seed*/*.json")):
        if f.name == "UNIT_SHA256.json":
            continue
        payload = json.loads(f.read_text(encoding="utf-8"))
        if f.name == "split.json":
            splits.append(payload)
        else:
            records.append(payload)
    rows = pd.DataFrame([r["row"] for r in records])
    return rows, records, splits


def verify_raw_metrics(records: List[Dict[str, Any]], tol: float = 1e-9) -> Dict[str, Any]:
    """Recomputa accuracy, balanced accuracy, macro-F1 e fidelity a partir de y_true/predições brutas e compara com as métricas."""
    checked, worst, mism = 0, 0.0, []
    oracle_of = {}
    for r in records:
        pr = r["predictions"]
        if pr["surrogate_predictions"] is None:
            continue
        yt, yp = np.asarray(pr["y_true"]), np.asarray(pr["surrogate_predictions"])
        minority = int(float(r["minority_label"])) if r.get("minority_label") not in (None, "") else None
        bundle = classification_bundle(yt, yp, labels=r["labels"], minority_label=minority)
        row = r["row"]
        for k, v in bundle.items():
            if row.get(k) is not None and v is not None and not (isinstance(v, float) and np.isnan(v)):
                d = abs(float(row[k]) - float(v)); worst = max(worst, d); checked += 1
                if d > tol:
                    mism.append((r["arm"], r["split_id"], k, row[k], v))
        if row.get("fidelity_to_oracle") is not None and row.get("oracle_name") in pr["oracle_predictions"]:
            f = fidelity_to_oracle(np.asarray(pr["oracle_predictions"][row["oracle_name"]]), yp, OracleInfo(str(row["oracle_name"]), "x", 1))
            d = abs(float(row["fidelity_to_oracle"]) - f); worst = max(worst, d); checked += 1
            if d > tol:
                mism.append((r["arm"], r["split_id"], "fidelity_to_oracle", row["fidelity_to_oracle"], f))
    return {"ok": not mism and checked > 0, "checked_values": checked, "max_abs_diff": worst, "mismatches": mism[:10]}
