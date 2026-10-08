"""Registo NÃO destrutivo de resultados históricos potencialmente afectados pela colisão entre ontologias OWL (mundo owlready2 partilhado).

Nada é apagado nem alterado: o registo (JSON) acrescenta, por ficheiro, ``sha256``, a classificação e o estado ``NOT_VERIFIED_OWL_ISOLATION``.
Critérios (conservadores; o processo que gerou cada resultado NÃO está registado, logo não é possível provar isolamento):

* ``not_applicable`` — o ficheiro não envolve ontologia (sem marcadores OWL/semânticos);
* ``potentially_contaminated_multi_ontology`` — o artefacto/execução contém ≥2 ontologias/datasets semânticos (provável carga sequencial no mesmo
  processo): resultados dos braços semânticos NÃO verificados;
* ``unverified_single_ontology`` — uma só ontologia, mas sem prova de que o processo não carregou outra antes: NÃO verificado (risco baixo).

Reverificar = repetir a unidade com o código corrigido (um World isolado por carga) e comparar; ver docs/ONTOLOGY_ISOLATION_AUDIT.md.
Uso: ``python -m validation.historical_results_registry --out docs/evidence/ontology_isolation/historical_results_registry.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

STATUS = "NOT_VERIFIED_OWL_ISOLATION"
MARKERS = re.compile(r"ontolog|\.owl|owl_|semantic|reloaded|efsr|knowledge_source", re.I)
MULTI_HINT = re.compile(r"datasets?|dataset_id|dataset_name|ontologies", re.I)
SKIP_SUFFIX = {".pyc", ".png", ".svg", ".pdf", ".pkl", ".joblib"}
MAX_BYTES = 40_000_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _distinct_values(text: str, key_pattern: str) -> set:
    return set(re.findall(rf'"?{key_pattern}"?\s*[:,=]\s*"([^"]+)"', text))


def classify(path: Path, text: str) -> Dict[str, Any]:
    rel = path.as_posix()
    involves = bool(MARKERS.search(text)) or bool(MARKERS.search(rel))
    if not involves:
        return {"classification": "not_applicable", "reason": "sem marcadores de ontologia/braços semânticos"}
    ids = _distinct_values(text, r"(?:dataset_id|dataset_name|dataset)") | _distinct_values(text, r"ontology_(?:id|file|path|name)")
    ids = {i for i in ids if i and i.lower() not in {"none", "null"}}
    if len(ids) >= 2:
        return {"classification": "potentially_contaminated_multi_ontology",
                "reason": f"contém {len(ids)} identificadores distintos de dataset/ontologia no mesmo artefacto (carga sequencial provável)",
                "distinct_ids": sorted(ids)[:20]}
    return {"classification": "unverified_single_ontology",
            "reason": "uma ontologia, mas o processo gerador não está registado: isolamento não demonstrável"}


def build(root: Path) -> Dict[str, Any]:
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    rows: List[Dict[str, Any]] = []
    for rel in files:
        p = root / rel
        top = rel.split("/")[0]
        in_scope = (top == "results" or ("/" not in rel) or (top == "counterfactuals" and "resultados" in rel)) and rel.endswith((".json", ".csv"))
        if not in_scope or p.suffix.lower() in SKIP_SUFFIX or not p.is_file() or p.stat().st_size > MAX_BYTES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        info = classify(p.relative_to(root), text)
        rows.append({"path": rel, "sha256": _sha256(p), "status": STATUS if info["classification"] != "not_applicable" else "NOT_AFFECTED_NO_ONTOLOGY", **info})
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    return {
        "schema": 1,
        "purpose": "marcar (sem eliminar nem alterar) resultados históricos potencialmente afectados pela colisão de ontologias OWL",
        "fix_reference": "docs/ONTOLOGY_ISOLATION_AUDIT.md",
        "limitation": "o processo gerador de cada resultado não foi registado; classificação por conteúdo é conservadora. NOT_VERIFIED ≠ errado: significa não demonstrado.",
        "counts": counts, "files": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reg = build(Path(a.root).resolve())
    Path(a.out).write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(reg["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
