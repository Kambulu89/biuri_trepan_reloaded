"""H2 (Doc. 62 §2): integridade de hashes com DUAS identificações distintas e nunca trocadas em silêncio.
* ``sha256_raw``: bytes tal como estão no ficheiro desta cópia de trabalho (proveniência; depende do checkout: no Windows ``autocrlf`` pode converter LF→CRLF).
* ``sha256_canonical``: após a normalização textual EXPLÍCITA ``CRLF→LF`` (``\\r\\n``→``\\n``, nada mais); identifica o conteúdo independentemente da plataforma. Só se aplica a ficheiros de texto;
  para binários ``canonicalization = none`` e canónico = bruto.
Também verifica que um conjunto de caminhos protegidos (dados originais, baselines, evidência histórica, ontologia congelada, ``core/``) NÃO foi alterado entre uma referência git e o HEAD."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".data", ".owl", ".json", ".md", ".py", ".txt", ".csv", ".arff", ".xml", ".yml", ".yaml", ".doc"}
PROTECTED = ["core", "validation/counterfactual_research", "docs/evidence/b1b4/b31", "docs/evidence/b1b4/b32", "docs/evidence/b1b4/b33", "docs/b1b4/manifest_v3/BENCHMARK_MANIFEST_v3.DRAFT.json",
             "docs/b1b4/manifest_v2/BENCHMARK_MANIFEST_v2.DRAFT.json", "validation/benchmark/BENCHMARK_MANIFEST_v1.json", "validation/benchmark/ontologies/german_credit.owl"]


def hashes(path) -> Dict[str, Any]:
    b = Path(path).read_bytes()
    text = Path(path).suffix.lower() in TEXT_SUFFIXES
    canon = b.replace(b"\r\n", b"\n") if text else b
    raw = hashlib.sha256(b).hexdigest()
    can = hashlib.sha256(canon).hexdigest()
    return {"sha256_raw": raw, "sha256_canonical": can, "canonicalization": "CRLF->LF (\\r\\n -> \\n), apenas" if text else "none", "raw_equals_canonical": raw == can, "n_crlf": b.count(b"\r\n"), "size_bytes": len(b)}


def git_blob_sha256(rel: str) -> str:
    """SHA-256 dos bytes do blob no índice git (o que o repositório guarda), independente do checkout."""
    out = subprocess.run(["git", "show", f":{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
    return hashlib.sha256(out).hexdigest()


def unchanged_since(ref: str, paths: List[str]) -> Dict[str, Any]:
    r = subprocess.run(["git", "diff", "--name-status", ref, "HEAD", "--"] + paths, cwd=ROOT, capture_output=True, text=True, check=True)
    changed = [l for l in r.stdout.splitlines() if l.strip()]
    return {"reference": ref, "protected_paths": paths, "changed": changed, "unchanged": not changed}


def main(out: str, ref: str, files: List[str]):
    rows = []
    for f in files:
        h = hashes(ROOT / f)
        try:
            h["sha256_git_blob"] = git_blob_sha256(f)
        except Exception:
            h["sha256_git_blob"] = None
        h["file"] = f
        rows.append(h)
    res = {"policy": "raw (proveniência) e canónico (CRLF->LF) são identificações distintas; nenhum substitui o outro", "files": rows, "protected_unchanged": unchanged_since(ref, PROTECTED)}
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("protegidos inalterados desde", ref, ":", res["protected_unchanged"]["unchanged"], res["protected_unchanged"]["changed"][:3])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
