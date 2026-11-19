"""Executor da FASE 3 da auditoria (EXPERIMENTAL; só DEV): ontologias v2 (relações declaradas) sob a referência com amostrador conjunto.

Reutiliza, sem os alterar, os executores ``run_sampler_c`` e ``run_controls_m29``. Só muda duas coisas, ambas declaradas:
1. a pasta das ontologias passa a ``validation/benchmark/ontologies/v2`` (só existem v2 para alguns datasets; os restantes não se correm);
2. a verificação do manifesto v1 aceita que o ``ontology_hash`` do dataset da unidade difira (é esse o objeto da fase); qualquer outro
   problema continua a abortar. O hash da ontologia v2 usada fica registado em ``<out>/ontology_v2_hashes.json``.

    python -m validation.benchmark.run_fase3 sampler  --unit iris 42 --out DIR [--reps 3] [--reuse-checkpoint-from DIR]
    python -m validation.benchmark.run_fase3 controls --unit iris 42 --out DIR --sc-dir DIR_SAMPLER [--reps 3] [--reuse-checkpoint-from DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
from validation.benchmark import datasets as ds_mod
from validation.benchmark import manifest as mf

V2_DIR = Path(__file__).resolve().parent / "ontologies" / "v2"
ONTOLOGY_HASH_PROBLEM = "ontology_hash mudou desde o manifesto"


def available_datasets():
    return sorted(p.stem for p in V2_DIR.glob("*.owl"))


def activate_v2(dataset: str, out: Path) -> str:
    path = V2_DIR / f"{dataset}.owl"
    if not path.exists():
        raise SystemExit(f"Não existe ontologia v2 para '{dataset}'. Disponíveis: {available_datasets()}")
    ds_mod.ONTOLOGY_DIR = V2_DIR
    original_verify = mf.verify_manifest

    def verify(path_=mf.MANIFEST_PATH, **kwargs):
        report = original_verify(path_, **kwargs)
        kept = [p for p in report["problems"] if not (p.startswith(f"{dataset}:") and p.endswith(ONTOLOGY_HASH_PROBLEM))]
        report = dict(report, problems=kept, ok=not kept, ontology_version="v2")
        return report

    mf.verify_manifest = verify
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    record = out / "ontology_v2_hashes.json"
    import fcntl
    with open(out / ".ontology_v2_hashes.lock", "w") as lock:          # unidades em paralelo escrevem o mesmo registo
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = record.read_text(encoding="utf-8") if record.exists() else ""
        known = json.loads(text) if text.strip() else {}
        if known.get(dataset, digest) != digest:
            raise SystemExit(f"A ontologia v2 de '{dataset}' mudou a meio da campanha.")
        known[dataset] = digest
        tmp = record.with_suffix(".tmp")
        tmp.write_text(json.dumps(known, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(record)
    return digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sampler", "controls"])
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reuse-checkpoint-from", default=None)
    ap.add_argument("--sc-dir", default=None)
    a = ap.parse_args()
    dataset, seed, out = a.unit[0], int(a.unit[1]), Path(a.out)
    activate_v2(dataset, out)
    if a.reuse_checkpoint_from:
        dst = out / "checkpoints"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (Path(a.reuse_checkpoint_from) / "checkpoints").glob(f"{dataset}_seed{seed}.ckpt*"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    if a.mode == "sampler":
        from validation.benchmark.run_sampler_c import run_unit_sampler
        run_unit_sampler(dataset, seed, out, a.reps, "none")
        return
    if not a.sc_dir:
        ap.error("--sc-dir é obrigatório no modo controls")
    from validation.benchmark import run_controls_m29 as M
    M.run_unit_m29(dataset, seed, out, a.reps, Path(a.sc_dir), M.KINDS)
    res = M.runtime_comparability(out, Path(a.sc_dir), dataset, seed)
    if not res["all_comparable"]:
        print("COMPARABILIDADE FALHADA:", json.dumps(res, default=str)[:2000], file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    sys.exit(main())
