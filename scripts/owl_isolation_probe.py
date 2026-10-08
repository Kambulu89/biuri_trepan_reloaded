"""Sonda de isolamento OWL: carrega ontologias em SEQUÊNCIA no mesmo processo (ex.: ABA, BAB) e mostra, para cada carga, as features derivadas
e um digest dos valores. Com o isolamento correto, o mesmo ficheiro dá sempre o MESMO digest, seja qual for a ordem.

Uso: python scripts/owl_isolation_probe.py --csv dados.csv --target classe --owl A=a.owl B=b.owl --order ABA BAB
(genérico: as colunas numéricas do CSV, exceto o alvo, são as features; nenhum dataset é conhecido).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ontology_processor import OntologyProcessor  # noqa: E402
from core.ontology_quality import OntologyQualityGate  # noqa: E402
from core.trepan_reloaded_extractor import TrepanReloadedExtractor  # noqa: E402


def step(tag: str, path: str, frame: pd.DataFrame) -> dict:
    onto = TrepanReloadedExtractor.load_ontology_file(path)           # caminho de carga do produto (mundo OWL isolado)
    gate = OntologyQualityGate()
    report = gate.evaluate(list(frame.columns), onto, require_reasoner=False)
    proc = OntologyProcessor(onto, quality_gate=gate)
    proc.fit(frame, accepted_matches=[m for m in report.matches if m.get("accepted")], log=False)
    kinds: dict = {}
    for spec in proc.feature_specs_:
        kinds[spec["kind"]] = kinds.get(spec["kind"], 0) + 1
    out = proc.transform(frame)
    digest = hashlib.sha256(np.ascontiguousarray(out.to_numpy(float)).tobytes()).hexdigest()[:12]
    return {"step": tag, "kinds": kinds, "n_out": int(out.shape[1]), "values_sha": digest}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--owl", nargs="+", required=True, help="TAG=caminho.owl (tags de uma letra)")
    ap.add_argument("--order", nargs="+", default=["ABA", "BAB"])
    args = ap.parse_args()
    frame = pd.read_csv(args.csv)
    frame = frame.drop(columns=[args.target]).select_dtypes("number")
    owls = dict(item.split("=", 1) for item in args.owl)
    result = {order: [step(tag, owls[tag], frame) for tag in order] for order in args.order}
    print(json.dumps(result, indent=1))
    stable = all(len({s["values_sha"] for s in steps if s["step"] == tag}) == 1
                 for steps in result.values() for tag in {s["step"] for s in steps})
    print("ESTÁVEL entre ordens:", stable)
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
