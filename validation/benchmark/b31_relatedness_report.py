"""B3.1 (Doc. 35): compara a relatedness histórica com a taxonómica (Lin, Wu–Palmer) nas ontologias DEV e em ontologias sintéticas controladas. Só atributos/ontologia; sem rótulos."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from validation.benchmark import datasets as D
from validation.benchmark.b30_axiom_probe import dprop, klass, write
from validation.benchmark.b31_facts import compile_registry
from validation.benchmark.b31_relatedness import legacy_matrix, relatedness_matrix, semantic_units


def stats(M, groups):
    n = len(groups)
    iu = np.triu_indices(n, 1)
    same = np.array([groups[i] is not None and groups[i] == groups[j] for i, j in zip(*iu)])
    v = M[iu]
    return {"distinct_values": int(len(set(np.round(v, 4)))), "values_sample": sorted(set(np.round(v, 4).tolist()))[:6], "mean_same_concept": float(v[same].mean()) if same.any() else None,
            "mean_cross_concept": float(v[~same].mean()) if (~same).any() else None, "n_pairs": int(len(v)), "n_same": int(same.sum())}


def dev():
    out = {}
    for d in ("iris", "wine", "breast_cancer", "digits"):
        spec = D.REGISTRY[d]; ds, _, _ = spec.load(); F = [str(n) for n in ds.feature_names]
        reg = compile_registry(spec.ontology_path, F)
        grp = []
        for f in F:
            c = [fa.entities[1] for fa in reg.by_kind(__import__('validation.benchmark.b31_facts', fromlist=['Kind']).Kind.ATTRIBUTE_APPLIES_TO_CONCEPT) if fa.attributes[0] == f and fa.derivation == "ASSERTED"]
            grp.append(c[0] if c else None)
        r = {"legacy": stats(legacy_matrix(str(spec.ontology_path), F), grp)}
        for m in ("lin", "wup"):
            M, _ = relatedness_matrix(reg, F, m); r[m] = stats(M, grp)
        r["units"] = [(u.concept, len(u.features), round(u.ic, 3)) for u in semantic_units(reg, F)]
        r["registry"] = reg.summary()
        out[d] = r
    return out


def synthetic():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        body = (klass("Root") + klass("Mid1", ["Root"]) + klass("Mid2", ["Root"]) + klass("L1", ["Mid1"]) + klass("L2", ["Mid1"]) + klass("L3", ["Mid2"])
                + dprop("temperature_mean", "L1") + dprop("temperature_peak", "L1") + dprop("humidity_level", "L2") + dprop("pressure_value", "L3"))
        F = ["temperature_mean", "temperature_peak", "humidity_level", "pressure_value"]
        p = write(t, "h", body); reg = compile_registry(p, F)
        out["hierarchy3"] = {"features": F, "legacy": np.round(legacy_matrix(str(p), F), 4).tolist(), "lin": np.round(relatedness_matrix(reg, F, "lin")[0], 4).tolist(), "wup": np.round(relatedness_matrix(reg, F, "wup")[0], 4).tolist()}
    return out


if __name__ == "__main__":
    res = {"dev": dev(), "synthetic": synthetic()}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=str))
    for d, r in res["dev"].items():
        print(d, {m: (r[m]["distinct_values"], r[m]["mean_same_concept"], r[m]["mean_cross_concept"]) for m in ("legacy", "lin", "wup")}, r["units"][:4])
