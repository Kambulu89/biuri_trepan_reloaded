"""B3.1 (Doc. 36): contrastes emparelhados de fidelidade (TRUE vs controlos) com IC bootstrap e custo; complementa o relatório dos critérios pré-registados."""
import json, sys
from collections import defaultdict
import numpy as np


def main(results, out):
    res = json.load(open(results))
    rng = np.random.default_rng(0)
    out_d = {}
    for p, rows in res["problems"].items():
        by = defaultdict(dict)
        for r in rows:
            by[r["variant"]][r["seed"]] = r
        seeds = sorted(by["ORIG"])
        d = {}
        pairs = [("TRUE", "PERM"), ("TRUE", "RAND"), ("TRUE", "CONV"), ("TRUE", "WIDE"), ("TRUE", "ORIG"), ("HIERARCHY", "ORIG"), ("HIERARCHY", "TRUE"), ("PERM", "ORIG"), ("RAND", "ORIG")]
        for a, b in pairs:
            x = np.array([by[a][s]["fidelity_test"] - by[b][s]["fidelity_test"] for s in seeds])
            bs = [float(np.mean(rng.choice(x, len(x)))) for _ in range(4000)]
            d[f"{a}-{b}"] = {"mean_diff_fidelity": float(x.mean()), "ci95_bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             "wins_losses_ties": [int((x > 1e-12).sum()), int((x < -1e-12).sum()), int((abs(x) <= 1e-12).sum())]}
        t = {v: float(np.mean([by[v][s]["time_s"] for s in seeds])) for v in by}
        d["mean_time_s"] = t
        d["queries_equal_to_orig_all_variants"] = all(by[v][s]["queries"] == by["ORIG"][s]["queries"] for v in by for s in seeds if v not in ("WIDE",))
        out_d[p] = d
    json.dump(out_d, open(out, "w"), indent=1)
    for p, d in out_d.items():
        print(p)
        for k, v in d.items():
            if "-" in k and isinstance(v, dict):
                print(f"  {k:18s} Δfid {v['mean_diff_fidelity']:+.4f} IC95 [{v['ci95_bootstrap'][0]:+.4f}; {v['ci95_bootstrap'][1]:+.4f}] V/D/E {v['wins_losses_ties']}")
        print("  queries iguais ao Original (todas exceto WIDE):", d["queries_equal_to_orig_all_variants"], "| tempo médio (s):", {k: round(v, 2) for k, v in list(d["mean_time_s"].items())[:5]})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
