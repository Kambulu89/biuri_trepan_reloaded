"""Reanálise (só leitura) de split_audit_allnodes_v2.json: decomposição do score Reloaded e do gate EFSR. Reproduz os números do Doc. 1 §3 de docs/b1b4."""
import json
import pathlib

import numpy as np

path = pathlib.Path(__file__).with_name("split_audit_allnodes_v2.json")
runs = json.loads(path.read_text(encoding="utf-8"))
rows = [x for r in runs for x in r["all_nodes"]]


def col(key):
    return np.array([x[key] if x.get(key) is not None else np.nan for x in rows], float)


score, alpha, beta = col("semantic_selection_score"), col("alpha"), col("beta")
additive = alpha * col("onto_depth") + beta * col("error_coverage")      # termo semântico aditivo (B1)
base = score - additive                                                  # IG normalizado * peso^k (componente estatística)
print(f"nós={len(rows)}  aditivo mediana={np.median(additive):.2f} [{additive.min():.2f},{additive.max():.2f}]  "
      f"estatística mediana={np.median(base):.2f} [{base.min():.2f},{base.max():.2f}]")
print(f"razão aditivo/estatística mediana={np.median(additive / base):.2f}  aditivo>estatística em {(additive > base).sum()}/{len(rows)}")
print(f"semantic_bonus legado (score - IG cru) mediana={np.nanmedian(col('semantic_bonus')):.2f}  (escalas mistas)")
changed = col("decision_changed") == 1
gain = col("local_fidelity_gain")
print(f"propostas diferentes do data-only={int((col('proposed_decision_changed') == 1).sum())}/{len(rows)}  aceites={int(changed.sum())}  "
      f"aceites com ganho local<0={int((gain[changed] < 0).sum())}  amostra de decisão mediana={np.nanmedian(col('decision_sample_size')):.0f}")
for onto in sorted({r["ontology"] for r in runs}):
    d = np.array([r["fid_full"] - r["fid_core"] for r in runs if r["ontology"] == onto]) * 100
    print(f"{onto}: Δfidelidade Reloaded+OWL − Core = {d.mean():+.2f} pp (DP {d.std(ddof=1):.2f}, n={len(d)})")
