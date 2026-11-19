import json, glob, sys, os
import numpy as np, pandas as pd
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(root + "/raw/*/seed*/*.json")):
    b = os.path.basename(f)
    if b in ("split.json", "UNIT_SHA256.json"): continue
    r = json.load(open(f))["row"]
    rows.append({k: r.get(k) for k in ("dataset", "seed", "arm", "accuracy_real_labels", "macro_f1_real_labels", "weighted_f1_real_labels",
                 "precision_macro_real_labels", "recall_macro_real_labels", "balanced_accuracy_real_labels", "fidelity_to_oracle",
                 "node_count", "leaf_count", "rule_count", "depth", "average_rule_literals", "m_of_n_count", "tree_training_time", "arm_time_s",
                 "membership_queries", "total_oracle_samples", "semantic_split_count", "semantic_features_used", "ontology_influenced_splits",
                 "tree_max_nodes", "node_cap_reached", "oracle_accuracy_real_labels", "semantic_effect_observed", "semantic_attempt_status")})
d = pd.DataFrame(rows)
d.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 250)
print("unidades:", d.groupby("dataset").seed.nunique().to_dict())
g = d.groupby(["dataset", "arm"])[["accuracy_real_labels", "macro_f1_real_labels", "fidelity_to_oracle", "node_count", "rule_count", "depth", "arm_time_s"]].agg(["mean", "std"]).round(3)
print(g.to_string())
p = d.pivot_table(index=["dataset", "seed"], columns="arm", values=["accuracy_real_labels", "fidelity_to_oracle", "node_count"])
a, f = p["accuracy_real_labels"], p["fidelity_to_oracle"]
chk = pd.DataFrame({
 "acc R>O": a.reloaded_owl_full > a.trepan_original, "acc R>C45": a.reloaded_owl_full > a.c45, "acc O>C45": a.trepan_original > a.c45,
 "fid R>O": f.reloaded_owl_full > f.trepan_original, "fid R>C45": f.reloaded_owl_full > f.c45,
 "acc R>Shuf": a.reloaded_owl_full > a.reloaded_owl_shuffled, "fid R>Shuf": f.reloaded_owl_full > f.reloaded_owl_shuffled,
 "acc MLP>C45": a.mlp_original > a.c45})
chk["JOINT"] = chk[["acc R>O", "acc R>C45", "acc O>C45", "fid R>O", "fid R>C45"]].all(axis=1)
print(chk.astype(int).to_string())
print("por dataset (fração de unidades):"); print(chk.groupby(level=0).mean().round(2).to_string())
print("global:", chk.mean().round(2).to_dict())
