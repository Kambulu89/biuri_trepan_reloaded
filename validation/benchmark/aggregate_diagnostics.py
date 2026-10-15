"""Agrega as saídas de plausibility_study e audit_representation (diagnóstico DEV, exploratório)."""
import argparse, glob, json, os
import numpy as np


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def plausibility(d, prefix="", exclude=("mixed",)):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(f)
        if prefix and not b.startswith(prefix):
            continue
        if not prefix and any(b.startswith(e) for e in exclude):
            continue
        runs.append(json.load(open(f)))
    out = {"n_runs": len(runs), "strategies": {}}
    if not runs:
        return out
    for s in runs[0]["strategies"]:
        keys = runs[0]["strategies"][s].keys()
        out["strategies"][s] = {k: _mean([r["strategies"][s].get(k) for r in runs]) for k in keys}
    return out


def audit(d):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
    res = {"n_runs": len(runs)}
    for name in ("real", "shuffled", "conventional"):
        acc = {}
        for r in runs:
            x = r[name]
            acc.setdefault("n_derived", []).append(x["n_derived"])
            a = x["groups_alignment_with_data_structure"]
            acc.setdefault("mean_abs_spearman_within_groups", []).append(a["mean_abs_spearman_within_groups"])
            acc.setdefault("mean_z_vs_random_groups", []).append(a["mean_z_vs_random_groups"])
            acc.setdefault("share_groups_above_null_p95", []).append(a["share_groups_above_null_p95"])
            acc.setdefault("share_derived_nearly_single_original", []).append(x["share_derived_nearly_a_single_original(|r|>0.95)"])
            acc.setdefault("share_derived_mi_above_best_member", []).append(x["share_derived_mi_above_best_member"])
            acc.setdefault("conditional_info_Y_D_given_X", []).append(x["conditional_info_Y_D_given_X"])
            t = x["tree_utility"]
            acc.setdefault("share_bootstrap_derived_wins_root", []).append(t["share_bootstrap_subsamples_where_derived_wins"])
            acc.setdefault("mean_ig_gain_derived_minus_original", []).append(t["mean_ig_gain_derived_minus_original"])
            c = x["candidates"]
            acc.setdefault("candidate_ratio_enriched_over_original", []).append(c["enriched_space"] / max(c["original_space"], 1))
        res[name] = {k: _mean(v) for k, v in acc.items()}
    res["per_unit_real"] = {f"{r['dataset']}:{r['seed']}": {
        "matching": r["matching"],
        "n_derived": r["real"]["n_derived"],
        "cond_info": r["real"]["conditional_info_Y_D_given_X"],
        "nearly_single": r["real"]["share_derived_nearly_a_single_original(|r|>0.95)"],
        "mean_z": r["real"]["groups_alignment_with_data_structure"]["mean_z_vs_random_groups"],
    } for r in runs}
    res["stability"] = {"min_corr": min(r["stability_under_80pct_refit"]["min_corr"] for r in runs),
                        "max_rel_std_change": max(r["stability_under_80pct_refit"]["max_rel_std_change"] for r in runs)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pl"); ap.add_argument("--au"); ap.add_argument("--out")
    a = ap.parse_args()
    o = {"plausibility_dev": plausibility(a.pl), "plausibility_mixed": plausibility(a.pl, prefix="mixed"), "audit": audit(a.au)}
    json.dump(o, open(a.out, "w"), indent=1)
    print(json.dumps(o, indent=1))
