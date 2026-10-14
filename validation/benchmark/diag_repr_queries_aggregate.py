"""Agrega os JSON de ``diag_repr_queries`` (por dataset, média das sementes). Só leitura.  python -m ... --dir DIR"""
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    by = defaultdict(list)
    for f in sorted(Path(a.dir).glob("*_*.json")):
        d = json.loads(f.read_text())
        if d.get("available"):
            by[d["dataset"]].append(d)
    mean = lambda v: (st.mean([x for x in v if x is not None]) if [x for x in v if x is not None] else None)
    out = {}
    for ds, L in by.items():
        r = lambda path: mean([_get(d, path) for d in L])
        out[ds] = {"units": len(L), "n_original": L[0]["representation"]["n_original"], "n_derived": L[0]["representation"]["n_derived"],
                   "oracle_projection_matches_all": all(d["oracle_projection_matches"] for d in L),
                   "transform_finite_all": all(d["representation"]["finite_all"] for d in L), "originals_unchanged_all": all(d["representation"]["originals_unchanged"] for d in L),
                   "deterministic_all": all(d["representation"]["deterministic"] for d in L),
                   "projector_fixpoint_max_err": max(d["representation"]["projector_fixpoint_max_abs_err"] for d in L),
                   "share_derived_near_duplicate_r2_gt_0.95": r(("representation", "derived_summary", "share_near_duplicate_r2_gt_0.95")),
                   "share_derived_mi_above_null95": r(("representation", "derived_summary", "share_mi_above_null95")),
                   "share_derived_near_constant": r(("representation", "derived_summary", "share_near_constant")),
                   "mi_median_derived": r(("representation", "derived_summary", "mi_median")), "mi_median_original": r(("representation", "originals_scale", "mi_median")),
                   "std_median_derived": r(("representation", "derived_summary", "std_median")), "std_median_original": r(("representation", "originals_scale", "std_median")),
                   "cand_original": r(("candidate_space", "n_candidates_original_space")), "cand_enriched": r(("candidate_space", "n_candidates_enriched_space")),
                   "root_top12_share_derived": r(("candidate_space", "root_top12_share_derived")),
                   "root_best_is_derived": r(("candidate_space", "root_best_is_derived")),
                   "marginal_draw_inconsistency_mean": r(("draw_consistency", "marginal_draw_derived_inconsistency_mean_std_units")),
                   "projected_draw_inconsistency_mean": r(("draw_consistency", "projected_draw_derived_inconsistency_mean")),
                   "derived_literals_in_inert_tree": r(("derived_literals_in_final_inert_tree",))}
        for k in ("mean_ks_base_vs_sem", "near_boundary_base", "near_boundary_sem", "label_entropy_base", "label_entropy_sem", "share_same_choice", "share_same_choice_Do_vs_Dalt",
                  "fid_choiceDo_on_Do", "fid_choiceDo_on_Ds", "fid_choiceDs_on_Do", "fid_choiceDs_on_Ds", "fid_choiceDo_on_real", "fid_choiceDs_on_real", "fid_choiceDalt_on_real",
                  "fid_choiceDo_on_Dalt", "fid_choiceDalt_on_Dalt"):
            out[ds]["q_" + k] = r(("query_summary", k))
    s = json.dumps(out, indent=1, default=str)
    if a.out: Path(a.out).write_text(s)
    print(s)


def _get(d, path):
    for p in path:
        d = d.get(p) if isinstance(d, dict) else None
        if d is None:
            return None
    return float(d) if isinstance(d, (bool, int, float)) else d


if __name__ == "__main__":
    main()
