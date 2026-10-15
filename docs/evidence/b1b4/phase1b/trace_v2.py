"""Instrumentação do fluxo COMPLETO de consultas ao oráculo no caso V2 (mirror). SÓ LEITURA: não altera código de produção.

Para cada chamada regista: método, forma, hash do conteúdo, hash da resposta, fase (reloaded-pass | mirror-rerun | original-alone),
cadeia de chamadores (funções), e se o conteúdo (bytes exatos) coincide com alguma chamada anterior do mesmo método.
"""
import hashlib, json, pathlib, sys, traceback, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
from eq_support import ClosedFormOracle, make_problem, scenario_fit_kwargs, BUDGET
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


def _h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()[:16]


class Tracer:
    def __init__(self, inner):
        self.inner, self.calls, self.classes_ = inner, [], inner.classes_

    def _rec(self, kind, X):
        fr = []
        phase = "reloaded-pass"
        for f in traceback.extract_stack()[:-2][::-1]:
            if f.filename.endswith(("trepan_original.py", "trepan_reloaded_historical.py")):
                fr.append(f.name)
        # fase: dentro de Original.fit cujo self é um Original puro => rerun do mirror (ou Original isolado)
        import inspect
        for fi in inspect.stack()[2:]:
            if fi.function == "fit" and fi.filename.endswith("trepan_original.py"):
                phase = "plain-original" if type(fi.frame.f_locals.get("self")) is TrepanOriginalClassifier else "reloaded-pass"
                break
        out = getattr(self.inner, kind)(X)
        a = np.asarray(X, dtype=float)
        self.calls.append({"i": len(self.calls), "method": kind, "shape": list(a.shape), "in_hash": _h(a), "out_hash": _h(out),
                           "phase": phase, "callers": fr[:4]})
        return out

    def predict(self, X): return self._rec("predict", X)
    def predict_proba(self, X): return self._rec("predict_proba", X)


def run(scenario, seed, flags, p=8, k=3, kind="continuous"):
    X, names, oracle = make_problem(seed, p=p, k=k, kind=kind)
    Xtr = X[:200]
    t_o = Tracer(oracle); o = TrepanOriginalClassifier(**BUDGET, random_state=seed).fit(Xtr, oracle=t_o, feature_names=names)
    for c in t_o.calls: c["phase"] = "original-alone"
    t_r = Tracer(oracle)
    r = TrepanReloadedClassifier(**BUDGET, random_state=seed, **flags).fit(Xtr, oracle=t_r, feature_names=names, **scenario_fit_kwargs(scenario, p))
    return o, t_o, r, t_r


def analyse(t_o, t_r):
    rer = [c for c in t_r.calls if c["phase"] == "plain-original"]
    first = [c for c in t_r.calls if c["phase"] == "reloaded-pass"]
    orig_in = [(c["method"], c["in_hash"]) for c in t_o.calls]
    seen_first = {(c["method"], c["in_hash"]): c for c in first}
    ident_rerun_vs_first = [c for c in rer if (c["method"], c["in_hash"]) in seen_first]
    return {
        "original_alone": {"calls": len(t_o.calls), "samples": sum(c["shape"][0] for c in t_o.calls)},
        "reloaded_pass": {"calls": len(first), "samples": sum(c["shape"][0] for c in first),
                          "by_method": {m: sum(1 for c in first if c["method"] == m) for m in ("predict", "predict_proba")}},
        "mirror_rerun": {"calls": len(rer), "samples": sum(c["shape"][0] for c in rer)},
        "rerun_stream_equals_original_alone": [(c["method"], c["in_hash"]) for c in rer] == orig_in,
        "rerun_calls_with_input_identical_to_a_reloaded_pass_call": [
            {"method": c["method"], "shape": c["shape"], "rerun_i": c["i"], "first_i": seen_first[(c["method"], c["in_hash"])]["i"],
             "same_response": c["out_hash"] == seen_first[(c["method"], c["in_hash"])]["out_hash"]} for c in ident_rerun_vs_first],
        "samples_redundant_exact": sum(c["shape"][0] for c in ident_rerun_vs_first),
        "first_pass_callers": sorted({(c["method"], tuple(c["callers"][:3])) for c in first}),
    }


if __name__ == "__main__":
    rows = []
    for sc in ("projector", "weights", "groups", "relatedness", "combined", "depths"):
        for seed in (3, 11):
            o, t_o, r, t_r = run(sc, seed, {})
            if not r.semantic_effect_mirror_applied_:
                continue
            a = analyse(t_o, t_r)
            a.update(scenario=sc, seed=seed, mirror_applied=True, rules_equal=o.export_rules() == r.export_rules())
            rows.append(a)
            print(json.dumps({k: v for k, v in a.items() if k != "first_pass_callers"}, default=str)[:900], flush=True)
    pathlib.Path(__file__).with_name("trace_v2_result.json").write_text(json.dumps(rows, indent=1, default=str))
