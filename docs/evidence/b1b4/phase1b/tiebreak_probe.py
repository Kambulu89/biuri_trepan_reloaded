"""Sonda do desempate m-of-n por ruído de último bit (Fase 1b, tarefa associada). SÓ LEITURA: subclasses de ensaio, o Original não é alterado.

Compara o Original com uma variante cujo score é EXATAMENTE simétrico (IG calculado sempre sobre a máscara canónica), isolando o efeito do ruído de ulp
na escolha da orientação (teste vs complemento). Mede: regras idênticas? previsões idênticas? nº de nós onde a orientação difere?
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
from eq_support import make_problem, BUDGET, SpyOracle
from core.trepan_original import TrepanOriginalClassifier, _information_gain


class SymOriginal(TrepanOriginalClassifier):
    def _split_selection_score(self, y, mask, test=None):
        mask = np.asarray(mask, dtype=bool)
        key = mask if mask.tobytes() <= (~mask).tobytes() else ~mask      # orientação canónica: score(mask) ≡ score(~mask) bit-a-bit
        return _information_gain(y, key, self.classes_)


def nodes(tree):
    return [(n.test.m, tuple((l.feature, round(l.threshold, 12), l.greater) for l in n.test.literals)) for n in tree.iter_nodes() if getattr(n, "test", None) is not None]


def canon(t):
    m, lits = t
    comp = (len(lits) - m + 1, tuple((f, th, not g) for f, th, g in lits))
    return min((m, lits), comp)


rows = []
for seed in range(1, 41):
    for kind, p, k in (("continuous", 8, 3), ("mixed", 6, 2), ("continuous", 10, 4)):
        X, names, o = make_problem(seed, p=p, k=k, kind=kind)
        a = TrepanOriginalClassifier(**BUDGET, random_state=seed).fit(X[:200], oracle=o, feature_names=names)
        b = SymOriginal(**BUDGET, random_state=seed).fit(X[:200], oracle=o, feature_names=names)
        na, nb = nodes(a), nodes(b)
        rows.append({"seed": seed, "problem": f"{kind}-p{p}-k{k}", "rules_equal": a.export_rules() == b.export_rules(),
                     "pred_equal": bool(np.array_equal(a.predict(X[200:]), b.predict(X[200:]))), "n_nodes": len(na),
                     "tests_identical": na == nb, "tests_equal_up_to_complement": [canon(t) for t in na] == [canon(t) for t in nb],
                     "oriented_differently": sum(1 for s, t in zip(na, nb) if s != t)})
summary = {"runs": len(rows), "rules_equal": sum(r["rules_equal"] for r in rows), "pred_equal": sum(r["pred_equal"] for r in rows),
           "tests_identical": sum(r["tests_identical"] for r in rows), "tests_equal_up_to_complement": sum(r["tests_equal_up_to_complement"] for r in rows),
           "nodes_total": sum(r["n_nodes"] for r in rows), "nodes_oriented_differently": sum(r["oriented_differently"] for r in rows)}
print(json.dumps(summary))
pathlib.Path(__file__).with_name("tiebreak_probe_result.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
