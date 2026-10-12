"""Sonda de equivalência Reloaded(sem ontologia) vs Original — SÓ LEITURA, não altera código.

Compara, com o mesmo MLP/oráculo congelado, dados, parâmetros e seed: regras, predições, nº de membership queries, **nº de amostras/chamadas consultadas ao
oráculo** e o *hash do fluxo de entradas enviadas ao oráculo* (E4 do contrato). Uso: ``python docs/evidence/b1b4/equivalence_probe.py > out.jsonl``.
"""
import hashlib
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

from core.mlp_factory import build_mlp_for_data
from core.scientific_experiment_contract import freeze_oracle
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


class StreamHashingOracle:
    """Delega no FrozenOracle e regista hash/contagem de tudo o que o oráculo recebe (predict e predict_proba)."""

    def __init__(self, inner):
        self.inner, self.h, self.calls, self.samples = inner, hashlib.sha256(), 0, 0

    def _see(self, kind, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.h.update(kind.encode() + str(X.shape).encode() + X.tobytes())
        self.calls += 1
        self.samples += len(X)

    def predict(self, X):
        self._see("predict", X)
        return self.inner.predict(X)

    def predict_proba(self, X):
        self._see("proba", X)
        return self.inner.predict_proba(X)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def fit(cls, kw, budget, Xtr, mlp, names, tag):
    fo = freeze_oracle(mlp, Xtr, builder="probe")
    spy = StreamHashingOracle(fo)
    with fo.scope(tag):
        tree = cls(**budget, **kw).fit(Xtr, oracle=spy, feature_names=names)
    return tree, spy


def run(name, data, seed):
    X, y = data.data, data.target
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    mlp = build_mlp_for_data(Xtr, ytr, random_state=seed).fit(Xtr, ytr)
    names = [f"f{i}" for i in range(X.shape[1])]
    budget = dict(min_sample=max(len(Xtr), 200), max_queries=2000, max_nodes=9, max_depth=4, max_n=3, beam_width=2, min_samples_leaf=2,
                  max_features_per_node=min(12, X.shape[1]), purity_epsilon=0.05, random_state=seed)
    armD = dict(alpha=0.35, beta=0.20, error_focused_refinement=True, semantic_gain_strength=1.0, semantic_group_strength=0.15,
                semantic_active_query_fraction=0.65, mirror_when_no_semantic_effect=False)       # = braço D do benchmark (sem informação ontológica)
    t_o, s_o = fit(TrepanOriginalClassifier, {}, budget, Xtr, mlp, names, "o")
    res = {"dataset": name, "seed": seed, "original": {"oracle_samples": s_o.samples, "oracle_calls": s_o.calls, "membership_queries": int(t_o.membership_queries_)}}
    for tag, kw in (("reloaded_armD", armD), ("reloaded_default", {})):
        t_r, s_r = fit(TrepanReloadedClassifier, kw, budget, Xtr, mlp, names, "r")
        res[tag] = {
            "E1_E2_rules_equal": t_o.export_rules() == t_r.export_rules(),
            "E5_predictions_equal": bool(np.array_equal(t_o.predict(Xte), t_r.predict(Xte))),
            "E4_membership_queries_equal": int(t_o.membership_queries_) == int(t_r.membership_queries_),
            "E4_oracle_samples": [s_o.samples, s_r.samples], "E4_oracle_calls": [s_o.calls, s_r.calls],
            "E4_oracle_input_stream_equal": s_o.h.hexdigest() == s_r.h.hexdigest(),
            "E7_nodes": [len(list(t_o.iter_nodes())), len(list(t_r.iter_nodes()))],
        }
    return res


if __name__ == "__main__":
    for nm, ld in (("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer)):
        for sd in (42, 7):
            print(json.dumps(run(nm, ld(), sd)), flush=True)
