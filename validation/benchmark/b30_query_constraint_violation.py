"""B3.0 (Doc. 33 §7): diagnóstico OBSERVACIONAL (só atributos de treino; sem rótulos nem teste) da fração de consultas sintéticas que violam uma relação de ordem
DEFINICIONAL entre atributos («máximo ≥ média» dentro de cada família de medições), para o amostrador marginal do Original e para o amostrador C.
A relação é satisfeita em 100 % das linhas reais de treino; uma violação é um ponto de consulta impossível para qualquer indivíduo consistente com a definição."""
import json, sys
import numpy as np


def main(out):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from core.query_sampling import make_joint_model_class
    from core.trepan_original import FeatureDistributionModel
    d = load_breast_cancer(); X = d.data; n = list(d.feature_names)
    res = {}
    for seed in (42, 7, 123, 2024, 11):
        Xtr, _, _, _ = train_test_split(X, d.target, test_size=0.25, stratify=d.target, random_state=seed)
        fams = sorted({s.split(" ", 1)[1] for s in n if s.startswith("mean ")})
        pairs = [(n.index(f"mean {f}"), n.index(f"worst {f}")) for f in fams]
        rate = lambda Z: float(np.mean(np.any(np.column_stack([Z[:, w] < Z[:, m] for m, w in pairs]), axis=1)))
        cls, _ = make_joint_model_class(replica=0)
        zc = cls(random_state=seed, discrete_max_unique=10).fit(Xtr)._draw_unconstrained(5000)
        zo = FeatureDistributionModel(random_state=seed).fit(Xtr)._draw_unconstrained(5000)
        res[str(seed)] = {"train_rows": rate(Xtr), "original_marginal": rate(zo), "sampler_C": rate(zc), "n_families": len(pairs)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
