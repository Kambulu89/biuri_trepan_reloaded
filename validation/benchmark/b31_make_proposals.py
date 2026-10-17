"""B3.1 (Doc. 37): gera pacotes de PROPOSTA de axiomas (não validados) com diagnósticos só sobre atributos de treino. Revisões: DOMAIN_REVIEW_PENDING.
Os textos de fonte são citados de memória e marcados ``verification_required``: o revisor de domínio tem de confirmar a citação exata."""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parents[2] / "docs/evidence/b1b4/b31/proposals"
REV = {r: {"status": "REVIEW_PENDING", "reviewer_name": None, "date": None, "signature_ref": None} for r in ("domain_expert", "ontology_engineer", "methodology")}
REV["domain_expert"]["status"] = "DOMAIN_REVIEW_PENDING"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_breast_cancer(); n = list(d.feature_names)
    fams = sorted({s.split(" ", 1)[1] for s in n if s.startswith("mean ")})
    rates = {}
    for seed in (42, 7, 123, 2024, 11):
        Xtr, _, _, _ = train_test_split(d.data, d.target, test_size=0.25, stratify=d.target, random_state=seed)
        for f in fams:
            a, b = n.index(f"mean {f}"), n.index(f"worst {f}")
            rates.setdefault(f, []).append(float(np.mean(Xtr[:, a] > Xtr[:, b])))
    axioms = []
    for f in fams:
        axioms.append({"id": f"wdbc_order_{f.replace(' ', '_')}", "axiom_text": f"worst({f}) >= mean({f})  [mesma grandeza, mesmo conjunto de observações por imagem]", "type": "ORDER_RELATION",
                       "attributes": [f"mean {f}", f"worst {f}"], "relation_class": "FORMAL_DOMAIN_CONSTRAINT",
                       "justification": "«worst» = média dos três maiores valores dos núcleos da imagem; «mean» = média de todos os núcleos; para ≥ 3 núcleos a média dos 3 maiores ≥ média de todos (identidade decorrente das definições).",
                       "sources": [{"citation": "UCI ML Repository — Breast Cancer Wisconsin (Diagnostic): informação de atributos (wdbc.names)", "locator": "Attribute Information: média, erro padrão e 'worst' (média dos três maiores valores)",
                                    "verification_required": True}],
                       "conditions_analysis": {"same_quantity": "sim (mesma medição por núcleo)", "same_observation_set": "sim (mesma imagem)", "compatible_statistical_definitions": "sim, se a definição de 'worst' for a citada (a confirmar)",
                                               "preprocessing_preserves_relation": "SÓ no espaço bruto: padronização por coluna (médias/DP diferentes) NÃO preserva a ordem; aplicar apenas a consultas em escala original",
                                               "missing_values_handled": "sem valores em falta no conjunto; relação só aplicável a linhas completas"},
                       "empirical_check_training_only": {"violation_rate_per_split": rates[f], "interpretation": "diagnóstico; NÃO substitui fonte nem revisão"}, "status_requested": "DOMAIN_DOCUMENTED", "status_now": "HYPOTHESIS_UNTIL_REVIEW"})
    pkg = {"package_id": "wdbc_order_relations_v0", "ontology": {"file": None, "sha256": None}, "axioms": axioms, "reviews": REV,
           "preregistration": {"frozen_before_evaluation": False, "freeze_commit": None, "note": "a congelar só após as 3 revisões"},
           "comparison_arms": ["original", "enriched", "permuted", "random", "empty"],
           "claude_role": "preparou a proposta e os diagnósticos; NÃO valida (DOMAIN_REVIEW_PENDING)"}
    (OUT / "wdbc_order_relations").mkdir(parents=True, exist_ok=True)
    (OUT / "wdbc_order_relations/proposal.json").write_text(json.dumps(pkg, indent=1))
    w = load_wine(); nw = list(w.feature_names)
    Xw = w.data
    a, b = nw.index("flavanoids"), nw.index("total_phenols")
    neg = {"package_id": "wine_phenol_subset_rejected_v0", "axioms": [{"id": "wine_flav_le_phenols", "axiom_text": "flavanoids <= total_phenols", "type": "ORDER_RELATION", "attributes": ["flavanoids", "total_phenols"],
           "relation_class": "HYPOTHESIS", "justification": "flavonoides são fenóis; a inclusão conceptual não implica ordem entre as MEDIDAS (métodos analíticos diferentes)",
           "sources": [], "empirical_check_training_only": {"violation_rate_all_rows": float(np.mean(Xw[:, a] > Xw[:, b])), "interpretation": "regularidade que falha: NÃO é axioma"},
           "status_requested": "none", "status_now": "REJECTED_AS_AXIOM"}], "reviews": REV, "comparison_arms": []}
    (OUT / "wine_phenol_subset_rejected").mkdir(parents=True, exist_ok=True)
    (OUT / "wine_phenol_subset_rejected/proposal.json").write_text(json.dumps(neg, indent=1))
    print(json.dumps({"wdbc_max_violation": max(max(v) for v in rates.values()), "wine_violation": neg["axioms"][0]["empirical_check_training_only"]["violation_rate_all_rows"]}))


if __name__ == "__main__":
    main()
