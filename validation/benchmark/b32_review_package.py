"""B3.2 (Doc. 39): gera o PACOTE DE REVISÃO das ontologias/axiomas candidatos (propostas, formulários em branco, hashes). Nenhum nome, assinatura ou aprovação é inventado:
todos os estados de revisão ficam DOMAIN_REVIEW_PENDING / ONTOLOGY_REVIEW_PENDING / METHODOLOGY_REVIEW_PENDING. Diagnósticos empíricos usam SÓ atributos (nunca rótulos)
e servem de verificação de consistência, nunca de prova de universalidade."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/evidence/b1b4/b32/review_package"
PENDING = {"domain_expert": {"status": "DOMAIN_REVIEW_PENDING", "reviewer_name": None, "date": None, "signature_ref": None},
           "ontology_engineer": {"status": "ONTOLOGY_REVIEW_PENDING", "reviewer_name": None, "date": None, "signature_ref": None},
           "methodology": {"status": "METHODOLOGY_REVIEW_PENDING", "reviewer_name": None, "date": None, "signature_ref": None}}
ARMS = ["original", "enriched", "permuted", "random", "empty"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wdbc():
    d = load_breast_cancer()
    n = list(d.feature_names)
    descr = d.DESCR
    i = descr.index("The mean, standard error")
    excerpt = descr[i:i + 330].replace("\n    ", " ").strip()
    fams = sorted({s.split(" ", 1)[1] for s in n if s.startswith("mean ")})
    diag = {}
    for f in fams:
        a, b = n.index(f"mean {f}"), n.index(f"worst {f}")
        diag[f] = {"violation_rate_all_569_rows": float(np.mean(d.data[:, a] > d.data[:, b])), "min_gap_worst_minus_mean": float((d.data[:, b] - d.data[:, a]).min())}
    src = {"citation": f"Texto de atributos incluído em scikit-learn {sklearn.__version__} (`load_breast_cancer().DESCR`), cópia do UCI ML Breast Cancer Wisconsin (Diagnostic)",
           "locator": "DESCR → «Attribute Information»", "excerpt": excerpt, "descr_sha256": sha(descr.encode()), "verification_required": "o revisor deve conferir contra o UCI original (wdbc.names)"}
    order = {"package_id": "wdbc_order_relations_v1", "ontology": {"file": None, "sha256": None}, "claude_role": "preparou a proposta e os diagnósticos; NÃO valida", "reviews": PENDING,
             "preregistration": {"frozen_before_evaluation": False, "freeze_commit": None}, "comparison_arms": ARMS,
             "axioms": [{"id": f"wdbc_worst_ge_mean_{f.replace(' ', '_')}", "axiom_text": f"worst({f}) >= mean({f})", "type": "ORDER_RELATION", "attributes": [f"mean {f}", f"worst {f}"],
                         "relation_class": "MATHEMATICAL_IDENTITY_CONDITIONAL",
                         "justification": "Pela definição documentada, «mean» = média dos valores de todos os núcleos da imagem e «worst» = média dos três maiores valores dos MESMOS núcleos; a média dos k maiores (k ≤ nº de núcleos) é ≥ à média de todos. Identidade decorrente das definições, condicional a: (i) «largest» ser o maior valor numérico; (ii) o comportamento quando a imagem tem < 3 núcleos; (iii) nenhum arredondamento que inverta a ordem.",
                         "sources": [src],
                         "conditions_analysis": {"same_quantity": "sim (mesma medição por núcleo)", "same_observation_set": "sim, segundo a descrição (mesma imagem)", "compatible_statistical_definitions": "sim, se «worst» = média dos 3 maiores (a confirmar)",
                                                 "preprocessing_preserves_relation": "SÓ no espaço bruto; padronizar cada coluna com média/DP diferentes NÃO preserva a ordem → qualquer uso tem de ocorrer na escala original, nunca em colunas padronizadas",
                                                 "missing_values_handled": "sem valores em falta no conjunto; a relação aplica-se apenas a linhas completas"},
                         "empirical_check_attributes_only": diag[f], "status_requested": "DOMAIN_DOCUMENTED (identidade condicional)", "status_now": "HYPOTHESIS_UNTIL_DOMAIN_REVIEW",
                         "open_questions_for_domain_reviewer": ["Qual é o nº mínimo de núcleos por imagem? «worst» com < 3 núcleos usa os disponíveis?", "«largest» é sempre o maior valor numérico, incluindo para simetria e dimensão fractal?", "A cópia do UCI foi arredondada de forma que possa inverter a ordem?"]}
                        for f in fams]}
    comp = {"package_id": "wdbc_compactness_identity_rejected_v1", "reviews": PENDING, "comparison_arms": [], "axioms": [{
        "id": "wdbc_compactness_mean_identity", "axiom_text": "compactness_mean = perimeter_mean^2 / area_mean - 1", "type": "MATHEMATICAL_IDENTITY",
        "attributes": ["mean compactness", "mean perimeter", "mean area"], "relation_class": "EMPIRICAL_REGULARITY",
        "justification": "A definição «perimeter^2 / area − 1» vale POR NÚCLEO; «mean» é a média sobre núcleos, e a média de um quociente não é o quociente das médias (desigualdade de Jensen) → NÃO é identidade para atributos agregados.",
        "sources": [{"citation": src["citation"], "locator": "DESCR → compactness (perimeter^2 / area - 1.0)"}], "status_requested": "none", "status_now": "REJECTED_AS_IDENTITY_FOR_AGGREGATED_ATTRIBUTES",
        "empirical_check_attributes_only": {"note": "correlação elevada mas erro relativo mediano grande (ver Doc. 33 §7 e `b30`); regularidade empírica, não restrição"}}]}
    geo = {"package_id": "wdbc_geometric_approximations_v1", "reviews": PENDING, "comparison_arms": [], "axioms": [{
        "id": "wdbc_perimeter_radius_circle", "axiom_text": "perimeter ≈ 2π·radius ; area ≈ π·radius²", "type": "ORDER_OF_MAGNITUDE_RELATION", "attributes": ["mean perimeter", "mean radius", "mean area"],
        "relation_class": "EMPIRICAL_REGULARITY", "justification": "Exato só para contornos circulares; os núcleos não o são → aproximação, no máximo preferência/hipótese; nunca restrição.",
        "sources": [], "status_requested": "none", "status_now": "HYPOTHESIS",
        "empirical_check_attributes_only": {"median_perimeter_over_2pi_radius": float(np.median(d.data[:, n.index("mean perimeter")] / (2 * np.pi * d.data[:, n.index("mean radius")]))),
                                            "median_area_over_pi_radius2": float(np.median(d.data[:, n.index("mean area")] / (np.pi * d.data[:, n.index("mean radius")] ** 2)))}}]}
    return {"wdbc_order_relations": order, "wdbc_compactness_identity_rejected": comp, "wdbc_geometric_approximations": geo}


def wine():
    w = load_wine()
    nw = list(w.feature_names)
    a, b = nw.index("flavanoids"), nw.index("total_phenols")
    return {"wine_phenol_subset_rejected": {"package_id": "wine_phenol_subset_rejected_v1", "reviews": PENDING, "comparison_arms": [], "axioms": [{
        "id": "wine_flav_le_phenols", "axiom_text": "flavanoids <= total_phenols", "type": "ORDER_RELATION", "attributes": ["flavanoids", "total_phenols"], "relation_class": "HYPOTHESIS",
        "justification": "Flavonoides são fenóis, mas a inclusão conceptual não implica ordem entre as MEDIDAS (métodos analíticos diferentes). Mantido REJEITADO como restrição universal; usado como TESTE NEGATIVO de validação semântica.",
        "sources": [], "status_requested": "none", "status_now": "REJECTED_AS_UNIVERSAL_CONSTRAINT",
        "empirical_check_attributes_only": {"violation_rate_all_178_rows": float(np.mean(w.data[:, a] > w.data[:, b])), "interpretation": "regularidade que falha: nunca axioma; não forçar na ontologia"}}]}}


def digits():
    d = load_digits()
    X = d.data
    names = list(d.feature_names)
    idx = {(int(n.split("_")[1]), int(n.split("_")[2])): i for i, n in enumerate(names)}
    adj, non = [], []
    rng = np.random.default_rng(0)
    for (r, c), i in idx.items():
        for (r2, c2), j in idx.items():
            if (r2, c2) <= (r, c):
                continue
            cc = np.corrcoef(X[:, i], X[:, j])[0, 1] if X[:, i].std() > 0 and X[:, j].std() > 0 else np.nan
            if np.isnan(cc):
                continue
            (adj if abs(r - r2) + abs(c - c2) == 1 else non).append(cc)
    return {"digits_spatial_topology": {"package_id": "digits_spatial_topology_v1", "reviews": PENDING, "comparison_arms": ARMS, "axioms": [{
        "id": "digits_pixel_grid_adjacency", "axiom_text": "pixel_r_c e pixel_r2_c2 são adjacentes (4-vizinhança) sse |r−r2|+|c−c2| = 1", "type": "SPATIAL_ADJACENCY (tipo de facto ainda inexistente no registo)",
        "attributes": [f"pixel_{r}_{c}" for r, c in sorted(idx)[:2]] + ["…(64 atributos)"], "relation_class": "FORMAL_DOMAIN_CONSTRAINT",
        "justification": "Estrutura do próprio conjunto: «8x8 image of integer pixels» obtida por contagem de píxeis ligados em blocos 4×4 não sobrepostos de um bitmap 32×32; os nomes dos atributos codificam a posição. A relação de vizinhança é geometria do suporte (não depende dos rótulos).",
        "sources": [{"citation": f"scikit-learn {sklearn.__version__} `load_digits().DESCR`; UCI Optical Recognition of Handwritten Digits", "locator": "DESCR → Attribute Information; descrição do pré-processamento", "verification_required": True}],
        "conditions_analysis": {"nota": "é relação ESTRUTURAL entre atributos (aplicável como preferência/relatedness), não uma restrição sobre valores; não elimina candidatos nem consultas"},
        "empirical_check_attributes_only": {"mean_abs_correlation_adjacent_pairs": float(np.mean(np.abs(adj))), "mean_abs_correlation_non_adjacent_pairs": float(np.mean(np.abs(non))), "interpretation": "regularidade empírica compatível; não prova"},
        "status_requested": "DOMAIN_DOCUMENTED (estrutura)", "status_now": "HYPOTHESIS_UNTIL_DOMAIN_REVIEW"}]}}


FORM_DOMAIN = """# Formulário de REVISÃO DE DOMÍNIO — {pkg}
Estado: **DOMAIN_REVIEW_PENDING** (nenhum revisor designado; nenhuma aprovação registada).
Revisor (nome humano): ______  Data: ______  Referência de assinatura: ______   [NÃO preenchido por Claude]

Para CADA axioma do pacote (`proposal.json`) responder, com evidência:
1. A relação é **verdadeira** no domínio? (identidade matemática / restrição formal / regularidade empírica / hipótese)
2. Qual é a **fonte** (citação exata e localizador)? Confirmar a citação indicada (`verification_required`).
3. As **definições** dos atributos são as assumidas (mesma grandeza, mesmo conjunto de observações, definições estatísticas compatíveis)?
4. **Âmbito de validade** e **pressupostos** (unidades, escala, pré-processamento que preserva a relação, valores em falta, arredondamentos, nº mínimo de observações por agregado)?
5. Existem **contra-exemplos** conhecidos? A observação em 100 % das linhas de treino NÃO é prova de universalidade.
6. Decisão: APPROVED_AS_IDENTITY / APPROVED_AS_FORMAL_CONSTRAINT / APPROVED_AS_PREFERENCE_ONLY / REJECTED / NEEDS_MORE_EVIDENCE.
"""
FORM_ONTO = """# Formulário de REVISÃO ONTOLÓGICA — {pkg}
Estado: **ONTOLOGY_REVIEW_PENDING**. Revisor: ______  Data: ______  Assinatura: ______   [NÃO preenchido por Claude]

1. A modelação em **OWL 2 DL** é válida (perfil DL verificado com ferramenta formal; o proxy do repositório NÃO substitui)?
2. **Consistência** e **satisfazibilidade** (reasoner independente, versão registada)?
3. **Inferências** esperadas vs obtidas; alguma inferência inesperada?
4. A **interpretação dos axiomas** para atributos tabulares é a pretendida (domínio ≠ validador de linhas; disjunção ≠ incompatibilidade de combinação)?
5. **Mundo aberto:** nenhuma regra assume negação por ausência?
6. **Matching:** nomes/rótulos não ambíguos (sem empates); cobertura; ver Doc. 40.
7. Decisão: APPROVED / APPROVED_WITH_CHANGES / REJECTED.
"""
FORM_METH = """# Formulário de REVISÃO METODOLÓGICA — {pkg}
Estado: **METHODOLOGY_REVIEW_PENDING**. Revisor: ______  Data: ______  Assinatura: ______   [NÃO preenchido por Claude]

1. **Independência dos dados de teste**: nenhum axioma deriva de rótulos, de resultados ou de dados de teste?
2. **Ausência de leakage** (treino/validação/teste; sementes; calibração só com o Original)?
3. **Pré-registo**: problemas, critérios, braços e hash do código congelados ANTES da avaliação (commit de congelamento)?
4. **Versões e hashes** (ontologia, dados, código, ambiente) verificáveis e reproduzíveis?
5. **Validade dos controlos**: permutado ao nível da relação, aleatório, vazio, convencional, orçamento igualado, Original WIDE; um controlo inválido nunca conta como evidência.
6. Decisão: APPROVED / APPROVED_WITH_CHANGES / REJECTED.
"""


def main():
    pk = {}
    pk.update(wdbc()); pk.update(wine()); pk.update(digits())
    (OUT / "proposals").mkdir(parents=True, exist_ok=True)
    (OUT / "review_forms").mkdir(parents=True, exist_ok=True)
    for k, v in pk.items():
        (OUT / "proposals" / k).mkdir(parents=True, exist_ok=True)
        (OUT / "proposals" / k / "proposal.json").write_text(json.dumps(v, indent=1, ensure_ascii=False))
    for name, tpl in (("DOMAIN_REVIEW", FORM_DOMAIN), ("ONTOLOGY_REVIEW", FORM_ONTO), ("METHODOLOGY_REVIEW", FORM_METH)):
        (OUT / "review_forms" / f"{name}.md").write_text(tpl.format(pkg="<id do pacote>"))
    # hashes (conteúdo normalizado para LF)
    files = sorted(list(OUT.rglob("*.json")) + list(OUT.rglob("*.md")))
    ext = [ROOT / "validation/benchmark/ontologies" / f for f in ("iris.owl", "wine.owl", "breast_cancer.owl", "digits.owl", "german_credit.owl", "synthetic_binary.owl", "synthetic_3class.owl")]
    ext += [ROOT / "docs/evidence/b1b4/b31/problems_frozen.json", ROOT / "docs/evidence/b1b4/b31/problems_frozen_v2.json", ROOT / "docs/evidence/b1b4/b31/freeze_B30.json"]
    h = {str(f.relative_to(ROOT)): sha(f.read_bytes().replace(b"\r\n", b"\n")) for f in files + ext if f.exists() and f.name != "hashes.json"}
    h["_environment"] = {"scikit-learn": sklearn.__version__, "numpy": np.__version__}
    (OUT / "hashes.json").write_text(json.dumps(h, indent=1))
    print(len(pk), "pacotes;", len(h) - 1, "ficheiros com hash")


if __name__ == "__main__":
    main()
