"""Contrato da hierarquia científica em DOIS NÍVEIS (requisito obrigatório; CLAUDE.md §2; Doc. 28). Funções puras, sem ML, sem acesso a dados.

Requisito final (inalterado): ACCURACY (rótulos reais) ``A_R > A_O > A_C`` e FIDELIDADE ao MESMO MLP ``F_R > F_O`` e ``F_R > F_C``, em conjunto, nas MESMAS instâncias.
``accuracy`` NÃO é ``precision``: a métrica preditiva principal é a accuracy; ``precision_macro`` e ``macro_f1`` são complementares (só relatório).

* NÍVEL 1 — ganho ontológico DIRETO (``evaluate_direct_gain``): Reloaded+OWL contra o Original EQUIVALENTE (mesmo amostrador, oráculo, partições, orçamento).
  Mede se a extensão ontológica acrescenta valor ao Original; ``attribute_to_ontology`` exige ainda ganho conjunto sobre os controlos (ontologia permutada, grupos
  aleatórios, engenharia convencional de features). Não substitui o Nível 2.
* NÍVEL 2 — superioridade GLOBAL (``evaluate_global``): cadeia completa U1–U5. Estado lógico por unidade + sinalizadores ORTOGONAIS (ganho direto, superioridade de
  accuracy, de fidelidade, compromissos, inviabilidade da cadeia). Unidades com ``A_O <= A_C`` são ``HIERARCHY_INFEASIBLE`` (só significa que a cadeia não pode ser
  satisfeita com os resultados fixados dos baselines nessa unidade; NÃO é impossibilidade universal e a unidade NÃO é excluída dos agregados).
* NÃO-INFERIORIDADE (``A_R >= A_O`` e ``F_R >= F_O``) é um filtro de elegibilidade de seleção em validação interna; SUPERIORIDADE ESTRITA (``>``) é o que se declara. Igualdade
  nunca é superioridade. Os baselines nunca são alterados; a seleção usa só treino/validação interna.
* EVIDÊNCIA (``classify_evidence``) é separada do estado lógico: ``OBSERVED_JOINT_GAIN`` / ``SUPPORTED_JOINT_GAIN`` / ``TRADEOFF`` / ``NOT_SATISFIED`` / ``INSUFFICIENT_EVIDENCE``.
  A unidade de inferência é o DATASET (as sementes/réplicas dentro de um dataset não substituem datasets independentes); nunca confirmatória sobre DEV já observado.
* ``check_comparability`` verifica que os braços usam as mesmas instâncias de teste e o mesmo MLP/oráculo antes de qualquer veredito."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

# --- estados lógicos por unidade (Nível 2)
JOINT_HIERARCHY_SATISFIED = "JOINT_HIERARCHY_SATISFIED"
TRADEOFF = "TRADEOFF"
NOT_SATISFIED = "NOT_SATISFIED"
HIERARCHY_INFEASIBLE = "HIERARCHY_INFEASIBLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
# --- estados do Nível 1
ONTOLOGICAL_JOINT_GAIN = "ONTOLOGICAL_JOINT_GAIN"
ACCURACY_ONLY_GAIN = "ACCURACY_ONLY_GAIN"
FIDELITY_ONLY_GAIN = "FIDELITY_ONLY_GAIN"
NO_GAIN = "NO_GAIN"
EQUIVALENT = "EQUIVALENT"
# --- níveis de evidência estatística
OBSERVED_JOINT_GAIN = "OBSERVED_JOINT_GAIN"
SUPPORTED_JOINT_GAIN = "SUPPORTED_JOINT_GAIN"
# --- atribuição à ontologia
ATTRIBUTED, NOT_ATTRIBUTED, NOT_ASSESSABLE = "ATTRIBUTED", "NOT_ATTRIBUTED", "NOT_ASSESSABLE"

CONDITIONS = ("U1_acc_R_gt_O", "U2_acc_O_gt_C", "U3_acc_R_gt_C", "U4_fid_R_gt_O", "U5_fid_R_gt_C")
REQUIRED_CONTROLS = ("permuted_ontology", "random_groups", "conventional_features")


def _bad(*vals) -> bool:
    return any(v is None or v != v for v in vals)


def evaluate_direct_gain(acc_R, acc_Oeq, fid_R, fid_Oeq, *, delta_acc: float = 0.0, delta_fid: float = 0.0, nu_acc: float = 0.0, nu_fid: float = 0.0) -> Dict[str, object]:
    """Nível 1. ``non_inferior``: dA >= -nu_acc e dF >= -nu_fid (filtro de elegibilidade; ν=0 ⇒ A_R>=A_O e F_R>=F_O). ``strict_*``: dA > delta_acc, dF > delta_fid (superioridade)."""
    if _bad(acc_R, acc_Oeq, fid_R, fid_Oeq):
        return {"level": 1, "state": INSUFFICIENT_EVIDENCE, "non_inferior": None, "strict_joint_gain": None}
    dA, dF = acc_R - acc_Oeq, fid_R - fid_Oeq
    gA, gF = dA > delta_acc, dF > delta_fid
    non_inf = (dA >= -nu_acc) and (dF >= -nu_fid)
    if gA and gF:
        state = ONTOLOGICAL_JOINT_GAIN
    elif (gA and dF < 0) or (gF and dA < 0):
        state = TRADEOFF
    elif gA:
        state = ACCURACY_ONLY_GAIN
    elif gF:
        state = FIDELITY_ONLY_GAIN
    elif dA == 0 and dF == 0:
        state = EQUIVALENT
    else:
        state = NO_GAIN
    return {"level": 1, "state": state, "non_inferior": bool(non_inf), "strict_acc_gain": bool(gA), "strict_fid_gain": bool(gF), "strict_joint_gain": bool(gA and gF),
            "deltas": {"acc": dA, "fid": dF}}


def attribute_to_ontology(direct: Mapping[str, object], controls: Mapping[str, Optional[tuple]], *, required: Sequence[str] = REQUIRED_CONTROLS,
                          delta_acc: float = 0.0, delta_fid: float = 0.0) -> Dict[str, object]:
    """O ganho só se atribui à ontologia se (i) há ganho conjunto estrito sobre o Original equivalente e (ii) há ganho conjunto estrito sobre CADA controlo exigido
    (``controls[nome] = (dAcc, dFid)`` = Reloaded+OWL − controlo). Controlo ausente ⇒ ``NOT_ASSESSABLE`` (nunca se atribui por omissão)."""
    per, missing = {}, []
    for name in required:
        c = controls.get(name)
        if c is None or _bad(*c):
            missing.append(name)
        else:
            per[name] = c[0] > delta_acc and c[1] > delta_fid
    extra = {k: (v[0] > delta_acc and v[1] > delta_fid) for k, v in controls.items() if k not in required and v is not None and not _bad(*v)}
    if direct.get("state") == INSUFFICIENT_EVIDENCE:
        label = NOT_ASSESSABLE
    elif not direct.get("strict_joint_gain"):
        label = NOT_ATTRIBUTED                       # sem ganho conjunto sobre o Original equivalente não há nada a atribuir
    elif missing:
        label = NOT_ASSESSABLE
    else:
        label = ATTRIBUTED if all(per.values()) else NOT_ATTRIBUTED
    return {"attribution": label, "beats_control": per, "extra_controls": extra, "missing_required_controls": missing}


def evaluate_global(acc_R, acc_O, acc_C, fid_R, fid_O, fid_C, *, delta_acc: float = 0.0, delta_fid: float = 0.0) -> Dict[str, object]:
    """Nível 2: cadeia completa. Devolve ``state`` (precedência: INSUFFICIENT > SATISFIED > INFEASIBLE > TRADEOFF > NOT_SATISFIED) e ``flags`` ortogonais."""
    if _bad(acc_R, acc_O, acc_C, fid_R, fid_O, fid_C):
        return {"level": 2, "state": INSUFFICIENT_EVIDENCE, "conditions": {}, "flags": {}, "mode": "MISSING_DATA"}
    c = {"U1_acc_R_gt_O": acc_R - acc_O > delta_acc, "U2_acc_O_gt_C": acc_O - acc_C > delta_acc, "U3_acc_R_gt_C": acc_R - acc_C > delta_acc,
         "U4_fid_R_gt_O": fid_R - fid_O > delta_fid, "U5_fid_R_gt_C": fid_R - fid_C > delta_fid}
    tr_o = (acc_R > acc_O and fid_R < fid_O) or (fid_R > fid_O and acc_R < acc_O)
    tr_c = (acc_R > acc_C and fid_R < fid_C) or (fid_R > fid_C and acc_R < acc_C)
    direct = evaluate_direct_gain(acc_R, acc_O, fid_R, fid_O, delta_acc=delta_acc, delta_fid=delta_fid)
    flags = {"ontological_direct_gain_state": direct["state"], "accuracy_superiority": bool(c["U1_acc_R_gt_O"] and c["U2_acc_O_gt_C"] and c["U3_acc_R_gt_C"]),
             "fidelity_superiority": bool(c["U4_fid_R_gt_O"] and c["U5_fid_R_gt_C"]), "tradeoff_vs_original": bool(tr_o), "tradeoff_vs_c45": bool(tr_c),
             "chain_infeasible_by_baselines": not c["U2_acc_O_gt_C"]}
    if all(c.values()):
        state, mode = JOINT_HIERARCHY_SATISFIED, "JOINT_GAIN"
    elif not c["U2_acc_O_gt_C"]:
        state, mode = HIERARCHY_INFEASIBLE, "ORIGINAL_NOT_ABOVE_C45_IN_ACCURACY"
    elif tr_o or tr_c:
        state, mode = TRADEOFF, "TRADEOFF_VS_ORIGINAL" if tr_o else "TRADEOFF_VS_C45"
    else:
        state, mode = NOT_SATISFIED, "NO_JOINT_GAIN"
    return {"level": 2, "state": state, "conditions": c, "flags": flags, "mode": mode, "direct": direct,
            "deltas": {"acc_R-O": acc_R - acc_O, "acc_O-C": acc_O - acc_C, "acc_R-C": acc_R - acc_C, "fid_R-O": fid_R - fid_O, "fid_R-C": fid_R - fid_C}}


evaluate_joint = evaluate_global                     # nome anterior (compatibilidade)


def is_scientific_success(verdict: Mapping[str, object]) -> bool:
    """Sucesso global numa unidade = ``JOINT_HIERARCHY_SATISFIED``. O Nível 1 sozinho nunca autoriza declarar sucesso global."""
    return verdict.get("state") == JOINT_HIERARCHY_SATISFIED


STRICT_PROPERTIES = ("instance_identity_and_order", "true_labels", "class_mapping_and_order", "partitioning", "dataset_identity", "base_representation_and_preprocessing",
                     "oracle_identity", "oracle_predictions_on_test", "model_version", "experimental_configuration", "query_budget")


def _eq(vals):
    vals = list(vals)
    return all(v == vals[0] for v in vals[1:])


def check_comparability(records: Mapping[str, Mapping[str, object]], *, level: str = "strict", relaxed: Iterable[str] = (), oracle_key: str = "oracle_id",
                        c45_oracle_key: str = "fidelity_oracle_id") -> Dict[str, object]:
    """Comparabilidade ESTRITA entre braços. ``records``: nome do braço -> registo bruto completo (``raw_store.write_unit``: ``row``, ``predictions``, ``labels``, ...).
    Propriedades verificadas diretamente nos dados ou por identificadores criptográficos cujo contrato as garante:
      instance_identity_and_order   ``predictions.test_row_index`` idênticos (mesma sequência, mesma ordem);
      true_labels                    ``predictions.y_true`` idênticos;
      class_mapping_and_order        ``labels`` e ``minority_label`` idênticos;
      partitioning                   ``split_id``, ``split_hash`` (hash dos índices de treino e teste), ``n_train``, ``n_test``;
      dataset_identity               ``dataset_content_hash`` e ``dataset_hash``;
      base_representation_and_preprocessing  ``base_representation_hash`` (matriz de treino) e ``preprocessing_id``; as representações ENRIQUECIDAS (``feature_space``, ``ontology_hash``)
                                     são diferenças DOCUMENTADAS (devolvidas em ``documented_differences``), nunca ocultas;
      oracle_identity                ``oracle_id`` igual em todas as árvores TREPAN e igual ao ``fidelity_oracle_id`` do C4.5;
      oracle_predictions_on_test     previsões do oráculo sobre as instâncias de teste idênticas (o mesmo MLP produz as mesmas saídas);
      model_version                  ``oracle_version``/``oracle_name`` idênticos;
      experimental_configuration     ``config_hash`` e ``manifest_sha256`` idênticos;
      query_budget                   ``tree_max_nodes``, ``query_budget``, ``tree_max_n`` iguais nas árvores TREPAN (orçamento estrutural controlado).
    Propriedade não verificável (campo ausente) ⇒ ``ok=None`` ⇒ NÃO comparável (``unverifiable``): igualdade do nº de amostras nunca basta. ``relaxed`` isenta propriedades
    nomeadas (p.ex. ``experimental_configuration`` ao juntar réplicas de execuções diferentes) e regista a isenção. ``level="basic"`` (legado) verifica apenas
    ``split_hash``/``n_test``/oráculo sobre registos mínimos."""
    if level == "basic":
        return _check_basic(records, oracle_key, c45_oracle_key)
    relaxed = set(relaxed)
    rows = {k: (v.get("row") or v) for k, v in records.items()}
    top = {k: v for k, v in records.items()}
    pred = {k: (v.get("predictions") or {}) for k, v in records.items()}
    trepan = [k for k, r in rows.items() if r.get(oracle_key)]
    checks: Dict[str, Dict[str, object]] = {}

    def put(name, ok, detail=""):
        checks[name] = {"ok": ok, "detail": detail, "relaxed": name in relaxed}

    def same(getter, name, *, subset=None):
        keys = subset if subset is not None else list(records)
        vals = [getter(k) for k in keys]
        if any(v is None for v in vals) or not keys:
            put(name, None, "campo ausente em " + ",".join(k for k, v in zip(keys, vals) if v is None))
        else:
            put(name, _eq(vals), "" if _eq(vals) else "valores diferentes")

    same(lambda k: tuple(pred[k].get("test_row_index") or ()) or None, "instance_identity_and_order")
    same(lambda k: tuple(pred[k].get("y_true") or ()) or None, "true_labels")
    same(lambda k: (tuple(top[k].get("labels") or ()) or None, str(rows[k].get("minority_label"))) if top[k].get("labels") else None, "class_mapping_and_order")
    same(lambda k: (rows[k].get("split_id"), rows[k].get("split_hash"), rows[k].get("n_train"), rows[k].get("n_test")) if rows[k].get("split_hash") else None, "partitioning")
    same(lambda k: (rows[k].get("dataset_content_hash"), rows[k].get("dataset_hash")) if rows[k].get("dataset_hash") else None, "dataset_identity")
    same(lambda k: (rows[k].get("base_representation_hash"), rows[k].get("preprocessing_id")) if rows[k].get("base_representation_hash") else None, "base_representation_and_preprocessing")
    oracles = {rows[k].get(oracle_key) for k in trepan} | {rows[k].get(c45_oracle_key) for k in rows if rows[k].get(c45_oracle_key)}
    put("oracle_identity", (len(oracles) == 1) if oracles else None, "" if len(oracles) == 1 else f"oráculos: {sorted(map(str, oracles))}")
    op = {}
    for k in records:
        d = pred[k].get("oracle_predictions")
        op[k] = None if not d else tuple(sorted((str(n), tuple(v)) for n, v in d.items()))
    same(lambda k: op[k], "oracle_predictions_on_test")
    same(lambda k: (rows[k].get("oracle_version"), rows[k].get("oracle_name")) if rows[k].get("oracle_version") else None, "model_version", subset=trepan)
    same(lambda k: (rows[k].get("config_hash"), top[k].get("manifest_sha256")) if rows[k].get("config_hash") else None, "experimental_configuration")
    same(lambda k: tuple(((top[k].get("config") or {}).get("structural") or {}).get(f) for f in ("tree_max_nodes", "query_budget", "tree_max_n")), "query_budget", subset=trepan)
    unver = [n for n, c in checks.items() if c["ok"] is None and n not in relaxed]
    fail = [n for n, c in checks.items() if c["ok"] is False and n not in relaxed]
    doc_diff = {"feature_space": sorted({str(rows[k].get("feature_space")) for k in rows}), "ontology_hash": sorted({str(rows[k].get("ontology_hash")) for k in rows})}
    return {"comparable": not unver and not fail, "level": "strict", "checks": checks, "failed": fail, "unverifiable": unver, "relaxed": sorted(relaxed),
            "documented_differences": doc_diff, "problems": [f"{n}: {checks[n]['detail']}" for n in fail + unver]}


def _check_basic(rows, oracle_key, c45_oracle_key):
    problems: List[str] = []
    splits = {r.get("split_hash") for r in rows.values()}
    ntest = {r.get("n_test") for r in rows.values()}
    if len(splits) != 1 or None in splits:
        problems.append(f"instâncias de teste diferentes ou desconhecidas (split_hash={sorted(map(str, splits))})")
    if len(ntest) != 1 or None in ntest:
        problems.append(f"n_test diferente ou desconhecido ({sorted(map(str, ntest))})")
    oracles = {r.get(oracle_key) for r in rows.values() if r.get(oracle_key)} | {r.get(c45_oracle_key) for r in rows.values() if r.get(c45_oracle_key)}
    if len(oracles) != 1:
        problems.append(f"oráculo/MLP de referência diferente ou ausente ({sorted(map(str, oracles))})")
    return {"comparable": not problems, "level": "basic", "problems": problems}


def _t_ci(vals: Sequence[float], alpha: float):
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    from scipy.stats import t as T
    h = float(T.ppf(1 - alpha / 2, n - 1)) * se
    return {"mean": mean, "se": se, "lo": mean - h, "hi": mean + h, "between_dataset_sd": sd}


def _boot_ci(vals: Sequence[float], alpha: float, n_boot: int, seed: int):
    import numpy as np
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    means = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def classify_evidence(diffs_by_dataset: Mapping[str, Mapping[str, Sequence[float]]], *, confirmatory: bool = False, min_datasets: int = 12, alpha: float = 0.05,
                      n_boot: int = 4000, seed: int = 0) -> Dict[str, object]:
    """Nível de evidência SEPARADO do estado lógico. ``diffs_by_dataset[dataset] = {"acc": [diferenças pareadas por unidade], "fid": [...]}`` (Reloaded − referência).
    A unidade de inferência é o DATASET (média das sementes/réplicas dentro do dataset); a variabilidade dentro do dataset reporta-se à parte. ``SUPPORTED`` exige:
    avaliação confirmatória, ≥ ``min_datasets`` datasets independentes (Doc. 2: N=12) e limite inferior dos ICs (t e bootstrap entre datasets) > 0 em accuracy E fidelidade.
    Seleção retroativa de configurações sobre DEV já observado nunca é confirmatória (``confirmatory=False`` ⇒ no máximo ``OBSERVED_JOINT_GAIN``)."""
    import numpy as np
    per = {}
    for ds, d in diffs_by_dataset.items():
        a, f = list(d.get("acc", [])), list(d.get("fid", []))
        if a and f:
            per[ds] = {"acc": float(np.mean(a)), "fid": float(np.mean(f)), "n_units": len(a), "within_dataset_sd_acc": float(np.std(a, ddof=1)) if len(a) > 1 else None,
                       "within_dataset_sd_fid": float(np.std(f, ddof=1)) if len(f) > 1 else None}
    n = len(per)
    out: Dict[str, object] = {"n_datasets": n, "per_dataset": per, "confirmatory": bool(confirmatory), "unit_of_inference": "dataset", "reasons": []}
    if n < 2:
        out.update(evidence=INSUFFICIENT_EVIDENCE)
        out["reasons"].append("menos de 2 datasets")
        return out
    stats = {}
    for m in ("acc", "fid"):
        v = [per[d][m] for d in per]
        stats[m] = {"t": _t_ci(v, alpha), "bootstrap_ci": _boot_ci(v, alpha, n_boot, seed), "mean": float(np.mean(v))}
    out["across_datasets"] = stats
    mA, mF = stats["acc"]["mean"], stats["fid"]["mean"]
    if mA > 0 and mF > 0:
        ci_pos = all(stats[m]["t"]["lo"] > 0 and stats[m]["bootstrap_ci"][0] > 0 for m in ("acc", "fid"))
        if confirmatory and n >= min_datasets and ci_pos:
            out["evidence"] = SUPPORTED_JOINT_GAIN
        else:
            out["evidence"] = OBSERVED_JOINT_GAIN
            if not confirmatory:
                out["reasons"].append("avaliação não confirmatória (DEV/exploratória)")
            if n < min_datasets:
                out["reasons"].append(f"{n} < {min_datasets} datasets independentes")
            if not ci_pos:
                out["reasons"].append("IC95 entre datasets inclui 0 em accuracy e/ou fidelidade")
    elif (mA > 0 and mF < 0) or (mF > 0 and mA < 0):
        out["evidence"] = TRADEOFF
    else:
        out["evidence"] = NOT_SATISFIED
    return out
