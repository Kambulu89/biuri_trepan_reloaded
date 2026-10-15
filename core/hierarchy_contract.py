"""Contrato CONJUNTO da hierarquia científica (requisito obrigatório; CLAUDE.md §2; Doc. 5 revisão 2). Funções puras, sem ML, sem acesso a dados.

Condições (todas medidas nas MESMAS instâncias de teste e, para a fidelidade, contra o MESMO MLP/oráculo de referência):
  accuracy (rótulos reais):  U1  A_R > A_O      U2  A_O > A_C      U3  A_R > A_C
  fidelidade ao MLP:         U4  F_R > F_O      U5  F_R > F_C
``accuracy`` NÃO é ``precision``: a métrica preditiva principal é a accuracy; ``precision_macro`` e ``macro_f1`` são complementares (só relatório).
Uma intervenção só é cientificamente bem-sucedida quando U1–U5 se verificam em conjunto: melhorar a accuracy à custa da fidelidade (ou o inverso) é um
COMPROMISSO (``TRADEOFF_*``), nunca um sucesso. Os baselines nunca são alterados: se U2 falha, o estado é ``INFEASIBLE`` (a cadeia é inalcançável com estes
baselines) e isso é reportado, não corrigido. Este módulo não é usado para selecionar nada com o teste: a seleção usa só treino/validação interna."""
from __future__ import annotations

from typing import Dict, Optional

SATISFIED = "HIERARCHY_SATISFIED"
NOT_SATISFIED = "HIERARCHY_NOT_SATISFIED"
INFEASIBLE = "HIERARCHY_INFEASIBLE"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
CONDITIONS = ("U1_acc_R_gt_O", "U2_acc_O_gt_C", "U3_acc_R_gt_C", "U4_fid_R_gt_O", "U5_fid_R_gt_C")


def evaluate_joint(acc_R: Optional[float], acc_O: Optional[float], acc_C: Optional[float], fid_R: Optional[float], fid_O: Optional[float], fid_C: Optional[float],
                   *, delta_acc: float = 0.0, delta_fid: float = 0.0) -> Dict[str, object]:
    vals = (acc_R, acc_O, acc_C, fid_R, fid_O, fid_C)
    if any(v is None or v != v for v in vals):
        return {"status": INSUFFICIENT, "conditions": {}, "mode": "MISSING_DATA"}
    c = {"U1_acc_R_gt_O": acc_R - acc_O > delta_acc, "U2_acc_O_gt_C": acc_O - acc_C > delta_acc, "U3_acc_R_gt_C": acc_R - acc_C > delta_acc,
         "U4_fid_R_gt_O": fid_R - fid_O > delta_fid, "U5_fid_R_gt_C": fid_R - fid_C > delta_fid}
    acc_gain, fid_gain = c["U1_acc_R_gt_O"], c["U4_fid_R_gt_O"]
    if all(c.values()):
        status, mode = SATISFIED, "JOINT_GAIN"
    elif not c["U2_acc_O_gt_C"]:
        status, mode = INFEASIBLE, "ORIGINAL_NOT_ABOVE_C45_IN_ACCURACY"          # propriedade dos baselines congelados; não se corrige
    else:
        status = NOT_SATISFIED
        if acc_R > acc_O and fid_R < fid_O:
            mode = "TRADEOFF_ACCURACY_UP_FIDELITY_DOWN"
        elif fid_R > fid_O and acc_R < acc_O:
            mode = "TRADEOFF_FIDELITY_UP_ACCURACY_DOWN"
        elif acc_gain and not c["U4_fid_R_gt_O"]:
            mode = "ACCURACY_ONLY_GAIN"
        elif fid_gain and not acc_gain:
            mode = "FIDELITY_ONLY_GAIN"
        elif not acc_gain and not fid_gain:
            mode = "NO_GAIN_OVER_ORIGINAL"
        else:
            mode = "GAIN_OVER_ORIGINAL_BUT_NOT_OVER_C45"
    return {"status": status, "conditions": c, "mode": mode,
            "deltas": {"acc_R-O": acc_R - acc_O, "acc_O-C": acc_O - acc_C, "acc_R-C": acc_R - acc_C, "fid_R-O": fid_R - fid_O, "fid_R-C": fid_R - fid_C}}


def is_scientific_success(verdict: Dict[str, object]) -> bool:
    """Sucesso científico = hierarquia conjunta satisfeita. Sozinho, nenhum ganho unilateral (nem compromisso) conta."""
    return verdict.get("status") == SATISFIED
