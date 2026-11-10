"""Extrai de cada log do BIURI: estado estrutural, gate de utilidade, razão do professor, pesos híbridos,
métricas de teste (MLP, C4.5, Trepan Original, Trepan Reloaded) e complexidade. Só lê logs (resultados brutos imutáveis).
Uso: python parse_runs.py <dir_com_logs> > resumo.csv"""
import re, sys, glob, os, json, csv

def grab(t, pat, cast=float, default=None):
    m = re.search(pat, t, re.S)
    return cast(m.group(1)) if m else default

def block(t, title):
    m = re.search(re.escape(title) + r":\n(.*?)(?:\n\n|\Z)", t, re.S)
    return m.group(1) if m else ""

def parse(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    fa = t[t.find("[FINAL MODEL AUDIT]"):] if "[FINAL MODEL AUDIT]" in t else ""
    row = {"run": os.path.basename(path)[4:-4]}
    row["struct"] = grab(t, r"'status': '(VALID_[A-Z_]+|REJECT[A-Z_]*|GENERIC[A-Z_]*|INSUFFICIENT[A-Z_]*|[A-Z_]*ONTOLOGY[A-Z_]*)'", str)
    row["utility"] = grab(t, r"'status': '((?:ACCEPT|REJECT)_[A-Z_]+)'", str)
    row["utility_reason"] = grab(t, r"'reason': '(validated_predictive_gain|positive_predictive_gain_below_complexity_cost|[a-z_]+)', ?'?[a-z_]*'?", str)
    row["teacher_reason"] = grab(t, r"(ZERO_ONTOLOGY_MASS_IN_SELECTED_HYBRID|STRICT_NONINFERIORITY_[A-Z]+|ONTOLOGY_FEATURE_GATE_REJECTED)", str)
    row["w_orig"] = grab(t, r"'original': ([0-9.]+)"); row["w_onto"] = grab(t, r"'ontological': ([0-9.]+)"); row["w_res"] = grab(t, r"'residual': ([0-9.]+)")
    row["n_candidates"] = grab(t, r"candidate_semantic_indices': \[([0-9, ]+)\]", lambda s: len(s.split(",")))
    row["selected"] = grab(t, r"'selected_candidate': \{.*?'semantic_names': \[([^\]]*)\]", str)
    for name, key in (("MLP Original", "mlp"), ("C4.5-Nativo", "c45"), ("Trepan Original", "orig"), ("Trepan Reloaded", "rel")):
        b = block(fa, name)
        row[key + "_acc"] = grab(b, r"accuracy: ([0-9.]+)")
        row[key + "_f1"] = grab(b, r"f1: ([0-9.]+)")
        row[key + "_ba"] = grab(b, r"balanced_accuracy: ([0-9.]+)")
        row[key + "_fid"] = grab(b, r"fidelity[^:]*: ([0-9.]+)")
        row[key + "_nodes"] = grab(b, r"number of nodes: ([0-9]+)", int)
    m = re.search(r"OOF MLP base — Acc: ([0-9,]+)%.*?OOF MLP \+ OWL — Acc: ([0-9,]+)%", json.dumps(t) and t, re.S)
    row["oof_base"], row["oof_owl"] = (m.group(1), m.group(2)) if m else (None, None)
    return row

if __name__ == "__main__":
    rows = [parse(p) for p in sorted(glob.glob(os.path.join(sys.argv[1], "log_*.txt")))]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k))
    w = csv.DictWriter(sys.stdout, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
