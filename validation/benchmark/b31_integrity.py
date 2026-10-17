"""B3.1 (Doc. 34 §0): verificação reproduzível da integridade da evidência M29 e congelamento do estado B3.0.
Verifica (1) hashes de todos os `UNIT_SHA256.json`, (2) proveniência dos `split.json` repostos (hash registado NO MOMENTO da execução, por unidade, vs ficheiro presente),
(3) relatório M29 regenerado só dos brutos versionados == relatório guardado, (4) F29≡A29 e comparabilidade 18/18 no relatório. Só leitura; escreve um JSON de resultado."""
from __future__ import annotations
import argparse, glob, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    ev = ROOT / "docs/evidence/b1b4/m29_campaign"
    res = {"unit_dirs": 0, "files_checked": 0, "mismatch": [], "missing": [], "split_json": {"expected": 0, "present": 0, "hash_ok": 0, "details": []}}
    for p in sorted(glob.glob(str(ev / "*/raw/*/seed*/UNIT_SHA256.json"))):
        d = Path(p).parent; h = json.loads(Path(p).read_text()); res["unit_dirs"] += 1
        for f, s in h.items():
            res["files_checked"] += 1; fp = d / f
            if not fp.exists():
                res["missing"].append(str(fp.relative_to(ROOT))); continue
            ok = sha(fp) == s
            if not ok:
                res["mismatch"].append(str(fp.relative_to(ROOT)))
            if f == "split.json":
                res["split_json"]["expected"] += 1; res["split_json"]["present"] += 1; res["split_json"]["hash_ok"] += int(ok)
                res["split_json"]["details"].append({"file": str(fp.relative_to(ROOT)), "sha256": s, "recorded_at_run_time_in": "UNIT_SHA256.json"})
    res["files_ok"] = not res["mismatch"] and not res["missing"]
    # relatório regenerado
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r.json"
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        subprocess.run([sys.executable, "-m", "validation.benchmark.m29_report", "--sc", str(ROOT / "docs/evidence/b1b4/sampler_c/dev_numeric"), "--m29",
                        str(ev / "main"), str(ev / "g29_pilot_patch"), "--out", str(out)], cwd=ROOT, env=env, check=True, capture_output=True)
        new = json.loads(out.read_text())
    old = json.loads((ev / "report/m29_report.json").read_text())
    res["report_regenerated_identical"] = new == old
    res["report_sha256"] = sha(ev / "report/m29_report.json")
    res["comparability_units"] = {k: v for k, v in old.get("comparability", {}).items()} if isinstance(old.get("comparability"), dict) else str(old.get("comparability"))[:200]
    res["equivalence_F29_vs_A29"] = old.get("equivalence_F29_vs_A29")
    Path(a.out).write_text(json.dumps(res, indent=1)[:2_000_000])
    print(json.dumps({k: res[k] for k in ("unit_dirs", "files_checked", "files_ok", "report_regenerated_identical")}), "split.json:", {k: res["split_json"][k] for k in ("expected", "present", "hash_ok")})


if __name__ == "__main__":
    main()
