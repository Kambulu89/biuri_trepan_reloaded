#!/usr/bin/env python3
from __future__ import annotations
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def no_cart_runtime():
    offenders=[]
    for folder in ('core','gui','counterfactuals','scripts'):
        for path in (ROOT/folder).rglob('*.py'):
            try:
                tree=ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            except SyntaxError:
                offenders.append(str(path.relative_to(ROOT))+':syntax_error')
                continue
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module=='sklearn.tree':
                    if any(alias.name=='DecisionTreeClassifier' for alias in node.names):
                        offenders.append(str(path.relative_to(ROOT)))
    return sorted(set(offenders))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gui',action='store_true'); ap.add_argument('--owl',action='store_true'); ap.add_argument('--clear',action='store_true'); ap.add_argument('--json',type=Path); a=ap.parse_args()
    expected=json.loads((ROOT/'docs/CONFIRMATORY_V7_EXPECTED_SHA256.json').read_text())
    frozen={k:sha(ROOT/'results/confirmatory_v7'/k) for k in expected}
    offenders=no_cart_runtime()
    from core.training_config import PRODUCTION_TRAINING_PRESET
    from core.controlled_trepan_experiment import ControlledTrepanConfig
    efsr_cfg=ControlledTrepanConfig()
    efsr_ready=bool(
        efsr_cfg.error_focused_refinement
        and efsr_cfg.error_focus_strength > 0
        and efsr_cfg.error_focus_min_disagreement >= 0
        and efsr_cfg.error_focus_min_local_fidelity_gain >= 0
        and efsr_cfg.error_focus_min_real_fidelity_gain >= 0
        and efsr_cfg.error_focus_top_k >= 1
        and efsr_cfg.error_focus_min_regions >= 1
        and efsr_cfg.mirror_when_no_semantic_effect
        and (ROOT/'core/error_focused_semantic_refinement.py').exists()
        and (ROOT/'core/ontology_semantic_graph.py').exists()
        and (ROOT/'tests/test_error_focused_semantic_refinement_v92.py').exists()
        and (ROOT/'tests/test_semantic_real_gain_production_v92.py').exists()
    )
    semantic_real_gain_contract=bool(
        efsr_cfg.mirror_when_no_semantic_effect
        and efsr_cfg.error_focus_top_k >= efsr_cfg.error_focus_min_regions >= 1
        and efsr_cfg.error_focus_min_real_fidelity_gain >= 0
    )
    checks={'confirmatory_v7_unchanged':frozen==expected,'compileall':False,'preflight':False,'no_cart_runtime':not offenders,'scientific_mode_locked':PRODUCTION_TRAINING_PRESET=='scientific','error_focused_semantic_refinement':efsr_ready,'semantic_real_gain_contract':semantic_real_gain_contract}
    comp=subprocess.run([sys.executable,'-m','compileall','-q','core','gui','counterfactuals','tests','scripts'],cwd=ROOT)
    checks['compileall']=comp.returncode==0
    cmd=[sys.executable,'scripts/environment_preflight.py']
    if a.gui: cmd.append('--gui')
    if a.owl: cmd.append('--owl')
    if a.clear: cmd.append('--clear')
    pre=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    checks['preflight']=pre.returncode==0
    report={'checks':checks,'ready':all(checks.values()),'cart_offenders':offenders,'preflight_output':pre.stdout[-8000:]}
    rendered=json.dumps(report,indent=2,ensure_ascii=False); print(rendered)
    if a.json: a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(rendered+'\n',encoding='utf-8')
    return 0 if report['ready'] else 2
if __name__=='__main__': raise SystemExit(main())
