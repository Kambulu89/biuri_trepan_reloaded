import ast
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def test_no_bare_except_in_product_code():
    bad=[]
    for base in ('core','gui','counterfactuals'):
        for p in (ROOT/base).rglob('*.py'):
            try: tree=ast.parse(p.read_text(encoding='utf-8',errors='ignore'))
            except SyntaxError: continue
            for n in ast.walk(tree):
                if isinstance(n,ast.ExceptHandler) and n.type is None: bad.append(f'{p.relative_to(ROOT)}:{n.lineno}')
    assert not bad,bad

def test_no_dataset_names_in_core_gui_outside_benchmark_exceptions():
    pattern=re.compile(r'\b(wine|wdbc|sonar|german|hepatitis|adult|breast_cancer|digits|diabetes)\b',re.I)
    allowed=set()   # sem exceções: catálogos de datasets vivem em validation/, fora do núcleo científico
    bad=[]
    for base in ('core','gui','counterfactuals'):
        for p in (ROOT/base).rglob('*.py'):
            rel=p.relative_to(ROOT).as_posix()
            if rel in allowed: continue
            if pattern.search(p.read_text(encoding='utf-8',errors='ignore')): bad.append(rel)
    assert not bad,bad
