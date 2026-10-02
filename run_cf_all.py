#!/usr/bin/env python3
"""Executa pipeline completo (train + CF + improve) para todos os datasets e consolida."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    datasets = sys.argv[1:] or None
    cmds = [
        [sys.executable, str(ROOT / 'run_cf_train.py')] + (datasets or []),
        [sys.executable, str(ROOT / 'run_cf_counterfactuals.py')] + (datasets or []),
        [sys.executable, str(ROOT / 'run_cf_improve.py')] + (datasets or []),
        [sys.executable, str(ROOT / 'counterfactuals' / 'analisis' / 'consolidar_resultados.py')],
    ]
    env = {**dict(__import__('os').environ), 'PYTHONPATH': str(ROOT)}
    for cmd in cmds:
        print('\n>>>', ' '.join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    print('\nPipeline completo concluído.')


if __name__ == '__main__':
    main()
