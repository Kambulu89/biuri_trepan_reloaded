"""Acesso ao protocolo confirmatório V7 congelado.

A execução confirmatória única já foi consumida. Em produção este script serve
apenas para validar o ambiente e recordar onde estão os resultados congelados;
``--execute`` é deliberadamente recusado.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from environment_preflight import build_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/confirmatory_v7"),
        help="Directório dos artefactos confirmatórios congelados.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Mantido apenas por compatibilidade; a execução é bloqueada em produção.",
    )
    args = parser.parse_args()

    preflight = build_preflight(require_gui=False)
    print(json.dumps(preflight, ensure_ascii=False, indent=2, default=str))
    if args.execute:
        print(
            "BLOQUEADO: o benchmark confirmatório V7 é imutável e a execução única já foi consumida. "
            "Consulte results/confirmatory_v7/."
        )
        return 3
    print(f"Confirmatório congelado disponível em: {args.output}")
    return 0 if preflight.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
