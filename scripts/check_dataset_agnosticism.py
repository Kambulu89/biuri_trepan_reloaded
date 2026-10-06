#!/usr/bin/env python3
"""Guardrail arquitetural: o núcleo é AGNÓSTICO a datasets. Só biblioteca padrão (corre na CI antes de instalar dependências).

Verifica, em ``core/``, ``gui/`` e ``counterfactuals/`` (o sistema genérico de produção), por análise estática (AST + texto):

  R1  nomes de datasets conhecidos (lista fixa + todos os ``dataset_id`` registados em ``validation/benchmark/datasets.py``);
  R2  ramificação pela identidade do dataset (``dataset_name == "..."``, ``dataset_id in (...)``, ``file_name.startswith(...)`` ...);
  R3  tabelas/políticas indexadas pela identidade (``CONFIG[dataset_name]``, ``resource_policy[dataset_id]``, ``.get(dataset_name)``);
  R4  formas conhecidas (``n_features == 30``, ``n_classes == 10`` ...);
  R6  política de recursos/limites/hiperparâmetros específica por dataset (``resource_policy[dataset_name]``, ``LIMITS.get(dataset_id)``),
      verificada também em ``validation/benchmark`` (os datasets de validação podem existir, as políticas têm de ser globais);
  R5  dependência proibida ``core -> validation`` (e ``scripts``/``tests``): só é permitido ``validation -> core``.

``IRI``/``IRIs``/``IRIS`` (API do OWL) NÃO são o dataset Iris: a regra do nome exige a palavra isolada «iris» (minúscula ou Iris).
Uso:  python scripts/check_dataset_agnosticism.py [--root RAIZ]      (código de saída 1 se houver violações)
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

PRODUCTION_DIRS = ("core", "gui", "counterfactuals")
FORBIDDEN_IMPORT_ROOTS = ("validation", "scripts", "tests")

# Nomes de datasets de validação/conhecidos (palavras isoladas). «iris» tratado à parte (ver IRIS_DATASET).
KNOWN_DATASET_WORDS = (
    "wine", "wdbc", "sonar", "german", "hepatitis", "adult", "census", "income", "breast[_ ]?cancer", "digits", "diabetes", "titanic",
    "mnist", "heart[_ ]disease", "mushroom", "credit", "setosa", "versicolor", "virginica", "malignant", "benign", "wisconsin", "pima",
    "ionosphere", "vehicle", "glass", "synthetic_binary", "synthetic_3class",
)
IRIS_DATASET = re.compile(r"(?<![A-Za-z_])(iris|Iris)(?![A-Za-z_])")          # exclui IRI/IRIs/IRIS da API do OWL
IDENTITY_TOKENS = ("dataset_name", "dataset_id", "data_set_name", "dataset_key", "dataset_label", "dataset_slug", "relation_name",
                   "file_name", "filename", "dataset_path", "data_path", "arff_name", "csv_name", "dataset_title", "ds_name", "ds_id", "data_name")
SHAPE_RULE = re.compile(r"(n_features|n_classes|shape\[1\]|len\((feature_names|class_names|classes)\))\s*(==|!=)\s*\d{2,}")


def known_dataset_regex(root: Path) -> re.Pattern:
    """Lista fixa + ``dataset_id`` registados em ``validation/benchmark/datasets.py`` (extensível sem tocar neste ficheiro)."""
    words = list(KNOWN_DATASET_WORDS)
    registry = root / "validation" / "benchmark" / "datasets.py"
    if registry.exists():
        try:
            tree = ast.parse(registry.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "DatasetSpec" and node.args \
                        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    if node.args[0].value.lower() != "iris":          # «iris» tem regra própria (distingue a API IRI/IRIS do OWL)
                        words.append(re.escape(node.args[0].value))
        except SyntaxError:
            pass
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(dict.fromkeys(words)) + r")(?![A-Za-z0-9])", re.I)


def _identity_in(expr: ast.AST, constant_keys: bool = True) -> bool:
    """A expressão envolve a identidade nominal do dataset (variável, atributo, chave de dict ou chamada sobre eles).

    ``constant_keys=False`` ignora chaves literais (``result["dataset_name"] = x`` só GUARDA metadados; não é uma tabela indexada por identidade)."""
    for node in ast.walk(expr):
        ident = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if ident and any(tok in ident.lower() for tok in IDENTITY_TOKENS):
            return True
        if constant_keys and isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.lower() in IDENTITY_TOKENS:
            return True               # ex.: meta["file_name"] == "x"
    return False


def _is_literal_collection(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, str)
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        return bool(expr.elts) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in expr.elts)
    return False


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body[0].value.value = ""


Violation = Tuple[str, int, str, str]       # (ficheiro, linha, regra, descrição)


def check_source(rel: str, text: str, names: re.Pattern) -> List[Violation]:
    out: List[Violation] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        m = names.search(line) or IRIS_DATASET.search(line)
        if m:
            out.append((rel, lineno, "R1", f"nome de dataset conhecido «{m.group(0)}»"))
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return out + [(rel, exc.lineno or 0, "R0", f"sintaxe inválida: {exc.msg}")]
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):                                                   # R2: if dataset_name == "x" / in (...)
            parts = [node.left, *node.comparators]
            if any(_identity_in(p) for p in parts) and any(_is_literal_collection(p) for p in parts):
                out.append((rel, node.lineno, "R2", "ramificação pela identidade do dataset"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("startswith", "endswith") \
                and _identity_in(node.func.value) and node.args and _is_literal_collection(node.args[0]):
            out.append((rel, node.lineno, "R2", "ramificação pela identidade do dataset (startswith/endswith)"))
        elif isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant) and _identity_in(node.slice, constant_keys=False):                  # R3: POLICY[dataset_name]
            out.append((rel, node.lineno, "R3", "tabela/política indexada pela identidade do dataset"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "pop", "setdefault") \
                and node.args and not isinstance(node.args[0], ast.Constant) and _identity_in(node.args[0], constant_keys=False):
            out.append((rel, node.lineno, "R3", "consulta de tabela pela identidade do dataset (.get)"))
    stripped = ast.parse(text)
    _strip_docstrings(stripped)
    code = ast.unparse(stripped)
    for m in SHAPE_RULE.finditer(code):
        out.append((rel, 0, "R4", f"forma conhecida codificada «{m.group(0)}»"))
    return out


POLICY_TOKENS = ("resource", "policy", "limit", "timeout", "wall", "budget", "hyper", "param", "setting", "config", "cfg", "capacity",
                 "grid", "threshold", "max_", "min_")
POLICY_DIRS = PRODUCTION_DIRS + ("validation/benchmark",)


def _container_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _container_name(node.func)
    if isinstance(node, ast.Subscript):
        return _container_name(node.value)
    return ""


def check_policies(rel: str, text: str) -> List[Violation]:
    """R6: recursos/limites/hiperparâmetros indexados pela identidade do dataset (``policy[dataset_name]``, ``limits.get(dataset_id)``)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: List[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant) and _identity_in(node.slice, constant_keys=False) \
                and any(t in _container_name(node.value).lower() for t in POLICY_TOKENS):
            out.append((rel, node.lineno, "R6", f"política/limite/hiperparâmetro indexado pelo dataset: {_container_name(node.value)}[<dataset>]"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "pop", "setdefault") and node.args \
                and not isinstance(node.args[0], ast.Constant) and _identity_in(node.args[0], constant_keys=False) \
                and any(t in _container_name(node.func.value).lower() for t in POLICY_TOKENS):
            out.append((rel, node.lineno, "R6", f"política/limite/hiperparâmetro consultado pelo dataset: {_container_name(node.func.value)}.get(<dataset>)"))
    return out


def scan_policies(root: Path, dirs: Iterable[str] = POLICY_DIRS) -> List[Violation]:
    out: List[Violation] = []
    for d in dirs:
        for path in sorted((root / d).rglob("*.py")):
            out += check_policies(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="ignore"))
    return out


def imports_of(tree: ast.AST) -> Iterable[Tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module


def check_imports(rel: str, text: str) -> List[Violation]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = [(rel, lineno, "R5", f"o sistema genérico importa «{name}» (só validation → core é permitido)")
           for lineno, name in imports_of(tree) if name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS]
    for node in ast.walk(tree):                                                              # caminhos/strings para validation/
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"(^|[\\/])validation[\\/]", node.value):
            out.append((rel, node.lineno, "R5", f"o sistema genérico referencia o caminho «{node.value[:40]}»"))
    return out


def scan(root: Path, dirs: Iterable[str] = PRODUCTION_DIRS) -> List[Violation]:
    names = known_dataset_regex(root)
    violations: List[Violation] = []
    for d in dirs:
        for path in sorted((root / d).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            violations += check_source(rel, text, names)
            violations += check_imports(rel, text)
    return violations


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    violations = scan(Path(args.root)) + scan_policies(Path(args.root))
    for rel, line, rule, msg in violations:
        print(f"{rel}:{line}: [{rule}] {msg}")
    if violations:
        print(f"\nFALHOU: {len(violations)} violação(ões) do contrato de agnosticismo ao dataset (ver CLAUDE.md, secção 1).")
        return 1
    print("OK: núcleo agnóstico (sem nomes de datasets, ramificações/tabelas pela identidade, formas fixas nem dependência de validation/).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
