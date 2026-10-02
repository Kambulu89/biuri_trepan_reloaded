# Quickstart — V9.2 Produção sem CART

## 1. Ambiente recomendado

- Python 3.11 ou 3.12 x64
- Java 17 para OWL/HermiT
- Graphviz

Crie e active um ambiente virtual, depois instale:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-test.txt
```

## 2. Verificação obrigatória

```powershell
python scripts/production_readiness_v92.py --gui --owl --json production_readiness.json
```

Para promover a release, os checks relevantes devem ficar `true`, incluindo:

```text
confirmatory_v7_unchanged
compileall
no_cart_runtime
scientific_mode_locked
preflight
```

## 3. Guard de árvore

```powershell
python -m pytest -q tests/test_production_no_cart_v92.py
```

A build falha se `DecisionTreeClassifier` for reintroduzido no runtime.

## 4. GUI

```powershell
python run_biuri.py
```

O treino normal fica bloqueado no modo **Científico**.

## 5. Política dos modelos

```text
MLP Original        → oráculo
C4.5-Nativo         → baseline supervisionado com y real
TREPAN Original     → TrepanOriginalClassifier histórico
TREPAN Reloaded     → TrepanReloadedClassifier histórico + OWL opcional
```

Nenhum fallback para outra família de árvore é permitido.

## 6. Bundle de produção

Bundles novos usam:

```text
biuri-v9.2-production-2-no-cart
historical_trepan_only_no_cart
```

Artefactos antigos com uma família de árvore incompatível são recusados pelo loader e devem ser regenerados.
