$ErrorActionPreference = "Stop"
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONUTF8 = "1"

python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
python -m pip install -e . --no-deps
python scripts/environment_preflight.py --gui --json results/environment_preflight_windows.json
python -m compileall -q core gui counterfactuals tests
python -m pytest -q
