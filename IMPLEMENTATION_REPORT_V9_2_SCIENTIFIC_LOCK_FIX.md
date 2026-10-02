# BIURI / TREPAN Reloaded V9.2 — Scientific Mode Lock Fix

## Problema reproduzido

No modo **Científico**, `TrainingPreset.trepan_max_depth` é `None`, que significa não impor um limite de profundidade mais apertado que o limite estrutural da árvore. O caminho histórico do TREPAN Reloaded fazia `int(None)` ao construir `TrepanReloadedClassifier`, causando:

`int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`

## Correcções

- `resolve_trepan_structure_limits()` converte limites opcionais para um contrato seguro e igual para Original/Reloaded.
- `trepan_max_depth=None` resolve para `max_nodes` no modo científico.
- `TrepanOriginalClassifier` também aceita `max_depth=None` sem erro em chamadas directas.
- O TREPAN Reloaded já não faz `int(None)`.
- Corrigida recursão acidental em `_legacy_cart_baseline()`.
- A GUI apresenta apenas **Científico**, bloqueado para edição.
- `_get_selected_training_preset`, `_execute_training_pipeline` e `TrainingWorker` impõem o preset científico.
- `TrepanReloaded.train_mlp`, `train_mlp_onto` e `train_mlp_residual_onto` impõem o preset científico.
- Defaults internos de `MLPTrainer`, `mlp_factory` e `mlp_optimizer` mudaram de `balanced` para `scientific`.
- CLI pública restringe `--preset` a `scientific`.
- Falhas do worker são registadas com `logger.exception` antes de serem apresentadas ao utilizador.

Os presets `fast` e `balanced` permanecem no módulo de configuração apenas para testes de regressão e desenvolvimento interno; não são seleccionáveis pelo fluxo normal de produção.

## Validação executada neste ambiente

Ambiente disponível: Python 3.13.5. PyQt6, Owlready2/HermiT, TensorFlow e dtreeviz não estão instalados neste runtime, por isso a GUI Windows e reasoner OWL não foram executados aqui.

Comandos concluídos:

- `pytest -q tests/test_scientific_mode_lock_v92.py` → **6 passed**
- bateria científica/ontologia/TREPAN → **49 passed**
- contrato/preprocessamento/dataset-agnostic/artifacts → **22 passed**
- regressões GUI/fontes → **16 passed, 1 skipped**
- mirror/context/oracle TREPAN → **8 passed**
- integração dos três caminhos MLP do orquestrador → **23 passed**
- `python -m compileall -q core gui counterfactuals tests scripts` → **OK**

O erro `int(None)` ficou coberto por teste de regressão.

## Limitação honesta

A execução visual exacta da captura deve ser repetida no Windows do utilizador com Python 3.11/3.12 + PyQt6 + Owlready2/Java/HermiT para validar todo o fluxo GUI/OWL no ambiente alvo. O código não é marcado como validado nesses componentes neste runtime.
