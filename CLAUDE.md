# BIURI / TREPAN Reloaded — regras permanentes do projeto

## 1. O sistema é AGNÓSTICO a datasets (regra inviolável)
- Em produção serão usados **outros datasets**, desconhecidos. Iris, Wine, Breast Cancer, Digits e os sintéticos são **apenas** o conjunto de
  validação/regressão offline; nunca definem o comportamento do núcleo.
- Proibido no núcleo (`core/`, `gui/`): `if dataset_name == ...`, hiperparâmetros por dataset, nomes de datasets conhecidos, nº fixo de
  features/classes, limites de recursos ou de capacidade ligados à identidade do dataset, atalhos "só para o Digits/Wine/...".
- Os datasets conhecidos vivem só em `validation/` (benchmarks em `validation/benchmark/`). Acrescentar um dataset = acrescentar **dados +
  metadados** (`DatasetSpec`), nunca alterar o algoritmo.
- Qualquer correção motivada por um dataset só entra no núcleo se for uma propriedade **geral** do problema/dados. Um problema que aparece
  só num dataset investiga-se como violação de uma regra genérica, não com um ramo específico.
- Otimizações de desempenho: genéricas (vectorização, cache com chave matemática exata, retoma/checkpoint), nunca por dataset; políticas de
  recursos são globais e configuráveis (`--max-wall-time-per-unit`), nunca trocam silenciosamente a configuração científica.
- Testes que impõem isto: `tests/test_dataset_agnosticism.py`, `tests/test_v92_static_guards.py`,
  `tests/test_benchmark_manifest_and_c45_guard.py::test_validation_benchmark_has_no_dataset_specific_branching`.

## 2. Desenho experimental congelado
- Não alterar (nem em resposta a resultados): tuning científico (CV 5×3, purity_epsilon, max_nodes, política de expansão, critérios
  lexicográficos e de estabilidade, bootstrap diagnóstico) nem o contrato do oráculo (FrozenOracle/oracle_id).
- Braços A–F, protocolo estrutural comum e ontology-blind, mirror desligado, C4.5 canónico: ver `validation/benchmark/BENCHMARK_MANIFEST_v*.json`.
- Nunca otimizar nem codificar para produzir a ordem esperada Reloaded > Original > C4.5; resultados inesperados são evidência a investigar.

## 3. Processo
- Commits pequenos e lógicos; `git status` antes de commitar; não versionar caches/temporários (`.pyc`, `results/` pesados).
- Merge para `main` só a pedido explícito. Não criar PR sem pedido.
- Resultados brutos são imutáveis e as tabelas reconstroem-se só deles; equivalência de otimizações provada com
  `validation/benchmark/equivalence_report.py` (comparação exata).
