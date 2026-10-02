# Migração para BIURI / TREPAN Reloaded V9.2

## Artefactos

A V9.2 introduz um formato de artefacto em directório:

- `manifest.json`
- `model.joblib`

O manifest contém a versão do formato, versões relevantes de bibliotecas, hashes, schema/contrato, configuração do preprocessor, classes, seeds, métricas e informação de convergência/calibração quando disponível.

`core.artifacts.load_artifact` valida compatibilidade e pode lançar `ArtifactCompatibilityError`.

## Pickles antigos

Pickles históricos podem depender de imports relativos ao antigo `sys.path` e não são garantidamente portáveis. Use:

```bash
python scripts/migrate_legacy_pickles_v92.py
```

A ferramenta apenas audita/tenta recuperar; não fabrica resultados. Nesta execução foram encontrados artefactos incompatíveis; veja `results/legacy_pickle_audit_v9_2.json` e o relatório de implementação.

## TREPAN Original e política de produção

A build de produção já não contém a antiga implementação de árvore destilada no caminho `core.trepan_extractor`. `TREPANExtractor` é apenas um alias de compatibilidade que instancia o TREPAN Original histórico (`core.trepan_original.TrepanOriginalExtractor`).

Não existe fallback silencioso para outra família de árvore. Artefactos de produção exigem `TrepanOriginalClassifier` e `TrepanReloadedClassifier` e usam a política `historical_trepan_only_no_cart`. Consulte `docs/PRODUCTION_TREE_POLICY_V9_2.md`.

## Hepatitis

O preparador histórico de Hepatitis usava a coluna errada como alvo e fazia imputação global antes do split. O alvo foi corrigido para a primeira coluna e a imputação global removida. Resultados históricos dependentes desse processamento foram marcados como obsoletos em `counterfactuals/experimentos/hepatitis/STALE.md` e não foram reescritos.

## Bootstrap de contrafactuais

`counterfactuals/_bootstrap.py` deixou de chamar `os.chdir()`. Caminhos de biblioteca devem ser resolvidos por `Path(__file__)`, evitando efeitos globais sobre o processo.

## Cache

A assinatura do dataset passa a considerar todo o conteúdo do DataFrame, em vez de uma amostra parcial de 256 linhas. Uma alteração numa única linha invalida a chave.

## Compatibilidade

O alvo declarado no `pyproject.toml` é Python 3.11–3.12. A matriz do CI contém ambas as versões. Nesta sessão local não estavam instalados Python 3.11/3.12; portanto essa compatibilidade não foi executada aqui.
