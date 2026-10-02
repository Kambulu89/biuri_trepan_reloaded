# BIURI / TREPAN Reloaded V9.2 — Produção

## Fluxo principal

`dados -> contrato -> split -> preprocessing fit no treino -> MLP Original -> TREPAN Original histórico -> TREPAN Reloaded histórico + OWL opcional -> avaliação pareada -> artefacto`

A OWL é opcional. Sem OWL, o Reloaded é o espelho do TREPAN Original. Com OWL, o mesmo MLP, seed, dados e orçamento são mantidos; apenas a estrutura semântica pode alterar/reinforçar a procura de splits.

## Estados independentes

1. **Ontology Quality**: validade lógica/schema/ABox.
2. **Semantic Feature Utility**: se features derivadas ajudam o professor em OOF.
3. **Semantic Contribution**: se a OWL alterou/reforçou decisões do TREPAN Reloaded e qual foi o efeito final.

Nunca interpretar `feature utility rejected` como `ontology invalid`.

## Métricas semânticas

- `Ontology Usage Rate`: fração de splits alterados ou reforçados.
- `Semantic Decision Impact`: fração de splits cujo vencedor mudou.
- `Mean Semantic Bonus`: bónus médio observado no score dos splits.
- `SemanticContributionGate`: combina impacto, fidelidade, desempenho e crescimento de complexidade.

## CLI

```powershell
python scripts/biuri_cli.py train-production --data dados.arff --target classe --out results/run_001
python scripts/biuri_cli.py train-production --data dados.arff --target classe --owl dominio.owl --out results/run_owl_001
python scripts/biuri_cli.py predict-production --artifact results/run_001 --data novos.csv --model mlp
python scripts/biuri_cli.py explain-production --artifact results/run_001 --data novos.csv --model trepan_reloaded
python scripts/run_reloaded_ablation_v92.py --data dados.arff --target classe --owl dominio.owl --out results/ablation_v9_2_prod --seeds 11,23,42,67,101
```

## Dependências

- núcleo: `requirements-core.txt`
- GUI: `requirements-gui.txt`
- OWL/HermiT: `requirements-owl.txt` + Java 17
- CLEAR/TensorFlow: `requirements-clear.txt`
- Excel/Parquet: `requirements-excel-parquet.txt`
- testes: `requirements-test.txt`

Preflight:

```powershell
python scripts/environment_preflight.py --gui --owl
```

Python 3.11 e 3.12 são os alvos de produção/CI.

## Claim guard

Resultados pontuais não autorizam alegação de superioridade. `production_report.json` inclui `scientific_validation.claim_guard`; somente `superiority_supported` autoriza essa formulação no protocolo correspondente.
